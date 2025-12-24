from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

import torch
from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, Gemma3ForCausalLM
from trl import GRPOTrainer, GRPOConfig

from rewards_en_ta_dupo import CometDuPOReward


@dataclass
class RunCfg:
    model_id: str = "google/gemma-3-1b-it"

    # C4 streaming source (won't download full dataset)
    dataset_name: str = "allenai/c4"
    dataset_config: str = "en"
    dataset_split: str = "train"
    text_field: str = "text"

    # Only materialize this many samples into a real Dataset
    num_train_samples: int = 2000   # set 1000–3000
    shuffle_buffer: int = 10_000    # streaming shuffle buffer; reduce if RAM is tight

    # Prompt truncation to keep examples small + stable
    max_chars: int = 800

    output_dir: str = "./grpo_gemma_en_ta_ckpts"
    max_steps: int = 200

    group_size: int = 4
    temperature: float = 0.8
    max_new_tokens: int = 256

    lr: float = 1e-6
    kl_beta: float = 0.05
    gradient_accumulation_steps: int = 4
    seed: int = 42


def make_gemma_prompt(english: str) -> str:
    english = " ".join((english or "").split()).strip()
    return (
        "<start_of_turn>user\n"
        "Only reply with the Tamil translation. Do not add explanations.\n\n"
        "Translate the following English text into natural, fluent Tamil.\n\n"
        f"English: {english}\n"
        "Tamil:\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def materialize_small_dataset(cfg: RunCfg) -> Dataset:
    """
    Streams C4, shuffles approximately, takes N, then builds a small map-style Dataset.
    This avoids downloading the entire C4 Arrow dataset to disk.
    """
    stream = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.dataset_split,
        streaming=True,
    )

    if cfg.shuffle_buffer and cfg.shuffle_buffer > 0:
        stream = stream.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)

    rows: List[Dict[str, str]] = []
    for ex in stream.take(cfg.num_train_samples):
        text = (ex.get(cfg.text_field) or "").strip()
        if not text:
            continue
        text = text[: cfg.max_chars]
        rows.append({"prompt": make_gemma_prompt(text)})

    if len(rows) == 0:
        raise RuntimeError("No examples were materialized. Check dataset/text_field settings.")

    return Dataset.from_list(rows)


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = RunCfg()
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    # Tokenizer / model (bf16 on A100)
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = Gemma3ForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,  # dtype (not torch_dtype)
    ).to(device)

    # Build a tiny map-style dataset from streaming
    ds = materialize_small_dataset(cfg)

    reward_fn = CometDuPOReward(
        cycle_metric_name="Unbabel/wmt22-comet-da",
        qe_metric_name="Unbabel/wmt22-cometkiwi-da",
        dual_model_name="facebook/nllb-200-distilled-1.3B",
        normalize="identity",
        reward_device=device,
        dual_device=device,
        dual_dtype_bf16=(device == "cuda"),
        dual_use_safetensors=True,
    )

    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,
        learning_rate=cfg.lr,
        beta=cfg.kl_beta,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        num_generations=cfg.group_size,
        temperature=cfg.temperature,
        max_completion_length=cfg.max_new_tokens,

        bf16=(device == "cuda"),
        logging_steps=10,
        save_steps=50,
        report_to="tensorboard",
        seed=cfg.seed,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=ds,          # <-- now a standard Dataset, GRPOTrainer accepts it
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()