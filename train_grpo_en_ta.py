from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, Gemma3ForCausalLM
from trl import GRPOTrainer, GRPOConfig

from rewards_en_ta_dupo import CometDuPOReward


@dataclass
class RunCfg:
    # Model
    model_id: str = "google/gemma-3-1b-it"

    # Monolingual English data (stream then materialize only N rows)
    dataset_name: str = "allenai/c4"
    dataset_config: str = "en"
    dataset_split: str = "train"
    text_field: str = "text"
    num_train_samples: int = 2000          # set 1000–3000
    shuffle_buffer: int = 10_000
    max_chars: int = 800

    # GRPO run
    output_dir: str = "./grpo_gemma_en_ta_ckpts"
    max_steps: int = 200
    group_size: int = 4
    temperature: float = 0.8
    max_new_tokens: int = 256
    lr: float = 1e-6
    kl_beta: float = 0.05
    gradient_accumulation_steps: int = 4
    seed: int = 42

    # ---- Weights & Biases logging ----
    use_wandb: bool = True

    # Paste your key here if you want (DO NOT COMMIT THIS FILE if you do)
    wandb_api_key: str = ""  # e.g. "abcd1234..."; leave "" to use env var / existing login

    wandb_project: str = "rl-nmt-en-ta-dupo"
    wandb_entity: str = ""   # optional; set if your org requires it
    wandb_run_name: str = "gemma3-1b-it-grpo-en-ta"
    wandb_mode: str = "online"  # "online" or "offline"


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

    if not rows:
        raise RuntimeError("No examples materialized; check dataset/text_field.")

    return Dataset.from_list(rows)


def setup_wandb(cfg: RunCfg) -> None:
    if not cfg.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return

    # Avoid tokenizer fork warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    if cfg.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", cfg.wandb_entity)
    os.environ.setdefault("WANDB_MODE", cfg.wandb_mode)

    # If user pasted the key, use it (don’t print it)
    if cfg.wandb_api_key:
        os.environ["WANDB_API_KEY"] = cfg.wandb_api_key

        # Optional explicit login (more reliable than relying on env alone)
        import wandb_reward_callback
        wandb_reward_callback.login(key=cfg.wandb_api_key, relogin=True)


def main() -> None:
    cfg = RunCfg()
    os.makedirs(cfg.output_dir, exist_ok=True)

    setup_wandb(cfg)

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
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

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
        seed=cfg.seed,
        remove_unused_columns=False,

        # W&B
        report_to=(["wandb"] if cfg.use_wandb else ["none"]),
        run_name=cfg.wandb_run_name,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=ds,
        processing_class=tok,
    )
    from wandb_reward_callback import WandbRewardCallback
    trainer.add_callback(WandbRewardCallback(reward_fn=reward_fn, log_every=1, ema_alpha=0.05))

    trainer.train()
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()