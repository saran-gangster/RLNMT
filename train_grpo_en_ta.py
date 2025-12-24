# train_grpo_gemma_en_ta.py
# GRPO training for English->Tamil with Gemma-3-1B-IT, DuPO reward (cycle COMET-DA + QE COMETKiwi)

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from datasets import load_dataset

from transformers import AutoTokenizer, Gemma3ForCausalLM

from trl import GRPOTrainer, GRPOConfig

from rewards_en_ta_dupo import CometDuPOReward


@dataclass
class RunCfg:
    model_id: str = "google/gemma-3-1b-it"

    # Monolingual English source dataset (swap to WMT News monolingual if desired)
    dataset_name: str = "allenai/c4"
    dataset_config: str = "en"
    dataset_split: str = "train[:20000]"
    text_field: str = "text"

    output_dir: str = "./grpo_gemma_en_ta_ckpts"
    max_steps: int = 200

    # GRPO / sampling
    group_size: int = 4
    temperature: float = 0.8
    max_new_tokens: int = 256

    # optimization
    lr: float = 1e-6
    kl_beta: float = 0.05
    gradient_accumulation_steps: int = 4

    seed: int = 42


def make_gemma_prompt(english: str) -> str:
    """
    Gemma IT models prefer the <start_of_turn> format.
    No system role: include "system-like" rules inside the user turn.
    """
    english = " ".join((english or "").split()).strip()
    # Keep prompts sane; you can also truncate by chars here if needed.
    return (
        "<start_of_turn>user\n"
        "Only reply with the Tamil translation. Do not add explanations.\n\n"
        "Translate the following English text into natural, fluent Tamil.\n\n"
        f"English: {english}\n"
        "Tamil:\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def main() -> None:
    cfg = RunCfg()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Tokenizer / model (bf16 on A100)
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = Gemma3ForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    # Dataset
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)

    def to_prompt(ex):
        en = ex.get(cfg.text_field, "")
        return {"prompt": make_gemma_prompt(en)}

    ds = ds.map(to_prompt, remove_columns=ds.column_names)

    # Reward function (Fix #1 applied)
    reward_fn = CometDuPOReward(
        cycle_metric_name="Unbabel/wmt22-comet-da",
        qe_metric_name="Unbabel/wmt22-cometkiwi-da",
        dual_model_name="facebook/nllb-200-distilled-1.3B",
        w_cycle=0.7,
        w_qe=0.3,
        normalize="linear",   # less squashed than sigmoid; change if you prefer
        reward_device="cuda" if torch.cuda.is_available() else "cpu",
        dual_device="cuda" if torch.cuda.is_available() else "cpu",
        comet_batch_size=16,
        dual_dtype_bf16=True,
    )

    # TRL config (Fix #2 applied)
    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,
        learning_rate=cfg.lr,
        beta=cfg.kl_beta,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        num_generations=cfg.group_size,
        temperature=cfg.temperature,

        # new-ish TRL name; some versions call this max_completion_length
        max_completion_length=cfg.max_new_tokens,

        bf16=True,
        logging_steps=10,
        save_steps=50,
        report_to="tensorboard",
        seed=cfg.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=ds,
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)


if __name__ == "__main__":
    main()