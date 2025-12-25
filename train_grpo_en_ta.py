from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Sequence

import statistics
import re

import torch
from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, Gemma3ForCausalLM
from trl import GRPOTrainer, GRPOConfig

from rewards_en_ta_dupo import CometDuPOReward


END_TURN = "<end_of_turn>"
CTRL_RE = re.compile(r"<start_of_turn>|<end_of_turn>|user|model")


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
    group_size: int = 8
    temperature: float = 0.8
    max_new_tokens: int = 256
    lr: float = 1e-6
    kl_beta: float = 0.05
    gradient_accumulation_steps: int = 4
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 200           # set to max_steps to save only once
    save_total_limit: int = 1       # keep disk use low
    resume_from_checkpoint: str = ""  # local path or HF repo id

    # Eval / benchmarking
    num_eval_samples: int = 64
    eval_batch_size: int = 12
    eval_temperature: float = 0.8
    eval_max_new_tokens: int = 256
    eval_seed: int = 1234

    # ---- Weights & Biases logging ----
    use_wandb: bool = True

    # Paste your key here if you want (DO NOT COMMIT THIS FILE if you do)
    wandb_api_key: str = ""  # e.g. "abcd1234..."; leave "" to use env var / existing login

    wandb_project: str = "rl-nmt-en-ta-dupo"
    wandb_entity: str = ""   # optional; set if your org requires it
    wandb_run_name: str = "gemma3-1b-it-grpo-en-ta"
    wandb_mode: str = "online"  # "online" or "offline"

    # ---- Hugging Face Hub push ----
    push_to_hub: bool = True
    hub_model_id: str = "Saran-Gangster/gemma3-en-ta-grpo"       # e.g. "yourname/gemma3-en-ta-grpo"
    hub_token: str = ""          # optional; else uses HF_TOKEN / ~/.huggingface
    hub_private: bool = True


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


def materialize_eval_dataset(cfg: RunCfg) -> Dataset:
    stream = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.dataset_split,
        streaming=True,
    )
    if cfg.shuffle_buffer and cfg.shuffle_buffer > 0:
        stream = stream.shuffle(seed=cfg.eval_seed, buffer_size=cfg.shuffle_buffer)

    rows: List[Dict[str, str]] = []
    for ex in stream.take(cfg.num_eval_samples):
        text = (ex.get(cfg.text_field) or "").strip()
        if not text:
            continue
        text = text[: cfg.max_chars]
        rows.append({"prompt": make_gemma_prompt(text)})

    if not rows:
        raise RuntimeError("No eval examples materialized; check dataset/text_field.")

    return Dataset.from_list(rows)


def _strip_control(text: str) -> str:
    text = text.replace("\u200b", "")
    text = CTRL_RE.sub("", text).strip()
    if END_TURN in text:
        text = text.split(END_TURN, 1)[0]
    return " ".join(text.split()).strip()


@torch.inference_mode()
def benchmark_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    reward_fn: CometDuPOReward,
    prompts: Sequence[str],
    cfg: RunCfg,
    device: str,
    label: str,
) -> Dict[str, float]:
    rewards: List[float] = []
    model.eval()

    for idx in range(0, len(prompts), cfg.eval_batch_size):
        batch_prompts = prompts[idx : idx + cfg.eval_batch_size]
        enc = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        input_len = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=cfg.eval_temperature,
            max_new_tokens=cfg.eval_max_new_tokens,
        )

        gen_ids = gen[:, input_len:]
        completions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        completions = [_strip_control(c) for c in completions]

        batch_rewards = reward_fn(list(batch_prompts), completions)
        rewards.extend(batch_rewards)

    mean_r = statistics.mean(rewards) if rewards else 0.0
    std_r = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0

    print(f"[benchmark:{label}] n={len(rewards)} mean={mean_r:.4f} std={std_r:.4f}")

    return {
        "label": label,
        "count": float(len(rewards)),
        "mean": float(mean_r),
        "std": float(std_r),
    }


def _append_benchmark_line(metrics: Dict[str, float], path: str) -> None:
    # Append a single benchmark line for easy offline inspection
    header = "label\tcount\tmean\tstd\n"
    line = f"{metrics['label']}\t{metrics['count']:.0f}\t{metrics['mean']:.4f}\t{metrics['std']:.4f}\n"
    needs_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if needs_header:
            f.write(header)
        f.write(line)


def setup_wandb(cfg: RunCfg):
    if not cfg.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return None

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

    import wandb

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        name=cfg.wandb_run_name,
        config={
            "seed": cfg.seed,
            "max_steps": cfg.max_steps,
            "model_id": cfg.model_id,
            "dataset": cfg.dataset_name,
        },
        reinit=True,
    )

    return run


def main() -> None:
    cfg = RunCfg()
    os.makedirs(cfg.output_dir, exist_ok=True)

    wandb_run = setup_wandb(cfg)

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
    eval_ds = materialize_eval_dataset(cfg)
    eval_prompts = list(eval_ds["prompt"])
    bench_path = os.path.join(cfg.output_dir, "benchmark.txt")

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

    pre_metrics = benchmark_model(
        model=model,
        tokenizer=tok,
        reward_fn=reward_fn,
        prompts=eval_prompts,
        cfg=cfg,
        device=device,
        label="pretrain",
    )
    _append_benchmark_line(pre_metrics, bench_path)

    if cfg.use_wandb:
        import wandb

        wandb.log(
            {
                "benchmark/pre/mean": pre_metrics["mean"],
                "benchmark/pre/std": pre_metrics["std"],
                "benchmark/pre/count": pre_metrics["count"],
            },
            step=0,
        )

    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,
        learning_rate=cfg.lr,
        beta=cfg.kl_beta,
        generation_batch_size=cfg.group_size,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.group_size,
        temperature=cfg.temperature,
        max_completion_length=cfg.max_new_tokens,
        bf16=(device == "cuda"),
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        remove_unused_columns=False,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id or None,
        hub_token=cfg.hub_token or None,
        hub_private_repo=cfg.hub_private,

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

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint or None)
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)

    post_metrics = benchmark_model(
        model=trainer.model,
        tokenizer=tok,
        reward_fn=reward_fn,
        prompts=eval_prompts,
        cfg=cfg,
        device=device,
        label="posttrain",
    )
    _append_benchmark_line(post_metrics, bench_path)

    if cfg.use_wandb:
        import wandb

        wandb.log(
            {
                "benchmark/post/mean": post_metrics["mean"],
                "benchmark/post/std": post_metrics["std"],
                "benchmark/post/count": post_metrics["count"],
                "benchmark/pre/mean": pre_metrics["mean"],
            },
            step=trainer.state.global_step,
        )

    if wandb_run is not None:
        wandb_run.finish()

    if cfg.push_to_hub:
        print(f"Pushing to hub repo: {cfg.hub_model_id} (private={cfg.hub_private})")
        trainer.push_to_hub(
            repo_id=cfg.hub_model_id or None,
            token=cfg.hub_token or None,
            private=cfg.hub_private,
        )


if __name__ == "__main__":
    main()