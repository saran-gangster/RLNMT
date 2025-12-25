# train_grpo_gemma_en_ta_opus100_flores.py
from __future__ import annotations

import os
import re
import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, Gemma3ForCausalLM, set_seed
from trl import GRPOTrainer, GRPOConfig

from rewards_en_ta_refcomet import CometRefQeReward

END_TURN = "<end_of_turn>"
CTRL_RE = re.compile(r"<start_of_turn>|<end_of_turn>|user|model")


@dataclass
class RunCfg:
    # Model
    model_id: str = "google/gemma-3-1b-it"

    # Train dataset: OPUS-100 en-ta
    train_dataset_id: str = "Helsinki-NLP/opus-100"
    train_config: str = "en-ta"
    train_split: str = "train"

    # ---- Small-data regime (2k–5k) ----
    num_train_samples: int = 3000
    shuffle_buffer: int = 10_000

    # Sentence-ish filters
    min_chars: int = 5
    max_chars: int = 220

    # GRPO
    output_dir: str = "./grpo_gemma_en_ta_opus100"
    max_steps: int = 0  # 0 => auto: ceil(num_train_samples / gradient_accumulation_steps)
    group_size: int = 8
    temperature: float = 0.6
    max_new_tokens: int = 128
    lr: float = 5e-7
    kl_beta: float = 0.15
    gradient_accumulation_steps: int = 4
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 0       # 0 => auto: save once at end
    save_total_limit: int = 2
    resume_from_checkpoint: str = ""

    # Eval: FLORES-200 eng_Latn-tam_Taml
    flores_dataset_candidates: Tuple[str, ...] = ("facebook/flores", "facebook/flores200", "Muennighoff/flores200")
    flores_pair_config: str = "eng_Latn-tam_Taml"
    flores_split: str = "devtest"  # "dev" or "devtest"
    num_eval_samples: int = 128
    eval_batch_size: int = 16
    eval_max_new_tokens: int = 128
    eval_seed: int = 1234

    # W&B via HF/TRL only
    use_wandb: bool = True
    wandb_api_key: str = ""
    wandb_project: str = "rl-nmt-en-ta-dupo"
    wandb_entity: str = ""
    wandb_run_name: str = "gemma3-1b-it-grpo-opus100-en-ta"
    wandb_mode: str = "online"

    # Hub
    push_to_hub: bool = True
    hub_model_id: str = "Saran-Gangster/gemma3-en-ta-grpo-opus100"
    hub_token: str = ""
    hub_private: bool = True


def setup_wandb_env(cfg: RunCfg) -> None:
    if not cfg.use_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    if cfg.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", cfg.wandb_entity)
    os.environ.setdefault("WANDB_MODE", cfg.wandb_mode)
    if cfg.wandb_api_key:
        os.environ["WANDB_API_KEY"] = cfg.wandb_api_key


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


def _strip_control(text: str) -> str:
    text = (text or "").replace("\u200b", "")
    text = CTRL_RE.sub("", text).strip()
    if END_TURN in text:
        text = text.split(END_TURN, 1)[0]
    return " ".join(text.split()).strip()


def _get_en_ta_from_opus(ex: Dict[str, Any]) -> Tuple[str, str]:
    # Handles both:
    # 1) {"translation": {"en": "...", "ta": "..."}}
    # 2) {"en": "...", "ta": "..."}
    if isinstance(ex.get("translation"), dict):
        en = ex["translation"].get("en", "") or ""
        ta = ex["translation"].get("ta", "") or ""
        return en, ta
    return (ex.get("en", "") or ""), (ex.get("ta", "") or "")


def _load_dataset_trust(*args, **kwargs):
    """
    Newer `datasets` requires `trust_remote_code=True` to run hub dataset scripts (like FLORES).
    For older versions that don't accept it, we remove it.
    """
    try:
        return load_dataset(*args, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        return load_dataset(*args, **kwargs)


def materialize_opus100_train(cfg: RunCfg) -> Dataset:
    stream = _load_dataset_trust(
        cfg.train_dataset_id,
        cfg.train_config,
        split=cfg.train_split,
        streaming=True,
    )

    if cfg.shuffle_buffer and cfg.shuffle_buffer > 0:
        stream = stream.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)

    rows: List[Dict[str, str]] = []
    take_cap = cfg.num_train_samples * 8

    for ex in stream.take(take_cap):
        en, ta = _get_en_ta_from_opus(ex)
        en = " ".join((en or "").split()).strip()
        ta = " ".join((ta or "").split()).strip()

        if not en or not ta:
            continue
        if len(en) < cfg.min_chars or len(en) > cfg.max_chars:
            continue

        rows.append({"prompt": make_gemma_prompt(en), "ref_ta": ta})
        if len(rows) >= cfg.num_train_samples:
            break

    if not rows:
        raise RuntimeError("No train examples materialized from OPUS-100. Check config/split.")
    return Dataset.from_list(rows)


def load_flores200_stream(cfg: RunCfg):
    last_err: Optional[Exception] = None
    for ds_id in cfg.flores_dataset_candidates:
        try:
            return _load_dataset_trust(
                ds_id,
                cfg.flores_pair_config,
                split=cfg.flores_split,
                streaming=True,
            )
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Could not load FLORES-200 from any of: {cfg.flores_dataset_candidates} "
        f"with config={cfg.flores_pair_config} split={cfg.flores_split}. Last error: {last_err}"
    )


def materialize_flores_eval(cfg: RunCfg) -> Dataset:
    stream = load_flores200_stream(cfg)

    rows: List[Dict[str, str]] = []
    take_cap = cfg.num_eval_samples * 5

    for ex in stream.take(take_cap):
        # For paired config eng_Latn-tam_Taml, these are the common column names
        en = (ex.get("sentence_eng_Latn") or "").strip()
        ta = (ex.get("sentence_tam_Taml") or "").strip()
        if not en or not ta:
            continue

        en = " ".join(en.split()).strip()
        ta = " ".join(ta.split()).strip()

        if len(en) < cfg.min_chars:
            continue
        en = en[: cfg.max_chars].strip()
        if len(en) < cfg.min_chars:
            continue

        rows.append({"prompt": make_gemma_prompt(en), "ref_ta": ta})
        if len(rows) >= cfg.num_eval_samples:
            break

    if not rows:
        raise RuntimeError("No FLORES eval examples materialized; check dataset/config/split.")
    return Dataset.from_list(rows)


def _from_pretrained_with_dtype(model_id: str, dtype, device: str):
    # Your env warns torch_dtype is deprecated, so prefer dtype=...
    try:
        return Gemma3ForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    except TypeError:
        # fallback for older transformers
        return Gemma3ForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)


@torch.inference_mode()
def benchmark_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    reward_fn: CometRefQeReward,
    prompts: Sequence[str],
    refs: Sequence[str],
    cfg: RunCfg,
    device: str,
    label: str,
) -> Dict[str, float]:
    model.eval()
    set_seed(cfg.eval_seed)

    end_turn_id = tokenizer.convert_tokens_to_ids(END_TURN)
    eos_ids = [tokenizer.eos_token_id]
    if isinstance(end_turn_id, int) and end_turn_id >= 0 and end_turn_id != tokenizer.unk_token_id:
        eos_ids.append(end_turn_id)

    rewards: List[float] = []

    for i in range(0, len(prompts), cfg.eval_batch_size):
        batch_prompts = list(prompts[i : i + cfg.eval_batch_size])
        batch_refs = list(refs[i : i + cfg.eval_batch_size])

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        input_len = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            do_sample=False,
            num_beams=4,
            max_new_tokens=cfg.eval_max_new_tokens,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        gen_ids = gen[:, input_len:]
        completions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        completions = [_strip_control(c) for c in completions]

        batch_rewards = reward_fn(batch_prompts, completions, ref_ta=batch_refs)
        rewards.extend(batch_rewards)

    mean_r = statistics.mean(rewards) if rewards else 0.0
    std_r = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    print(f"[benchmark:{label}] n={len(rewards)} mean={mean_r:.4f} std={std_r:.4f}")
    return {"label": label, "count": float(len(rewards)), "mean": float(mean_r), "std": float(std_r)}


def _append_benchmark_line(metrics: Dict[str, float], path: str) -> None:
    header = "label\tcount\tmean\tstd\n"
    line = f"{metrics['label']}\t{metrics['count']:.0f}\t{metrics['mean']:.4f}\t{metrics['std']:.4f}\n"
    needs_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if needs_header:
            f.write(header)
        f.write(line)


def main() -> None:
    cfg = RunCfg()
    os.makedirs(cfg.output_dir, exist_ok=True)
    setup_wandb_env(cfg)
    set_seed(cfg.seed)

    # auto steps for 2k–5k regime
    if cfg.max_steps <= 0:
        cfg.max_steps = int(math.ceil(cfg.num_train_samples / max(1, cfg.gradient_accumulation_steps)))
    if cfg.save_steps <= 0:
        cfg.save_steps = cfg.max_steps

    print(f"[config] num_train_samples={cfg.num_train_samples} grad_accum={cfg.gradient_accumulation_steps} -> max_steps={cfg.max_steps}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = _from_pretrained_with_dtype(cfg.model_id, dtype=dtype, device=device).eval()

    # Encourage stopping at <end_of_turn>
    end_turn_id = tok.convert_tokens_to_ids(END_TURN)
    eos_ids = [tok.eos_token_id]
    if isinstance(end_turn_id, int) and end_turn_id >= 0 and end_turn_id != tok.unk_token_id:
        eos_ids.append(end_turn_id)
    model.generation_config.eos_token_id = eos_ids
    model.generation_config.pad_token_id = tok.pad_token_id

    train_ds = materialize_opus100_train(cfg)
    eval_ds = materialize_flores_eval(cfg)

    eval_prompts = list(eval_ds["prompt"])
    eval_refs = list(eval_ds["ref_ta"])
    bench_path = os.path.join(cfg.output_dir, "benchmark.txt")

    reward_fn = CometRefQeReward(
        ref_metric_name="Unbabel/wmt22-comet-da",
        qe_metric_name="Unbabel/wmt22-cometkiwi-da",
        normalize="auto",
        reward_device=device,
        comet_batch_size=32,
        ref_key="ref_ta",
        w_ref=0.85,
        w_qe=0.15,
        penalty_weight_script=0.25,
        penalty_weight_lenratio=0.05,
    )

    pre = benchmark_model(
        model=model,
        tokenizer=tok,
        reward_fn=reward_fn,
        prompts=eval_prompts,
        refs=eval_refs,
        cfg=cfg,
        device=device,
        label="pretrain_flores",
    )
    _append_benchmark_line(pre, bench_path)

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

        report_to=(["wandb"] if cfg.use_wandb else ["none"]),
        run_name=cfg.wandb_run_name,

        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id or None,
        hub_token=cfg.hub_token or None,
        hub_private_repo=cfg.hub_private,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_ds,
        processing_class=tok,
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint or None)
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)

    post = benchmark_model(
        model=trainer.model,
        tokenizer=tok,
        reward_fn=reward_fn,
        prompts=eval_prompts,
        refs=eval_refs,
        cfg=cfg,
        device=device,
        label="posttrain_flores",
    )
    _append_benchmark_line(post, bench_path)

    if cfg.push_to_hub:
        print(f"Pushing to hub repo: {cfg.hub_model_id} (private={cfg.hub_private})")
        trainer.push_to_hub(
            repo_id=cfg.hub_model_id or None,
            token=cfg.hub_token or None,
            private=cfg.hub_private,
        )


if __name__ == "__main__":
    main()