# train_grpo_gemma_en_ta_opus100_flores.py
from __future__ import annotations

import os
import re
import math
import statistics
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional, Set

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, Gemma3ForCausalLM, set_seed
from trl import GRPOTrainer, GRPOConfig

from rewards_en_ta_refcomet import CometRefQeReward

END_TURN = "<end_of_turn>"
CTRL_RE = re.compile(r"<start_of_turn>|<end_of_turn>|user|model")
_CHECKPOINT_DIR_RE = re.compile(r"(?:^|/)(checkpoint-[0-9A-Za-z_.-]+)(?:/|$)")
_LOCAL_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

@dataclass
class RunCfg:
    model_id: str = "google/gemma-3-1b-it"

    # training dataset
    train_dataset_id: str = "Helsinki-NLP/opus-100"
    train_config: str = "en-ta"
    train_split: str = "train"

    num_train_samples: int = 3000
    shuffle_buffer: int = 10_000
    min_chars: int = 5
    max_chars: int = 220

    # GRPO / training
    output_dir: str = "./grpo_gemma_en_ta_opus100"
    max_steps: int = 0
    group_size: int = 8
    temperature: float = 0.6
    max_new_tokens: int = 128
    lr: float = 5e-7
    kl_beta: float = 0.15
    gradient_accumulation_steps: int = 4
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 0
    save_total_limit: int = 2
    resume_from_checkpoint: str = ""  

    # FLORES eval
    flores_local_dir: str = "./flores200_dataset"
    flores_lang_en: str = "eng_Latn"
    flores_lang_ta: str = "tam_Taml"
    flores_split: str = "devtest"
    num_eval_samples: int = 128
    eval_batch_size: int = 16
    eval_max_new_tokens: int = 128
    eval_seed: int = 1234

    # wandb/hub
    use_wandb: bool = True
    wandb_api_key: str = ""
    wandb_project: str = "rl-nmt-en-ta-dupo"
    wandb_entity: str = ""
    wandb_run_name: str = "gemma3-1b-it-grpo-opus100-en-ta"
    wandb_mode: str = "online"

    push_to_hub: bool = True
    hub_model_id: str = "Saran-Gangster/gemma3-en-ta-grpo-opus100"
    hub_token: str = ""  

    push_checkpoint_to_hub: bool = True


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
    if isinstance(ex.get("translation"), dict):
        en = ex["translation"].get("en", "") or ""
        ta = ex["translation"].get("ta", "") or ""
        return en, ta
    return (ex.get("en", "") or ""), (ex.get("ta", "") or "")


def _load_dataset_trust(*args, **kwargs):
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


def download_flores200(local_dir: str) -> None:
    import tarfile
    import urllib.request

    flores_data_dir = os.path.join(local_dir, "flores200_dataset")
    if os.path.exists(flores_data_dir):
        print(f"[flores] Found existing directory: {flores_data_dir}")
        return

    print(f"[flores] Downloading FLORES-200 to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    tar_url = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
    tar_path = os.path.join(local_dir, "flores200_dataset.tar.gz")

    try:
        print(f"[flores] Downloading from {tar_url}...")
        urllib.request.urlretrieve(tar_url, tar_path)
        print(f"[flores] Extracting {tar_path}...")

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=local_dir)

        os.remove(tar_path)
        print(f"[flores] Download complete: {flores_data_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to download FLORES-200: {e}")


def load_flores200_local(cfg: RunCfg) -> List[Tuple[str, str]]:
    download_flores200(cfg.flores_local_dir)

    flores_base = os.path.join(cfg.flores_local_dir, "flores200_dataset", cfg.flores_split)
    en_file = os.path.join(flores_base, f"{cfg.flores_lang_en}.{cfg.flores_split}")
    ta_file = os.path.join(flores_base, f"{cfg.flores_lang_ta}.{cfg.flores_split}")

    if not os.path.exists(en_file) or not os.path.exists(ta_file):
        raise FileNotFoundError(
            f"Could not find FLORES files:\n  EN: {en_file}\n  TA: {ta_file}"
        )

    with open(en_file, "r", encoding="utf-8") as f:
        en_lines = [line.strip() for line in f if line.strip()]

    with open(ta_file, "r", encoding="utf-8") as f:
        ta_lines = [line.strip() for line in f if line.strip()]

    if len(en_lines) != len(ta_lines):
        raise ValueError(
            f"Mismatched line counts: EN={len(en_lines)}, TA={len(ta_lines)}"
        )

    return list(zip(en_lines, ta_lines))


def materialize_flores_eval(cfg: RunCfg) -> Dataset:
    pairs = load_flores200_local(cfg)

    rows: List[Dict[str, str]] = []
    for en, ta in pairs[: cfg.num_eval_samples]:
        en = " ".join(en.split()).strip()
        ta = " ".join(ta.split()).strip()

        if not en or not ta:
            continue
        if len(en) < cfg.min_chars:
            continue

        en = en[: cfg.max_chars].strip()
        if len(en) < cfg.min_chars:
            continue

        rows.append({"prompt": make_gemma_prompt(en), "ref_ta": ta})

    if not rows:
        raise RuntimeError("No FLORES eval examples materialized.")

    print(f"[flores] Materialized {len(rows)} eval examples")
    return Dataset.from_list(rows)


def _from_pretrained_with_dtype_and_subfolder(model_id: str, dtype, device: str, subfolder: Optional[str] = None, **kwargs):
    load_kwargs = kwargs.copy()
    if subfolder:
        load_kwargs["subfolder"] = subfolder
    try:
        return Gemma3ForCausalLM.from_pretrained(model_id, dtype=dtype, **load_kwargs).to(device)
    except TypeError:
        return Gemma3ForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **load_kwargs).to(device)


def find_checkpoint_subfolder_in_hub(hub_model_id: str, token: Optional[str] = None) -> Optional[str]:
    try:
        from huggingface_hub import HfApi
    except Exception:
        print("[hub] huggingface_hub not installed; cannot inspect hub for checkpoints.")
        return None

    if token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=hub_model_id)
    except Exception as e:
        print(f"[hub] Failed to list repo files for {hub_model_id}: {e}")
        return None

    checkpoint_names: Set[str] = set()
    for p in files:
        m = _CHECKPOINT_DIR_RE.search(p)
        if m:
            checkpoint_names.add(m.group(1))

    if not checkpoint_names:
        return None

    numeric = []
    non_numeric = []
    for name in checkpoint_names:
        m = re.match(r"checkpoint-(\d+)$", name)
        if m:
            numeric.append((int(m.group(1)), name))
        else:
            non_numeric.append(name)

    if numeric:
        numeric.sort(key=lambda x: x[0], reverse=True)
        return numeric[0][1]
    else:
        non_numeric.sort(reverse=True)
        return non_numeric[0]


def find_latest_local_checkpoint(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for d in os.listdir(output_dir):
        full = os.path.join(output_dir, d)
        if os.path.isdir(full) and d.startswith("checkpoint-"):
            m = _LOCAL_CKPT_RE.match(d)
            if m:
                candidates.append((int(m.group(1)), d))
            else:
                candidates.append((None, d))
    if not candidates:
        return None
    # prefer numeric largest
    numeric = [c for c in candidates if c[0] is not None]
    if numeric:
        numeric.sort(key=lambda x: x[0], reverse=True)
        return os.path.join(output_dir, numeric[0][1])
    # fallback to lexicographic
    candidates.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(output_dir, candidates[0][1])


def upload_checkpoint_folder_to_hub(hub_model_id: str, local_ckpt_path: str, token: Optional[str] = None) -> bool:
    """
    Uploads the entire local checkpoint folder to <hub_model_id>/<checkpoint-folder-name>.
    Returns True on success, False otherwise.
    """
    try:
        from huggingface_hub import HfApi
    except Exception:
        print("[hub-upload] huggingface_hub not installed; cannot upload checkpoint.")
        return False

    if token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
    api = HfApi()

    ckpt_name = os.path.basename(local_ckpt_path.rstrip("/"))
    print(f"[hub-upload] Uploading local checkpoint {local_ckpt_path} to repo {hub_model_id} as path '{ckpt_name}'")

    try:
        api.upload_folder(
            folder_path=local_ckpt_path,
            repo_id=hub_model_id,
            path_in_repo=ckpt_name,
            repo_type="model",
            token=token,
            commit_message=f"Upload checkpoint folder {ckpt_name}",
        )
        print(f"[hub-upload] Successfully uploaded checkpoint folder '{ckpt_name}' to {hub_model_id}")
        return True
    except Exception as e:
        print(f"[hub-upload] Error uploading checkpoint folder: {e}")
        return False


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

    if cfg.max_steps <= 0:
        cfg.max_steps = int(math.ceil(cfg.num_train_samples / max(1, cfg.gradient_accumulation_steps)))
    if cfg.save_steps <= 0:
        cfg.save_steps = cfg.max_steps

    print(f"[config] num_train_samples={cfg.num_train_samples} grad_accum={cfg.gradient_accumulation_steps} -> max_steps={cfg.max_steps}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    checkpoint_subfolder = None
    if cfg.hub_model_id:
        checkpoint_subfolder = find_checkpoint_subfolder_in_hub(cfg.hub_model_id, token=cfg.hub_token)
        if checkpoint_subfolder:
            print(f"[hub] Found checkpoint subfolder in hub: {checkpoint_subfolder}")
        else:
            print(f"[hub] No checkpoint-* subfolder in hub; will use base model {cfg.model_id}")

    local_resume = cfg.resume_from_checkpoint or None
    if local_resume is None:
        latest_local_ckpt = find_latest_local_checkpoint(cfg.output_dir)
        if latest_local_ckpt:
            local_resume = latest_local_ckpt
            print(f"[local] Found local checkpoint to resume: {local_resume}")

    tok_source = cfg.hub_model_id if checkpoint_subfolder else cfg.model_id
    tok_subfolder = checkpoint_subfolder if checkpoint_subfolder else None
    print(f"[model] Loading tokenizer from: {tok_source} subfolder={tok_subfolder}")
    try:
        tok = AutoTokenizer.from_pretrained(tok_source, subfolder=tok_subfolder, use_fast=True)
    except TypeError:
        if tok_subfolder and cfg.hub_model_id:
            print("[model] AutoTokenizer failed with subfolder param; loading tokenizer from hub root.")
            tok = AutoTokenizer.from_pretrained(cfg.hub_model_id, use_fast=True)
        else:
            tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_load_source = cfg.hub_model_id if checkpoint_subfolder else cfg.model_id
    model_subfolder = checkpoint_subfolder if checkpoint_subfolder else None
    print(f"[model] Loading model from: {model_load_source} subfolder={model_subfolder}")
    try:
        model = _from_pretrained_with_dtype_and_subfolder(model_load_source, dtype=dtype, device=device, subfolder=model_subfolder)
    except Exception as e:
        print(f"[model] Warning: failed to load from {model_load_source} (subfolder={model_subfolder}): {e}")
        model = _from_pretrained_with_dtype_and_subfolder(cfg.model_id, dtype=dtype, device=device, subfolder=None)

    model.eval()

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
        hub_private_repo=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_ds,
        processing_class=tok,
    )

    resume_arg = local_resume if local_resume else None
    if resume_arg:
        print(f"[train] Resuming trainer from local checkpoint: {resume_arg}")
    else:
        if checkpoint_subfolder:
            print(f"[train] Loaded model weights from hub checkpoint '{checkpoint_subfolder}' as base weights.")
        else:
            print("[train] Starting from base weights (no resume).")

    trainer.train(resume_from_checkpoint=resume_arg)
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
        print(f"[hub] Pushing final model files to hub repo: {cfg.hub_model_id}")
        try:
            trainer.push_to_hub()
            print("[hub] Trainer push_to_hub completed.")
        except Exception as e:
            print(f"[hub] Error in trainer.push_to_hub(): {e}")

    if cfg.push_checkpoint_to_hub:
        latest_ckpt_path = find_latest_local_checkpoint(cfg.output_dir)
        if latest_ckpt_path:
            print(f"[local] Latest checkpoint to push: {latest_ckpt_path}")
            success = upload_checkpoint_folder_to_hub(cfg.hub_model_id, latest_ckpt_path, token=cfg.hub_token or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
            if not success:
                print("[hub-upload] Failed to upload checkpoint folder. You can upload manually using `huggingface_hub` CLI or HfApi.upload_folder.")
        else:
            print("[local] No local checkpoint folder found to push to hub.")
    else:
        print("[local] push_checkpoint_to_hub disabled; not uploading local checkpoint folder.")

    print("[done] training and hub upload complete.")


if __name__ == "__main__":
    main()