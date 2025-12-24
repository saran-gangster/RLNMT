# rewards_en_ta_dupo.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Union, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from comet import download_model, load_from_checkpoint

PromptType = Union[str, List[Dict[str, Any]]]
CompletionType = Union[str, List[Dict[str, Any]]]

_LATIN_RE = re.compile(r"[A-Za-z]")
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
_GEMMA_CTRL = re.compile(r"<start_of_turn>|<end_of_turn>|user|model")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            else:
                parts.append(str(c))
        return "".join(parts)
    return str(content)


def _to_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict):
        return "\n".join(_content_to_text(m.get("content", "")) for m in x if isinstance(m, dict))
    return str(x)


def _strip_control(text: str) -> str:
    text = text.replace("\u200b", "")
    text = _GEMMA_CTRL.sub("", text)
    return " ".join(text.split()).strip()


def _extract_english_from_prompt(prompt_text: str) -> str:
    head, sep, tail = prompt_text.rpartition("English:")
    if not sep:
        return prompt_text.strip()
    en, sep2, _ = tail.partition("Tamil:")
    return (en if sep2 else tail).strip()


@torch.inference_mode()
def _nllb_backtranslate_ta_to_en(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    ta_texts: List[str],
    device: torch.device,
    max_new_tokens: int = 256,
) -> List[str]:
    tokenizer.src_lang = "tam_Taml"
    forced_bos = tokenizer.convert_tokens_to_ids("eng_Latn")

    batch = tokenizer(
        ta_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    gen = model.generate(
        **batch,
        do_sample=False,
        num_beams=4,
        max_new_tokens=max_new_tokens,
        forced_bos_token_id=forced_bos,
    )
    out = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [_strip_control(s) for s in out]


def _norm_identity_clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def _length_ratio_penalty(src_en: str, mt_ta: str) -> float:
    en_len = max(1, len(src_en.split()))
    ta_len = max(1, len(mt_ta.split()))
    ratio = ta_len / en_len
    return float(abs(math.log(ratio)))


def _script_penalty(mt_ta: str) -> float:
    p = 0.0
    if _LATIN_RE.search(mt_ta):
        p += 1.0
    if not _TAMIL_RE.search(mt_ta):
        p += 1.0
    return p


@dataclass
class CometDuPOReward:
    dual_model_name: str = "facebook/nllb-200-distilled-1.3B"
    cycle_metric_name: str = "Unbabel/wmt22-comet-da"
    qe_metric_name: str = "Unbabel/wmt22-cometkiwi-da"

    w_cycle: float = 0.7
    w_qe: float = 0.3

    penalty_weight_script: float = 0.10
    penalty_weight_lenratio: float = 0.05

    normalize: str = "identity"

    reward_device: str = "cuda"
    dual_device: str = "cuda"
    comet_batch_size: int = 16

    dual_dtype_bf16: bool = True
    dual_use_safetensors: bool = True

    # ---- logging state (used by callback) ----
    last_log: Optional[Dict[str, float]] = None

    @property
    def __name__(self) -> str:
        # TRL expects reward_funcs[i].__name__
        return "comet_dupo_reward"

    def __post_init__(self) -> None:
        self._reward_device = torch.device(self.reward_device)
        self._dual_device = torch.device(self.dual_device)

        cycle_path = download_model(self.cycle_metric_name)
        self.comet_cycle = load_from_checkpoint(cycle_path).eval()

        qe_path = download_model(self.qe_metric_name)
        self.comet_qe = load_from_checkpoint(qe_path).eval()

        for m in (self.comet_cycle, self.comet_qe):
            try:
                m.to(self._reward_device)
            except Exception:
                pass

        self.bt_tok = AutoTokenizer.from_pretrained(self.dual_model_name, use_fast=True)
        dtype = torch.bfloat16 if (self.dual_dtype_bf16 and self._dual_device.type == "cuda") else None

        self.bt_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.dual_model_name,
            dtype=dtype,  # avoids torch_dtype deprecation in your env
            use_safetensors=self.dual_use_safetensors,
            low_cpu_mem_usage=True,
        ).to(self._dual_device).eval()

        for p in self.bt_model.parameters():
            p.requires_grad_(False)

    def _predict_scores(self, metric, data: List[Dict[str, str]]) -> torch.Tensor:
        out = None
        try:
            out = metric.predict(
                data,
                batch_size=self.comet_batch_size,
                gpus=1 if self._reward_device.type == "cuda" else 0,
            )
        except TypeError:
            try:
                out = metric.predict(
                    data,
                    batch_size=self.comet_batch_size,
                    device=str(self._reward_device),
                )
            except TypeError:
                out = metric.predict(data, batch_size=self.comet_batch_size)

        if isinstance(out, dict) and "scores" in out:
            scores = out["scores"]
        else:
            scores = getattr(out, "scores", out)

        return torch.tensor(scores, dtype=torch.float32)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize == "identity":
            return _norm_identity_clamp01(x)
        raise ValueError("Use normalize='identity' for these COMET models.")

    def __call__(
        self,
        prompts: Sequence[PromptType],
        completions: Sequence[CompletionType],
        **kwargs: Any,
    ) -> List[float]:
        prompt_texts = [_to_text(p) for p in prompts]
        ta_hyps = [_strip_control(_to_text(c)) for c in completions]
        sources_en = [_extract_english_from_prompt(t) for t in prompt_texts]

        # backtranslation
        back_en = _nllb_backtranslate_ta_to_en(
            self.bt_model, self.bt_tok, ta_hyps, device=self._dual_device
        )

        # cycle (COMET-DA)
        cycle_data = [{"src": s, "mt": b, "ref": s} for s, b in zip(sources_en, back_en)]
        r_cycle_raw = self._predict_scores(self.comet_cycle, cycle_data)
        r_cycle = self._normalize(r_cycle_raw)

        # QE (COMETKiwi)
        qe_data = [{"src": s, "mt": t} for s, t in zip(sources_en, ta_hyps)]
        r_qe_raw = self._predict_scores(self.comet_qe, qe_data)
        r_qe = self._normalize(r_qe_raw)

        # penalties
        p_script_vals = []
        p_len_vals = []
        for s, t in zip(sources_en, ta_hyps):
            p_script_vals.append(self.penalty_weight_script * _script_penalty(t))
            p_len_vals.append(self.penalty_weight_lenratio * _length_ratio_penalty(s, t))

        penalties = torch.tensor(p_script_vals, dtype=torch.float32) + torch.tensor(p_len_vals, dtype=torch.float32)

        total = self.w_cycle * r_cycle + self.w_qe * r_qe - penalties
        total = torch.clamp(total, 0.0, 1.0)

        # ---- store stats for W&B callback ----
        self.last_log = {
            "reward/total_mean": float(total.mean().item()),
            "reward/total_std": float(total.std(unbiased=False).item()) if total.numel() > 1 else 0.0,
            "reward/cycle_mean": float(r_cycle.mean().item()),
            "reward/qe_mean": float(r_qe.mean().item()),
            "reward/cycle_raw_mean": float(r_cycle_raw.mean().item()),
            "reward/qe_raw_mean": float(r_qe_raw.mean().item()),
            "reward/penalty_script_mean": float(torch.tensor(p_script_vals).mean().item()),
            "reward/penalty_len_mean": float(torch.tensor(p_len_vals).mean().item()),
            "reward/penalty_total_mean": float(penalties.mean().item()),
        }

        return total.detach().cpu().tolist()