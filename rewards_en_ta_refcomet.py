# rewards_en_ta_refcomet.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from comet import download_model, load_from_checkpoint

PromptType = Union[str, List[Dict[str, Any]]]
CompletionType = Union[str, List[Dict[str, Any]]]

END_TURN = "<end_of_turn>"

_LATIN_RE = re.compile(r"[A-Za-z]")
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
_GEMMA_CTRL = re.compile(r"<start_of_turn>|<end_of_turn>|user|model")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
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
    text = (text or "").replace("\u200b", "")
    text = _GEMMA_CTRL.sub("", text)
    if END_TURN in text:
        text = text.split(END_TURN, 1)[0]
    return " ".join(text.split()).strip()


def _extract_english_from_prompt(prompt_text: str) -> str:
    head, sep, tail = prompt_text.rpartition("English:")
    if not sep:
        return prompt_text.strip()
    en, sep2, _ = tail.partition("Tamil:")
    return (en if sep2 else tail).strip()


def _length_ratio_penalty(src_en: str, mt_ta: str) -> float:
    en_len = max(1, len((src_en or "").split()))
    ta_len = max(1, len((mt_ta or "").split()))
    ratio = ta_len / en_len
    return float(abs(math.log(ratio)))


def _latin_tolerant_script_penalty(mt_ta: str, latin_frac_threshold: float = 0.35) -> float:
    """
    Penalize only if:
      - no Tamil characters at all (strong penalty)
      - or output is mostly Latin (fraction over threshold)
    This avoids punishing valid tokens like 'svg', 'NASA', names, etc.
    """
    s = mt_ta or ""
    has_tamil = bool(_TAMIL_RE.search(s))
    has_latin = bool(_LATIN_RE.search(s))

    if not has_tamil:
        return 1.0  # must contain Tamil

    if not has_latin:
        return 0.0

    # approximate latin fraction over letters
    latin_chars = len(re.findall(r"[A-Za-z]", s))
    tamil_chars = len(re.findall(r"[\u0B80-\u0BFF]", s))
    denom = max(1, latin_chars + tamil_chars)
    latin_frac = latin_chars / denom

    if latin_frac <= latin_frac_threshold:
        return 0.0
    # scale up smoothly above threshold
    return float(min(1.0, (latin_frac - latin_frac_threshold) / (1.0 - latin_frac_threshold)))


@dataclass
class CometRefQeReward:
    ref_metric_name: str = "Unbabel/wmt22-comet-da"
    qe_metric_name: str = "Unbabel/wmt22-cometkiwi-da"

    w_ref: float = 0.85
    w_qe: float = 0.15

    penalty_weight_script: float = 0.08       # lowered because OPUS has legit Latin tokens
    penalty_weight_lenratio: float = 0.03

    # "affine": map roughly [-1, 1] -> [0, 1] via 0.5*(x+1) then clamp
    normalize: str = "affine"

    reward_device: str = "cuda"
    comet_batch_size: int = 64

    ref_key: str = "ref_ta"

    # optional fallback index: prompt -> ref_ta
    prompt_to_ref: Optional[Dict[str, str]] = None

    @property
    def __name__(self) -> str:
        return "comet_ref_qe_reward"

    def __post_init__(self) -> None:
        self._reward_device = torch.device(self.reward_device)

        ref_path = download_model(self.ref_metric_name)
        self.comet_ref = load_from_checkpoint(ref_path).eval()

        qe_path = download_model(self.qe_metric_name)
        self.comet_qe = load_from_checkpoint(qe_path).eval()

        for m in (self.comet_ref, self.comet_qe):
            try:
                m.to(self._reward_device)
            except Exception:
                pass

    def index_references(self, prompts: Sequence[str], refs: Sequence[str]) -> None:
        """
        Build a fallback map so training still uses references even if TRL doesn't
        pass ref_ta through kwargs.
        """
        m: Dict[str, str] = {}
        for p, r in zip(prompts, refs):
            ps = str(p)
            # only keep first if duplicates occur
            if ps not in m:
                m[ps] = str(r)
        self.prompt_to_ref = m

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

        x = torch.tensor(scores, dtype=torch.float32)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize == "identity":
            return torch.clamp(x, 0.0, 1.0)
        if self.normalize == "auto":
            if (x.min().item() < 0.0) or (x.max().item() > 1.0):
                x = torch.sigmoid(x)
            return torch.clamp(x, 0.0, 1.0)
        if self.normalize == "affine":
            # stable spread when raw scores are around [-1, 1]
            return torch.clamp(0.5 * (x + 1.0), 0.0, 1.0)
        raise ValueError("normalize must be 'identity', 'auto', or 'affine'.")

    @torch.inference_mode()
    def __call__(
        self,
        prompts: Sequence[PromptType],
        completions: Sequence[CompletionType],
        **kwargs: Any,
    ) -> List[float]:
        prompt_texts = [_to_text(p) for p in prompts]
        ta_hyps = [_strip_control(_to_text(c)) for c in completions]
        src_en = [_extract_english_from_prompt(t) for t in prompt_texts]

        # Get refs from kwargs if provided; else fallback to prompt_to_ref index.
        ref_ta_in: Optional[Sequence[str]] = kwargs.get(self.ref_key, None)
        if ref_ta_in is not None and len(ref_ta_in) == len(ta_hyps):
            ta_refs = [_strip_control(str(r)) for r in ref_ta_in]
        else:
            ta_refs = []
            if self.prompt_to_ref:
                for p in prompt_texts:
                    ta_refs.append(_strip_control(self.prompt_to_ref.get(p, "")))
            else:
                ta_refs = [""] * len(ta_hyps)

        # QE always
        qe_data = [{"src": s, "mt": t} for s, t in zip(src_en, ta_hyps)]
        r_qe = self._normalize(self._predict_scores(self.comet_qe, qe_data))

        # Reference-based COMET where refs exist; else 0.
        has_ref_mask = torch.tensor([1.0 if (r and r.strip()) else 0.0 for r in ta_refs], dtype=torch.float32)
        if has_ref_mask.sum().item() > 0:
            ref_data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src_en, ta_hyps, ta_refs)]
            r_ref_all = self._normalize(self._predict_scores(self.comet_ref, ref_data))
            r_ref = r_ref_all * has_ref_mask
        else:
            r_ref = torch.zeros_like(r_qe)

        # penalties
        p_script_vals: List[float] = []
        p_len_vals: List[float] = []
        for s, t in zip(src_en, ta_hyps):
            p_script_vals.append(self.penalty_weight_script * _latin_tolerant_script_penalty(t))
            p_len_vals.append(self.penalty_weight_lenratio * _length_ratio_penalty(s, t))
        penalties = torch.tensor(p_script_vals, dtype=torch.float32) + torch.tensor(p_len_vals, dtype=torch.float32)

        total = self.w_ref * r_ref + self.w_qe * r_qe - penalties
        total = torch.clamp(total, 0.0, 1.0)
        return total.detach().cpu().tolist()