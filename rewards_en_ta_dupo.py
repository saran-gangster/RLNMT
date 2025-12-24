# rewards_en_ta_dupo.py
# DuPO-style compound reward:
#   cycle (COMET-DA, reference-based) + QE/fluency (COMETKiwi-DA, reference-free)
# with oracle dual back-translation (NLLB Ta->En, frozen)

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

# Gemma control tokens sometimes appear in decoded text
_GEMMA_CTRL = re.compile(r"<start_of_turn>|<end_of_turn>|user|model")


def _content_to_text(content: Any) -> str:
    """Gemma messages may store content as a string OR as a list of {"type":"text","text":...} chunks."""
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
    """Handles TRL passing strings or chat-like lists of messages."""
    if isinstance(x, str):
        return x
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict):
        # join all message contents in order
        return "\n".join(_content_to_text(m.get("content", "")) for m in x if isinstance(m, dict))
    return str(x)


def _strip_control(text: str) -> str:
    text = text.replace("\u200b", "")
    text = _GEMMA_CTRL.sub("", text)
    # remove repeated whitespace/newlines
    return " ".join(text.split()).strip()


def _extract_english_from_prompt(prompt_text: str) -> str:
    """
    Extracts the *last* 'English:' ... 'Tamil:' block to avoid accidental earlier occurrences.
    Falls back to full prompt if markers not found.
    """
    t = prompt_text
    # rpartition is safer than split if "English:" appears elsewhere
    head, sep, tail = t.rpartition("English:")
    if sep:
        en_block = tail
        # stop at Tamil:
        en, sep2, _ = en_block.partition("Tamil:")
        if sep2:
            return en.strip()
        return en_block.strip()
    return t.strip()


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


def _norm_linear_clip(x: torch.Tensor) -> torch.Tensor:
    # Common heuristic for COMET-ish scores: map ~[-1, +1] to [0, 1]
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def _norm_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


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
    # Dual/oracle backtranslation
    dual_model_name: str = "facebook/nllb-200-distilled-1.3B"

    # Metric split (your Fix #1)
    cycle_metric_name: str = "Unbabel/wmt22-comet-da"          # reference-based
    qe_metric_name: str = "Unbabel/wmt22-cometkiwi-da"         # reference-free

    # weights
    w_cycle: float = 0.7
    w_qe: float = 0.3

    # penalties
    penalty_weight_script: float = 0.10
    penalty_weight_lenratio: float = 0.05

    # normalization: "sigmoid" or "linear"
    normalize: str = "linear"

    # devices (best-effort; COMET .predict device selection depends on comet version)
    reward_device: str = "cuda"
    dual_device: str = "cuda"

    comet_batch_size: int = 16

    # If you ever need memory relief on a single GPU, you can set this True to keep NLLB in bf16
    dual_dtype_bf16: bool = True

    def __post_init__(self) -> None:
        self._reward_device = torch.device(self.reward_device)
        self._dual_device = torch.device(self.dual_device)

        # --- Load COMET metrics ---
        cycle_path = download_model(self.cycle_metric_name)
        self.comet_cycle = load_from_checkpoint(cycle_path)
        self.comet_cycle.eval()

        qe_path = download_model(self.qe_metric_name)
        self.comet_qe = load_from_checkpoint(qe_path)
        self.comet_qe.eval()

        # Move to device if possible (COMET sometimes manages internally)
        for m in (self.comet_cycle, self.comet_qe):
            try:
                m.to(self._reward_device)
            except Exception:
                pass

        # --- Load dual back-translation model (frozen) ---
        self.bt_tok = AutoTokenizer.from_pretrained(self.dual_model_name, use_fast=True)

        dtype = torch.bfloat16 if (self.dual_dtype_bf16 and self._dual_device.type == "cuda") else None

        try:
            # Force safetensors to avoid torch.load(.bin) restriction on torch<2.6
            self.bt_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.dual_model_name,
                dtype=dtype,                 # torch_dtype is deprecated in your setup
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dual model '{self.dual_model_name}' with safetensors.\n"
                f"To proceed, either:\n"
                f"  1) ensure the repo has a *.safetensors weight file (preferred), or\n"
                f"  2) upgrade torch to >= 2.6 to allow loading *.bin safely.\n"
                f"Original error: {repr(e)}"
            )

        self.bt_model.to(self._dual_device)
        self.bt_model.eval()
        for p in self.bt_model.parameters():
            p.requires_grad_(False)

    def _predict_scores(self, metric, data: List[Dict[str, str]]) -> torch.Tensor:
        """
        Robust to COMET version differences.
        Tries multiple signatures and extracts scores.
        """
        out = None

        # Try to steer to GPU if available; API differs by COMET versions.
        # If your COMET build ignores device placement, it will still work (just may run on default GPU/CPU).
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
        if self.normalize == "sigmoid":
            return _norm_sigmoid(x)
        return _norm_linear_clip(x)

    def __call__(
        self,
        prompts: Sequence[PromptType],
        completions: Sequence[CompletionType],
        **kwargs: Any,
    ) -> List[float]:
        # 1) Convert to plain text
        prompt_texts = [_to_text(p) for p in prompts]
        comp_texts = [_to_text(c) for c in completions]
        ta_hyps = [_strip_control(c) for c in comp_texts]

        # 2) Extract English sources from prompt
        sources_en = [_extract_english_from_prompt(t) for t in prompt_texts]

        # 3) Dual/oracle: backtranslate Tamil -> English
        back_en = _nllb_backtranslate_ta_to_en(
            self.bt_model, self.bt_tok, ta_hyps, device=self._dual_device
        )

        # 4) Cycle reward: COMET-DA with reference
        cycle_data = [{"src": s, "mt": b, "ref": s} for s, b in zip(sources_en, back_en)]
        r_cycle_raw = self._predict_scores(self.comet_cycle, cycle_data)

        # 5) QE reward: COMETKiwi (reference-free)
        qe_data = [{"src": s, "mt": t} for s, t in zip(sources_en, ta_hyps)]
        r_qe_raw = self._predict_scores(self.comet_qe, qe_data)

        # 6) Normalize + combine
        r_cycle = self._normalize(r_cycle_raw)
        r_qe = self._normalize(r_qe_raw)
        total = self.w_cycle * r_cycle + self.w_qe * r_qe

        # 7) Penalties
        penalties = []
        for s, t in zip(sources_en, ta_hyps):
            p = 0.0
            p += self.penalty_weight_script * _script_penalty(t)
            p += self.penalty_weight_lenratio * _length_ratio_penalty(s, t)
            penalties.append(p)

        total = total - torch.tensor(penalties, dtype=torch.float32)
        total = torch.clamp(total, min=0.0, max=1.0)
        return total.detach().cpu().tolist()