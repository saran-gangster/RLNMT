# smoke_test_rewards.py
from __future__ import annotations

import argparse
import os
import re
import torch

from transformers import AutoTokenizer, Gemma3ForCausalLM

from rewards_en_ta_refcomet import CometRefQeReward

END_TURN = "<end_of_turn>"
CTRL_RE = re.compile(r"<start_of_turn>|<end_of_turn>|user|model")
LATIN_RE = re.compile(r"[A-Za-z]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")


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


def strip_ctrl(text: str) -> str:
    text = (text or "").replace("\u200b", "")
    text = CTRL_RE.sub("", text).strip()
    if END_TURN in text:
        text = text.split(END_TURN, 1)[0]
    return " ".join(text.split()).strip()


def extract_english_from_prompt(prompt_text: str) -> str:
    head, sep, tail = prompt_text.rpartition("English:")
    if not sep:
        return prompt_text.strip()
    en, sep2, _ = tail.partition("Tamil:")
    return (en if sep2 else tail).strip()


def script_penalty(mt_ta: str) -> float:
    p = 0.0
    if LATIN_RE.search(mt_ta or ""):
        p += 1.0
    if not TAMIL_RE.search(mt_ta or ""):
        p += 1.0
    return p


def length_ratio_penalty(src_en: str, mt_ta: str) -> float:
    # same definition as training reward: abs(log(ta_len/en_len))
    import math

    en_len = max(1, len((src_en or "").split()))
    ta_len = max(1, len((mt_ta or "").split()))
    ratio = ta_len / en_len
    return float(abs(math.log(ratio)))


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="google/gemma-3-1b-it")
    ap.add_argument("--english", default="I will visit Chennai next week for a conference.")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)

    ap.add_argument("--no_generate", action="store_true", help="Skip generation and use --tamil as completion.")
    ap.add_argument("--tamil", default="அடுத்த வாரம் மாநாட்டிற்காக நான் சென்னைக்கு செல்வேன்.")
    ap.add_argument(
        "--ref_tamil",
        default="அடுத்த வாரம் மாநாட்டிற்காக நான் சென்னைக்கு செல்வேன்.",
        help="Gold Tamil reference for reference-based COMET.",
    )

    ap.add_argument("--comet_batch_size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    prompt = make_gemma_prompt(args.english)

    # completion
    if args.no_generate:
        completion_ta = args.tamil
    else:
        tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        model = Gemma3ForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device).eval()

        # stop at <end_of_turn> if token exists
        end_turn_id = tok.convert_tokens_to_ids(END_TURN)
        eos_ids = [tok.eos_token_id]
        if isinstance(end_turn_id, int) and end_turn_id >= 0 and end_turn_id != tok.unk_token_id:
            eos_ids.append(end_turn_id)

        enc = tok(prompt, return_tensors="pt").to(device)
        input_len = enc["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=tok.pad_token_id,
            )

        gen_ids = out[:, input_len:]
        completion_ta = tok.batch_decode(gen_ids, skip_special_tokens=True)[0]
        completion_ta = strip_ctrl(completion_ta)

    ref_ta = strip_ctrl(args.ref_tamil)

    print("\n=== PROMPT (rendered) ===")
    print(prompt)
    print("\n=== COMPLETION TAMIL ===")
    print(completion_ta)
    print("\n=== REFERENCE TAMIL ===")
    print(ref_ta)

    # Reward function (reference-based COMET + QE)
    reward_fn = CometRefQeReward(
        ref_metric_name="Unbabel/wmt22-comet-da",
        qe_metric_name="Unbabel/wmt22-cometkiwi-da",
        w_ref=0.85,
        w_qe=0.15,
        normalize="auto",
        reward_device=device,
        comet_batch_size=args.comet_batch_size,
        ref_key="ref_ta",
        penalty_weight_script=0.25,
        penalty_weight_lenratio=0.05,
    )

    # Compute components explicitly (for transparency)
    src_en = extract_english_from_prompt(prompt)

    # QE
    qe_data = [{"src": src_en, "mt": completion_ta}]
    r_qe_raw = float(reward_fn._predict_scores(reward_fn.comet_qe, qe_data)[0].item())
    r_qe_norm = float(reward_fn._normalize(torch.tensor([r_qe_raw], dtype=torch.float32))[0].item())

    # Reference COMET
    ref_data = [{"src": src_en, "mt": completion_ta, "ref": ref_ta}]
    r_ref_raw = float(reward_fn._predict_scores(reward_fn.comet_ref, ref_data)[0].item())
    r_ref_norm = float(reward_fn._normalize(torch.tensor([r_ref_raw], dtype=torch.float32))[0].item())

    # Penalties (match reward)
    p_script = reward_fn.penalty_weight_script * script_penalty(completion_ta)
    p_len = reward_fn.penalty_weight_lenratio * length_ratio_penalty(src_en, completion_ta)
    p_tot = p_script + p_len

    total_manual = max(
        0.0,
        min(
            1.0,
            reward_fn.w_ref * r_ref_norm + reward_fn.w_qe * r_qe_norm - p_tot,
        ),
    )

    # Reward call path (like TRL will do)
    total_call = reward_fn([prompt], [completion_ta], ref_ta=[ref_ta])[0]

    print("\n=== REWARD BREAKDOWN ===")
    print(f"Ref (COMET-DA) raw:           {r_ref_raw: .4f}")
    print(f"Ref (COMET-DA) normalized:    {r_ref_norm: .4f}")
    print(f"QE (COMETKiwi) raw:           {r_qe_raw: .4f}")
    print(f"QE (COMETKiwi) normalized:    {r_qe_norm: .4f}")
    print(f"Penalty script:               {p_script: .4f}")
    print(f"Penalty length-ratio:         {p_len: .4f}")
    print(f"Penalty total:                {p_tot: .4f}")
    print(f"Total (manual):               {total_manual: .4f}")
    print(f"Total (reward_fn __call__):   {total_call: .4f}")


if __name__ == "__main__":
    main()