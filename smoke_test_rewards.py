# smoke_test_rewards.py
from __future__ import annotations

import argparse
import os
import re
import torch

from transformers import AutoTokenizer, Gemma3ForCausalLM

from rewards_en_ta_dupo import CometDuPOReward


END_TURN = "<end_of_turn>"
CTRL_RE = re.compile(r"<start_of_turn>|<end_of_turn>|user|model")


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
    text = text.replace("\u200b", "")
    text = CTRL_RE.sub("", text).strip()
    if END_TURN in text:
        text = text.split(END_TURN, 1)[0]
    return " ".join(text.split()).strip()


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="google/gemma-3-1b-it")
    ap.add_argument("--english", default="I will visit Chennai next week for a conference.")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--no_generate", action="store_true", help="Skip generation and use --tamil as completion.")
    ap.add_argument("--tamil", default="அடுத்த வாரம் மாநாட்டிற்காக நான் சென்னைக்கு வருவேன்.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    prompt = make_gemma_prompt(args.english)

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

        enc = tok(prompt, return_tensors="pt").to(device)
        input_len = enc["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )

        gen_ids = out[:, input_len:]
        completion_ta = tok.batch_decode(gen_ids, skip_special_tokens=True)[0]
        completion_ta = strip_ctrl(completion_ta)

    print("\n=== PROMPT (rendered) ===")
    print(prompt)
    print("\n=== GENERATED/PROVIDED TAMIL ===")
    print(completion_ta)

    reward_fn = CometDuPOReward(
        cycle_metric_name="Unbabel/wmt22-comet-da",
        qe_metric_name="Unbabel/wmt22-cometkiwi-da",
        dual_model_name="facebook/nllb-200-distilled-1.3B",
        w_cycle=0.7,
        w_qe=0.3,
        normalize="identity",  # IMPORTANT FIX
        reward_device=device,
        dual_device=device,
        comet_batch_size=8,
        dual_dtype_bf16=(device == "cuda"),
        dual_use_safetensors=True,
    )

    comps = reward_fn.score_components([prompt], [completion_ta])

    back_en = comps["back_en"][0]
    r_cycle_raw = comps["cycle_raw"][0]
    r_cycle_norm = comps["cycle_norm"][0]
    r_qe_raw = comps["qe_raw"][0]
    r_qe_norm = comps["qe_norm"][0]
    p_script = comps["penalty_script"][0]
    p_len = comps["penalty_lenratio"][0]
    p_tot = comps["penalty_total"][0]

    total_manual = max(
        0.0,
        min(1.0, 0.7 * r_cycle_norm + 0.3 * r_qe_norm - p_tot),
    )
    total_call = reward_fn([prompt], [completion_ta])[0]

    print("\n=== BACK-TRANSLATED ENGLISH (Ta->En via NLLB) ===")
    print(back_en)

    print("\n=== REWARD BREAKDOWN ===")
    print(f"Cycle (COMET-DA) raw:         {r_cycle_raw: .4f}")
    print(f"Cycle (COMET-DA) normalized:  {r_cycle_norm: .4f}   (identity clamp)")
    print(f"QE (COMETKiwi) raw:           {r_qe_raw: .4f}")
    print(f"QE (COMETKiwi) normalized:    {r_qe_norm: .4f}   (identity clamp)")
    print(f"Penalty script:               {p_script: .4f}")
    print(f"Penalty length-ratio:         {p_len: .4f}")
    print(f"Penalty total:                {p_tot: .4f}")
    print(f"Total (manual):               {total_manual: .4f}")
    print(f"Total (reward_fn __call__):   {total_call: .4f}")


if __name__ == "__main__":
    main()