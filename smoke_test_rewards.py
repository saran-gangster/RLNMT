# smoke_test_rewards.py
from __future__ import annotations

import argparse
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
    text = CTRL_RE.sub("", text)
    text = text.strip()
    # If model emits end_of_turn, truncate there
    if END_TURN in text:
        text = text.split(END_TURN, 1)[0]
    return " ".join(text.split()).strip()


def extract_english(prompt: str) -> str:
    # Safe “last occurrence” extraction
    head, sep, tail = prompt.rpartition("English:")
    if not sep:
        return prompt.strip()
    en, sep2, _ = tail.partition("Tamil:")
    return (en if sep2 else tail).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="google/gemma-3-1b-it")
    ap.add_argument("--english", default="The weather is nice today, but I forgot my umbrella.")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--no_generate", action="store_true", help="Skip generation and use --tamil as completion.")
    ap.add_argument("--tamil", default="இன்று வானிலை நல்லதாக இருக்கிறது, ஆனால் நான் குடையை மறந்துவிட்டேன்.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Build prompt
    prompt = make_gemma_prompt(args.english)
    src_en = extract_english(prompt)

    # 2) Optionally generate Tamil completion with Gemma
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
    print("\n=== SOURCE ENGLISH ===")
    print(src_en)
    print("\n=== GENERATED/PROVIDED TAMIL ===")
    print(completion_ta)

    # 3) Load reward fn (this loads: COMET-DA + COMETKiwi + NLLB dual)
    reward_fn = CometDuPOReward(
        cycle_metric_name="Unbabel/wmt22-comet-da",
        qe_metric_name="Unbabel/wmt22-cometkiwi-da",
        dual_model_name="facebook/nllb-200-distilled-1.3B",
        w_cycle=0.7,
        w_qe=0.3,
        normalize="linear",
        reward_device=device,
        dual_device=device,
        comet_batch_size=8,
        dual_dtype_bf16=(device == "cuda"),
    )

    # 4) Compute components (using the same internals as the reward class)
    #    This avoids guessing TRL’s call format.
    with torch.inference_mode():
        # Backtranslate Ta->En
        back_en = reward_fn.bt_tok  # just to make it obvious what we use
        bt = reward_fn  # alias

        backtrans_en_list = bt._CometDuPOReward__dict__ if False else None  # no-op to keep linters quiet

        # Use the class's own helper
        back_en = bt.bt_tok  # tokenizer
        back_model = bt.bt_model

        # Call the module-level helper via the instance method route:
        # (We use the public pipeline instead: invoke reward_fn once, then recompute detailed pieces.)
        # Recompute pieces directly for printing:
        from rewards_en_ta_dupo import _nllb_backtranslate_ta_to_en, _script_penalty, _length_ratio_penalty

        backtrans_en = _nllb_backtranslate_ta_to_en(
            back_model, back_en, [completion_ta], device=torch.device(device)
        )[0]

        # Cycle raw (COMET-DA expects src, mt, ref)
        cycle_data = [{"src": src_en, "mt": backtrans_en, "ref": src_en}]
        r_cycle_raw = bt._predict_scores(bt.comet_cycle, cycle_data)[0].item()
        r_cycle_norm = bt._normalize(torch.tensor([r_cycle_raw]))[0].item()

        # QE raw (Kiwi expects src, mt)
        qe_data = [{"src": src_en, "mt": completion_ta}]
        r_qe_raw = bt._predict_scores(bt.comet_qe, qe_data)[0].item()
        r_qe_norm = bt._normalize(torch.tensor([r_qe_raw]))[0].item()

        # Penalties
        p_script = bt.penalty_weight_script * _script_penalty(completion_ta)
        p_len = bt.penalty_weight_lenratio * _length_ratio_penalty(src_en, completion_ta)
        penalty_total = p_script + p_len

        # Total (same formula as reward_fn)
        total = bt.w_cycle * r_cycle_norm + bt.w_qe * r_qe_norm - penalty_total
        total = max(0.0, min(1.0, total))

        # Sanity: also call the reward_fn as TRL would
        total_via_call = bt([prompt], [completion_ta])[0]

    print("\n=== BACK-TRANSLATED ENGLISH (Ta->En via NLLB) ===")
    print(backtrans_en)

    print("\n=== REWARD BREAKDOWN ===")
    print(f"Cycle (COMET-DA) raw:        {r_cycle_raw: .4f}")
    print(f"Cycle (COMET-DA) normalized: {r_cycle_norm: .4f}")
    print(f"QE (COMETKiwi) raw:          {r_qe_raw: .4f}")
    print(f"QE (COMETKiwi) normalized:   {r_qe_norm: .4f}")
    print(f"Penalty script:              {p_script: .4f}")
    print(f"Penalty length-ratio:        {p_len: .4f}")
    print(f"Total (manual):              {total: .4f}")
    print(f"Total (reward_fn __call__):  {total_via_call: .4f}")


if __name__ == "__main__":
    main()