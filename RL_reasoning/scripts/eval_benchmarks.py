"""Evaluate a HF model on GSM8K and MATH-500 with vLLM, write a CSV row.

Usage:
    python eval_benchmarks.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
        --tag baseline \
        --out /content/drive/MyDrive/OMSCS/results.csv

For trained checkpoints, point --model at the local checkpoint dir.
GSM8K is sampled to 500 problems by default (full set is 1319) to keep eval
time bounded; pass --gsm8k_n 1319 for the full split.
"""
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams


def extract_boxed(text: str) -> str | None:
    """Pull the last \\boxed{...} content, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def normalize(ans: str) -> str:
    if ans is None:
        return ""
    a = ans.strip().rstrip(".").replace(" ", "").replace(",", "")
    a = a.replace("\\$", "").replace("$", "").replace("%", "")
    return a


def gsm8k_gold(example):
    # GSM8K answers look like "...\n#### 42"
    m = re.search(r"####\s*(-?\d[\d,]*)", example["answer"])
    return m.group(1).replace(",", "") if m else None


def math500_gold(example):
    # MATH-500 has "answer" pre-extracted.
    return example.get("answer") or extract_boxed(example["solution"])


def build_prompts(tokenizer, questions):
    prompts = []
    for q in questions:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        ))
    return prompts


def score(preds, golds):
    correct = 0
    lengths = []
    for p, g in zip(preds, golds):
        ans = extract_boxed(p)
        lengths.append(len(p))
        if ans is not None and g is not None and normalize(ans) == normalize(g):
            correct += 1
    return correct / max(len(preds), 1), sum(lengths) / max(len(lengths), 1)


def run_benchmark(llm, tokenizer, name, questions, golds, max_new_tokens, out_dir):
    prompts = build_prompts(tokenizer, questions)
    params = SamplingParams(
        temperature=0.6, top_p=0.95,
        max_tokens=max_new_tokens, n=1,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    preds = [o.outputs[0].text for o in outputs]
    acc, mean_len = score(preds, golds)

    # Persist raw generations for inspection.
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / f"{name}_generations.jsonl", "w") as f:
        for q, g, p in zip(questions, golds, preds):
            f.write(json.dumps({"q": q, "gold": g, "pred": p}) + "\n")
    return acc, mean_len, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="row label, e.g. baseline / step_500")
    ap.add_argument("--out", required=True, help="CSV path; appended if exists")
    ap.add_argument("--gen_dir", default=None, help="dir to dump raw generations")
    ap.add_argument("--gsm8k_n", type=int, default=500,
                    help="GSM8K problems to evaluate (full split is 1319)")
    ap.add_argument("--math500_n", type=int, default=500,
                    help="MATH-500 problems to evaluate (full split is 500)")
    ap.add_argument("--max_new_tokens", type=int, default=3072)
    ap.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16",
                    help="Use float16 on Turing GPUs (T4) which lack bf16 support")
    args = ap.parse_args()

    gen_dir = args.gen_dir or os.path.join(os.path.dirname(args.out), f"gens_{args.tag}")

    print(f"Loading {args.model} (dtype={args.dtype})")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    print("Loading GSM8K")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm = gsm.shuffle(seed=0).select(range(min(args.gsm8k_n, len(gsm))))
    gsm_q = [x["question"] for x in gsm]
    gsm_gold = [gsm8k_gold(x) for x in gsm]

    print("Loading MATH-500")
    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    math = math.shuffle(seed=0).select(range(min(args.math500_n, len(math))))
    math_q = [x["problem"] for x in math]
    math_gold = [math500_gold(x) for x in math]

    rows = []
    for name, qs, gs in [("gsm8k", gsm_q, gsm_gold), ("math500", math_q, math_gold)]:
        print(f"\n=== {name} (n={len(qs)}) ===")
        acc, mean_len, elapsed = run_benchmark(
            llm, tokenizer, name, qs, gs, args.max_new_tokens, gen_dir,
        )
        print(f"  acc={acc:.4f}  mean_chars={mean_len:.0f}  time={elapsed:.0f}s")
        rows.append({
            "tag": args.tag, "model": args.model, "benchmark": name,
            "n": len(qs), "accuracy": f"{acc:.4f}",
            "mean_pred_chars": f"{mean_len:.0f}", "seconds": f"{elapsed:.0f}",
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(args.out).exists()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
