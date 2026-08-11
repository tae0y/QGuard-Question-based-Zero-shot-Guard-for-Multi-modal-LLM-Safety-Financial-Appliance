#!/usr/bin/env python3
"""MM-SafetyBench Text_only, all 13 categories, paper-faithful replication.

No condition augmentation, no benign set, no context injection — each raw
`question` goes straight into the QGuard pipeline exactly as the paper does.
(That is `run_financial_advice_experiment.py`'s C0 arm, minus everything else.)

One JSONL per category under `<out_dir>/`, one line per benchmark row, each
line carrying every per-guard-question number: yes/no logit, yes/no prob,
decoded token, plus the PageRank risk score, threshold and prediction.

Resumable: rows already present in a category file are skipped, so a Colab
disconnect costs only the row in flight.

Usage (Colab, T4/L4/A100):
  uv run experiments/run_mmsafetybench_textonly.py \
    --out_dir "/content/drive/MyDrive/qguard_results/mmsafety_full"
"""
import argparse
import json
import os
import time
from typing import Any, Dict, List, Set

import torch
from datasets import load_dataset

from qguard.config import RunConfig
from qguard.modeling import load_internvl_text_or_mm
from qguard.pipeline import evaluate_prompt_with_pagerank
from qguard.seed import set_seed

# https://huggingface.co/api/datasets/PKU-Alignment/MM-SafetyBench — every
# config carries the same four splits; Text_only is the only image-free one.
CATEGORIES = [
    "EconomicHarm", "Financial_Advice", "Fraud", "Gov_Decision", "HateSpeech",
    "Health_Consultation", "Illegal_Activitiy", "Legal_Opinion",
    "Malware_Generation", "Physical_Harm", "Political_Lobbying",
    "Privacy_Violence", "Sex",
]


def done_ids(path: str) -> Set[str]:
    """Row ids already written, so a re-run resumes instead of duplicating."""
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["row_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # truncated last line from a hard kill
    return seen


def run_category(
    model, tokenizer, guard_qs: Dict[str, List[str]], category: str,
    threshold: float, out_path: str, n_samples: int = None,
) -> int:
    ds = load_dataset("PKU-Alignment/MM-SafetyBench", category, split="Text_only")
    rows = [r for r in ds if r.get("question")]
    if n_samples:
        rows = rows[:n_samples]

    skip = done_ids(out_path)
    written = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            row_id = str(row.get("id", i))
            if row_id in skip:
                continue
            t0 = time.time()
            result = evaluate_prompt_with_pagerank(
                model, tokenizer, row["question"], guard_qs, threshold=threshold,
            )
            f.write(json.dumps({
                "category": category,
                "row_id": row_id,
                "prompt": row["question"],
                "label": 1,  # every MM-SafetyBench row is an unsafe prompt
                "threshold": threshold,
                "risk_score": result["risk_score"],
                "prediction": result["prediction"],
                "questions": result["questions"],
                "elapsed_sec": round(time.time() - t0, 3),
            }, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="OpenGVLab/InternVL2_5-4B")
    parser.add_argument("--guard_questions_json", default="files/guard_questions.json")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--n_samples", type=int, default=None, help="Per-category cap; omit for the full split")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = RunConfig(model_path=args.model_path, guard_questions_json=args.guard_questions_json,
                    threshold=args.threshold, seed=args.seed)
    cfg.apply_env()
    set_seed(cfg.seed)

    print(f"CUDA available: {torch.cuda.is_available()}")
    model, tokenizer = load_internvl_text_or_mm(cfg.model_path)
    with open(cfg.guard_questions_json, "r", encoding="utf-8") as f:
        guard_qs = json.load(f)

    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "n_guard_questions": sum(len(v) for v in guard_qs.values())}, f, indent=2)

    for cat in args.categories:
        out_path = os.path.join(args.out_dir, f"{cat}.jsonl")
        n = run_category(model, tokenizer, guard_qs, cat, cfg.threshold, out_path, args.n_samples)
        print(f"{cat}: +{n} rows -> {out_path}", flush=True)

    print(f"\nDone -> {args.out_dir}")


if __name__ == "__main__":
    main()
