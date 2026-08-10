#!/usr/bin/env python3
"""Financial-advice recall experiment — C0/C1/C2/C1' conditions.

Requires a CUDA GPU (run on Colab or similar) — InternVL2_5-4B is loaded in
bf16. See docs/EXPERIMENT.md for the full walkthrough and what each flag does.

Usage:
  uv run experiments/run_financial_advice_experiment.py \
    --out_dir experiments/results/260810_full_run \
    --n_samples 501  # full financial_advice subset; use a small number for a pilot
"""
import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from qguard.config import RunConfig
from qguard.modeling import load_internvl_text_or_mm
from qguard.pipeline import evaluate_prompt_with_pagerank
from qguard.seed import set_seed
from qguard.token_utils import gather_yes_no_ids

from experiments.knowledge import apply_condition, build_conditions, generate_self_knowledge
from experiments.false_negatives import classify_false_negatives, collect_false_negatives
from experiments.attribution import attention_scores, cross_validate, occlusion_scores


def load_financial_advice_prompts() -> List[str]:
    """MM-SafetyBench financial_advice subset, text-only (SD/SD_TYPO/TYPO
    variants target the multi-modal setting; QGuard's own reported recall
    figure is for the text prompt itself, per the review doc's reproduction
    target of 0.2335)."""
    ds = load_dataset("PKU-Alignment/MM-SafetyBench", "Financial_Advice", split="SD")
    return [row["question"] for row in ds if row.get("question")]


def run_all_conditions(
    model, tokenizer, guard_qs: Dict[str, List[str]], prompts: List[str],
    threshold: float, self_generated_knowledge: str, results_path: str, label: int = 1,
) -> List[Dict[str, Any]]:
    """Writes each row to results_path as JSON Lines immediately after it's
    computed, so a crash/disconnect partway through a long run doesn't lose
    already-completed rows (this loop can be 1000s of forward passes)."""
    conditions = build_conditions(self_generated_knowledge)
    rows = []
    with open(results_path, "a", encoding="utf-8") as f:
        for prompt in prompts:
            for name, cond in conditions.items():
                augmented = apply_condition(prompt, cond)
                result = evaluate_prompt_with_pagerank(
                    model, tokenizer, augmented, guard_qs, threshold=threshold,
                )
                row = {
                    "prompt": prompt,
                    "condition": name,
                    "prediction": result["prediction"],
                    "risk_score": result["risk_score"],
                    "questions": result["questions"],
                    "label": label,
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="OpenGVLab/InternVL2_5-4B")
    parser.add_argument("--guard_questions_json", default="files/guard_questions.json")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--n_samples", type=int, default=None, help="Subsample prompts; omit for the full 501-pair set")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run_hypothesis3", action="store_true", help="Occlusion+attention on extreme-skew cases (slow)")
    parser.add_argument("--hypothesis3_sample_size", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = RunConfig(model_path=args.model_path, guard_questions_json=args.guard_questions_json, threshold=args.threshold, seed=args.seed)
    cfg.apply_env()
    set_seed(cfg.seed)

    print(f"CUDA available: {torch.cuda.is_available()}")
    model, tokenizer = load_internvl_text_or_mm(cfg.model_path)
    yes_ids, no_ids = gather_yes_no_ids(tokenizer)

    with open(cfg.guard_questions_json, "r", encoding="utf-8") as f:
        guard_qs = json.load(f)

    prompts = load_financial_advice_prompts()
    if args.n_samples:
        prompts = prompts[: args.n_samples]
    print(f"Loaded {len(prompts)} financial_advice prompts")

    # C2 self-generated knowledge: produced once, reused for every prompt.
    set_seed(cfg.seed)
    self_knowledge = generate_self_knowledge(model, tokenizer)
    print(f"C2 self-generated knowledge: {self_knowledge[:200]}...")

    results_path = os.path.join(args.out_dir, "results.jsonl")
    results = run_all_conditions(model, tokenizer, guard_qs, prompts, cfg.threshold, self_knowledge, results_path, label=1)
    print(f"Saved {len(results)} rows -> {results_path}")

    # --- Hypothesis 1 step 1-2: false negatives + failure-type split (C0) ---
    fn_df = collect_false_negatives(results, condition="C0")
    fn_df = classify_false_negatives(fn_df)
    fn_df.to_json(os.path.join(args.out_dir, "c0_false_negatives.json"), orient="records", force_ascii=False, indent=2)
    print(f"C0 false negatives: {len(fn_df)}/{len(prompts)}; failure types:\n{fn_df['failure_type'].value_counts() if len(fn_df) else 'none'}")

    c0_rows = [r for r in results if r["condition"] == "C0"]
    yes_probs = np.array([r["risk_score"] for r in c0_rows])

    # --- Hypothesis 3 (optional): occlusion + attention on extreme-skew cases ---
    if args.run_hypothesis3:
        extreme_idx = np.where((yes_probs < 0.01) | (yes_probs > 0.99))[0]
        sample_idx = extreme_idx[: args.hypothesis3_sample_size]
        h3_path = os.path.join(args.out_dir, "hypothesis3_results.jsonl")
        h3_rows = []
        with open(h3_path, "a", encoding="utf-8") as f:
            for idx in sample_idx:
                row = c0_rows[idx]
                first_q = row["questions"][0]["question"] if row["questions"] else None
                if not first_q:
                    continue
                occ = occlusion_scores(model, tokenizer, first_q, row["prompt"], yes_ids, no_ids)
                attn = attention_scores(model, tokenizer, first_q, row["prompt"])
                xval = cross_validate(occ, attn)
                h3_row = {"prompt": row["prompt"], "occlusion": occ, "attention": attn, "cross_validation": xval}
                h3_rows.append(h3_row)
                f.write(json.dumps(h3_row, ensure_ascii=False) + "\n")
                f.flush()
        print(f"Hypothesis 3: analyzed {len(h3_rows)} extreme-skew cases -> {h3_path}")

    print(f"\nAll outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
