#!/usr/bin/env python3
"""Financial-advice recall experiment — C0/C1/C2/C1' x calibration diagnostics.

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
from experiments.calibration import (
    corp_decomposition,
    expected_calibration_error,
    extreme_mass_fraction,
    normalized_calibration_error,
    plot_reliability_diagram,
    temperature_sweep,
    threshold_sweep,
)
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
    threshold: float, self_generated_knowledge: str,
) -> List[Dict[str, Any]]:
    conditions = build_conditions(self_generated_knowledge)
    rows = []
    for prompt in prompts:
        for name, cond in conditions.items():
            augmented = apply_condition(prompt, cond)
            result = evaluate_prompt_with_pagerank(
                model, tokenizer, augmented, guard_qs, threshold=threshold,
            )
            rows.append({
                "prompt": prompt,
                "condition": name,
                "prediction": result["prediction"],
                "risk_score": result["risk_score"],
                "questions": result["questions"],
                "label": 1,  # financial_advice subset is entirely harmful
            })
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

    results = run_all_conditions(model, tokenizer, guard_qs, prompts, cfg.threshold, self_knowledge)
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} rows -> {results_path}")

    # --- Hypothesis 1 step 1-2: false negatives + failure-type split (C0) ---
    fn_df = collect_false_negatives(results, condition="C0")
    fn_df = classify_false_negatives(fn_df)
    fn_df.to_json(os.path.join(args.out_dir, "c0_false_negatives.json"), orient="records", force_ascii=False, indent=2)
    print(f"C0 false negatives: {len(fn_df)}/{len(prompts)}; failure types:\n{fn_df['failure_type'].value_counts() if len(fn_df) else 'none'}")

    # --- Hypothesis 2 step 1: ECE/NCE pre-diagnosis + reliability diagram (C0) ---
    c0_rows = [r for r in results if r["condition"] == "C0"]
    yes_probs = np.array([r["risk_score"] for r in c0_rows])
    labels = np.array([r["label"] for r in c0_rows])
    ece = expected_calibration_error(yes_probs, labels)
    nce = normalized_calibration_error(yes_probs, labels)
    extreme_frac = extreme_mass_fraction(yes_probs)
    print(f"C0 ECE={ece:.4f} NCE={nce:.4f} extreme_mass_fraction={extreme_frac:.4f}")
    plot_reliability_diagram(yes_probs, labels, out_path=os.path.join(args.out_dir, "reliability_diagram_c0.png"))

    diagnostics = {"ece": ece, "nce": nce, "extreme_mass_fraction": extreme_frac}
    with open(os.path.join(args.out_dir, "calibration_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    # --- Hypothesis 2 step 2-5: temperature sweep, CORP, threshold sweep ---
    temp_rows = temperature_sweep(yes_probs, labels)
    corp_before = corp_decomposition(yes_probs, labels)
    best_t = min(temp_rows, key=lambda r: r["ece"])["temperature"]
    from experiments.calibration import apply_temperature
    corp_after = corp_decomposition(apply_temperature(yes_probs, best_t), labels)
    thresh_rows = threshold_sweep(yes_probs, labels)

    with open(os.path.join(args.out_dir, "hypothesis2_results.json"), "w") as f:
        json.dump({
            "temperature_sweep": temp_rows,
            "best_temperature": best_t,
            "corp_before": corp_before,
            "corp_after": corp_after,
            "threshold_sweep": thresh_rows,
        }, f, indent=2)
    print(f"CORP before: {corp_before}\nCORP after (T={best_t}): {corp_after}")

    # --- Hypothesis 3 (optional): occlusion + attention on extreme-skew cases ---
    if args.run_hypothesis3:
        extreme_idx = np.where((yes_probs < 0.01) | (yes_probs > 0.99))[0]
        sample_idx = extreme_idx[: args.hypothesis3_sample_size]
        h3_rows = []
        for idx in sample_idx:
            row = c0_rows[idx]
            first_q = row["questions"][0]["question"] if row["questions"] else None
            if not first_q:
                continue
            occ = occlusion_scores(model, tokenizer, first_q, row["prompt"], yes_ids, no_ids)
            attn = attention_scores(model, tokenizer, first_q, row["prompt"])
            xval = cross_validate(occ, attn)
            h3_rows.append({"prompt": row["prompt"], "occlusion": occ, "attention": attn, "cross_validation": xval})
        with open(os.path.join(args.out_dir, "hypothesis3_results.json"), "w") as f:
            json.dump(h3_rows, f, ensure_ascii=False, indent=2)
        print(f"Hypothesis 3: analyzed {len(h3_rows)} extreme-skew cases")

    print(f"\nAll outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
