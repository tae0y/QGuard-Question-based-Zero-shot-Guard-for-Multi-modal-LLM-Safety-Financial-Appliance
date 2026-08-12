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

Benign set (--with_benign) is MMInstruct QA + FiQA, not the paper's
caption+QA split — decided 2026-08-18 after the caption task turned out to
collapse to ~20-30 boilerplate templates once text-only (see
`mminstruct_benign.py` deviation 4). Composition matches the harmful set 1:1
and weights FiQA (financial-domain benign) by the harmful set's own
Financial_Advice+EconomicHarm share, so the benign class isn't skewed
away from the categories this study cares about most. See ECONOMIC_RATIO.

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

from experiments.benign_prompts import load_benign_prompts
from experiments.mminstruct_benign import load_mminstruct_qa_benign

# https://huggingface.co/api/datasets/PKU-Alignment/MM-SafetyBench — every
# config carries the same four splits; Text_only is the only image-free one.
CATEGORIES = [
    "EconomicHarm", "Financial_Advice", "Fraud", "Gov_Decision", "HateSpeech",
    "Health_Consultation", "Illegal_Activitiy", "Legal_Opinion",
    "Malware_Generation", "Physical_Harm", "Political_Lobbying",
    "Privacy_Violence", "Sex",
]

# Measured 2026-08-12 from the full Text_only split: EconomicHarm=122,
# Financial_Advice=167, all categories combined=1680. Both categories are
# economic-domain content (EconomicHarm: illicit finance schemes;
# Financial_Advice: personal-finance guidance) — decided 2026-08-18 to count
# both, not just Financial_Advice, when weighting the benign mix.
HARMFUL_TOTAL = 1680
ECONOMIC_HARMFUL = 122 + 167
ECONOMIC_RATIO = ECONOMIC_HARMFUL / HARMFUL_TOTAL


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


def load_category(category: str, n_samples: int = None) -> List[Dict[str, Any]]:
    # Point at the Text_only parquet directly. Naming the config instead would
    # pull SD/SD_TYPO/TYPO too — ~60MB of images per category we never open.
    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/PKU-Alignment/MM-SafetyBench/data/{category}/Text_only.parquet",
        split="train",
    )
    rows = [r for r in ds if r.get("question")]
    if n_samples:
        rows = rows[:n_samples]
    # every MM-SafetyBench row is an unsafe prompt
    return [{"category": category, "row_id": str(r.get("id", i)), "prompt": r["question"], "label": 1}
            for i, r in enumerate(rows)]


def run_prompts(
    model, tokenizer, guard_qs: Dict[str, List[str]], items: List[Dict[str, Any]],
    threshold: float, out_path: str,
    group_coupling: float = 1.0, intra_group_q_coupling: float = 0.3,
) -> int:
    """Score each item and append to out_path. Items carry their own
    row_id/label/category, so harmful and benign run through one code path —
    they must see identical guard questions and couplings or theta is meaningless."""
    skip = done_ids(out_path)
    written = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for item in items:
            if item["row_id"] in skip:
                continue
            t0 = time.time()
            result = evaluate_prompt_with_pagerank(
                model, tokenizer, item["prompt"], guard_qs, threshold=threshold,
                group_coupling=group_coupling,
                intra_group_q_coupling=intra_group_q_coupling,
            )
            f.write(json.dumps({
                **item,
                "threshold": threshold,
                "group_coupling": group_coupling,
                "intra_group_q_coupling": intra_group_q_coupling,
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
    # Paper Sec 3.4/4.1.3: theta is tuned per dataset, and no value is given for
    # MM-SafetyBench (only ~0.75 ToxicChat / ~0.7 WildGuardMix in Sec 5.3). The
    # default here is a placeholder — risk_score is what matters, and every
    # per-question probability is logged so theta can be re-swept afterwards.
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--group_coupling", type=float, default=1.0, help="Group-group edge weight (paper Sec 4.1.3: 1.0)")
    parser.add_argument("--intra_group_q_coupling", type=float, default=0.3, help="Question-question edge weight (paper Sec 4.1.3: 0.3)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--n_samples", type=int, default=None, help="Per-category cap; omit for the full split")
    parser.add_argument("--with_benign", action="store_true",
                        help="Also score benign (label=0): MMInstruct QA + FiQA — required before theta can be swept")
    parser.add_argument("--skip_harmful", action="store_true", help="Benign only (run the two halves in separate sessions)")
    parser.add_argument("--n_benign_total", type=int, default=HARMFUL_TOTAL,
                        help="Total benign count, split into FiQA/QA by ECONOMIC_RATIO. Default matches harmful 1:1.")
    parser.add_argument("--per_scenario_cap", type=int, default=61,
                        help="MMInstruct QA: max unique prompts per domain (23 domains * 61 >= default n_qa)")
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

    couplings = dict(group_coupling=args.group_coupling,
                     intra_group_q_coupling=args.intra_group_q_coupling)

    if not args.skip_harmful:
        for cat in args.categories:
            out_path = os.path.join(args.out_dir, f"{cat}.jsonl")
            items = load_category(cat, args.n_samples)
            n = run_prompts(model, tokenizer, guard_qs, items, cfg.threshold, out_path, **couplings)
            print(f"{cat}: +{n} rows -> {out_path}", flush=True)

    if args.with_benign:
        n_total = args.n_samples * len(args.categories) if args.n_samples else args.n_benign_total
        n_fiqa = round(n_total * ECONOMIC_RATIO)
        n_qa = n_total - n_fiqa

        qa_benign = load_mminstruct_qa_benign(n=n_qa, per_scenario_cap=args.per_scenario_cap, seed=cfg.seed)
        qa_items = [{"category": f"benign_{b['scenario']}", "row_id": f"qa:{i}",
                     "prompt": b["prompt"], "label": 0, "task": "qa", "scenario": b["scenario"]}
                    for i, b in enumerate(qa_benign)]

        fiqa_prompts = load_benign_prompts(n_fiqa, seed=cfg.seed)
        fiqa_items = [{"category": "benign_fiqa", "row_id": f"fiqa:{i}", "prompt": p, "label": 0,
                       "task": "fiqa", "scenario": "fiqa"} for i, p in enumerate(fiqa_prompts)]

        actual_counts = {"n_benign_qa_actual": len(qa_items), "n_benign_fiqa_actual": len(fiqa_items)}
        for task, items in (("qa", qa_items), ("fiqa", fiqa_items)):
            out_path = os.path.join(args.out_dir, f"benign_{task}.jsonl")
            n = run_prompts(model, tokenizer, guard_qs, items, cfg.threshold, out_path, **couplings)
            print(f"benign_{task}: +{n} rows -> {out_path}", flush=True)
        with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({**vars(args), "n_guard_questions": sum(len(v) for v in guard_qs.values()),
                       **actual_counts}, f, indent=2)

    print(f"\nDone -> {args.out_dir}")


if __name__ == "__main__":
    main()
