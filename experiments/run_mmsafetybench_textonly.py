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

from experiments.mminstruct_benign import load_mminstruct_benign

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
                        help="Also score MMInstruct benign (label=0) — required before theta can be swept")
    parser.add_argument("--skip_harmful", action="store_true", help="Benign only (run the two halves in separate sessions)")
    parser.add_argument("--n_benign_caption", type=int, default=901, help="Paper Sec 4.1.1: 901")
    parser.add_argument("--n_benign_qa", type=int, default=1100, help="Paper Sec 4.1.1: 1100")
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
        n_cap, n_qa = args.n_benign_caption, args.n_benign_qa
        if args.n_samples:  # pilot: keep the benign half the same size as the harmful half
            n_cap = n_qa = args.n_samples * len(args.categories) // 2
        benign = load_mminstruct_benign(n_caption=n_cap, n_qa=n_qa, seed=cfg.seed)
        actual_counts = {}
        for task in ("caption", "qa"):
            items = [{"category": f"benign_{b['scenario']}", "row_id": f"{task}:{i}",
                      "prompt": b["prompt"], "label": 0, "task": task, "scenario": b["scenario"]}
                     for i, b in enumerate(benign) if b["task"] == task]
            actual_counts[f"n_benign_{task}_actual"] = len(items)
            out_path = os.path.join(args.out_dir, f"benign_{task}.jsonl")
            n = run_prompts(model, tokenizer, guard_qs, items, cfg.threshold, out_path, **couplings)
            print(f"benign_{task}: +{n} rows -> {out_path}", flush=True)
        with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({**vars(args), "n_guard_questions": sum(len(v) for v in guard_qs.values()),
                       **actual_counts}, f, indent=2)

    print(f"\nDone -> {args.out_dir}")


if __name__ == "__main__":
    main()
