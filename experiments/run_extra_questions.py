#!/usr/bin/env python3
"""Score *additional* guard questions against an already-finished run.

The base run stored every prompt and its 35 per-question probabilities, so
testing "what if we add one advisory-type question?" only needs the new
question scored — 1 forward pass per row instead of 36. On the full balanced
set that is ~1 hour instead of ~11.

Output is a separate directory of jsonl, same row_id keys as the base run.
`merge_and_score.py` then joins the two and recomputes PageRank over 36
questions. Nothing overwrites the base run.

Usage:
  uv run experiments/run_extra_questions.py \
    --base_dir  "/content/drive/MyDrive/qguard_results/mmsafety_full" \
    --out_dir   "/content/drive/MyDrive/qguard_results/extra_s1" \
    --guard_questions_json files/guard_questions_s1_advisory.json
"""
import argparse
import glob
import json
import os
import time
from typing import Any, Dict, List

import torch

from qguard.config import RunConfig
from qguard.modeling import load_internvl_text_or_mm
from qguard.scoring import score_guard_question_yes_prob
from qguard.seed import set_seed
from qguard.token_utils import gather_yes_no_ids

from experiments.run_mmsafetybench_textonly import done_ids


def base_prompts(base_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """{source filename -> [{row_id, prompt, category, label}]} from a finished run."""
    out = {}
    for path in sorted(glob.glob(os.path.join(base_dir, "*.jsonl"))):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append({k: r[k] for k in ("row_id", "prompt", "category", "label")})
        if rows:
            out[os.path.basename(path)] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base_dir", required=True, help="Finished run to take prompts from")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--guard_questions_json", required=True, help="Only the NEW questions")
    ap.add_argument("--model_path", default="OpenGVLab/InternVL2_5-4B")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = RunConfig(model_path=args.model_path, guard_questions_json=args.guard_questions_json, seed=args.seed)
    cfg.apply_env()
    set_seed(cfg.seed)

    with open(cfg.guard_questions_json, "r", encoding="utf-8") as f:
        extra_qs = json.load(f)
    n_extra = sum(len(v) for v in extra_qs.values())

    by_file = base_prompts(args.base_dir)
    total = sum(len(v) for v in by_file.values())
    if not total:
        raise SystemExit(f"{args.base_dir} 에 결과 jsonl이 없습니다")
    print(f"base: {total} rows / extra questions: {n_extra} -> {total * n_extra} forward passes")

    print(f"CUDA available: {torch.cuda.is_available()}")
    model, tokenizer = load_internvl_text_or_mm(cfg.model_path)
    yes_ids, no_ids = gather_yes_no_ids(tokenizer)

    with open(os.path.join(args.out_dir, "extra_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "questions": extra_qs}, f, indent=2, ensure_ascii=False)

    for fname, rows in by_file.items():
        out_path = os.path.join(args.out_dir, fname)
        skip = done_ids(out_path)
        written = 0
        with open(out_path, "a", encoding="utf-8") as f:
            for item in rows:
                if item["row_id"] in skip:
                    continue
                t0 = time.time()
                scored = []
                for group, qs in extra_qs.items():
                    for q in qs:
                        res = score_guard_question_yes_prob(
                            model, tokenizer, q, item["prompt"], yes_ids, no_ids)
                        res["category"] = group
                        scored.append(res)
                f.write(json.dumps({**item, "questions": scored,
                                    "elapsed_sec": round(time.time() - t0, 3)},
                                   ensure_ascii=False) + "\n")
                f.flush()
                written += 1
        print(f"{fname}: +{written} rows -> {out_path}", flush=True)

    print(f"\nDone -> {args.out_dir}\n"
          f"다음: uv run experiments/merge_and_score.py {args.base_dir} {args.out_dir} --out_dir <합친결과>")


if __name__ == "__main__":
    main()
