#!/usr/bin/env python3
"""Join a base run with extra-question runs and recompute PageRank over the union.

CPU only. The merged directory has the same shape as a normal run, so
`sweep_theta.py` and `report_tables.py` read it unchanged — theta is re-derived
from the merged scores, exactly as the paper derives it per dataset.

  # S1 = 35 base questions + 1 advisory question
  uv run experiments/merge_and_score.py <base_dir> <extra_dir> --out_dir <merged_dir>

  # then, as usual
  uv run experiments/report_tables.py <merged_dir>

Rows present in base but missing from extra are dropped, with a count printed:
scoring them over 35 questions while the rest use 36 would make the two sets
incomparable.
"""
import argparse
import glob
import json
import os
from typing import Any, Dict, List

from qguard.graph import build_graph_from_results, pagerank_risk_score


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # truncated tail
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir")
    ap.add_argument("extra_dirs", nargs="+", help="One or more extra-question run dirs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.50, help="Placeholder; theta is swept afterwards")
    ap.add_argument("--group_coupling", type=float, default=1.0)
    ap.add_argument("--intra_group_q_coupling", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    merged_total = dropped_total = 0

    for base_path in sorted(glob.glob(os.path.join(args.base_dir, "*.jsonl"))):
        fname = os.path.basename(base_path)

        # row_id -> extra questions, unioned across every extra dir
        extra: Dict[str, List[Dict[str, Any]]] = {}
        for d in args.extra_dirs:
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                continue
            for r in read_jsonl(p):
                extra.setdefault(r["row_id"], []).extend(r["questions"])

        out_rows, dropped = [], 0
        for r in read_jsonl(base_path):
            add = extra.get(r["row_id"])
            if not add:
                dropped += 1
                continue
            qs = r["questions"] + add
            gb = {q["question"]: q["category"] for q in qs}
            score = pagerank_risk_score(build_graph_from_results(
                [{"question": q["question"], "yes_prob": q["yes_prob"]} for q in qs], gb,
                group_coupling=args.group_coupling,
                intra_group_q_coupling=args.intra_group_q_coupling))
            out_rows.append({**r, "questions": qs, "n_questions": len(qs),
                             "risk_score": score,
                             "prediction": "harmful" if score > args.threshold else "unharmful",
                             "threshold": args.threshold,
                             "group_coupling": args.group_coupling,
                             "intra_group_q_coupling": args.intra_group_q_coupling})

        if out_rows:
            with open(os.path.join(args.out_dir, fname), "w", encoding="utf-8") as f:
                for r in out_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        merged_total += len(out_rows)
        dropped_total += dropped
        note = f"  (extra 누락 {dropped}건 제외)" if dropped else ""
        print(f"{fname}: {len(out_rows)} rows{note}")

    n_q = "?"
    if merged_total:
        first = read_jsonl(sorted(glob.glob(os.path.join(args.out_dir, "*.jsonl")))[0])
        n_q = first[0]["n_questions"] if first else "?"
    print(f"\n{merged_total} rows merged, {n_q} questions each -> {args.out_dir}")
    if dropped_total:
        print(f"경고: extra에 없는 {dropped_total}건을 제외했습니다. "
              f"extra 실행이 끝나지 않았다면 이어서 돌린 뒤 다시 병합하십시오.")
    print(f"다음: uv run experiments/report_tables.py {args.out_dir}")


if __name__ == "__main__":
    main()
