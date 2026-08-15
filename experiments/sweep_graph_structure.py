#!/usr/bin/env python3
"""Sweep PageRank graph-structure knobs (edge weights, directionality, damping)
over a finished run's stored yes_prob, before touching guard questions.

Diagnoses whether the degenerate theta documented in the 8/18 review (recall
1.0 across every category, because F1-argmax lands at the dataset's global
risk_score minimum) traces back to the aggregation formula itself, rather
than the question set or theta-selection procedure.

CPU only -- no model, no GPU. Rebuilds the PageRank graph from each row's
stored `questions` (yes_prob), exactly like merge_and_score.py and
metrics.py::temperature_sweep already do.

  uv run experiments/sweep_graph_structure.py <results_dir>
  uv run experiments/sweep_graph_structure.py <results_dir> --group_coupling 0,0.1,0.5,1.0
"""
import argparse
import json
from typing import Any, Dict, List

from qguard.graph import build_graph_from_results, pagerank_risk_score
from experiments.sweep_theta import auroc, best_theta, load_rows

# The paper's top category (Illegal_Activitiy) and the one this study centers
# on (Financial_Advice) -- a one-line snapshot of whether the expected split
# survives a given graph configuration.
SNAPSHOT_CATS = ("Illegal_Activitiy", "Financial_Advice")


def rescore(rows: List[Dict[str, Any]], group_coupling: float, intra_group_q_coupling: float,
            symmetric_coupling: bool, alpha: float) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        qs = [{"question": q["question"], "yes_prob": q["yes_prob"]} for q in r["questions"]]
        gb = {q["question"]: q["category"] for q in r["questions"]}
        G = build_graph_from_results(
            qs, gb,
            group_coupling=group_coupling,
            intra_group_q_coupling=intra_group_q_coupling,
            symmetric_coupling=symmetric_coupling,
        )
        out.append({**r, "risk_score": pagerank_risk_score(G, alpha=alpha)})
    return out


def is_degenerate(rows: List[Dict[str, Any]], theta: float) -> bool:
    """True if every row -- harmful and benign alike -- scores above theta,
    i.e. the "predict everything harmful" collapse this sweep checks for."""
    return sum(r["risk_score"] > theta for r in rows) == len(rows)


def category_recall(rows: List[Dict[str, Any]], theta: float, category: str) -> float:
    cat_rows = [r for r in rows if r["label"] == 1 and r["category"] == category]
    if not cat_rows:
        return float("nan")
    return sum(r["risk_score"] > theta for r in cat_rows) / len(cat_rows)


def run_config(rows: List[Dict[str, Any]], label: str, group_coupling: float,
                intra_group_q_coupling: float, symmetric_coupling: bool, alpha: float) -> Dict[str, Any]:
    scored = rescore(rows, group_coupling, intra_group_q_coupling, symmetric_coupling, alpha)
    theta, p, rec, f1 = best_theta(scored)
    out = {
        "label": label,
        "group_coupling": group_coupling,
        "intra_group_q_coupling": intra_group_q_coupling,
        "symmetric_coupling": symmetric_coupling,
        "alpha": alpha,
        "auroc": auroc(scored),
        "theta": theta,
        "precision": p,
        "recall": rec,
        "f1": f1,
        "degenerate": is_degenerate(scored, theta),
    }
    for cat in SNAPSHOT_CATS:
        out[f"recall_{cat}"] = category_recall(scored, theta, cat)
    return out


def default_grid() -> List[Dict[str, Any]]:
    grid = [
        dict(label="baseline (paper 4.1.3)", group_coupling=1.0, intra_group_q_coupling=0.3,
             symmetric_coupling=False, alpha=0.85),
        dict(label="no-coupling (naive-mean ablation)", group_coupling=0.0, intra_group_q_coupling=0.0,
             symmetric_coupling=False, alpha=0.85),
        dict(label="symmetric(baseline)", group_coupling=1.0, intra_group_q_coupling=0.3,
             symmetric_coupling=True, alpha=0.85),
        dict(label="symmetric(no-coupling)", group_coupling=0.0, intra_group_q_coupling=0.0,
             symmetric_coupling=True, alpha=0.85),
    ]
    for gc in (0.1, 0.3, 0.5):
        grid.append(dict(label=f"group_coupling={gc}", group_coupling=gc, intra_group_q_coupling=0.3,
                          symmetric_coupling=False, alpha=0.85))
    for a in (0.5, 0.7, 0.95):
        grid.append(dict(label=f"alpha={a}", group_coupling=1.0, intra_group_q_coupling=0.3,
                          symmetric_coupling=False, alpha=a))
    return grid


def custom_grid(group_couplings: List[float], intra_group_q_coupling: float,
                 symmetric_coupling: bool, alphas: List[float]) -> List[Dict[str, Any]]:
    return [
        dict(label=f"gc={gc} iqc={intra_group_q_coupling} sym={symmetric_coupling} alpha={a}",
             group_coupling=gc, intra_group_q_coupling=intra_group_q_coupling,
             symmetric_coupling=symmetric_coupling, alpha=a)
        for gc in group_couplings for a in alphas
    ]


def print_table(results: List[Dict[str, Any]]) -> None:
    cols = ["설정", "gc", "iqc", "방향", "alpha", "AUROC", "최적θ", "퇴화",
            f"{SNAPSHOT_CATS[0]} recall", f"{SNAPSHOT_CATS[1]} recall"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in results:
        direction = "양방향" if r["symmetric_coupling"] else "단방향"
        degen = "예" if r["degenerate"] else "아니오"
        cells = [r["label"], r["group_coupling"], r["intra_group_q_coupling"], direction, r["alpha"],
                 f"{r['auroc']:.4f}", f"{r['theta']:.4f}", degen,
                 f"{r[f'recall_{SNAPSHOT_CATS[0]}']:.4f}", f"{r[f'recall_{SNAPSHOT_CATS[1]}']:.4f}"]
        print("| " + " | ".join(str(c) for c in cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--group_coupling", default=None,
                     help="Comma list (e.g. 0,0.1,0.5,1.0) -- replaces the default grid's coupling axis")
    ap.add_argument("--intra_group_q_coupling", type=float, default=0.3)
    ap.add_argument("--alpha", default=None,
                     help="Comma list (e.g. 0.5,0.85,0.95) -- replaces the default grid's alpha axis")
    ap.add_argument("--symmetric_coupling", action="store_true",
                     help="Only used together with --group_coupling/--alpha custom axes")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    rows = load_rows(args.results_dir)
    if not any(r["label"] == 0 for r in rows):
        print("benign 행이 없습니다 -- theta를 구할 수 없어 스윕이 불가합니다.")
        return

    if args.group_coupling or args.alpha:
        gcs = [float(x) for x in args.group_coupling.split(",")] if args.group_coupling else [1.0]
        alphas = [float(x) for x in args.alpha.split(",")] if args.alpha else [0.85]
        grid = custom_grid(gcs, args.intra_group_q_coupling, args.symmetric_coupling, alphas)
    else:
        grid = default_grid()

    n_pos = sum(r["label"] == 1 for r in rows)
    print(f"{len(rows)} rows: {n_pos} harmful / {len(rows) - n_pos} benign\n")

    results = [run_config(rows, **cfg) for cfg in grid]
    print_table(results)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n-> {args.out_json}")


if __name__ == "__main__":
    main()
