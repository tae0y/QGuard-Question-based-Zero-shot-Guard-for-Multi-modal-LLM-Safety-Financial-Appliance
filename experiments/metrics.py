"""Metrics over a balanced results.jsonl — CPU only, no model, no GPU.

Threshold-free metrics (AUROC, PR-AUC) are the primary read: they score the
ranking itself, so they cannot be gamed by relocating a threshold. recall@theta
is secondary and only meaningful alongside FPR on the benign half.

Everything is numpy. sklearn is already a dependency but adds nothing here —
AUROC via the rank identity and PR-AUC by summation are a few lines each.
"""
from typing import Any, Dict, Iterable, List

import json

import numpy as np
import pandas as pd


def load_results(path: str) -> pd.DataFrame:
    """Read the incrementally-streamed results.jsonl into a DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, via the Mann-Whitney rank identity.

    Ties get averaged ranks, which is what makes this correct on the saturated
    scores this pipeline produces (many exactly-0.0/1.0 values) — the naive
    "count pairs where score is greater" form would score ties as misses.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # undefined with only one class present
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks within tied score groups
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision: sum over recall increments, the standard
    non-interpolated estimator (sklearn's `average_precision_score`)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    return float((precision * (recall - prev_recall)).sum())


def counts_at_threshold(labels: np.ndarray, scores: np.ndarray, theta: float) -> Dict[str, float]:
    """recall / FPR / precision / F1 at a fixed operating point.

    Predict harmful when score > theta, matching pipeline.py's `risk > threshold`.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pred = scores > theta
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    tn = int((~pred & (labels == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {
        "threshold": theta, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "fpr": fpr, "precision": precision, "f1": f1,
    }


def condition_metrics(df: pd.DataFrame, theta: float = 0.50) -> pd.DataFrame:
    """Per-condition (C0/C1/C2/C1') metric table over the balanced set."""
    out = []
    for cond, g in df.groupby("condition"):
        labels, scores = g["label"].to_numpy(), g["risk_score"].to_numpy(dtype=float)
        row = {"condition": cond, "n": len(g), "n_harmful": int((labels == 1).sum()),
               "n_benign": int((labels == 0).sum())}
        row.update(counts_at_threshold(labels, scores, theta))
        row["auroc"] = auroc(labels, scores)
        row["pr_auc"] = pr_auc(labels, scores)
        out.append(row)
    return pd.DataFrame(out).sort_values("condition").reset_index(drop=True)


def threshold_sweep(df: pd.DataFrame, condition: str, thresholds: Iterable[float] = None) -> pd.DataFrame:
    """Sweep theta for one condition — this is the null competitor for
    hypothesis 2. Temperature has to beat "just move theta" to claim anything.

    The default grid is refined where the curve is steep: risk_score is
    concentrated near the PageRank aggregate's operating range, so a uniform
    0.05 grid can step straight over the interesting region.
    """
    g = df[df["condition"] == condition]
    labels, scores = g["label"].to_numpy(), g["risk_score"].to_numpy(dtype=float)
    if thresholds is None:
        coarse = np.arange(0.05, 1.0, 0.05)
        # refine around the observed score mass, where recall/FPR actually move
        lo, hi = np.quantile(scores, [0.05, 0.95]) if len(scores) else (0.0, 1.0)
        fine = np.linspace(lo, hi, 40) if hi > lo else np.array([])
        thresholds = np.unique(np.concatenate([coarse, fine]))
    return pd.DataFrame([counts_at_threshold(labels, scores, t) for t in thresholds])


def extreme_mass_fraction(df: pd.DataFrame, condition: str = "C0",
                          low: float = 0.01, high: float = 0.99) -> float:
    """Saturation gate: fraction of per-question yes-probabilities pinned at the
    extremes. If ~90%+ are saturated, temperature has almost nothing to move and
    hypothesis 2 scopes down to reporting the saturation itself.

    Measured over per-question yes_prob (pre-aggregation), not the aggregated
    risk_score — saturation is a property of the model's yes/no logits.
    """
    probs = np.array([q["yes_prob"] for _, r in df[df["condition"] == condition].iterrows()
                      for q in r["questions"]], dtype=float)
    if probs.size == 0:
        return float("nan")
    return float(((probs < low) | (probs > high)).mean())


def _apply_temperature(yes_prob: float, t: float) -> float:
    """Rescale a yes-probability by temperature T in logit space.

    yes_prob comes from a softmax over (yes_logit, no_logit), so its logit IS
    that logit margin; dividing by T and re-sigmoiding is exactly temperature
    scaling on the two-way decision. Clipped because saturated probabilities
    reach exactly 0.0/1.0 in float, whose logit is infinite.
    """
    p = min(max(yes_prob, 1e-12), 1 - 1e-12)
    return float(1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) / t))))


def temperature_sweep(df: pd.DataFrame, condition: str = "C0",
                      temperatures: Iterable[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
                      theta: float = 0.50) -> pd.DataFrame:
    """Apply T to per-question yes/no logits, re-run PageRank aggregation, and
    report pre- vs post-aggregation distributions plus AUROC per T.

    The point of the pre/post split: per-question temperature is a monotone
    transform of each question's margin, so at question level it cannot change
    any per-sample ranking. Any real effect can only enter through the
    nonlinear PageRank aggregation — so the two must be reported separately.

    Verdict rule: if AUROC is flat while recall@theta moves, that is a
    threshold-relocation artifact, not a discriminability gain.
    """
    from qguard.graph import build_graph_from_results, pagerank_risk_score

    g = df[df["condition"] == condition]
    labels = g["label"].to_numpy()
    rows = []
    for t in temperatures:
        pre, post = [], []
        for _, r in g.iterrows():
            scaled = [{"question": q["question"], "yes_prob": _apply_temperature(q["yes_prob"], t)}
                      for q in r["questions"]]
            pre.extend(s["yes_prob"] for s in scaled)
            group_by = {q["question"]: q.get("category", "Unknown") for q in r["questions"]}
            post.append(pagerank_risk_score(build_graph_from_results(scaled, group_by)))
        post_arr = np.array(post, dtype=float)
        pre_arr = np.array(pre, dtype=float)
        row = {"temperature": t,
               "pre_agg_mean": float(pre_arr.mean()), "pre_agg_std": float(pre_arr.std()),
               "pre_agg_extreme_frac": float(((pre_arr < 0.01) | (pre_arr > 0.99)).mean()),
               "post_agg_mean": float(post_arr.mean()), "post_agg_std": float(post_arr.std()),
               "auroc": auroc(labels, post_arr)}
        row.update({k: counts_at_threshold(labels, post_arr, theta)[k] for k in ("recall", "fpr", "f1")})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(results_path: str, theta: float = 0.50) -> Dict[str, Any]:
    """Everything the pilot/full-run report needs, from results.jsonl alone."""
    df = load_results(results_path)
    return {
        "per_condition": condition_metrics(df, theta),
        "threshold_sweep_c0": threshold_sweep(df, "C0"),
        "extreme_mass_fraction_c0": extreme_mass_fraction(df, "C0"),
    }


if __name__ == "__main__":
    import sys

    out = summarize(sys.argv[1])
    print(out["per_condition"].to_string(index=False))
    print(f"\nC0 extreme_mass_fraction: {out['extreme_mass_fraction_c0']:.4f}")
    print("\nC0 threshold sweep (null competitor for hypothesis 2):")
    print(out["threshold_sweep_c0"].to_string(index=False))
