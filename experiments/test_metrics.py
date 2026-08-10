"""CPU-only checks on the metric math. No GPU, no model download, no framework.

    uv run experiments/test_metrics.py
"""
import numpy as np

from experiments.metrics import (
    _apply_temperature, auroc, counts_at_threshold, extreme_mass_fraction, pr_auc,
)
import pandas as pd


def test_separable():
    """Perfectly separable scores: AUROC 1.0, and at theta=0.5 every harmful
    fires and no benign does."""
    labels = np.array([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert auroc(labels, scores) == 1.0
    assert pr_auc(labels, scores) == 1.0
    m = counts_at_threshold(labels, scores, 0.5)
    assert (m["recall"], m["fpr"], m["precision"], m["f1"]) == (1.0, 0.0, 1.0, 1.0)


def test_inverted():
    """Ranking exactly backwards is AUROC 0.0 — catches a flipped comparison."""
    labels = np.array([1, 1, 0, 0])
    assert auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0


def test_random_is_chance():
    """Random scores sit at chance. Loose bound: this is a sanity check on the
    estimator, not a distributional test."""
    rng = np.random.default_rng(1234)
    labels = np.array([1] * 500 + [0] * 500)
    assert abs(auroc(labels, rng.random(1000)) - 0.5) < 0.05


def test_all_tied_is_half():
    """Every score identical = no information = 0.5. This is the case the
    naive pair-counting AUROC gets wrong (it would return 0.0), and saturated
    guard-question outputs produce real ties, so it must hold."""
    assert auroc(np.array([1, 1, 0, 0]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_single_class_is_nan():
    """The degenerate case this whole redesign exists to avoid: with only
    label=1 present, AUROC is undefined rather than silently 1.0."""
    assert np.isnan(auroc(np.array([1, 1, 1]), np.array([0.9, 0.5, 0.1])))


def test_pr_auc_baseline():
    """With no signal, average precision approaches the positive rate (0.25)."""
    rng = np.random.default_rng(7)
    labels = np.array([1] * 250 + [0] * 750)
    assert abs(pr_auc(labels, rng.random(1000)) - 0.25) < 0.05


def test_threshold_is_strict_greater():
    """Matches pipeline.py's `risk > threshold`: a score exactly at theta is
    predicted unharmful. Off-by-one here would silently shift every recall."""
    m = counts_at_threshold(np.array([1, 1]), np.array([0.5, 0.51]), 0.5)
    assert (m["tp"], m["fn"]) == (1, 1)


def test_temperature_is_monotone_per_question():
    """The premise of hypothesis 2: temperature cannot reorder scores at
    question level, so any effect must come from PageRank aggregation."""
    probs = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    for t in (0.5, 1.0, 2.0, 5.0):
        scaled = [_apply_temperature(p, t) for p in probs]
        assert scaled == sorted(scaled)
    assert abs(_apply_temperature(0.73, 1.0) - 0.73) < 1e-9  # T=1 is identity
    # T>1 pulls toward 0.5, T<1 pushes toward the extremes
    assert _apply_temperature(0.9, 3.0) < 0.9
    assert _apply_temperature(0.9, 0.5) > 0.9
    # saturated inputs stay finite rather than producing inf/nan
    assert 0.0 < _apply_temperature(1.0, 2.0) < 1.0


def test_extreme_mass_fraction():
    """3 of 4 per-question probabilities are pinned at the extremes."""
    df = pd.DataFrame([{
        "condition": "C0",
        "questions": [{"yes_prob": 0.999}, {"yes_prob": 0.0001},
                      {"yes_prob": 0.5}, {"yes_prob": 0.995}],
    }])
    assert extreme_mass_fraction(df, "C0") == 0.75


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall metric checks passed")
