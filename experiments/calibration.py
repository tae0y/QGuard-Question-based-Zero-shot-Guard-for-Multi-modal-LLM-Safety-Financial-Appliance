"""Hypothesis 2 — temperature/threshold calibration diagnostics.

ECE/NCE pre-diagnosis, temperature sweep, CORP (MCB/DSC/UNC) decomposition,
and threshold sweep. All operate on the yes_prob values QGuard already
produces per guard question — no change to QGuard's own scoring/graph/PageRank.

See: 100 Project/.../~260811 실험설계및수행 리뷰.md §2 가설2
"""
import os
from typing import Dict, List

os.environ["MPLBACKEND"] = "Agg"  # script/headless run; Colab's inline backend isn't installed in this venv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE over the max-class confidence."""
    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def normalized_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """NCE — like ECE but signed per bin before aggregating magnitude, so it
    also captures *directional* skew (over- vs under-confidence), not just
    magnitude. See arXiv:2405.02917 for the definition this follows.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    numerator = 0.0
    denom = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        if not mask.any():
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        w = mask.sum() / n
        numerator += w * (bin_conf - bin_acc)
        denom += w * bin_conf * (1 - bin_conf)
    return float(numerator / denom) if denom > 0 else 0.0


def extreme_mass_fraction(probs: np.ndarray, low: float = 0.01, high: float = 0.99) -> float:
    """Fraction of probabilities piled up at the extremes (p<low or p>high) —
    the concrete number the review doc's "상위 90% 이상" check refers to.
    """
    return float(np.mean((probs < low) | (probs > high)))


def reliability_diagram_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, np.ndarray]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_acc, bin_conf, bin_count = [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        bin_centers.append((lo + hi) / 2)
        bin_count.append(int(mask.sum()))
        bin_acc.append(float(labels[mask].mean()) if mask.any() else np.nan)
        bin_conf.append(float(probs[mask].mean()) if mask.any() else np.nan)
    return {
        "bin_centers": np.array(bin_centers),
        "accuracy": np.array(bin_acc),
        "confidence": np.array(bin_conf),
        "count": np.array(bin_count),
    }


def pseudo_logit_gap(yes_prob: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """QGuard only keeps the softmax'd yes/no probabilities, not the raw
    logits — but for a 2-class softmax only the logit *difference* matters,
    and that's recoverable from the probability alone:
    yes_prob = sigmoid(logit_yes - logit_no)  =>  gap = logit(yes_prob).
    Temperature scaling divides this gap by T, which is equivalent to
    scaling the original logit pair by T (any shared additive term cancels).
    """
    p = np.clip(yes_prob, eps, 1 - eps)
    return np.log(p / (1 - p))


def apply_temperature(yes_prob: np.ndarray, temperature: float) -> np.ndarray:
    """Re-derive yes_prob after dividing the yes/no logit gap by T."""
    gap = pseudo_logit_gap(yes_prob) / temperature
    return 1 / (1 + np.exp(-gap))


def temperature_sweep(
    yes_prob: np.ndarray, labels: np.ndarray,
    temperatures: List[float] = None, n_bins: int = 10,
) -> List[Dict[str, float]]:
    temperatures = temperatures or list(np.arange(0.1, 1.01, 0.1))
    rows = []
    for t in temperatures:
        probs_t = apply_temperature(yes_prob, t)
        preds_t = (probs_t >= 0.5).astype(int)
        recall = float((preds_t[labels == 1] == 1).mean()) if (labels == 1).any() else float("nan")
        rows.append({
            "temperature": t,
            "ece": expected_calibration_error(probs_t, labels, n_bins),
            "recall": recall,
        })
    return rows


def corp_decomposition(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """CORP (Consistent, Optimally binned, Reproducible) decomposition into
    MCB (miscalibration) / DSC (discrimination) / UNC (uncertainty), via
    isotonic-regression recalibration — the standard CORP estimator.
    See arXiv:2008.03033, arXiv:2108.03210.

    Brier = MCB - DSC + UNC.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    recalibrated = iso.fit_transform(probs, labels)

    brier = float(np.mean((probs - labels) ** 2))
    base_rate = float(labels.mean())
    unc = base_rate * (1 - base_rate)

    mcb = float(np.mean((probs - recalibrated) ** 2))
    dsc = mcb + unc - brier  # rearranged from Brier = MCB - DSC + UNC

    return {"brier": brier, "mcb": mcb, "dsc": dsc, "unc": unc}


def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10, out_path: str = None):
    data = reliability_diagram_data(probs, labels, n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.bar(data["bin_centers"], data["accuracy"], width=1 / n_bins, alpha=0.6, edgecolor="black", label="accuracy")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("empirical accuracy")
    ax.legend()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def threshold_sweep(
    scores: np.ndarray, labels: np.ndarray, thresholds: List[float] = None,
) -> List[Dict[str, float]]:
    thresholds = thresholds or list(np.arange(0.05, 1.0, 0.05))
    rows = []
    for th in thresholds:
        preds = (scores >= th).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        rows.append({"threshold": th, "precision": precision, "recall": recall})
    return rows
