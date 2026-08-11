#!/usr/bin/env python3
"""CPU self-check for the MM-SafetyBench runner: resume logic + logit/prob math.

  uv run experiments/test_mmsafetybench_run.py
"""
import json
import math
import os
import tempfile

import torch

from experiments.run_mmsafetybench_textonly import CATEGORIES, done_ids
from experiments.sweep_theta import auroc, best_theta, prf
from qguard.token_utils import yes_no_logits_from_logits, yes_no_probs_from_logits


def test_done_ids_survives_truncated_line():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"row_id": "0"}) + "\n")
        f.write(json.dumps({"row_id": "1"}) + "\n")
        f.write('{"row_id": "2", "prompt": "kill')  # process died mid-write
        path = f.name
    try:
        assert done_ids(path) == {"0", "1"}
        assert done_ids(path + ".nope") == set()
    finally:
        os.unlink(path)


def test_probs_match_logits():
    torch.manual_seed(0)
    logits = torch.randn(100)
    y, n = yes_no_logits_from_logits(logits, [1, 2], [3, 4])
    yp, np_ = yes_no_probs_from_logits(logits, [1, 2], [3, 4])
    assert abs(yp + np_ - 1.0) < 1e-6
    assert abs(yp - 1 / (1 + math.exp(n - y))) < 1e-6, (yp, y, n)


def test_categories_complete():
    assert len(CATEGORIES) == 13 and len(set(CATEGORIES)) == 13


def _rows(pairs):
    return [{"label": l, "risk_score": s, "category": "c"} for l, s in pairs]


def test_auroc_known_values():
    # perfectly separated / perfectly inverted / all-ties
    assert abs(auroc(_rows([(1, 0.9), (1, 0.8), (0, 0.2), (0, 0.1)])) - 1.0) < 1e-9
    assert abs(auroc(_rows([(1, 0.1), (1, 0.2), (0, 0.8), (0, 0.9)])) - 0.0) < 1e-9
    assert abs(auroc(_rows([(1, 0.5), (1, 0.5), (0, 0.5), (0, 0.5)])) - 0.5) < 1e-9


def test_best_theta_separates_when_separable():
    rows = _rows([(1, 0.9), (1, 0.8), (0, 0.2), (0, 0.1)])
    theta, p, r, f1 = best_theta(rows)
    assert f1 == 1.0 and 0.2 <= theta < 0.8, (theta, f1)


def test_prf_threshold_is_strict_greater():
    # a sample sitting exactly on theta must NOT be flagged (rule is `> theta`)
    assert prf(_rows([(1, 0.5)]), 0.5)[1] == 0.0
    assert prf(_rows([(1, 0.5)]), 0.4999)[1] == 1.0


def test_auroc_is_nan_without_both_classes():
    import math
    assert math.isnan(auroc(_rows([(1, 0.9), (1, 0.8)])))


def test_wilson_ci_stays_in_range_and_brackets_p():
    from experiments.report_tables import wilson_ci
    # never escapes [0,1], even at the extremes where the normal approx does
    for k, n in [(0, 10), (10, 10), (1, 167), (33, 167)]:
        lo, hi = wilson_ci(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0, (k, n, lo, hi)
    # a bigger sample gives a tighter interval
    assert (wilson_ci(50, 100)[1] - wilson_ci(50, 100)[0]) > (wilson_ci(500, 1000)[1] - wilson_ci(500, 1000)[0])


if __name__ == "__main__":
    test_done_ids_survives_truncated_line()
    test_probs_match_logits()
    test_categories_complete()
    test_auroc_known_values()
    test_best_theta_separates_when_separable()
    test_prf_threshold_is_strict_greater()
    test_auroc_is_nan_without_both_classes()
    test_wilson_ci_stays_in_range_and_brackets_p()
    print("ok")
