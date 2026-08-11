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


if __name__ == "__main__":
    test_done_ids_survives_truncated_line()
    test_probs_match_logits()
    test_categories_complete()
    print("ok")
