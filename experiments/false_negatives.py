"""ETGPO step 1 — collect C0 false negatives for financial_advice.

All financial_advice pairs in MM-SafetyBench are harmful (label=1), so a
false negative is any C0 result predicted "unharmful".

See: 100 Project/.../~260811 실험설계및수행 리뷰.md §2 가설1, 1단계
"""
from typing import Any, Dict, List

import pandas as pd


def collect_false_negatives(results: List[Dict[str, Any]], condition: str = "C0") -> pd.DataFrame:
    """results: list of dicts with keys prompt, condition, prediction, risk_score, questions."""
    rows = [r for r in results if r["condition"] == condition and r["prediction"] == "unharmful"]
    return pd.DataFrame(rows)


# Placeholder failure-type keywords for a first-pass heuristic split, to be
# replaced with judgment calls once real C0 false negatives are in hand
# (see 실험설계및수행 리뷰.md §2 1단계 — the review document only prescribes
# the *procedure*, not the taxonomy itself).
FAILURE_TYPE_KEYWORDS = {
    "ambiguous_regulatory_terms": ["guarantee", "risk-free", "insider", "tip"],
    "disclaimer_adjacent": ["not financial advice", "disclaimer", "no advice"],
    "implicit_vs_explicit_request": ["what should i", "my situation", "should i"],
}


def classify_failure_type(prompt: str) -> str:
    """First-pass heuristic classification by keyword match; falls back to
    'unclassified' when the prompt is not phrased in any recognized pattern.
    Intended as a starting point for manual review, not a final labeling.
    """
    lowered = prompt.lower()
    for failure_type, keywords in FAILURE_TYPE_KEYWORDS.items():
        if any(k in lowered for k in keywords):
            return failure_type
    return "unclassified"


def classify_false_negatives(fn_df: pd.DataFrame) -> pd.DataFrame:
    fn_df = fn_df.copy()
    if fn_df.empty:
        fn_df["failure_type"] = pd.Series(dtype=object)
        return fn_df
    fn_df["failure_type"] = fn_df["prompt"].apply(classify_failure_type)
    return fn_df
