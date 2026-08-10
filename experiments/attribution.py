"""Hypothesis 3 — input attribution for the Yes/No logit (occlusion + attention).

Diagnostic-only: explains *why* the Yes logit moved for a given guard
question / injected-knowledge case, after hypothesis 1/2 interventions.
Does not change QGuard's scoring, graph, or PageRank.

Limited to a small sample of extreme-skew cases (see review doc §2 가설3
"연산 비용" note) — occlusion needs one forward pass per masked unit.

See: 100 Project/.../~260811 실험설계및수행 리뷰.md §2 가설3
"""
from typing import Any, Dict, List

import numpy as np
import torch

from qguard.scoring import _first_step_logits_text_only
from qguard.token_utils import yes_no_probs_from_logits


def split_units(text: str, granularity: str = "token") -> List[str]:
    """(a) guard question -> token/phrase units, (b) injected knowledge block
    -> sentence units, per the review doc's rationale (short questions need
    finer granularity; multi-sentence knowledge blocks split at sentence level).
    """
    if granularity == "sentence":
        import re
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]
    return text.split()


def yes_logit_gap(model, tokenizer, question_text: str, yes_ids: List[int], no_ids: List[int]) -> float:
    step0, _ = _first_step_logits_text_only(model, tokenizer, question_text, max_new_tokens=4)
    y_prob, n_prob = yes_no_probs_from_logits(step0, yes_ids, no_ids)
    eps = 1e-7
    p = min(max(y_prob, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def occlusion_scores(
    model, tokenizer, guard_q: str, prompt_text: str,
    yes_ids: List[int], no_ids: List[int], granularity: str = "token",
) -> List[Dict[str, Any]]:
    """Mask each unit of `prompt_text` (the injected knowledge block or the
    guard question itself, whichever is under study) one at a time and
    measure the resulting change in the Yes-logit gap. Larger |delta| means
    that unit contributed more to the Yes/No decision.
    """
    q_full = f"{guard_q} (You must answer with only Yes or No).\n\nprompt: {prompt_text}"
    baseline = yes_logit_gap(model, tokenizer, q_full, yes_ids, no_ids)

    units = split_units(prompt_text, granularity)
    rows = []
    for i, unit in enumerate(units):
        masked_units = units[:i] + units[i + 1:]
        sep = " " if granularity == "token" else " "
        masked_prompt = sep.join(masked_units)
        masked_q = f"{guard_q} (You must answer with only Yes or No).\n\nprompt: {masked_prompt}"
        gap = yes_logit_gap(model, tokenizer, masked_q, yes_ids, no_ids)
        rows.append({"unit": unit, "index": i, "delta_yes_logit": baseline - gap})
    return rows


def attention_scores(
    model, tokenizer, guard_q: str, prompt_text: str, granularity: str = "token",
) -> List[Dict[str, Any]]:
    """Single forward pass: extract last-layer attention from the Yes/No
    decision token to each input token, then aggregate to `granularity`.

    Caveat (see review doc §2 가설3 리스크 1): attention weight is a softmax
    output already normalized by sqrt(d_k) — a peak here is not proof of
    causal contribution (Jain & Wallace, arXiv:1902.10186), which is why
    this must be cross-checked against occlusion_scores(), not used alone.
    """
    q_full = f"{guard_q} (You must answer with only Yes or No).\n\nprompt: {prompt_text}"
    inputs = tokenizer(q_full, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.language_model(
            **inputs, output_attentions=True, return_dict=True,
        )
    # last layer, average over heads, attention from the last (decision) token
    last_layer_attn = out.attentions[-1][0]  # (heads, seq, seq)
    decision_token_attn = last_layer_attn[:, -1, :].mean(dim=0)  # (seq,)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    prompt_start = q_full.index(prompt_text)
    prompt_prefix_tokens = len(tokenizer(q_full[:prompt_start])["input_ids"])

    units = split_units(prompt_text, granularity)
    if granularity == "token":
        weights = decision_token_attn[prompt_prefix_tokens:prompt_prefix_tokens + len(units)]
        return [{"unit": u, "index": i, "attention": float(w)} for i, (u, w) in enumerate(zip(units, weights))]

    # sentence granularity: sum sub-token weights within each sentence's char span
    rows, cursor = [], 0
    for i, sentence in enumerate(units):
        start = prompt_text.index(sentence, cursor)
        end = start + len(sentence)
        cursor = end
        tok_ids_in_span = tokenizer(prompt_text[start:end])["input_ids"]
        span_len = max(len(tok_ids_in_span), 1)
        w = float(decision_token_attn[prompt_prefix_tokens:prompt_prefix_tokens + span_len].sum())
        rows.append({"unit": sentence, "index": i, "attention": w})
    return rows


def cross_validate(occlusion: List[Dict[str, Any]], attention: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
    """Compare top-k units by occlusion vs. by attention; agreement fraction
    is the review doc's calibration check for whether attention is a
    trustworthy proxy for this case.
    """
    occ_top = {r["index"] for r in sorted(occlusion, key=lambda r: -abs(r["delta_yes_logit"]))[:top_k]}
    attn_top = {r["index"] for r in sorted(attention, key=lambda r: -r["attention"])[:top_k]}
    overlap = occ_top & attn_top
    return {
        "occlusion_top": sorted(occ_top),
        "attention_top": sorted(attn_top),
        "agreement_fraction": len(overlap) / top_k,
    }
