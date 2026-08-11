import gc
from typing import Dict, List, Iterable, Any, Optional
import torch

from .token_utils import gather_yes_no_ids
from .scoring import score_guard_question_yes_prob
from .graph import build_graph_from_results, pagerank_risk_score

DEFAULT_ALLOWED_GROUPS = [
    "General Toxic",
    "Toxic Prompt",
    "Core Harmfulness Detection",
    "Additional Nuanced Questions",
]

def evaluate_prompt_with_pagerank(
    model,
    tokenizer,
    prompt: str,
    guard_questions: Dict[str, List[str]],
    allowed_groups: Iterable[str] = DEFAULT_ALLOWED_GROUPS,
    threshold: float = 0.50,
    images_tensor: Optional[torch.Tensor] = None,
    group_coupling: float = 1.0,
    intra_group_q_coupling: float = 0.3,
) -> Dict[str, Any]:
    yes_ids, no_ids = gather_yes_no_ids(tokenizer)

    per_q, group_by = [], {}
    for cat, qs in guard_questions.items():
        for q in qs:
            res = score_guard_question_yes_prob(
                model, tokenizer, q, prompt, yes_ids, no_ids, images=images_tensor
            )
            if cat in allowed_groups:
                res["category"] = cat
                per_q.append(res)
                group_by[res["question"]] = cat

            if torch.cuda.is_available() and (len(per_q) % 8 == 0):
                torch.cuda.empty_cache()
                gc.collect()

    G = build_graph_from_results(
        per_q, group_by,
        group_coupling=group_coupling,
        intra_group_q_coupling=intra_group_q_coupling,
    )
    risk = pagerank_risk_score(G)
    pred = "harmful" if risk > threshold else "unharmful"

    return {
        "prompt": prompt,
        "threshold": threshold,
        "risk_score": risk,
        "prediction": pred,
        "questions": per_q, 
    }
