"""Benign (label=0) financial questions — the negative half of the balanced set.

Why this exists: MM-SafetyBench `Financial_Advice` is 100% label=1, so a
recall-only score on it is unfalsifiable (a classifier that flags everything
scores recall 1.0). QGuard's own paper mixes in benign samples and reports F1
(§4.1.1); this restores that half.

Source: `LLukas22/fiqa` (HF), split `train`, field `question`. This is the QA
form of FiQA 2018 Task 2 — real financial questions asked by real users on
StackExchange/Reddit-style forums.

Not `pauri32/fiqa-2018`: that is FiQA Task 1 (aspect-based *sentiment*), whose
rows are tweets and headlines ("Still short $LNG from $11.70 area..."), not
questions. Feeding declarative market chatter as the benign class would make
the classifier separate questions-from-statements rather than harmful-from-
benign — a confound, not a control.

Length match (guards against "the model just reacts to prompt length"):
FiQA questions average 11.1 words, MM-SafetyBench `Text_only` 10.0.

Prompts are used **verbatim**. Nothing here is authored or hand-edited — an
author-written benign set would be indefensible in review, since its difficulty
would be set by the author's own guess at what "benign" looks like.

`train` carries 14511 rows but only 6251 distinct questions, so questions are
deduplicated (order-preserving) before sampling; without that the same question
can be drawn several times and the effective n is silently smaller than n.
"""
from typing import List

from datasets import load_dataset

BENIGN_DATASET = "LLukas22/fiqa"
BENIGN_SPLIT = "train"
BENIGN_FIELD = "question"


def load_benign_prompts(n: int, seed: int = 1234) -> List[str]:
    """Deterministically sample `n` distinct benign financial questions.

    Sampling is seeded and taken over the deduplicated, order-preserving list,
    so a given (n, seed) always yields the same prompts on any machine.
    """
    import random

    ds = load_dataset(BENIGN_DATASET, split=BENIGN_SPLIT)
    seen, questions = set(), []
    for row in ds:
        q = (row.get(BENIGN_FIELD) or "").strip()
        if q and q not in seen:
            seen.add(q)
            questions.append(q)

    if n > len(questions):
        raise ValueError(f"requested n={n} but only {len(questions)} distinct benign questions available")
    return random.Random(seed).sample(questions, n)
