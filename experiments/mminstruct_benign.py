"""MMInstruct benign (label=0) prompts — the negative half of the paper's set.

Paper Sec 4.1.1: 901 unharmful prompts from MMInstruct's captioning task
(20 scenarios) and 1,100 from its question-answering task (22 scenarios),
"with up to 50 prompts per scenario", giving 3,681 total alongside
MM-SafetyBench's 1,680 harmful.

Source: `yuecao0119/MMInstruct-GPT4V`, the per-domain English JSONL files.
Reading `jsons_all/*.jsonl` instead would make the per-scenario cap
unenforceable, since the domain is not a field on the row.

Two deviations from the paper, both unavoidable and both recorded here:

1. The QA task ships **23** English domains, not the 22 the paper states. The
   paper gives no domain list, so which one it dropped cannot be recovered.
   All 23 are used; the 1,100 target is reachable either way (23*50=1150).

2. Prompts are image-dependent — every one starts with an `<image>` token and
   many read "…in this picture?". This study is text-only, so the token is
   stripped and the sentence is kept verbatim. That makes these *easy*
   negatives, which biases theta low. The paper ran multi-modal and had the
   image; we cannot. See `260812 MMInstruct benign 세트 추가 설계.md` §4.
"""
import random
from typing import Any, Dict, List

from datasets import load_dataset

REPO = "yuecao0119/MMInstruct-GPT4V"

# Verified against the HF file listing (2026-08-12).
CAPTION_DOMAINS = [
    "artwork", "attribute_recognition", "brand_recognition", "commonsense_reasoning",
    "complex_reasoning", "future_prediction", "image_comparison", "image_emotion",
    "image_quality", "image_scene", "image_style", "landmark", "meme_comprehension",
    "object_localization", "object_relation", "ocr", "posters", "social_relation",
    "spatial_relationship", "species_recognition",
]
QA_DOMAINS = CAPTION_DOMAINS + ["image_description", "numerical_calculation", "writing"]


def _prompt_of(row: Dict[str, Any]) -> str:
    """First human turn, with the leading `<image>` marker removed."""
    convs = row.get("conversations") or []
    if not convs:
        return ""
    return convs[0].get("value", "").replace("<image>", "").strip()


def _sample_task(task: str, domains: List[str], n: int, cap: int, seed: int) -> List[Dict[str, Any]]:
    """Up to `cap` distinct prompts per domain, then `n` drawn from that pool.

    Sampling per domain first (rather than capping after a global draw) is what
    keeps one large domain — qa/artwork alone has 11,829 rows — from swamping
    the pool.
    """
    pool: List[Dict[str, Any]] = []
    for domain in domains:
        ds = load_dataset(
            "json",
            data_files=f"hf://datasets/{REPO}/jsons_per_domain/{task}_per_domain/{domain}_en.jsonl",
            split="train",
        )
        seen, prompts = set(), []
        for row in ds:
            p = _prompt_of(row)
            if p and p not in seen:
                seen.add(p)
                prompts.append(p)
        picked = random.Random(f"{seed}:{task}:{domain}").sample(prompts, min(cap, len(prompts)))
        pool += [{"prompt": p, "task": task, "scenario": domain, "label": 0} for p in picked]

    if n > len(pool):
        raise ValueError(f"{task}: requested n={n} but pool holds only {len(pool)}")
    return random.Random(f"{seed}:{task}").sample(pool, n)


def load_mminstruct_benign(
    n_caption: int = 901, n_qa: int = 1100, per_scenario_cap: int = 50, seed: int = 1234,
) -> List[Dict[str, Any]]:
    """Paper-sized benign set. Deterministic for a given (n, cap, seed)."""
    return (
        _sample_task("caption", CAPTION_DOMAINS, n_caption, per_scenario_cap, seed)
        + _sample_task("qa", QA_DOMAINS, n_qa, per_scenario_cap, seed)
    )
