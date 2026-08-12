"""MMInstruct QA benign (label=0) prompts — the general-domain half of the
benign set (the other half is FiQA, see `benign_prompts.py`).

Paper Sec 4.1.1 describes a caption+QA benign set (901+1100). This
reproduction only uses the QA task; see deviation 5 below for why.

Source: `yuecao0119/MMInstruct-GPT4V`, the per-domain English JSONL files.
Reading `jsons_all/*.jsonl` instead would make the per-scenario cap
unenforceable, since the domain is not a field on the row.

Deviations from the paper, all unavoidable and recorded here:

1. The QA task ships **23** English domains, not the 22 the paper states. The
   paper gives no domain list, so which one it dropped cannot be recovered.
   All 23 are used.

2. Prompts are image-dependent — every one starts with an `<image>` token and
   many read "…in this picture?". This study is text-only, so the token is
   stripped and the sentence is kept verbatim. That makes these somewhat
   *easier* negatives than the paper's multi-modal setting, which had the
   image. See `260812 MMInstruct benign 세트 추가 설계.md` §4.

3. `_en` per-domain files are not purely English despite the name — a sample
   of the caption pool showed ~25% pure-Chinese rows (design doc §3.1(c)
   flagged this as an unconfirmed risk). QA pools measured <0.1% Chinese, but
   rows containing CJK characters are dropped before dedup/sampling anyway,
   since guard questions and the harmful set (MM-SafetyBench) are English.

4. The caption task is dropped entirely (not just short of target). Once
   `<image>` is stripped, caption prompts collapse to ~20-30 boilerplate
   "describe this image" templates repeated across domains — confirmed via
   `inspect_benign_pools.py` (326/344-ish of a sampled 126-row draw were one
   of 27 unique strings). QA prompts embed image-specific content in the
   question text itself (e.g. "How does the interaction between the elephant
   and the people in the water contribute to the overall scene?"), so they
   stay diverse without the image; per-domain unique ratios measured 45-90%.
   Discussed 2026-08-18; benign composition is now QA (general) + FiQA
   (financial-domain-weighted), see `run_mmsafetybench_textonly.py`.
"""
import random
import re
from typing import Any, Dict, List

from datasets import load_dataset

REPO = "yuecao0119/MMInstruct-GPT4V"

# Verified against the HF file listing (2026-08-12).
QA_DOMAINS = [
    "artwork", "attribute_recognition", "brand_recognition", "commonsense_reasoning",
    "complex_reasoning", "future_prediction", "image_comparison", "image_emotion",
    "image_quality", "image_scene", "image_style", "landmark", "meme_comprehension",
    "object_localization", "object_relation", "ocr", "posters", "social_relation",
    "spatial_relationship", "species_recognition",
    "image_description", "numerical_calculation", "writing",
]

_CJK_RE = re.compile(r"[一-鿿]")


def _prompt_of(row: Dict[str, Any]) -> str:
    """First human turn, with the leading `<image>` marker removed."""
    convs = row.get("conversations") or []
    if not convs:
        return ""
    return convs[0].get("value", "").replace("<image>", "").strip()


def _sample_domain(domain: str, cap: int, seed: int) -> List[Dict[str, Any]]:
    """Up to `cap` distinct, English-only prompts from one QA domain."""
    ds = load_dataset(
        "json",
        data_files=f"hf://datasets/{REPO}/jsons_per_domain/qa_per_domain/{domain}_en.jsonl",
        split="train",
    )
    seen, prompts = set(), []
    for row in ds:
        p = _prompt_of(row)
        if p and p not in seen and not _CJK_RE.search(p):
            seen.add(p)
            prompts.append(p)
    picked = random.Random(f"{seed}:qa:{domain}").sample(prompts, min(cap, len(prompts)))
    return [{"prompt": p, "task": "qa", "scenario": domain, "label": 0} for p in picked]


def load_mminstruct_qa_benign(n: int = 1391, per_scenario_cap: int = 61, seed: int = 1234) -> List[Dict[str, Any]]:
    """Deterministic sample of `n` MMInstruct QA benign prompts.

    `per_scenario_cap` must be large enough that `len(QA_DOMAINS) * cap >= n`
    (61 * 23 = 1403 for the current default n=1391) — every domain measured
    well over 1000 unique English prompts, so this is not the bottleneck
    caption was.
    """
    pool: List[Dict[str, Any]] = []
    for domain in QA_DOMAINS:
        pool += _sample_domain(domain, per_scenario_cap, seed)

    if n > len(pool):
        print(f"WARNING: qa: requested n={n} but pool holds only {len(pool)} unique prompts; using all {len(pool)}")
        n = len(pool)
    return random.Random(f"{seed}:qa").sample(pool, n)
