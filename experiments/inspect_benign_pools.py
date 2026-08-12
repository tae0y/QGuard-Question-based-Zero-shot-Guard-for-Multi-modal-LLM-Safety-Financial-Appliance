#!/usr/bin/env python3
"""Diversity check for the benign sources actually in use (MMInstruct QA,
FiQA) — how much of each pool is distinct text vs. the same sentence
repeated.

Written after `benign_caption.jsonl` (MMInstruct caption task) turned out to
be ~20-30 "describe this image" templates repeated across domains once the
`<image>` token is stripped for text-only scoring — confirmed here and led
to dropping the caption task entirely (see mminstruct_benign.py deviation 4).
QA and FiQA measured fine on the same check, which is why they're what's
actually used now.

Usage:
  uv run experiments/inspect_benign_pools.py --out_path benign_pool_diversity.md
"""
import argparse
from collections import Counter
from typing import Dict, List

from datasets import load_dataset

from experiments.benign_prompts import BENIGN_DATASET, BENIGN_FIELD, BENIGN_SPLIT
from experiments.mminstruct_benign import QA_DOMAINS, REPO, _prompt_of


def domain_stats(task: str, domain: str, top_n: int = 5) -> Dict:
    ds = load_dataset(
        "json",
        data_files=f"hf://datasets/{REPO}/jsons_per_domain/{task}_per_domain/{domain}_en.jsonl",
        split="train",
    )
    counts = Counter(p for row in ds if (p := _prompt_of(row)))
    return {
        "domain": domain,
        "total": sum(counts.values()),
        "unique": len(counts),
        "top": counts.most_common(top_n),
    }


def fiqa_stats(top_n: int = 5) -> Dict:
    ds = load_dataset(BENIGN_DATASET, split=BENIGN_SPLIT)
    counts = Counter(q for row in ds if (q := (row.get(BENIGN_FIELD) or "").strip()))
    return {"total": sum(counts.values()), "unique": len(counts), "top": counts.most_common(top_n)}


def render_task_section(title: str, rows: List[Dict]) -> str:
    lines = [f"## {title}", "", "| domain | total | unique | unique ratio |", "|---|---|---|---|"]
    for r in rows:
        ratio = r["unique"] / r["total"] if r["total"] else 0
        lines.append(f"| {r['domain']} | {r['total']} | {r['unique']} | {ratio:.3f} |")
    lines.append("")
    for r in rows:
        lines.append(f"### {r['domain']} — top repeats")
        for prompt, n in r["top"]:
            lines.append(f"- ({n}x) {prompt[:100]!r}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_path", default="benign_pool_diversity.md")
    parser.add_argument("--top_n", type=int, default=5)
    args = parser.parse_args()

    qa_rows = [domain_stats("qa", d, args.top_n) for d in QA_DOMAINS]
    fiqa = fiqa_stats(args.top_n)

    sections = [
        "# Benign pool diversity report",
        "",
        render_task_section("MMInstruct qa", qa_rows),
        "## FiQA",
        "",
        f"total={fiqa['total']} unique={fiqa['unique']} ratio={fiqa['unique']/fiqa['total']:.3f}",
        "",
        "top repeats:",
        *(f"- ({n}x) {q[:100]!r}" for q, n in fiqa["top"]),
        "",
    ]
    report = "\n".join(sections)

    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nWritten -> {args.out_path}")


if __name__ == "__main__":
    main()
