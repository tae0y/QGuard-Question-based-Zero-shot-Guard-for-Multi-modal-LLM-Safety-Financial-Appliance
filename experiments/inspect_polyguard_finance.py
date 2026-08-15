#!/usr/bin/env python3
"""Sample check for PolyGuard's finance_input config as a text-only harmful/
benign source for the financial guardrail experiment.

MM-SafetyBench's Financial_Advice puts the harmful noun inside the image
(Text_only split reads as bland once the image is dropped — see
docs/EXPERIMENT-ko.md). PolyGuard's finance_input is pure text and already
splits each of 5 sources (ALT/BIS/FINRA/OECD/USDT) into *_safe / *_unsafe,
same schema, matched by `category name` — a direct harmful/benign pair with
no image dependency. This script pulls real rows so the split can be judged
before committing GPU time to it.

CPU only, no model needed.

Usage:
  uv run experiments/inspect_polyguard_finance.py --n_samples 3
"""
import argparse

from datasets import load_dataset

SOURCES = ["ALT", "BIS", "FINRA", "OECD", "USDT"]


def render_split(ds, name: str, n: int) -> str:
    lines = [f"### {name}  (n={len(ds)})", ""]
    cats = ds.unique("category name")
    lines.append(f"categories ({len(cats)}): {', '.join(sorted(cats))}")
    lines.append("")
    for row in ds.select(range(min(n, len(ds)))):
        lines.append(f"- **category**: {row['category name']}")
        lines.append(f"  **rule**: {row['rule']}")
        lines.append(f"  **original**: {row['original instance'][:300]}")
        if row["rephrased instance"]:
            lines.append(f"  **rephrased**: {row['rephrased instance'][:300]}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_samples", type=int, default=3, help="rows to print per split")
    ap.add_argument("--out_path", default="polyguard_finance_samples.md")
    args = ap.parse_args()

    dsd = load_dataset("Virtue-AI-HUB/PolyGuard", "finance_input")

    sections = ["# PolyGuard finance_input — sample check", ""]
    for src in SOURCES:
        for label in ("unsafe", "safe"):
            split = f"{src}_{label}"
            if split not in dsd:
                continue
            sections.append(render_split(dsd[split], split, args.n_samples))

    report = "\n".join(sections)
    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\nWritten -> {args.out_path}")


if __name__ == "__main__":
    main()
