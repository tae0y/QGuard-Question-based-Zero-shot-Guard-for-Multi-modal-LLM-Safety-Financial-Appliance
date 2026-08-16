#!/usr/bin/env python3
"""표층 표현 최소쌍 채점 — I-5 반증 실험 2단계.

`generate_surface_form_variants.py`가 만든 (원본, advisory, procedural) 최소쌍
각각에 대해 S1 표적 질문(기본: `guard_questions_s1_advisory.json`) 하나만
재실행한다. 세 버전의 위해 내용은 동일하므로, yes_prob 차이는 표층 표현만으로
설명돼야 한다.

같은 카테고리·같은 원본에서 나온 세 버전을 한 행에 같이 적어 쌍대비교가
바로 되게 한다(run_extra_questions.py처럼 base_dir을 다시 조회하지 않음 —
variants_path 자체가 이미 세 버전을 다 담고 있음).

Usage:
  uv run experiments/run_surface_form_experiment.py \
    --variants_path files/surface_form_variants.jsonl \
    --guard_questions_json files/guard_questions_s1_advisory.json \
    --out_path /content/qguard_results/surface_form_scored.jsonl
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import torch

from qguard.modeling import load_internvl_text_or_mm
from qguard.scoring import score_guard_question_yes_prob
from qguard.seed import set_seed
from qguard.token_utils import gather_yes_no_ids

from experiments.attention_focus import load_guard_question
from experiments.run_mmsafetybench_textonly import done_ids

FORMS = [("original", "original_prompt"), ("advisory", "advisory_prompt"), ("procedural", "procedural_prompt")]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants_path", required=True)
    ap.add_argument("--guard_questions_json", default="files/guard_questions_s1_advisory.json")
    ap.add_argument("--model_path", default="OpenGVLab/InternVL2_5-4B")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    guard_q = load_guard_question(args.guard_questions_json)
    print(f"대상 질문: {guard_q}")

    variants = read_jsonl(args.variants_path)
    if not variants:
        raise SystemExit(f"{args.variants_path}에 데이터가 없습니다")

    print(f"CUDA available: {torch.cuda.is_available()}")
    model, tokenizer = load_internvl_text_or_mm(args.model_path)
    yes_ids, no_ids = gather_yes_no_ids(tokenizer)

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    skip = done_ids(args.out_path)

    with open(args.out_path, "a", encoding="utf-8") as f:
        for item in variants:
            row_id = item["row_id"]
            if row_id in skip:
                continue
            scored = {}
            for form_name, key in FORMS:
                res = score_guard_question_yes_prob(
                    model, tokenizer, guard_q, item[key], yes_ids, no_ids)
                scored[form_name] = {"yes_prob": res["yes_prob"], "prompt": item[key]}
            f.write(json.dumps({
                "row_id": row_id, "category": item["category"], "forms": scored,
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{item['category']}/{row_id}: "
                  f"original={scored['original']['yes_prob']:.3f} "
                  f"advisory={scored['advisory']['yes_prob']:.3f} "
                  f"procedural={scored['procedural']['yes_prob']:.3f}")

    rows = read_jsonl(args.out_path)
    by_cat = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for form_name, _ in FORMS:
            by_cat[r["category"]][form_name].append(r["forms"][form_name]["yes_prob"])

    print(f"\n=== 카테고리별 평균 yes_prob ({args.out_path}) ===")
    for cat, forms_dict in by_cat.items():
        n = len(forms_dict["original"])
        line = f"{cat} (n={n}): "
        line += " / ".join(f"{name}={sum(vals)/len(vals):.3f}" for name, vals in forms_dict.items())
        print(line)


if __name__ == "__main__":
    main()
