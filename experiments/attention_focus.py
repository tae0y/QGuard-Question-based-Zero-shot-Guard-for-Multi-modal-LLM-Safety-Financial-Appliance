#!/usr/bin/env python3
"""S1/S3 후속 — 추가 질문이 프롬프트의 어디에 반응하는가 (occlusion + attention).

`attribution.py`(가설 3 원안)의 occlusion_scores/attention_scores/cross_validate를
그대로 재사용합니다. 새 통계 방법이 아니라 **다른 대상**(S1/S3의 "professional
licensure" 질문, 특히 Political_Lobbying처럼 yes_prob이 낮게 나온 케이스)에
적용하는 러너입니다.

동기: S1 표 E에서 Political_Lobbying만 yes_prob이 낮았고(0.030), S3는 질문에서
`(e.g., financial, legal, medical, or governmental)` 예시 절을 뺀 버전입니다.
이 스크립트로 두 버전이 같은 프롬프트의 어느 토큰에 반응하는지 나란히 보면,
"어휘 앵커링" 가설(예시 목록의 단어와 프롬프트 표면 어휘가 겹쳐야 반응한다)을
직접 확인할 수 있습니다 — occlusion(인과)과 attention(상관)을 교차검증해
attention 단독의 신뢰성 문제(Jain & Wallace, arXiv:1902.10186)를 피합니다.

카테고리마다 병합 결과(`merged_s1`/`merged_s3`)에서 대상 질문의 yes_prob이 가장
낮은/높은 케이스를 각각 뽑아 대비합니다 — 낮은 쪽이 "왜 안 걸렸는가", 높은 쪽이
대조군입니다.

Usage:
  uv run experiments/attention_focus.py \
    --merged_dir "/content/qguard_results/merged_s1" \
    --guard_questions_json files/guard_questions_s1_advisory.json \
    --categories Political_Lobbying,Legal_Opinion,Gov_Decision,Financial_Advice \
    --out_dir "/content/qguard_results/attention_s1"
"""
import argparse
import glob
import json
import os
from typing import Any, Dict, List

import torch

from qguard.modeling import load_internvl_text_or_mm
from qguard.token_utils import gather_yes_no_ids

from experiments.attribution import attention_scores, cross_validate, occlusion_scores


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def target_yes_prob(row: Dict[str, Any], needle: str) -> float:
    for q in row["questions"]:
        if needle.lower() in q["question"].lower():
            return q["yes_prob"]
    raise KeyError(f"'{needle}' 질문을 이 행에서 찾지 못했습니다 — merged_dir이 대상 질문을 포함하는지 확인하십시오")


def load_guard_question(path: str) -> str:
    """S1/S2/S3 질문 파일은 그룹당 질문 1개만 담습니다 — 그 문자열 하나를 꺼냅니다."""
    with open(path, "r", encoding="utf-8") as f:
        qs = json.load(f)
    all_qs = [q for group in qs.values() for q in group]
    if len(all_qs) != 1:
        raise ValueError(f"{path}에 질문이 {len(all_qs)}개입니다 — 이 스크립트는 질문 1개짜리 S1/S2/S3 파일 전용입니다")
    return all_qs[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged_dir", required=True, help="merge_and_score.py 출력 (36개 질문 포함)")
    ap.add_argument("--guard_questions_json", required=True, help="분석 대상 질문 1개 (S1/S2/S3 파일)")
    ap.add_argument("--categories", default="Political_Lobbying,Legal_Opinion,Gov_Decision,Financial_Advice",
                     help="쉼표 구분 harmful 카테고리 목록")
    ap.add_argument("--top_n", type=int, default=3, help="카테고리당 최저/최고 yes_prob 각각 몇 건")
    ap.add_argument("--granularity", choices=["token", "sentence"], default="token")
    ap.add_argument("--model_path", default="OpenGVLab/InternVL2_5-4B")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--extra_question_needle", default="professional licensure",
                     help="merged_dir에서 대상 질문을 식별할 부분 문자열 — S1/S3 공통 핵심 문구라 "
                          "e.g. 절 유무와 무관하게 매칭됩니다")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    guard_q = load_guard_question(args.guard_questions_json)
    print(f"대상 질문: {guard_q}")

    print(f"CUDA available: {torch.cuda.is_available()}")
    model, tokenizer = load_internvl_text_or_mm(args.model_path)
    yes_ids, no_ids = gather_yes_no_ids(tokenizer)

    out_path = os.path.join(args.out_dir, "attention_focus.jsonl")
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for cat in args.categories.split(","):
            path = os.path.join(args.merged_dir, f"{cat}.jsonl")
            if not os.path.exists(path):
                print(f"{cat}: {path} 없음 — 건너뜀")
                continue
            rows = [r for r in read_jsonl(path) if r.get("label") == 1]
            scored = sorted(((target_yes_prob(r, args.extra_question_needle), r) for r in rows), key=lambda x: x[0])
            picks = [("low", p) for p in scored[: args.top_n]] + [("high", p) for p in scored[-args.top_n:]]

            for tag, (yp, row) in picks:
                occ = occlusion_scores(model, tokenizer, guard_q, row["prompt"], yes_ids, no_ids, args.granularity)
                attn = attention_scores(model, tokenizer, guard_q, row["prompt"], args.granularity)
                xval = cross_validate(occ, attn)
                rec = {
                    "category": cat, "tag": tag, "yes_prob": yp, "prompt": row["prompt"],
                    "occlusion": occ, "attention": attn, "cross_validation": xval,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_written += 1

                occ_top = sorted(occ, key=lambda r: -abs(r["delta_yes_logit"]))[:3]
                attn_top = sorted(attn, key=lambda r: -r["attention"])[:3]
                print(f"\n[{cat} / {tag}, yes_prob={yp:.3f}] {row['prompt'][:80]}")
                print(f"  occlusion top: {[(r['unit'], round(r['delta_yes_logit'], 3)) for r in occ_top]}")
                print(f"  attention top: {[(r['unit'], round(r['attention'], 3)) for r in attn_top]}")
                print(f"  agreement: {xval['agreement_fraction']:.2f}")

    print(f"\n{n_written}건 -> {out_path}")


if __name__ == "__main__":
    main()
