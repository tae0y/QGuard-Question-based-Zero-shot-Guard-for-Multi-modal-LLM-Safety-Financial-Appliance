#!/usr/bin/env python3
"""Emit the review document's tables as markdown, ready to paste.

CPU only — everything is recomputed from the stored per-question yes_prob, so
changing a coupling never needs the GPU pass re-run.

  uv run experiments/report_tables.py <results_dir>            # tables A-D
  uv run experiments/report_tables.py <results_dir> --table C  # just one
"""
import argparse
import statistics as st
from collections import defaultdict
from typing import Any, Dict, List

from experiments.sweep_theta import auroc, best_theta, load_rows, prf

# Paper Fig. 2, multi-modal SD setting. Only these two are stated numerically in
# the text; the rest of the figure is not transcribed in our paper note.
PAPER_RECALL = {"Financial_Advice": 0.2335}


def pr_auc(rows: List[Dict[str, Any]]) -> float:
    """Average precision: precision at each recall step, ranked by score."""
    ranked = sorted(rows, key=lambda r: -r["risk_score"])
    n_pos = sum(r["label"] == 1 for r in rows)
    if not n_pos:
        return float("nan")
    tp = 0
    total = 0.0
    for i, r in enumerate(ranked, 1):
        if r["label"] == 1:
            tp += 1
            total += tp / i
    return total / n_pos


def table_a(rows) -> str:
    theta, p, rec, f1 = best_theta(rows)
    n_pos = sum(r["label"] == 1 for r in rows)
    out = ["### 표 A — 전체 성능", "",
           "| 지표 | 값 |", "|---|---:|",
           f"| n (harmful / benign) | {n_pos} / {len(rows) - n_pos} |",
           f"| AUROC | {auroc(rows):.4f} |",
           f"| PR-AUC | {pr_auc(rows):.4f} |",
           f"| 최적 θ | {theta:.4f} |",
           f"| Precision @θ | {p:.4f} |",
           f"| Recall @θ | {rec:.4f} |",
           f"| F1 @θ | {f1:.4f} |"]
    return "\n".join(out)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval — holds up near 0 and 1, where the normal
    approximation runs off the end of [0,1]. Financial_Advice recall may well
    sit near 0.2 with n=167, so that matters here."""
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def table_b(rows) -> str:
    theta = best_theta(rows)[0]
    cats = defaultdict(list)
    for r in rows:
        if r["label"] == 1:
            cats[r["category"]].append(r)
    lines = [f"### 표 B — 카테고리별 recall (θ={theta:.4f} 고정)", "",
             "| 카테고리 | n | recall | 95% CI | risk_score 중앙값 | 논문(멀티모달) |",
             "|---|---:|---:|---:|---:|---:|"]
    for c, rs in sorted(cats.items(), key=lambda kv: -sum(x["risk_score"] > theta for x in kv[1]) / len(kv[1])):
        hit = sum(x["risk_score"] > theta for x in rs)
        lo, hi = wilson_ci(hit, len(rs))
        med = st.median(x["risk_score"] for x in rs)
        paper = f"{PAPER_RECALL[c]:.4f}" if c in PAPER_RECALL else "—"
        lines.append(f"| {c} | {len(rs)} | {hit / len(rs):.4f} | {lo:.3f}–{hi:.3f} | {med:.4f} | {paper} |")
    return "\n".join(lines)


def table_c(rows) -> str:
    """Category x guard-group mean yes_prob, plus the questions that swing most
    between the top and bottom category. 35 columns will not fit a page, so the
    per-question view is the ranked diff list rather than a full matrix."""
    by_cat_group = defaultdict(lambda: defaultdict(list))
    by_cat_q = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["label"] != 1:
            continue
        for q in r["questions"]:
            by_cat_group[r["category"]][q["category"]].append(q["yes_prob"])
            # the stored question text is "<guard q> (You must answer...)\n\nprompt: ..."
            by_cat_q[r["category"]][q["question"].split(" (You must answer")[0]].append(q["yes_prob"])

    groups = ["General Toxic", "Toxic Prompt", "Core Harmfulness Detection", "Additional Nuanced Questions"]
    order = sorted(by_cat_group, key=lambda c: -st.mean(
        v for g in groups for v in by_cat_group[c].get(g, [0])))

    lines = ["### 표 C-1 — 카테고리 × 가드질문 그룹 평균 yes_prob", "",
             "| 카테고리 | " + " | ".join(g.replace(" ", "&nbsp;") for g in groups) + " | 전체 |",
             "|---" * (len(groups) + 2) + "|"]
    for c in order:
        vals = [st.mean(by_cat_group[c][g]) if by_cat_group[c].get(g) else float("nan") for g in groups]
        allv = st.mean(v for g in groups for v in by_cat_group[c].get(g, []))
        lines.append(f"| {c} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {allv:.3f} |")

    top, bottom = order[0], order[-1]
    lines += ["", f"### 표 C-2 — 질문별 반응 차이 ({top} vs {bottom})", "",
              "| 가드 질문 | 상위 | 하위 | 차이 |", "|---|---:|---:|---:|"]
    diffs = []
    for q in by_cat_q[top]:
        if q in by_cat_q[bottom]:
            a, b = st.mean(by_cat_q[top][q]), st.mean(by_cat_q[bottom][q])
            diffs.append((a - b, q, a, b))
    for d, q, a, b in sorted(diffs, reverse=True)[:8]:
        lines.append(f"| {q[:70]} | {a:.3f} | {b:.3f} | {d:+.3f} |")
    lines.append("| … | | | |")
    for d, q, a, b in sorted(diffs)[:3]:
        lines.append(f"| {q[:70]} | {a:.3f} | {b:.3f} | {d:+.3f} |")
    return "\n".join(lines)


def table_d(rows, limit: int = 15) -> str:
    theta = best_theta(rows)[0]
    fn = [r for r in rows if r["label"] == 1 and r["risk_score"] <= theta]
    lines = [f"### 표 D — 오류 케이스 (false negative, θ={theta:.4f}) — {len(fn)}건 중 {min(limit, len(fn))}건", "",
             "| 카테고리 | prompt | risk | 최고 반응 질문 | 최저 |", "|---|---|---:|---|---:|"]
    for r in sorted(fn, key=lambda r: r["risk_score"])[:limit]:
        qs = sorted(r["questions"], key=lambda q: -q["yes_prob"])
        top = qs[0]["question"].split(" (You must answer")[0][:50]
        lines.append(f"| {r['category']} | {r['prompt'][:60]}… | {r['risk_score']:.3f} "
                     f"| {top} ({qs[0]['yes_prob']:.2f}) | {qs[-1]['yes_prob']:.2f} |")
    return "\n".join(lines)


def table_e(rows, needle: str) -> str:
    """Did the added question actually fire? Separates "the question does not
    respond" from "it responds but PageRank dilutes it" — a single new question
    among 36 can light up and still not move the final score."""
    per_cat = defaultdict(list)
    for r in rows:
        for q in r["questions"]:
            if needle.lower() in q["question"].lower():
                per_cat[(r["category"], r["label"])].append(q["yes_prob"])
                break
    if not per_cat:
        return f"### 표 E — 추가 질문 반응\n\n`{needle}` 를 포함한 질문을 찾지 못했습니다."
    lines = [f"### 표 E — 추가 질문 반응 (`{needle}`)", "",
             "| 카테고리 | label | n | 평균 yes_prob | ≥0.5 비율 |", "|---|---:|---:|---:|---:|"]
    for (c, lab), vals in sorted(per_cat.items(), key=lambda kv: -st.mean(kv[1])):
        lines.append(f"| {c} | {lab} | {len(vals)} | {st.mean(vals):.3f} "
                     f"| {sum(v >= 0.5 for v in vals) / len(vals):.3f} |")
    lines += ["", "판정: harmful(label=1) 조언형 카테고리에서 평균이 높은데 표 B recall이"
              " 오르지 않았다면, 질문이 아니라 **집계(PageRank 희석)** 가 원인입니다."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--table", choices=list("ABCDE"), default=None, help="Only this one")
    ap.add_argument("--extra_question", default="licensed expert",
                    help="표 E용: 추가한 질문을 식별할 부분 문자열")
    ap.add_argument("--group_coupling", type=float, default=None, help="Re-score from stored yes_prob")
    ap.add_argument("--symmetric_coupling", action="store_true",
                     help="Add group<->group and same-group question<->question edges in both directions")
    args = ap.parse_args()

    rows = load_rows(args.results_dir, args.group_coupling, symmetric_coupling=args.symmetric_coupling)
    has_extra = any(r.get("n_questions", 35) > 35 for r in rows)

    if args.table:
        picked = [args.table]
    elif not any(r["label"] == 0 for r in rows):
        # theta needs both classes; C and E are per-question and do not
        print("benign 행이 없습니다 — θ를 구할 수 없어 표 A/B/D를 만들 수 없습니다.\n"
              "러너를 --with_benign 으로 다시 돌리십시오.\n")
        picked = ["C", "E"] if has_extra else ["C"]
    else:
        picked = list("ABCDE") if has_extra else list("ABCD")
    has_benign = any(r["label"] == 0 for r in rows)
    for name in picked:
        if name in "ABD" and not has_benign:
            print(f"표 {name}: benign 행이 없어 θ를 구할 수 없습니다 — 생략\n")
            continue
        if name == "E":
            print(table_e(rows, args.extra_question))
        else:
            print({"A": table_a, "B": table_b, "C": table_c, "D": table_d}[name](rows))
        print()


if __name__ == "__main__":
    main()
