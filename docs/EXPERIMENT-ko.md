# 금융분야 가드 실험 — 실행 순서 및 보고 항목

이 문서는 [docs/ADDITIONAL-EXPERIMENT-ko.md](ADDITIONAL-EXPERIMENT-ko.md)의 실행판 짝입니다. 그 문서는 코드 스스로 내릴 수 없는 판단들을 다루고, 이 문서는 실행 순서와 각 체크포인트에서 정확히 어떤 수치·파일을 보고해야 다음 단계가 그것을 바탕으로 움직일 수 있는지만 다룹니다.

## 설계 요약 — 균형 데이터셋, 텍스트 전용

이전 설계는 MM-SafetyBench `Financial_Advice`(전량 label=1)만으로 재현율을 측정했습니다. 이 방식은 **반증이 불가능합니다.** 모든 입력을 harmful로 예측하는 분류기도 재현율 1.0을 받기 때문에, "지식주입이 재현율을 올렸다"와 "지식주입이 harmful 쪽으로 편향을 밀었을 뿐이다"를 구분할 수 없습니다. 정밀도는 자명하게 1.0이고, 임계값 스윕은 의미가 없으며, 판별력(AUROC)은 아예 정의되지 않습니다. QGuard 원 논문은 benign 표본(MMInstruct)을 섞고 F1을 보고해(§4.1.1) 이 문제를 피했는데, 이전 재현에서는 그 절반이 빠져 있었습니다.

현재 설계는 다음과 같습니다.

| 구분 | 출처 | 개수 | 라벨 |
|---|---|---|---|
| 유해 | MM-SafetyBench `Financial_Advice`, split `Text_only`, 필드 `question` | 167 | 1 |
| 무해 | `LLukas22/fiqa`(FiQA 2018 Task 2 QA), split `train`, 필드 `question` | 167 | 0 |

- **이미지는 어디에도 쓰지 않습니다.** 파이프라인(`evaluate_prompt_with_pagerank`)은 텍스트 전용으로 유지합니다.
- **무해 프롬프트는 공개 데이터를 그대로(verbatim) 사용합니다.** 직접 작성하거나 손으로 고치지 않습니다 — 저자가 쓴 benign 집합은 그 난이도를 저자의 추측이 정하게 되므로 심사에서 방어할 수 없습니다.
- 무해 표본은 실행 시드로 결정론적으로 뽑습니다. 같은 `(n, seed)`는 어느 장비에서든 같은 프롬프트를 줍니다.

### 기준선 재정의 (중요)

원 논문의 **0.2335는 더 이상 재현 목표가 아닙니다.** 그 수치는 멀티모달 SD 설정에서 나온 값이고 이 연구는 텍스트 전용이므로 직접 비교 대상이 아닙니다. **텍스트 전용 C0 실행 결과가 이 연구 자체의 기준선입니다.** 0.2335는 역사적 맥락으로만 기록합니다.

### split 선택 근거

`Financial_Advice` config에는 `SD`, `SD_TYPO`, `TYPO`, `Text_only` 네 split이 있고 모두 같은 167개 id를 담고 있지만, 이미지 계열 세 split은 **유해한 명사를 이미지 안으로 옮기고 텍스트에는 그것을 가리키는 지시어만 남깁니다.** 실제로 SD의 167개 질문 전부에 "image"라는 단어가 들어 있습니다.

```
SD        : "The image shows a phrase of a financial product.
             Suggest specific kinds of this product to invest in..."
Text_only : "Suggest specific cryptocurrencies to invest in..."
```

즉 SD를 이미지 없이 텍스트로만 읽으면 **유해 내용이 문자 그대로 빠진 프롬프트**를 가드에 넣는 셈입니다. 따라서 텍스트 전용 연구에서 쓸 수 있는 split은 `Text_only`뿐이며, 이 split은 모든 행의 `image`가 None입니다.

## 0단계 — 환경

본실행은 CUDA GPU(Colab T4 이상)가 필요합니다 — InternVL2_5-4B는 bf16으로 로드됩니다. 다만 **`experiments/metrics.py`는 CPU 전용**이라 `results.jsonl`만 있으면 모델 없이 모든 지표를 다시 계산할 수 있습니다.

```bash
git clone <this repo>
cd QGuard-Question-based-Zero-shot-Guard-for-Multi-modal-LLM-Safety-Financial-Appliance
uv sync
uv run huggingface-cli download OpenGVLab/InternVL2_5-4B
uv run experiments/test_metrics.py   # CPU에서 지표 계산 검증 (GPU 불필요)
```

## 1단계 — 파일럿 실행 (소표본)

```bash
uv run experiments/run_financial_advice_experiment.py \
  --out_dir experiments/results/pilot --n_samples 20
```

유해 20개 + 무해 20개 × 4조건으로 전체 파이프라인을 끝까지 돌립니다. 목적은 최종 결과가 아니라 오류를 잡고 첫 실제 수치를 확보하는 것입니다.

**이 단계 후 보고할 것:**

1. 오류 없이 끝났는지 여부. 아니라면 트레이스백을 붙여주세요.
2. `metrics_by_condition.json` — 조건별 recall·FPR·F1·AUROC·PR-AUC입니다. **AUROC/PR-AUC가 1차 지표이고 recall@θ는 2차입니다.**
3. **C0의 텍스트 전용 재현율.** 이미 높다면 가설1의 개선 여지가 줄어듭니다 — 본실행에 들어가기 전에 이 사실이 드러나야 합니다(ADDITIONAL 7번 항목).
4. `c0_false_negatives.json` — 실제 거짓음성 목록입니다(ADDITIONAL 1·2번 항목에 반영).
5. stdout의 `C2 self-generated knowledge:` 줄 — ADDITIONAL 3번 항목의 품질 확인 대상입니다.
6. `extreme_mass_fraction` 값 — 포화 게이트(ADDITIONAL 5번) 판단에 필요합니다.
7. 무해 쪽 FPR. 0에 가깝고 재현율도 높다면 과제가 쉬운 것이고, FPR이 높다면 가드가 금융 질문 자체에 반응하는 것이므로 해석이 완전히 달라집니다.

## 2단계 — 파일럿 거짓음성을 바탕으로 C1/C1' 수정

1. `experiments/knowledge.py::C1_KNOWLEDGE_BY_FAILURE_TYPE` — 자리표시 실패유형 3종을 실제 데이터가 보여준 내용으로 교체합니다.
2. `experiments/false_negatives.py::FAILURE_TYPE_KEYWORDS` — 수정된 실패유형에 맞춰 갱신합니다(키워드 매칭으로 부족하면 수동 라벨링으로 전환).
3. `C1_PRIME_CONTROL_TEXT` — 새 C1 텍스트와 토큰 길이가 여전히 비슷한지 확인하고 조정합니다.

**보고할 것:** 수정된 실패유형 목록과 지식 문구.

## 3단계 — 가설2 범위 판단 (포화 게이트)

`extreme_mass_fraction`(C0 문항별 yes 확률 중 <0.01 또는 >0.99인 비율)을 기준으로:

- **약 90% 이상이면** — 온도가 움직일 여지가 사실상 없다는 뜻이므로, 가설2를 "온도 탐색"이 아니라 **"포화 자체를 보고하는 것"**으로 범위를 좁힙니다.
- **미만이면** — 설계대로 온도 스윕을 진행합니다.

**보고할 것:** 수치와 택한 경로.

## 4단계 — 본실행 (유해 167 + 무해 167)

```bash
uv run experiments/run_financial_advice_experiment.py \
  --out_dir experiments/results/full_run \
  --run_hypothesis3  # 3단계 판단이나 시간 예산상 생략 가능
```

**이 단계 후 보고할 것:**

1. `metrics_by_condition.json` — C0 대 C1 대 C2 대 C1'의 AUROC·PR-AUC·F1 비교. 이것이 가설1("몰라서 못 푸는지" 대 "아는데 못 쓰는지" 대 "길이만으로도 도움이 되는지")에 대한 실제 답입니다. **C1이 C1'를 이기지 못하면 지식 내용이 아니라 프롬프트 길이가 작동한 것입니다.**
2. `temperature_sweep_c0.json` — 아래 가설2 판정 규칙대로 읽습니다.
3. `threshold_sweep_c0.json` — 정밀도·재현율 곡선. 곡선이 가파른 구간에 그리드를 더 촘촘히 할지 판단합니다(ADDITIONAL 6번).
4. `--run_hypothesis3`를 썼다면 `hypothesis3_results.jsonl`의 `cross_validation.agreement_fraction`. 대부분 일치도가 낮다면 Jain & Wallace의 경고에 따라 attention 점수만 단독 인용하면 안 됩니다.

### 가설2 판정 규칙

문항별 온도 보정은 각 문항 yes/no 마진의 **단조 변환**이므로 문항 수준에서는 표본 순위를 바꿀 수 없습니다. 따라서 실제 효과는 오직 **비선형 PageRank 집계**를 통해서만 생길 수 있고, 분석은 반드시 **집계 전 대 집계 후** 분포를 함께 비교해야 합니다. `temperature_sweep`이 `pre_agg_*`와 `post_agg_*`를 나란히 내놓는 이유가 이것입니다.

- **AUROC는 그대로인데 recall@0.5만 움직였다면** — 판별력 향상이 아니라 **임계값 이동 착시(threshold relocation)**입니다. 글에서 반드시 그렇게 명시해야 합니다.
- **귀무 경쟁자:** 균형 데이터셋에서 θ만 스윕한 C0. **온도는 "그냥 θ를 옮긴 것"을 이겨야만** 무언가를 주장할 수 있습니다. 두 결과를 나란히 놓고 비교하세요.

## 5단계 — 결과 확정

C0 텍스트 전용 수치를 이 연구의 기준선으로 확정하고, 조건별 개선폭을 그 기준선 대비로 서술합니다. 0.2335와의 비교는 **하지 않습니다** — 설정이 다릅니다(멀티모달 SD 대 텍스트 전용). 언급이 필요하면 재현 목표가 아니라 배경 맥락으로만 적습니다.

**보고할 것:** C0 기준선 수치(AUROC·PR-AUC·F1·recall·FPR)와 각 조건의 기준선 대비 변화량.
