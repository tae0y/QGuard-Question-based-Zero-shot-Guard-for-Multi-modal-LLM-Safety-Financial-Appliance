# 금융분야 재현율 실험 — 실행 순서 및 보고 항목

이 문서는 [docs/ADDITIONAL-EXPERIMENT-ko.md](ADDITIONAL-EXPERIMENT-ko.md)의 실행판 짝입니다. 그 문서는 아래 각 항목이 *왜* 사람의 판단이 필요한지를 설명하고, 이 문서는 실행 순서와 각 체크포인트에서 정확히 어떤 수치·파일을 보고하면 다음 단계(또는 다음 Claude 세션)가 그것을 바탕으로 행동할 수 있는지만 다룹니다.

## 0단계 — 환경

CUDA GPU(Colab T4 이상)에서 실행해야 합니다 — InternVL2_5-4B는 bf16으로 로드되고, `experiments/attribution.py`는 실제 forward pass가 필요합니다. `experiments/` 안의 코드는 CPU에서 실행되지 않습니다.

```bash
git clone <this repo>
cd QGuard-Question-based-Zero-shot-Guard-for-Multi-modal-LLM-Safety-Financial-Appliance
uv sync
uv run huggingface-cli download OpenGVLab/InternVL2_5-4B
```

## 1단계 — 파일럿 실행 (소표본, 자리표시 지식으로 C0/C1/C2/C1' 실행)

```bash
uv run experiments/run_financial_advice_experiment.py \
  --out_dir experiments/results/pilot --n_samples 20
```

프롬프트 20개 × 4조건으로 전체 파이프라인을 처음부터 끝까지 돌려보는 단계입니다. 목적은 최종 결과가 아니라 파이프라인 오류를 잡아내고 첫 실제 수치를 확보하는 것입니다.

**이 단계 후 보고할 것:**
1. 오류 없이 끝까지 실행됐는지 여부. 안 됐다면 트레이스백을 붙여주세요.
2. `experiments/results/pilot/c0_false_negatives.json` — 실제 거짓음성 프롬프트 목록입니다(ADDITIONAL-EXPERIMENT-ko.md 1번 항목인 C1 지식 내용, 2번 항목인 실패유형 분류기에 반영됩니다).
3. stdout에 출력되는 `C2 self-generated knowledge:` 줄 — ADDITIONAL-EXPERIMENT-ko.md 3번 항목의 품질 확인 대상입니다.
4. `experiments/results/pilot/calibration_diagnostics.json`의 `extreme_mass_fraction` 값 — ADDITIONAL-EXPERIMENT-ko.md 5번 항목(90% 기준선 판단)에 필요합니다.
5. `experiments/results/pilot/reliability_diagram_c0.png` — 보정 형태를 눈으로 확인하기 위한 자료입니다.

## 2단계 — 파일럿 거짓음성을 바탕으로 C1/C1' 수정

1단계에서 나온 실제 거짓음성을 읽은 뒤:

1. `experiments/knowledge.py::C1_KNOWLEDGE_BY_FAILURE_TYPE`을 수정합니다 — 세 가지 자리표시용 실패 유형과 문구를 실제 데이터가 보여준 내용으로 교체합니다.
2. `experiments/false_negatives.py::FAILURE_TYPE_KEYWORDS`를 수정된 실패 유형에 맞춰 갱신합니다(또는 키워드 매칭으로는 부족하다고 판단되면 다음 실행부터 수동 라벨링으로 전환합니다).
3. `experiments/knowledge.py::C1_PRIME_CONTROL_TEXT`가 새 C1 텍스트와 여전히 토큰 길이가 비슷한지 확인하고, 아니라면 지도학 문단을 늘리거나 줄입니다.

**이 단계 후 보고할 것:** 수정된 실패 유형 목록과 지식 문구입니다. 이를 바탕으로 리뷰 문서(`~260811 실험설계및수행 리뷰.md`)도 계획이 아니라 실제 구현된 내용으로 갱신할 수 있습니다.

## 3단계 — 가설2 범위 판단

1단계에서 나온 `extreme_mass_fraction`을 기준으로:

- 약 90%를 넘는다면(신승민 멘토의 우려가 재현된 경우): 본실행 전에 가설2 범위를 어떻게 좁힐지 결정합니다 — 예를 들어 전체 온도 스윕을 생략하고 극단 쏠림 결과만 보고하는 방향으로 가거나, 스윕은 유지하되 "탐색"이 아니라 "확인"으로 서술을 바꿉니다.
- 넘지 않는다면: 설계한 대로 가설2를 그대로 진행합니다.

**보고할 것:** extreme_mass_fraction 수치와 어느 경로를 택했는지입니다. 이에 따라 4단계 본실행의 플래그를 맞춰 설정합니다.

## 4단계 — 본실행 (501쌍)

```bash
uv run experiments/run_financial_advice_experiment.py \
  --out_dir experiments/results/full_run \
  --run_hypothesis3  # 3단계 판단이나 시간 예산상 생략한다면 이 옵션을 빼세요
```

**이 단계 후 보고할 것:**
1. `experiments/results/full_run/hypothesis2_results.json`의 `best_temperature`, `corp_before`, `corp_after`. `dsc`가 보정 전후로 거의 안 움직였는데 `temperature_sweep`의 `recall`은 움직였다면, 김기환 멘토가 우려한 "임계값 이동 착시"가 확인된 것이므로 글쓰기에서 명시적으로 짚어야 합니다.
2. 같은 파일 안의 `threshold_sweep` — 기본 0.05~0.95 그리드가 정밀도·재현율 트레이드오프 구간을 충분히 담고 있는지, 아니면 특정 구간에 더 촘촘한 해상도가 필요한지(ADDITIONAL-EXPERIMENT-ko.md 7번 항목).
3. `results.json`에서 뽑은 C0 대 C1 대 C2 대 C1'의 재현율·risk_score 비교 — 이것이 가설1("몰라서 못 푸는지" 대 "아는데 못 쓰는지" 대 "길이만으로도 도움이 되는지")에 대한 실제 답입니다.
4. `--run_hypothesis3`를 사용했다면 `hypothesis3_results.json`의 `cross_validation.agreement_fraction` 값들 — 대부분 사례에서 일치도가 낮다면, 리뷰 문서가 이미 명시한 Jain & Wallace의 경고에 따라 attention 점수만 단독으로 글에 인용하면 안 됩니다.

## 5단계 — 재현 여부 확인

본실행의 C0 재현율을 QGuard가 보고한 0.2335와 비교합니다. 오차 범위를 넘어서 차이가 난다면:

- `experiments/run_financial_advice_experiment.py::load_financial_advice_prompts`에 하드코딩된 `split="SD"`를 원 논문의 평가 프로토콜과 다시 대조합니다(ADDITIONAL-EXPERIMENT-ko.md 9번 항목) — `SD`가 재현하지 못한다면 MM-SafetyBench의 다른 split(`SD_TYPO`, `TYPO`, `Text_only`)을 시도합니다.

**보고할 것:** C0 재현율 수치와, 0.2335와 맞지 않는다면 어느 split이 맞는지(있다면)입니다.
