# Financial-advice guard experiment — run order and what to report

This is the execution-side companion to [docs/ADDITIONAL-EXPERIMENT.md](ADDITIONAL-EXPERIMENT.md). That document covers the judgments the code cannot make on its own; this one covers only the run order and exactly which numbers and files to report at each checkpoint so the next step can act on them.

## Design summary — balanced, text-only

The earlier design measured recall on MM-SafetyBench `Financial_Advice` alone, which is 100% label=1. That is **unfalsifiable**: a classifier that predicts harmful for everything scores recall 1.0, so "knowledge injection improved recall" cannot be distinguished from "knowledge injection shifted the bias toward harmful". Precision is degenerately 1.0, threshold sweeps are meaningless, and discriminability (AUROC) is undefined. QGuard's own paper avoids this by mixing in benign samples (MMInstruct) and reporting F1 (§4.1.1) — the earlier replication dropped that half.

The current design:

| Set | Source | n | Label |
|---|---|---|---|
| Harmful | MM-SafetyBench `Financial_Advice`, split `Text_only`, field `question` | 167 | 1 |
| Benign | `LLukas22/fiqa` (FiQA 2018 Task 2 QA), split `train`, field `question` | 167 | 0 |

- **No images anywhere.** The pipeline (`evaluate_prompt_with_pagerank`) stays text-only.
- **Benign prompts are used verbatim from public data.** Nothing is authored or hand-edited — an author-written benign set has its difficulty set by the author's own guess, which is indefensible in review.
- Benign sampling is seeded and deterministic: a given `(n, seed)` yields the same prompts on any machine.

### Baseline redefinition (important)

The original paper's **0.2335 is no longer a reproduction target.** That figure comes from the multimodal SD setting and this study is text-only, so it is not directly comparable. **The text-only C0 run is this study's own baseline.** Record 0.2335 as historical context only.

### Split decision and its evidence

The `Financial_Advice` config has four splits (`SD`, `SD_TYPO`, `TYPO`, `Text_only`) holding the same 167 ids, but the three image splits **move the harmful noun into the image** and leave the text pointing at it. All 167 SD questions literally contain the word "image":

```
SD        : "The image shows a phrase of a financial product.
             Suggest specific kinds of this product to invest in..."
Text_only : "Suggest specific cryptocurrencies to invest in..."
```

Reading SD text-only therefore feeds the guard a prompt whose harmful content is **literally absent**. `Text_only` is the only usable split for a text-only study, and its `image` field is None for every row.

## Step 0 — environment

The full run needs a CUDA GPU (Colab T4 or better) — InternVL2_5-4B loads in bf16. But **`experiments/metrics.py` is CPU-only**: given `results.jsonl` it recomputes every metric without the model.

```bash
git clone <this repo>
cd QGuard-Question-based-Zero-shot-Guard-for-Multi-modal-LLM-Safety-Financial-Appliance
uv sync
uv run huggingface-cli download OpenGVLab/InternVL2_5-4B
uv run experiments/test_metrics.py   # validates metric math on CPU, no GPU needed
```

## Step 1 — pilot run (small n)

```bash
uv run experiments/run_financial_advice_experiment.py \
  --out_dir experiments/results/pilot --n_samples 20
```

Runs 20 harmful + 20 benign through all four conditions end to end. The goal is catching pipeline errors and getting the first real numbers, not final results.

**Report after this step:**

1. Whether it ran to completion. If not, attach the traceback.
2. `metrics_by_condition.json` — recall, FPR, F1, AUROC, PR-AUC per condition. **AUROC/PR-AUC are primary; recall@θ is secondary.**
3. **C0's text-only recall.** If it is already high, hypothesis 1's improvement headroom shrinks — this must surface before the full run (ADDITIONAL item 7).
4. `c0_false_negatives.json` — the actual false negatives (feeds ADDITIONAL items 1-2).
5. The `C2 self-generated knowledge:` line on stdout — quality check for ADDITIONAL item 3.
6. `extreme_mass_fraction` — needed for the saturation gate (ADDITIONAL item 5).
7. FPR on the benign half. Near-zero FPR with high recall means the task is easy; high FPR means the guard reacts to financial topics as such, which changes the interpretation entirely.

## Step 2 — revise C1/C1' from the pilot's false negatives

1. `experiments/knowledge.py::C1_KNOWLEDGE_BY_FAILURE_TYPE` — replace the three placeholder failure types with what the data actually shows.
2. `experiments/false_negatives.py::FAILURE_TYPE_KEYWORDS` — update to match (or switch to manual labeling if keyword matching proves insufficient).
3. `C1_PRIME_CONTROL_TEXT` — confirm it still matches the new C1 text in token length, and adjust if not.

**Report:** the revised failure types and knowledge text.

## Step 3 — scope hypothesis 2 (saturation gate)

Based on `extreme_mass_fraction` (fraction of C0 per-question yes-probabilities below 0.01 or above 0.99):

- **~90% or above** — temperature has essentially nothing to move, so scope hypothesis 2 down from "temperature exploration" to **reporting the saturation itself**.
- **Below** — proceed with the temperature sweep as designed.

**Report:** the number and which path you took.

## Step 4 — full run (167 harmful + 167 benign)

```bash
uv run experiments/run_financial_advice_experiment.py \
  --out_dir experiments/results/full_run \
  --run_hypothesis3  # drop this if step 3 or the time budget says to skip it
```

**Report after this step:**

1. `metrics_by_condition.json` — AUROC/PR-AUC/F1 across C0 vs C1 vs C2 vs C1'. This is the actual answer to hypothesis 1 ("doesn't know" vs "knows but can't apply" vs "length alone helps"). **If C1 does not beat C1', what worked was prompt length, not knowledge content.**
2. `temperature_sweep_c0.json` — read it with the verdict rule below.
3. `threshold_sweep_c0.json` — the precision-recall curve; decide whether to refine the grid where it is steep (ADDITIONAL item 6).
4. If `--run_hypothesis3` was used, `cross_validation.agreement_fraction` in `hypothesis3_results.jsonl`. If agreement is mostly low, Jain & Wallace's caution applies and attention scores must not be cited alone.

### Hypothesis 2 verdict rule

Per-question temperature is a **monotone transform** of each question's yes/no margin, so it cannot change per-sample ranking at question level. Any real effect can only enter through the **nonlinear PageRank aggregation** — which is why the analysis must compare **pre- vs post-aggregation** distributions. That is what `temperature_sweep` reports side by side as `pre_agg_*` and `post_agg_*`.

- **If AUROC is unchanged but recall@0.5 moves**, that is a **threshold-relocation artifact**, not a discriminability gain. Say so explicitly in the write-up.
- **Null competitor:** threshold-tuned C0 (θ swept on the balanced set). **Temperature must beat "just move θ"** to claim anything. Put the two side by side.

## Step 5 — finalize

Fix the text-only C0 numbers as this study's baseline and describe every condition's change relative to it. **Do not compare against 0.2335** — different setting (multimodal SD vs text-only). If it is mentioned at all, mention it as background, not as a reproduction target.

**Report:** the C0 baseline (AUROC, PR-AUC, F1, recall, FPR) and each condition's delta against it.
