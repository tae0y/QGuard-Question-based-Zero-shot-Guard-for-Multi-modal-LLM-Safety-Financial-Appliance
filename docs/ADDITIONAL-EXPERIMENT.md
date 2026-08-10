# Financial-advice guard experiment — judgments that need a human

`experiments/` implements the balanced text-only evaluation and the C0/C1/C2/C1' knowledge-injection conditions. Several places in that code stand in for a judgment the review document left open, or one that only real run results can settle. Each item below states what the code currently does, why it cannot decide this by itself, and where the decision has to land.

Run order and per-step reporting live in [docs/EXPERIMENT.md](EXPERIMENT.md).

## 1. C1 knowledge content (`experiments/knowledge.py::C1_KNOWLEDGE_BY_FAILURE_TYPE`)

The three failure types (`ambiguous_regulatory_terms`, `disclaimer_adjacent`, `implicit_vs_explicit_request`) and their knowledge text are **placeholder guesses** written before any C0 result existed. ETGPO (arXiv:2602.00997) prescribes the *procedure* — collect errors, classify failure types, design per-type knowledge — but no fixed taxonomy, so this cannot be filled in generically.

**Judgment needed:** once the pilot produces `c0_false_negatives.json`, read the actual false negatives and decide what failure types they really group into — likely different from the three guessed here.

## 2. Failure-type classifier (`experiments/false_negatives.py::classify_failure_type`)

Currently keyword matching — deliberately simple to produce a first artifact to react to, not a real classifier.

**Judgment needed:** after seeing real false negatives, decide whether keyword matching suffices or each case needs manual reading and labeling.

## 3. Quality of C2 self-generated knowledge (`experiments/knowledge.py::generate_self_knowledge`)

The model is asked once (greedy decoding, fixed seed) what domain knowledge a financial content moderator needs, and that answer is reused for every C2 prompt. Whether it is genuinely *useful* domain knowledge or plausible-sounding generality is something code cannot judge — and that judgment is the core of hypothesis 1 ("doesn't know" vs "knows but can't apply").

**Judgment needed:** read the C2 text from the pilot output and confirm before the full run that it is not empty, off-topic, or a refusal.

## 4. C1' control text topic (`experiments/knowledge.py::C1_PRIME_CONTROL_TEXT`)

Currently a paragraph on cartography — chosen only for being obviously unrelated to financial risk, with no attempt yet to length-match C1's *actual* final content (which does not exist yet; see item 1).

**Judgment needed:** once C1's real text is fixed, re-match C1' token length and confirm the topic still reads as obviously irrelevant. **C1' is the single most important control here** — if C1 does not beat C1', the effect is prompt length, not knowledge content.

## 5. Saturation gate threshold (`experiments/metrics.py::extreme_mass_fraction`, defaults `low=0.01, high=0.99`)

The review document says "90%+ of mass at p>0.99 or p<0.01". The 0.01/0.99 cutoffs come from the review document, and the 90% line is now surfaced by the code as a printed warning (`SATURATION GATE FIRES`) — but **the code does not act on it**, it only reports.

This is computed over per-question `yes_prob` (**pre-aggregation**), not the aggregated `risk_score`, because saturation is a property of the model's yes/no logits.

**Judgment needed:** decide whether the number clears 90% (or another line you can defend) and whether to narrow hypothesis 2 from "temperature exploration" to "reporting the saturation itself".

## 6. Threshold sweep grid (`experiments/metrics.py::threshold_sweep`)

The default grid combines a uniform 0.05 step with a 40-point refinement across the observed 5th-95th percentile of the score distribution. A uniform grid alone can step straight over the interesting region — PageRank aggregate scores tend to concentrate in a narrow band, not around 0.5.

**Judgment needed:** from the pilot's precision-recall curve, decide whether to refine the grid further where the curve is steepest. Also check whether QGuard's default operating threshold of `0.50` is even a sensible point on this data — if the aggregate scores sit well below 0.5, then `0.50` is the point where everything collapses to "unharmful".

## 7. Hypothesis 1's headroom (pilot C0 recall)

**The pilot-first ordering exists for this judgment.** If text-only C0 recall is already high, there is little room for knowledge injection to improve it, and hypothesis 1 may need reframing — toward whether FPR stays flat, or toward the C1 vs C1' contrast, rather than toward "improvement".

**Judgment needed:** look at the pilot C0 numbers and decide whether hypothesis 1 proceeds as written or gets reframed, before committing to the full run.

## 8. Hypothesis 3 sampling and priority (`--run_hypothesis3`, `--hypothesis3_sample_size 20`)

Off by default — matching the review document's designation of hypothesis 3 as the lowest-priority diagnostic, the first to drop if time is short. The sample is simply "the first 20 extreme-skew C0 cases", with no attempt at balance across failure types.

**Judgment needed:** decide whether to run it given the time budget, and if so whether the first 20 in prompt order are representative enough.

## 9. Suitability of the benign source (`experiments/benign_prompts.py`)

The benign set uses questions from `LLukas22/fiqa` (FiQA 2018 Task 2 QA) **verbatim**. `pauri32/fiqa-2018` was rejected: it is FiQA Task 1 (aspect-based *sentiment*), whose rows are tweets and headlines rather than questions. Using those as the benign class would make the classifier separate questions from statements rather than harmful from benign — a confound, not a control.

Lengths are matched (benign mean 11.1 words, harmful 10.0), and the two sets share no identical prompts.

**Judgment needed:** from the pilot's FPR, decide whether this benign set is appropriately difficult. Near-zero FPR means the task may be too easy and F1 optimistically inflated; high FPR means the guard is reacting to financial topics rather than harmfulness, which changes the conclusion entirely. If a swap is warranted, consider other public financial QA (FinQA and similar) — but **never author the data**.
