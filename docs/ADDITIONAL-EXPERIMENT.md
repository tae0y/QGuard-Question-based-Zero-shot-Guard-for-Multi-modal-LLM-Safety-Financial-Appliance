# Financial-advice recall experiment — open decisions

`experiments/` implements the C0/C1/C2/C1' knowledge-injection conditions and
the ECE/NCE/CORP calibration diagnostics from the mentor-feedback review
(`~260811 실험설계및수행 리뷰.md`). Several places in that code stand in for a
judgment call the review doc leaves open, or that only the real C0 run can
resolve. Each item below: what the code currently does, why it can't decide
this on its own, and what the resolution should feed back into.

## 1. C1 knowledge content (`experiments/knowledge.py::C1_KNOWLEDGE_BY_FAILURE_TYPE`)

The three failure types (`ambiguous_regulatory_terms`, `disclaimer_adjacent`,
`implicit_vs_explicit_request`) and their knowledge text are a **placeholder
guess**, written before any real C0 output existed. ETGPO (arXiv:2602.00997)
prescribes the *procedure* — collect errors, cluster into failure types,
design knowledge per type — not a fixed taxonomy, so this can't be filled in
generically.

**Decision needed:** after the C0 pilot run produces `c0_false_negatives.json`,
read the actual false-negative prompts and judge what failure types they
really cluster into — likely different from the three guessed here — then
rewrite both the failure-type set and each knowledge snippet.

## 2. Failure-type classifier (`experiments/false_negatives.py::classify_failure_type`)

Currently a keyword-match heuristic (`FAILURE_TYPE_KEYWORDS`) — deliberately
crude, meant only to give a first pass to react to, not a real classifier.

**Decision needed:** after seeing real C0 false negatives, decide whether
keyword matching is good enough or whether each case needs manual reading
and labeling. If the failure types themselves change (see #1), the keyword
lists need to change with them.

## 3. C2 self-generated knowledge quality (`experiments/knowledge.py::generate_self_knowledge`)

The model is asked once (`C2_SELF_QUERY_PROMPT`, greedy decoding, fixed seed)
to describe what financial-domain knowledge a moderator would need, and that
output is reused for every C2 prompt. The code cannot judge whether the
model's self-generated answer is actually *useful* domain knowledge or generic
filler — that's the entire point of the C0 vs C1 vs C2 comparison (Hypothesis 1:
"doesn't know" vs "knows but doesn't use it").

**Decision needed:** after the pilot, read the printed C2 knowledge text
(logged to stdout and in `results.json`) and sanity-check it's not degenerate
(empty, off-topic, or a refusal) before committing to the full 501-pair run.

## 4. C1' control text topic (`experiments/knowledge.py::C1_PRIME_CONTROL_TEXT`)

Currently a paragraph about cartography — chosen only to be clearly unrelated
to financial risk, with no attempt made to length-match it to C1's *actual*
final content (which doesn't exist yet, see #1). This control only stays
valid as a "length/format matched, content-irrelevant" baseline if it's kept
roughly the same token length as C1's real text.

**Decision needed:** once C1's real knowledge text is written (#1), re-check
C1' is still length-matched (pad or trim the cartography paragraph), and
confirm the topic still reads as clearly irrelevant to a reviewer.

## 5. Extreme-skew thresholds for calibration diagnosis (`experiments/calibration.py::extreme_mass_fraction`, defaults `low=0.01, high=0.99`)

The review doc's own language is "상위 90% 이상이 p>0.99 또는 p<0.01" — the
0.01/0.99 cutoffs are already fixed by the review doc, but the **90% bar
for "confirmed"** is not encoded anywhere in the code. `run_financial_advice_experiment.py`
prints `extreme_mass_fraction` but doesn't compare it to a threshold or act on it.

**Decision needed:** after the pilot prints this number, judge whether it
clears 90% (or some other bar you're comfortable with) and decide, per the
review doc's own framing, whether to narrow Hypothesis 2's scope as a result.
This is explicitly named in the review doc as a "판단이 필요한 항목."

## 6. Best temperature selection (`run_financial_advice_experiment.py`, `best_t = min(temp_rows, key=...ece...)`)

The script picks the temperature that minimizes ECE and reports CORP
before/after using that single value. This is a reasonable default, not a
methodological requirement — the review doc's own point (mentor 김기환) is
that ECE improving doesn't by itself prove DSC (discrimination) improved.

**Decision needed:** after seeing the full temperature sweep and the
before/after DSC numbers, judge whether "best ECE" was the right temperature
to report, or whether a different point on the sweep (e.g. best DSC, or a
temperature that keeps recall roughly constant) tells a more honest story
for the writeup.

## 7. Threshold sweep range (`experiments/calibration.py::threshold_sweep`, default `0.05` to `0.95` step `0.05`)

Arbitrary default grid. QGuard's own default operating threshold is `0.50`
(`RunConfig.threshold`); nothing here checks whether the interesting behavior
(where recall/precision trade off) actually falls inside this grid or needs
finer resolution near a particular point.

**Decision needed:** after the pilot, look at the printed precision/recall
curve and decide whether the grid needs to be narrowed or made finer around
wherever the curve is steepest.

## 8. Hypothesis 3 sample selection and priority (`run_financial_advice_experiment.py`, `--run_hypothesis3`, `--hypothesis3_sample_size 20`)

Off by default (`action="store_true"`), matching the review doc's framing
that Hypothesis 3 is the lowest-priority, first-to-cut diagnostic if time
runs short during the 8/9-8/10 run. The sample is just "first 20 extreme-skew
cases from C0," with no attempt to balance across failure types or pick
illustrative examples.

**Decision needed:** decide whether to run this at all given the time
budget on the day (the review doc explicitly authorizes skipping it), and if
run, whether 20 cases picked in prompt order is representative enough or
whether specific cases should be hand-picked for the writeup.

## 9. `financial_advice` split choice (`run_financial_advice_experiment.py::load_financial_advice_prompts`, split `"SD"`)

MM-SafetyBench's `Financial_Advice` config has multiple splits (`SD`,
`SD_TYPO`, `TYPO`, `Text_only`); the code hardcodes `SD` as the closest match
to QGuard's original reported figure, but this mapping was not independently
re-verified against the original paper's exact eval protocol.

**Decision needed:** before treating any recall number from this pipeline as
comparable to QGuard's reported 0.2335, confirm `SD` is in fact the split the
original paper used (or run all splits and report which one reproduces
0.2335 on C0).
