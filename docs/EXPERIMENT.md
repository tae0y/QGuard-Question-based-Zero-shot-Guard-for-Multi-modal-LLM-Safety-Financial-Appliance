# Financial-advice recall experiment — run order and what to report back

This is the operational companion to
[docs/ADDITIONAL-EXPERIMENT.md](ADDITIONAL-EXPERIMENT.md), which explains
*why* each item below needs a human judgment call. This doc is just the
run order and, at each checkpoint, exactly which numbers/files to report
back so the next step (or the next Claude session) can act on them.

## Step 0 — environment

Run on a CUDA GPU (Colab T4 or better) — InternVL2_5-4B loads in bf16 and
`experiments/attribution.py` needs a real forward pass. Nothing in
`experiments/` runs on CPU.

```bash
git clone <this repo>
cd QGuard-Question-based-Zero-shot-Guard-for-Multi-modal-LLM-Safety-Financial-Appliance
uv sync
uv run huggingface-cli download OpenGVLab/InternVL2_5-4B
```

## Step 1 — pilot run (small sample, C0/C1/C2/C1' with placeholder knowledge)

```bash
uv run experiments/run_financial_advice_experiment.py \
  --out_dir experiments/results/pilot --n_samples 20
```

This exercises the full pipeline end to end on 20 prompts × 4 conditions —
its job is to catch pipeline errors and produce the first real numbers, not
to be a final result.

**Report back after this step:**
1. Did it run to completion without errors? If not, paste the traceback.
2. `experiments/results/pilot/c0_false_negatives.json` — the actual false-negative prompts (this feeds ADDITIONAL-EXPERIMENT.md item 1: C1 knowledge content, and item 2: failure-type classifier).
3. The `C2 self-generated knowledge:` line printed to stdout — sanity-check target for ADDITIONAL-EXPERIMENT.md item 3.
4. `experiments/results/pilot/calibration_diagnostics.json` — specifically `extreme_mass_fraction` (ADDITIONAL-EXPERIMENT.md item 5 — the 90% bar judgment call).
5. `experiments/results/pilot/reliability_diagram_c0.png` — visual gut-check on calibration shape.

## Step 2 — revise C1/C1' based on pilot false negatives

Once you've read the real false negatives from Step 1:

1. Edit `experiments/knowledge.py::C1_KNOWLEDGE_BY_FAILURE_TYPE` — replace
   the three placeholder failure types and their text with what the actual
   data showed.
2. Edit `experiments/false_negatives.py::FAILURE_TYPE_KEYWORDS` to match the
   revised failure types (or decide keyword matching isn't good enough and
   swap in manual labeling for the next run).
3. Check `experiments/knowledge.py::C1_PRIME_CONTROL_TEXT` is still roughly
   token-length-matched to the new C1 text; pad or trim the cartography
   paragraph if not.

**Report back after this step:** the revised failure-type list and knowledge
text, so the review doc (`~260811 실험설계및수행 리뷰.md`) can be updated to
match what was actually implemented, not just what was planned.

## Step 3 — decide Hypothesis 2 scope

Using the pilot's `extreme_mass_fraction` from Step 1:

- If it clears ~90% (mentor 신승민's concern reproduced): decide how to
  narrow Hypothesis 2 before the full run — e.g. skip the full temperature
  sweep and go straight to reporting the extreme-skew finding, or keep the
  sweep but frame it as confirming rather than exploring.
- If it doesn't: proceed with Hypothesis 2 as designed.

**Report back:** the extreme_mass_fraction number and which path you're
taking, so the full-run flags (Step 4) are set correctly.

## Step 4 — full run (501 pairs)

```bash
uv run experiments/run_financial_advice_experiment.py \
  --out_dir experiments/results/full_run \
  --run_hypothesis3  # omit if Step 3/time budget says skip it
```

**Report back after this step:**
1. `experiments/results/full_run/hypothesis2_results.json` — specifically
   `best_temperature`, `corp_before`, `corp_after`. Compare `dsc` before vs.
   after: if `dsc` barely moved but `recall` in `temperature_sweep` did,
   that's mentor 김기환's "threshold-shift illusion" concern confirmed —
   worth flagging explicitly in the writeup.
2. `experiments/results/full_run/threshold_sweep` (inside the same file) —
   whether the precision/recall tradeoff is well-covered by the default
   0.05–0.95 grid, or needs finer resolution somewhere (ADDITIONAL-EXPERIMENT.md item 7).
3. C0 vs C1 vs C2 vs C1' recall/risk_score comparison from `results.jsonl` —
   this is the actual answer to Hypothesis 1 ("doesn't know" vs "knows but
   doesn't use it" vs "length alone helps").
4. If `--run_hypothesis3` was used: `hypothesis3_results.jsonl`'s
   `cross_validation.agreement_fraction` values — low agreement across most
   cases means attention scores shouldn't be cited alone in the writeup
   (per the Jain & Wallace caveat already noted in the review doc).

## Step 5 — reproduction sanity check

Compare the full run's C0 recall to QGuard's reported 0.2335. If it's off by
more than noise:

- Re-check `experiments/run_financial_advice_experiment.py::load_financial_advice_prompts`'s
  hardcoded `split="SD"` against the original paper's eval protocol
  (ADDITIONAL-EXPERIMENT.md item 9) — try the other MM-SafetyBench splits
  (`SD_TYPO`, `TYPO`, `Text_only`) if `SD` doesn't reproduce it.

**Report back:** the C0 recall number and, if it doesn't match 0.2335, which
split (if any) does.
