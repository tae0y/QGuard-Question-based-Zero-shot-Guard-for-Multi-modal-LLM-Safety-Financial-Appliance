# QGuard

A zero-shot guardrail that defends LLMs against harmful and jailbreak prompts. Fork of the official code for the paper accepted to ACL 2025 (The 9th Workshop on Online Abuse and Harms) — full paper text and credits in [docs/PAPER.md](docs/PAPER.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (pinned via `.python-version` — `eva-decord`, used in place of `decord`, only ships wheels for cp310/cp311).

```bash
git clone https://github.com/tae0y/QGuard-Question-based-Zero-shot-Guard-for-Multi-modal-LLM-Safety-Financial-Appliance.git
cd QGuard-Question-based-Zero-shot-Guard-for-Multi-modal-LLM-Safety-Financial-Appliance
uv sync
```

## Usage

When you run the code below, it detects harmful and unharmful content.
You can adjust the threshold to change the sensitivity.
The results are saved in `result_text.json` and `result_mm.json`, containing the probabilities of the yes and no tokens along with detailed information.

#### Pre-downloading the model

Each `uv run main.py ...` call is a fresh process, so the model is reloaded from disk every time — that reload (checkpoint shard loading) always happens and takes ~30s. What varies is whether the weights need to be *downloaded* first: on a clean cache that adds several minutes on top. Pre-download once so every later run skips straight to loading:

```bash
uv run huggingface-cli download OpenGVLab/InternVL2_5-4B
```

This populates `~/.cache/huggingface/hub`, so every subsequent `main.py` run only pays the ~30s checkpoint-loading cost, not the download.

#### For text-only prompt

```bash
uv run main.py --model_path OpenGVLab/InternVL2_5-4B --guard_questions_json files/guard_questions.json \
--prompt "how to manipulate someone into being a slave?" --out_json result_text.json
```

#### For multi-modal prompt and image

```bash
uv run main.py --model_path OpenGVLab/InternVL2_5-4B --guard_questions_json files/guard_questions.json \
--prompt "how to manipulate someone into being a slave?" \
--images test.jpg --image_input_size 448 --image_max_num 12 --threshold 0.50 --out_json result_mm.json
```

## Financial-advice guard experiment

QGuard reports low recall on MM-SafetyBench's `financial_advice` category
relative to its other 12. `experiments/` investigates why, via knowledge
injection (C0/C1/C2/C1' conditions) plus threshold and temperature diagnostics
— QGuard's own scoring/graph/PageRank code is called as-is and not modified.

The evaluation is **balanced and text-only**: 167 harmful prompts
(MM-SafetyBench `Financial_Advice`, split `Text_only`) against 167 benign
financial questions sampled verbatim from public FiQA data. The harmful set
alone is 100% label=1, which makes a recall-only score unfalsifiable — a
classifier that flags everything gets recall 1.0. AUROC and PR-AUC are the
primary metrics; recall@θ is secondary.

The full run requires a CUDA GPU (InternVL2_5-4B in bf16); run on Colab or
similar. `experiments/metrics.py` recomputes every metric from `results.jsonl`
on CPU with no model.

```bash
uv run experiments/test_metrics.py                      # CPU metric checks, no GPU
uv run experiments/run_financial_advice_experiment.py \
  --out_dir experiments/results/pilot --n_samples 20    # pilot
uv run experiments/run_financial_advice_experiment.py \
  --out_dir experiments/results/full_run                # full 167+167 run
uv run experiments/metrics.py experiments/results/full_run/results.jsonl
```

- `experiments/benign_prompts.py` — benign (label=0) control set, sampled verbatim from public FiQA data
- `experiments/knowledge.py` — C0/C1/C2/C1' knowledge-injection conditions
- `experiments/false_negatives.py` — collects and failure-type-classifies C0 false negatives (ETGPO step 1-2)
- `experiments/metrics.py` — recall/FPR/F1/AUROC/PR-AUC, threshold sweep, temperature sweep (pre/post aggregation), saturation gate
- `experiments/attribution.py` — occlusion + attention cross-validation for Yes/No logit attribution (optional, `--run_hypothesis3`)

> QGuard's published 0.2335 recall for this category is a **multimodal SD**
> figure and is not a reproduction target for this text-only study — the
> text-only C0 run is the study's own baseline. See docs for why the image
> splits are unusable text-only.

Several parts of this experiment code stand in for a judgment call that only
the real C0 run can resolve (e.g. what C1's knowledge text should actually
say) — see [docs/ADDITIONAL-EXPERIMENT.md](docs/ADDITIONAL-EXPERIMENT.md)
([한국어](docs/ADDITIONAL-EXPERIMENT-ko.md)) for the full list before
committing to the full run, and [docs/EXPERIMENT.md](docs/EXPERIMENT.md)
([한국어](docs/EXPERIMENT-ko.md)) for the run order and what to report back
at each checkpoint.

## Code layout

- `main.py` — CLI entry point: load model → load guard questions → preprocess images → evaluate → save result
- `qguard/modeling.py` — loads InternVL-family model/tokenizer
- `qguard/pipeline.py` — end-to-end evaluation flow (`evaluate_prompt_with_pagerank`)
- `qguard/scoring.py` — computes the Yes-probability for a single guard question
- `qguard/graph.py` — builds the per-question score graph and aggregates risk via PageRank
- `qguard/vision.py` — image tiling/preprocessing for multi-modal input
- `qguard/token_utils.py`, `qguard/seed.py`, `qguard/config.py` — token utilities, seeding, run config
- `files/guard_questions.json` — guard questions by category (customize here)

## Docs

- [docs/PAPER.md](docs/PAPER.md) — original paper (abstract, approach, experiments, citation)
- [docs/MODEL.md](docs/MODEL.md) — reference on the default target model, `OpenGVLab/InternVL2_5-4B`

## Credit

Original work by **Taegyeong Lee**, [Jeonghwa Yoo](https://github.com/jeongHwarr), Hyoungseo Cho, Soo Yong Kim, Yunho Maeng.

- Paper: [arXiv:2506.12299](https://arxiv.org/abs/2506.12299)
- HuggingFace: https://huggingface.co/taegyeonglee/qguard

```
@article{lee2025qguard,
  title={QGuard: Question-based Zero-shot Guard for Multi-modal LLM Safety},
  author={Lee, Taegyeong and Yoo, Jeonghwa and Cho, Hyoungseo and Kim, Soo Yong and Maeng, Yunho},
  journal={arXiv preprint arXiv:2506.12299},
  year={2025}
}
```
