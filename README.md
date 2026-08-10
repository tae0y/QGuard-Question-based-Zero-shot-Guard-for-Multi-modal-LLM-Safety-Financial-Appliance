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
