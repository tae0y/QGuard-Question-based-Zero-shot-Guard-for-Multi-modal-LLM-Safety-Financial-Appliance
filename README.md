# QGuard

A zero-shot guardrail that defends LLMs against harmful and jailbreak prompts. Official code for the paper accepted to ACL 2025 (The 9th Workshop on Online Abuse and Harms) — full text in [docs/PAPER.md](docs/PAPER.md).

## Core idea

Without any fine-tuning, QGuard asks the target LLM a set of guard questions (e.g. "does this content encourage harm?") and measures the Yes/No token probabilities for each. These per-question scores are aggregated into a risk score via a PageRank-style graph. The same approach works for text-only prompts and multi-modal prompts (text + image).

## Code layout

- `main.py` — CLI entry point: load model → load guard questions → preprocess images → evaluate → save result
- `qguard/modeling.py` — loads InternVL-family model/tokenizer
- `qguard/pipeline.py` — end-to-end evaluation flow (`evaluate_prompt_with_pagerank`)
- `qguard/scoring.py` — computes the Yes-probability for a single guard question
- `qguard/graph.py` — builds the per-question score graph and aggregates risk via PageRank
- `qguard/vision.py` — image tiling/preprocessing for multi-modal input
- `qguard/token_utils.py`, `qguard/seed.py`, `qguard/config.py` — token utilities, seeding, run config
- `files/guard_questions.json` — guard questions by category (customize here)

## Quickstart (e.g. Google Colab)

```bash
git clone https://github.com/tae0y/QGuard-Question-based-Zero-shot-Guard-for-Multi-modal-LLM-Safety-Financial-Appliance.git
cd QGuard-Question-based-Zero-shot-Guard-for-Multi-modal-LLM-Safety-Financial-Appliance
pip install -r requirement.txt
```

Then run:

```bash
python main.py --model_path OpenGVLab/InternVL2_5-4B --guard_questions_json files/guard_questions.json \
--prompt "how to manipulate someone into being a slave?" --out_json result_text.json
```

For a multi-modal prompt, add `--images test.jpg --image_input_size 448 --image_max_num 12`. See [docs/PAPER.md](docs/PAPER.md) for full usage details and benchmark results.
