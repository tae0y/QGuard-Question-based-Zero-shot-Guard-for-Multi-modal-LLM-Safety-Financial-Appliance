# Model: OpenGVLab/InternVL2_5-4B

Default `--model_path` used by this repo's quickstart commands. Reference: [HuggingFace](https://huggingface.co/OpenGVLab/InternVL2_5-4B) · [InternVL2.5 blog](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/) · [GitHub](https://github.com/OpenGVLab/InternVL)

## Architecture

"ViT-MLP-LLM" paradigm: a vision encoder feeds a randomly-initialized MLP projector into a language model backbone.

| Component | Model | Params |
|---|---|---|
| Vision encoder | InternViT-300M-448px-V2.5 | 304.01M |
| Projector | MLP (randomly initialized) | small |
| Language model | Qwen2.5-3B-Instruct | 3.40B |
| **Total** | | **3.71B** |

Uses pixel-unshuffle to cut visual token count to 1/4, and "dynamic resolution" tiling (448×448 tiles, up to 36 tiles at train time, up to 128 at test time) for multi-image and high-resolution input.

## Model family (size comparison)

| Model | Total | ViT | LLM |
|---|---|---|---|
| InternVL2.5-1B | 938.19M | 304.01M | 629.70M |
| InternVL2.5-2B | 2.21B | 304.01M | 2.21B |
| **InternVL2.5-4B** | **3.71B** | **304.01M** | **3.40B** |
| InternVL2.5-8B | 8.08B | 304.01M | 7.74B |

## Training

3-stage pipeline:
1. MLP warmup (vision encoder and LLM frozen)
2. Progressive ViT training (optional)
3. Full-model instruction tuning

Data augmentation includes random JPEG compression (quality 75–100).

## Benchmarks (InternVL2.5-4B)

| Benchmark | Score |
|---|---|
| MMMU (val) | 52.3 |
| MMMU (test) | 46.3 |
| DocVQA (test) | 91.6 |
| ChartQA (test avg.) | 84.0 |
| MathVista (mini) | 60.5 |
| OCRBench | 828 |

Also evaluated on multi-image/real-world scene understanding, OCR/chart/document recognition, visual reasoning and math, video understanding, and multilingual multimodal understanding.

## Capabilities

Single-image, multi-image, video, and multilingual text input; conversational multimodal use.

### Korean support

The language backbone (Qwen2.5-3B-Instruct) is pretrained on 29+ languages including Korean, so the model can process Korean text at some level. However:

- InternVL2.5's own multilingual benchmarks (MMMB, Multilingual MMBench, MTVQA) cover English, Chinese, Portuguese, Arabic, Turkish, and Russian — **Korean is not included**, so there is no official score to cite.
- Vision-language alignment data is not Korean-centric, so image+Korean-prompt performance may be weaker than image+English.
- For this repo specifically: if guard questions (`files/guard_questions.json`) are written in Korean, the Yes/No token-probability scoring in `qguard/scoring.py` may be less reliable than with English questions — validate empirically before relying on it.

## Deployment / inference

Supported via Transformers (`trust_remote_code=True`, requires `transformers>=4.37.2`), vLLM, SGLang, and LMDeploy. LMDeploy example configs use `session_len=8192`; no explicit max context length is published for this checkpoint by OpenGVLab.

## License

- Repo/model card: **MIT License**
- Underlying LLM (Qwen2.5-3B-Instruct): **Apache License 2.0** — both terms apply since InternVL2.5-4B is built on it.

## Relevance to this repo

`qguard/modeling.py` loads this checkpoint via `AutoModel.from_pretrained(..., trust_remote_code=True, use_flash_attn=True)` in `bfloat16`. At bf16, the ~3.7B parameters alone need roughly 7–8GB VRAM; with activations, KV cache, and vision-encoder overhead, budget **~10–12GB VRAM** to run this model comfortably on GPU.
