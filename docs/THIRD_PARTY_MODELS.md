# Third-Party Models

This repository is Apache-2.0 licensed application code. It does not vendor,
redistribute, or embed third-party model weights. Runtime model downloads and
API calls remain subject to the upstream model providers' licenses and terms.

## Runtime Defaults

| Purpose | Model ID | Provider | Runtime path | License / terms |
|---|---|---|---|---|
| Full-document guard | `Qwen/Qwen3Guard-Gen-4B` | Qwen team | Hugging Face Transformers | Apache 2.0 |
| Streaming guard | `Qwen/Qwen3Guard-Stream-0.6B` | Qwen team | Hugging Face Transformers | Apache 2.0 |
| Custom-policy reasoning | `nvidia/Nemotron-Content-Safety-Reasoning-4B` | NVIDIA | Hugging Face Transformers | NVIDIA Open Model License + Gemma Terms of Use |
| Deep triage | `nvidia/nemotron-3-ultra-220b-a12b` | NVIDIA / OpenRouter | OpenAI-compatible chat API | Provider API terms |

## Source Links

- Qwen3Guard repository: https://github.com/QwenLM/Qwen3Guard
- Qwen3Guard-Gen-4B model card: https://huggingface.co/Qwen/Qwen3Guard-Gen-4B
- Qwen3Guard-Stream-0.6B model card: https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B
- Nemotron-Content-Safety-Reasoning-4B model card: https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B
- Nemotron-Content-Safety-Reasoning-4B NIM card: https://build.nvidia.com/nvidia/nemotron-content-safety-reasoning-4b/modelcard
- Full model survey and alternatives: `docs/MODEL_CATALOG.md`

## Submission Notes

- The Dockerfile intentionally skips model pre-download by default via
  `NEMOGUARDIAN_SKIP_PREDOWNLOAD=1`; this keeps the image redistributable and
  avoids bundling model artifacts.
- The GPU recording path should show `/health` so judges can see the actual
  configured model IDs, quantization settings, runtime device, and triage
  provider.
- Production deployments should complete a legal review before using the
  non-Apache NVIDIA/Gemma-licensed models in a commercial setting.
