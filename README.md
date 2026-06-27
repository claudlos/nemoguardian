# nemoguardian

<div align="center">

[![CI](https://github.com/claudlos/nemoguardian/actions/workflows/ci.yml/badge.svg)](https://github.com/claudlos/nemoguardian/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/claudlos/nemoguardian/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776ab.svg)](https://www.python.org)
[![Docker](https://img.shields.io/badge/docker-ready-2496ed.svg)](https://github.com/claudlos/nemoguardian/blob/main/docker-compose.yml)
[![Models: NVIDIA Nemotron](https://img.shields.io/badge/models-NVIDIA%20Nemotron-76B900.svg)](https://huggingface.co/nvidia)
[![Models: Qwen3Guard](https://img.shields.io/badge/models-Qwen3Guard-ff6f00.svg)](https://github.com/QwenLM/Qwen3Guard)
[![Stripe](https://img.shields.io/badge/billing-Stripe-635bff.svg)](https://stripe.com)

**Multi-model LLM moderation, plug-and-play.**
Qwen3Guard + NVIDIA Nemotron Content Safety Reasoning + optional Nemotron 3 Ultra triage.
Billable FastAPI service · self-hosted Docker · commodity GPUs from **$0.07/hr**.

[**Website**](https://claudlos.github.io/nemoguardian/) · [**Quickstart**](#quickstart) · [**Docs**](./docs/JUDGE_GUIDE.md) · [**Submission**](./SUBMISSION.md)

</div>

---

## What it is

nemoguardian is a small FastAPI service that classifies text as **safe / controversial / unsafe** by cascading two to three open-source models in series. Designed for:

- 🤖 Discord / Telegram / Slack bots
- 🎮 Twitch / Kick live-chat moderation (streaming mode)
- 🔌 Generic webhook moderation for any social platform
- 🛡️ LLM input/output guardrails (filter a prompt before it hits your chat model)

> **Hackathon:** Hermes Agent Accelerated Business Hackathon · NVIDIA × Stripe × Nous Research · June 30, 2026

---

## The cascade

```
[text or token stream]
        │
        ▼
 ┌────────────────────────────────────┐
 │ mode=fast                          │
 │ Qwen3Guard-Stream                  │
 └────────────────────────────────────┘
 ┌────────────────────────────────────┐
 │ mode=standard                      │
 │ Qwen3Guard-Gen + Nemotron-CSR      │
 └────────────────────────────────────┘
 ┌────────────────────────────────────┐
 │ mode=deep                          │
 │ standard + Nemotron 3 Ultra triage │
 └────────────────────────────────────┘
        ▼
   [aggregator: weighted verdict + per-model reasoning + audit trail]
        │
        ▼
   NemoClaw policy gate → {verdict, reasons, scores, audit_log}
```

The aggregator combines per-model verdicts with weights you control, returns a single JSON with the final decision, every contributing score, the reasoning traces, and the policy rule that fired (if any).

## Why a cascade?

| Tier | Model | Cost | Latency | Catches |
|---|---|---|---|---|
| Streaming | [Qwen3Guard-Stream 0.6B](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B) | ~1ms / token | streaming | obvious violations, PII, slurs as the LLM types |
| Full-doc | [Qwen3Guard-Gen 4B](https://huggingface.co/Qwen/Qwen3Guard-Gen-4B) | ~50 ms | per-message | nuanced violations, multilingual (119 languages) |
| Reasoning | [Nemotron-CSR 4B](https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B) | ~200 ms | per-message | custom policies ("no financial advice"), topic-following, jailbreak attempts |
| Deep triage | [Nemotron 3 Ultra (550B)](https://build.nvidia.com) | ~300 ms | per-batch | explains disagreements between the two local guard models |

The production API exposes `fast`, `standard`, and `deep` modes so callers can choose when to pay for the heavier reasoning path. The demo records `deep` mode explicitly to show the optional API-backed triage explanation.

---

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

# Start the server. The first moderation call downloads model weights.
NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT=1 python -m nemoguardian serve --port 8000

# Open the hackathon demo UI.
open http://localhost:8000/demo

# Or call the demo endpoint directly. This uses the real cascade without billing.
curl -X POST 'http://localhost:8000/demo/moderate?policy_preset=discord' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Hey @everyone, drop your SSN and I will send $100 to whoever DMs me first",
    "policy": "block PII and financial scams",
    "mode": "standard"
  }'
```

Convenience commands:

```bash
make verify                # ruff + pytest
make pre-submit-local      # build image, license check, lightweight host check
make serve                 # uvicorn on :8000
make docker-build          # self-hosted image
make docker-run            # docker run with --gpus all
```

> **Production note:** API calls use `POST /v1/moderate` with `Authorization: Bearer <nmg_...>` so billing and tier limits can run. The `/demo/moderate` endpoint is **disabled by default** — opt in only for controlled local or recording hosts with `NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT=1`. For self-hosted Docker, a non-placeholder `NEMOGUARDIAN_API_KEY` also bootstraps a local customer so `/v1/moderate` works on a fresh instance without manually seeding SQLite.

---

## Self-hosted on commodity GPUs

The same Docker image runs everywhere. Pick the price/perf trade-off:

| GPU | VRAM | $/hr | Provider | Notes |
|---|---|---|---|---|
| **RTX 3090** | 24 GB | **$0.07** | Vast.ai / Salad | Cheapest viable host. FP16 cascade. |
| L4 / RTX 3090 | 24 GB | $0.12 – 0.14 | RunPod | |
| A100 40 GB | 40 GB | $0.14 | Vast.ai | Mature vLLM, production-grade. |
| RTX 4090 | 24 GB | $0.18 | Vast.ai | |
| H100 80 GB | 80 GB | $0.19 | Vast.ai | Flagship. |
| RTX 5090 | 32 GB | $0.21 | Vast.ai | Best $/perf sweet spot. |
| L40S 48 GB | 48 GB | $0.50 | RunPod | |
| Hetzner GEX131 | 96 GB | $1.92 | Hetzner (DE) | Dedicated. EU GDPR. |
| DO H100 80 GB | 80 GB | $3.39 | DigitalOcean | Enterprise. |

> 💡 **Hero line:** the same Nemotron cascade runs on a $0.07/hr RTX 3090 as on a $3.39/hr H100 — pick your poison.

Total VRAM at FP16: **~17.5 GB minimum, 24 GB comfortable, 32 GB headroom.**

---

## What's real vs. simulated

**Real, shipping:**
- FastAPI service and cascade orchestration
- `/demo` moderation console backed by the real cascade
- Real Qwen3Guard-Gen 4B inference on CPU/GPU
- Real Nemotron-CSR 4B with custom-policy mode (Reasoning On)
- API-backed Nemotron 3 Ultra triage via NVIDIA / OpenRouter OpenAI-compatible API
- NemoClaw policy gate (YAML → decision)
- Discord bot adapter with token-level streaming
- Per-platform policy presets
- Stripe subscriptions + customer portal + metered billing + webhook
- 9-cloud provider registry (Vast.ai live, others stubbed with real catalog data)
- Docker image + docker-compose for self-hosted

**Simulated / cached for the demo video:**
- Non-Vast provider prices use static catalog rows unless their provider credentials are wired
- Stripe checkout and portal sessions fall back to demo URLs when Stripe secrets are absent

---

## Repo layout

```
nemoguardian/
├── README.md                  ← you are here
├── SUBMISSION.md              ← hackathon writeup
├── SECURITY.md                ← audit table + disclosure policy
├── CONTRIBUTING.md            ← dev setup, conventions, PR process
├── pyproject.toml
├── Dockerfile                 ← CUDA 12.4 base, predownload toggle
├── docker-compose.yml         ← self-hosted GPU stack
├── Makefile                   ← verify / serve / smoke / demo-check
├── nemoguardian/
│   ├── server.py              ← FastAPI app, routes
│   ├── cascade.py             ← orchestration: guard stack → triage → aggregator
│   ├── aggregator.py          ← weighted multi-model verdict combiner
│   ├── schemas.py             ← Pydantic v2 request/response models
│   ├── cli.py                 ← `nemoguardian serve` entrypoint
│   ├── models/                ← lazy Hugging Face / OpenAI-compatible wrappers
│   │   ├── qwen3_guard.py     ← Qwen3Guard-Gen + Stream wrappers
│   │   ├── nemotron_csr.py    ← Nemotron Content Safety Reasoning 4B
│   │   └── nemotron_triage.py ← Nemotron 3 Ultra deep triage
│   ├── policy/
│   │   ├── nemoclaw.py        ← YAML policy → verdict gate
│   │   └── presets.py         ← discord, twitch, generic presets
│   ├── adapters/              ← Discord / Twitch / generic webhook
│   ├── billing/               ← Stripe-style plans, auth, usage, webhooks
│   └── providers/             ← GPU provider registry (Vast.ai + stubs)
├── tests/                     ← pytest, asyncio_mode=auto
├── docs/                      ← judge guide, runbook, video script, model catalog
├── demo/                      ← browser moderation console (index.html)
└── scripts/                   ← real_model_smoke, framework_smoke, demo_host_check
```

---

## Documentation map

| File | Purpose |
|---|---|
| [`README.md`](./README.md) | This file — quickstart + architecture |
| [`SUBMISSION.md`](./SUBMISSION.md) | Hackathon writeup, NVIDIA + Stripe partnership pitch |
| [`docs/JUDGE_GUIDE.md`](./docs/JUDGE_GUIDE.md) | Fast review path for judges (no GPU needed) |
| [`docs/VASTAI_DEMO_RUNBOOK.md`](./docs/VASTAI_DEMO_RUNBOOK.md) | GPU host checklist — provision → boot → record |
| [`docs/PRE_SUBMIT_CHECKLIST.md`](./docs/PRE_SUBMIT_CHECKLIST.md) | Final submission gate |
| [`docs/VIDEO_SCRIPT.md`](./docs/VIDEO_SCRIPT.md) | 60–90 second shot list + narration |
| [`docs/SUBMISSION_FORM.md`](./docs/SUBMISSION_FORM.md) | Copy/paste hackathon form answers |
| [`docs/MODEL_CATALOG.md`](./docs/MODEL_CATALOG.md) | Full model survey + alternatives |
| [`docs/THIRD_PARTY_MODELS.md`](./docs/THIRD_PARTY_MODELS.md) | License & terms manifest for every runtime model |
| [`SECURITY.md`](./SECURITY.md) | Audit table + vulnerability disclosure |

---

## Modes

- `fast` — Qwen3Guard-Stream only. Streaming token-level. ~1 ms latency.
- `standard` — Qwen3Guard-Gen + Nemotron-CSR. Reasoning is controlled by `NEMOGUARDIAN_REASONING`.
- `deep` — Local guard stack + API-backed Nemotron 3 Ultra triage to explain disagreements.

---

## The submission video plan (60–90 seconds)

See [`docs/VIDEO_SCRIPT.md`](./docs/VIDEO_SCRIPT.md) for the exact shot list and narration.

1. Live demo: type a message into `/demo`
2. Show the verdict fly out within ~200 ms with the per-model breakdown
3. Rotate the policy from "default" to "no financial advice" — same input, different verdict
4. Show the audit log: every model's verdict + reasoning + latency
5. Close on the live Nemotron 3 Ultra triage explaining a multi-model disagreement

---

## Community

- 💬 [GitHub Discussions](https://github.com/claudlos/nemoguardian/discussions) — usage questions, ideas, show-and-tell
- 🐛 [Issue tracker](https://github.com/claudlos/nemoguardian/issues) — bug reports and feature requests
- 🔒 [Security disclosures](./SECURITY.md) — **please do not file as public issues**

---

## Contributing

We welcome PRs! See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for dev setup, conventions, and the review process. Open an issue first for non-trivial changes.

---

## License

Apache 2.0 for this repository's code. Model weights and hosted model APIs retain their own licenses and terms; see [`docs/THIRD_PARTY_MODELS.md`](./docs/THIRD_PARTY_MODELS.md) and [`docs/MODEL_CATALOG.md`](./docs/MODEL_CATALOG.md).

<div align="center">
<sub>Built for the Hermes Agent Accelerated Business Hackathon · NVIDIA × Stripe × Nous Research</sub>
</div>
