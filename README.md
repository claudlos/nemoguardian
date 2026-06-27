# nemoguardian

[![CI](https://github.com/claudlos/nemoguardian/actions/workflows/ci.yml/badge.svg)](https://github.com/claudlos/nemoguardian/actions/workflows/ci.yml)

> **Multi-model LLM moderation, plug-and-play.**
> Built on Nemotron Content Safety Reasoning + Qwen3Guard + optional Nemotron 3 Ultra triage.

**Hackathon:** Hermes Agent Accelerated Business Hackathon · NVIDIA × Stripe × Nous Research · June 30, 2026

---

## What it is

A small FastAPI service that classifies text as **safe / controversial / unsafe** by
cascading two to three open-source models in series. Designed for:

- Discord / Telegram / Slack bots
- Twitch / Kick live-chat moderation (streaming mode)
- Generic webhook moderation for any social platform
- LLM input/output guardrails (filter a prompt before it hits your chat model)

The architecture:

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

The aggregator combines per-model verdicts with weights you control, returns a single
JSON with the final decision, every contributing score, the reasoning traces, and
the policy rule that fired (if any).

## Why a cascade?

| Tier | Cost | Latency | Catches |
|---|---|---|---|
| Qwen3Guard-Stream 0.6B (token-level) | ~1ms/token | streaming | obvious violations, PII, slurs as the LLM types |
| Qwen3Guard-Gen 4B (full doc) | ~50ms | per-message | nuanced violations, multilingual (119 languages) |
| Nemotron-CSR 4B (reasoning on) | ~200ms | per-message | custom policies ("no financial advice"), topic-following, jailbreak attempts |
| Nemotron 3 Ultra triage | ~300ms | per-batch | explains disagreements between the two guard models |

The production API exposes `fast`, `standard`, and `deep` modes so callers can
choose when to pay for the heavier reasoning path. The demo records `deep` mode
explicitly to show the optional triage explanation.

## Quickstart

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

# Start the server. The first moderation call downloads model weights.
nemoguardian serve --port 8000

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

Convenience commands are also available:

```bash
make verify
make serve
make docker-build
make docker-run
```

Production API calls use `POST /v1/moderate` with `Authorization: Bearer <nmg_...>`
so billing and tier limits can run. The `/demo/moderate` endpoint is for the
hackathon recording path and can be disabled with `NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT=0`.
For self-hosted Docker, a non-placeholder `NEMOGUARDIAN_API_KEY` also bootstraps
a local self-hosted customer so `/v1/moderate` works on a fresh instance without
manually seeding SQLite.

## Real-model demo config

The June 30 demo target is a Vast.ai RTX 3090 with 24GB VRAM:

```bash
NEMOGUARDIAN_QUANTIZE=1
NEMOGUARDIAN_QWEN_MODEL=Qwen/Qwen3Guard-Gen-4B
NEMOGUARDIAN_QWEN_STREAM_MODEL=Qwen/Qwen3Guard-Stream-0.6B
NEMOGUARDIAN_CSR_MODEL=nvidia/Nemotron-Content-Safety-Reasoning-4B
NEMOGUARDIAN_TRIAGE_MODEL=nvidia/nemotron-3-ultra-220b-a12b

# Set one of these for deep-mode triage.
NVIDIA_API_KEY=...
# or
OPENROUTER_API_KEY=...
```

`GET /health` reports the runtime device, configured model IDs, quantization
flags, loaded model state, and triage provider so the recording can prove it is
running the intended real-model path.

Before recording on the GPU host, run:

```bash
python scripts/real_model_smoke.py
python scripts/real_model_smoke.py --deep  # requires NVIDIA_API_KEY or OPENROUTER_API_KEY
```

See `docs/VASTAI_DEMO_RUNBOOK.md` for the full GPU host checklist.
Use `docs/PRE_SUBMIT_CHECKLIST.md` as the final submission gate and
`docs/VIDEO_SCRIPT.md` for the 60-90 second recording. `docs/SUBMISSION_FORM.md`
has copy/paste answers for the final hackathon form. `docs/JUDGE_GUIDE.md`
is the fastest path for reviewers to verify the repo without a GPU.

After the GPU service is running, capture a submission evidence file with:

```bash
make demo-check DEMO_CHECK_FLAGS="--wait-seconds 120 --require-gpu --require-triage --moderate --deep --output demo-evidence.json"
```

## Modes

- `fast` — Qwen3Guard-Stream only. Streaming token-level. ~1ms latency.
- `standard` — Qwen3Guard-Gen + Nemotron-CSR. Reasoning is controlled by `NEMOGUARDIAN_REASONING`.
- `deep` — All three + Nemotron 3 Ultra triage to explain disagreements. ~500ms per message.

## What's real vs simulated for the demo

**Real, shipping:**
- The FastAPI service and cascade orchestration
- The `/demo` moderation console backed by the real cascade
- Real Qwen3Guard-Gen 4B inference on CPU/GPU
- Real Nemotron-CSR 4B with custom-policy mode (Reasoning On)
- Nemotron 3 Ultra triage via NVIDIA/OpenRouter OpenAI-compatible API
- NemoClaw policy gate (yaml → decision)
- Discord bot adapter with token-level streaming
- Per-platform policy presets
- Stripe subscriptions + metered billing + webhook
- 9-cloud provider registry (Vast.ai live, others stubbed with real catalog data)
- Docker image + docker-compose for self-hosted

**Simulated / cached for the demo video:**
- Multi-language demo runs (English shown live; cached snapshots for the other 118)
- The "live Argus dashboard" overlay in the demo video (real logs, mocked for some panels)

## Providers (self-hosted tier)

nemoguardian runs on **commodity GPUs starting at $0.07/hr**. The same Docker image
works on every option. See `GET /providers/offers` for live pricing.

| GPU | VRAM | $/hr | Provider | Notes |
|---|---|---|---|---|
| **RTX 3090** | 24GB | **$0.07** | Vast.ai / Salad | Cheapest viable host. FP16 cascade. |
| **L4 / RTX 3090** | 24GB | $0.12–0.14 | RunPod | |
| **A100 40GB** | 40GB | $0.14 | Vast.ai | Mature vLLM, production-grade. |
| **RTX 4090** | 24GB | $0.18 | Vast.ai | |
| **H100 80GB** | 80GB | $0.19 | Vast.ai | Flagship. |
| **RTX 5090** | 32GB | $0.21 | Vast.ai | Best $/perf sweet spot. |
| **L40S** | 48GB | $0.50 | RunPod | |
| **Hetzner GEX131** | 96GB | $1.92 | Hetzner (DE) | Dedicated. EU GDPR. |
| **DO H100** | 80GB | $3.39 | DigitalOcean | Enterprise. |

For the demo's hero line: **"the same Nemotron cascade runs on a $0.07/hr
RTX 3090 as on a $3.39/hr H100 — pick your poison."**


## Repo layout

```
nemoguardian/
├── README.md                  ← you are here
├── pyproject.toml
├── nemoguardian/
│   ├── __init__.py
│   ├── server.py              ← FastAPI app
│   ├── cascade.py             ← orchestration: guard stack → optional triage → aggregator
│   ├── aggregator.py          ← weighted multi-model verdict combiner
│   ├── models/                ← lazy Hugging Face / OpenAI-compatible wrappers
│   │   ├── qwen3_guard.py     ← Qwen3Guard-Gen + Stream wrappers
│   │   ├── nemotron_csr.py    ← Nemotron Content Safety Reasoning 4B
│   │   └── nemotron_triage.py ← Nemotron 3 Ultra deep triage
│   ├── policy/
│   │   ├── nemoclaw.py        ← YAML policy → verdict gate
│   │   └── presets.py         ← discord.yaml, twitch.yaml, generic.yaml
│   └── adapters/
│       ├── discord.py
│       ├── twitch.py
│       └── webhook.py
├── tests/
├── docs/
│   └── MODEL_CATALOG.md       ← full model survey (see docs/)
├── demo/
│   ├── index.html             ← browser moderation console
│   └── console_demo.py        ← terminal demo
└── scripts/
    ├── real_model_smoke.py    ← GPU-host smoke check
    └── demo_host_check.py     ← running-demo evidence check
```

## The submission video plan (60–90 seconds)

See `docs/VIDEO_SCRIPT.md` for the exact shot list and narration.

1. Live demo: type a message into `/demo`
2. Show the verdict fly out within ~200ms with the per-model breakdown
3. Rotate the policy from "default" to "no financial advice" — same input, different verdict
4. Show the audit log: every model's verdict + reasoning + latency
5. Close on the live Nemotron 3 Ultra triage explaining a multi-model disagreement

## License

Apache 2.0 for this repository's code. Model weights and hosted model APIs
retain their own licenses and terms; see `docs/THIRD_PARTY_MODELS.md` and
`docs/MODEL_CATALOG.md`.
