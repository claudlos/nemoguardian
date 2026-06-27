# Hackathon Submission Form Draft

Use this as copy/paste source for the final hackathon form. Replace only the
video URL and any form-specific fields that are not known yet.

## Project Name

nemoguardian

## Repository

https://github.com/claudlos/nemoguardian

## Demo Video

TBD after `docs/PRE_SUBMIT_CHECKLIST.md` passes on the GPU host.

## Short Description

Multi-model LLM moderation for agents, Discord, Twitch, and webhooks: Nemotron
Content Safety Reasoning plus Qwen3Guard, packaged as a billable FastAPI service
and self-hosted Docker image for commodity GPUs.

## Long Description

nemoguardian is a moderation layer for AI agents and user-generated text
pipelines. It cascades Qwen3Guard-Stream, Qwen3Guard-Gen, NVIDIA
Nemotron-Content-Safety-Reasoning-4B, and optional Nemotron 3 Ultra triage into
one auditable verdict. The API returns the final safe/controversial/unsafe
decision, per-model reasoning, latency, categories, and the policy rule that
fired.

The business wedge is self-hosted safety on commodity GPUs. The same Docker
image runs on RTX 3090, A100, H100, Hetzner, or on-prem hardware; the demo shows
GPU offers starting around $0.07/hr. Stripe-backed plans, metered usage, API
keys, and self-hosted provisioning turn the moderation cascade into a sellable
agent infrastructure product.

## NVIDIA Integration

- `nvidia/Nemotron-Content-Safety-Reasoning-4B` for custom-policy moderation.
- `nvidia/nemotron-3-ultra-220b-a12b` through NVIDIA/OpenRouter-compatible API
  for deep-mode disagreement triage.
- NemoClaw-style YAML policy gate for platform-specific enforcement.
- CUDA Docker image and GPU readiness checks for the real-model recording path.

## Stripe Integration

- Free, Pro, Scale, and Self-hosted plans.
- API-key auth tied to customer tiers.
- Checkout session creation with demo-mode fallback.
- Webhook handler for subscription lifecycle events.
- Metered usage recording for moderation and stream calls.
- Self-hosted provisioning endpoints gated by plan feature.

## What Is Real

- FastAPI service and generated OpenAPI docs.
- Production `/v1/moderate` API with billing auth and usage accounting.
- Protected `/v1/moderate/stream` API for token-level moderation.
- Unauthenticated `/demo/moderate` path for the recorded hackathon demo.
- Real model wrappers for Qwen3Guard, Nemotron-CSR, and Nemotron triage.
- Docker image and compose path for self-hosted GPU deployment.
- Vast.ai live API client plus provider catalog and cost comparison table.

## What Is Simulated Or Demo-Mode

- Non-Vast providers return static catalog entries unless their integrations are
  expanded later.
- Stripe checkout and webhooks run in demo mode when Stripe secrets are absent.
- Billing/provisioning can run without real cloud credentials for local demos.
- The final video should not be recorded until real-model smoke passes on the
  GPU host.

## Demo Instructions

1. Follow `docs/VASTAI_DEMO_RUNBOOK.md`.
2. Run `make verify` and `make docker-build`.
3. Start the GPU host with Docker or compose.
4. Run `python scripts/real_model_smoke.py`.
5. Run `python scripts/real_model_smoke.py --deep`.
6. Run:

```bash
make demo-check DEMO_CHECK_FLAGS="--wait-seconds 120 --require-gpu --require-triage --moderate --deep --output demo-evidence.json"
```

7. Record using `docs/VIDEO_SCRIPT.md`.

## Tags

AI safety, moderation, agents, NVIDIA, Nemotron, Qwen3Guard, Stripe, FastAPI,
Docker, self-hosted, GPU, Vast.ai

## Final Links

- Repo: https://github.com/claudlos/nemoguardian
- Model/license manifest: `docs/THIRD_PARTY_MODELS.md`
- Demo runbook: `docs/VASTAI_DEMO_RUNBOOK.md`
- Pre-submit checklist: `docs/PRE_SUBMIT_CHECKLIST.md`
- Video script: `docs/VIDEO_SCRIPT.md`
- Evidence file: `demo-evidence.json` from the GPU host
