# Judge Guide

This guide is for quickly evaluating `nemoguardian` without guessing which
parts require a GPU.

## 1. Fast Review Without GPU

Use this path to inspect the code, API surface, billing flow, provider catalog,
and tests. It does not load model weights.

```bash
git clone https://github.com/claudlos/nemoguardian
cd nemoguardian
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
make verify
```

Expected:

- Ruff passes.
- Pytest passes.
- GitHub Actions badge is green for `main`.

Useful files:

- `SUBMISSION.md` - project pitch and scoring narrative.
- `docs/SUBMISSION_FORM.md` - copy/paste hackathon form answers.
- `docs/THIRD_PARTY_MODELS.md` - model/license manifest.
- `docs/MODEL_CATALOG.md` - model selection notes.
- `nemoguardian/server.py` - FastAPI routes.
- `nemoguardian/cascade.py` - model orchestration.
- `nemoguardian/billing/` - Stripe-style plans, auth, usage, webhooks.
- `nemoguardian/providers/` - GPU provider registry and Vast.ai client.

To inspect the API routes without starting a server:

```bash
python - <<'PY'
from nemoguardian.server import app

for route in app.routes:
    methods = ",".join(sorted(getattr(route, "methods", []) or []))
    if getattr(route, "path", "").startswith(("/v1", "/demo", "/billing", "/providers", "/health")):
        print(f"{methods:12} {route.path}")
PY
```

## 2. Lightweight Demo Host Check

If the service is running, this checks health, the demo page, provider offers,
and production auth boundaries without loading model weights:

```bash
make demo-check DEMO_BASE_URL=http://localhost:8000 \
  DEMO_CHECK_FLAGS="--wait-seconds 30 --output demo-evidence-light.json"
```

Expected:

- `/health` returns JSON.
- `/demo` serves the moderation console.
- `/providers/offers` includes a fitting low-cost GPU.
- `/v1/moderate/stream` rejects unauthenticated requests.

## 3. Real GPU Verification

Use this path for the final scored demo. It requires a CUDA host with enough
VRAM and either `NVIDIA_API_KEY` or `OPENROUTER_API_KEY` for deep-mode triage.

```bash
cp .env.example .env
# Edit .env with a real NEMOGUARDIAN_API_KEY and NVIDIA_API_KEY or OPENROUTER_API_KEY.
docker compose up --build
```

In another shell on the GPU host:

```bash
python scripts/real_model_smoke.py
python scripts/real_model_smoke.py --deep
make demo-check DEMO_CHECK_FLAGS="--wait-seconds 120 --require-gpu --require-triage --moderate --deep --output demo-evidence.json"
```

Expected:

- `scripts/real_model_smoke.py` reports a CUDA device with enough VRAM.
- Standard smoke returns an unsafe verdict for the PII scam example.
- Deep smoke calls configured triage and returns without model errors.
- `demo-evidence.json` has `"passed": true`.

## 4. What Is Real vs Demo-Mode

Real:

- FastAPI service and OpenAPI docs.
- `/v1/moderate` production API with auth, tier gates, and usage accounting.
- `/v1/moderate/stream` production stream API with auth and usage accounting.
- `/demo/moderate` recording endpoint backed by the same cascade.
- Qwen3Guard, Nemotron-CSR, and Nemotron triage wrappers.
- NemoClaw-style policy presets.
- Stripe-style checkout, webhook, plans, usage, and API-key database.
- Vast.ai live client plus multi-provider offer registry.

Demo-mode or static:

- Stripe falls back to demo sessions when Stripe secrets are absent.
- Non-Vast providers return static catalog offers.
- Billing/provisioning can simulate local flows without cloud credentials.
- The final video and `demo-evidence.json` are not valid until real GPU smoke passes.

## 5. Final Artifacts To Submit

- Repository: https://github.com/claudlos/nemoguardian
- CI run: latest successful `CI` workflow on `main`.
- Demo video: recorded only after real GPU smoke passes.
- Evidence file: `demo-evidence.json` from the GPU host.
- Writeup source: `SUBMISSION.md`.
- Form draft: `docs/SUBMISSION_FORM.md`.
