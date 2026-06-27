# Pre-Submit Checklist

Use this as the final gate before sending `nemoguardian` to the hackathon form.

## 1. Repository

- Public repo opens: https://github.com/claudlos/nemoguardian
- Default branch is `main`.
- Local tree is clean:

```bash
git status -sb
```

- Latest commit is pushed:

```bash
git log --oneline --decorate -1
git status -sb
```

- Public CI is green for the latest commit:

```bash
gh run list --workflow ci.yml --limit 1
```

## 2. Local Verification

Run these from the repo root:

```bash
make verify
make docker-build
```

Expected:

- Ruff passes.
- Pytest passes.
- Docker image builds as `nemoguardian/self-hosted:latest`.

## 3. License And Model Terms

Confirm these files are present in the repo and Docker image:

- `LICENSE`
- `NOTICE`
- `docs/THIRD_PARTY_MODELS.md`
- `docs/MODEL_CATALOG.md`

Run:

```bash
docker run --rm --entrypoint sh nemoguardian/self-hosted:latest \
  -c 'test -s /opt/nemoguardian/LICENSE && test -s /opt/nemoguardian/NOTICE && test -s /opt/nemoguardian/docs/THIRD_PARTY_MODELS.md && test -s /opt/nemoguardian/docs/MODEL_CATALOG.md'
```

## 4. GPU Demo Host

Follow `docs/VASTAI_DEMO_RUNBOOK.md`.

Required proof before recording:

- `nvidia-smi` shows the target GPU.
- `GET /health` reports `runtime_device` as CUDA/GPU.
- `GET /health` shows the intended Qwen3Guard and Nemotron model IDs.
- `triage_configured` is `true` for the deep-mode recording.
- `scripts/real_model_smoke.py` preflight reports a CUDA device with enough VRAM.
- Both real-model smoke commands pass:

```bash
docker compose exec nemoguardian python scripts/real_model_smoke.py
docker compose exec nemoguardian python scripts/real_model_smoke.py --deep
```

If running the service from an activated host virtualenv instead of Docker,
`make smoke` and `make smoke-deep` are equivalent.

- The running service passes the host check and writes evidence:

```bash
make demo-check DEMO_CHECK_FLAGS="--wait-seconds 120 --require-gpu --require-triage --moderate --deep --output demo-evidence.json"
```

## 5. Recording

Use `docs/VIDEO_SCRIPT.md`.

Before recording:

- Hide terminal environment variables and API keys.
- Keep `/health` visible long enough to prove CUDA/model configuration.
- Keep the `/providers/offers` table or demo GPU table visible long enough to prove the commodity GPU cost story.
- Do not record a final take until both real-model smoke commands pass on the GPU host.

## 6. Submission Links

Submit these:

- Repository: https://github.com/claudlos/nemoguardian
- Model/license manifest: `docs/THIRD_PARTY_MODELS.md`
- Form draft: `docs/SUBMISSION_FORM.md`
- Demo video: final hosted recording URL
- Writeup: `SUBMISSION.md`, or a public post adapted from it
