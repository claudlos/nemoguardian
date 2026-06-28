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
make pre-submit-local
```

Expected:

- Ruff passes.
- Pytest passes.
- Docker image builds as `nemoguardian/self-hosted:latest`.
- The image includes license and model-term documents.
- The built image boots and passes the lightweight demo host check.
- `build/local-pre-submit-summary.json` and
  `build/docker-demo-evidence-light.json` are written.

## 3. License And Model Terms

Confirm these files are present in the repo and Docker image:

- `LICENSE`
- `NOTICE`
- `docs/THIRD_PARTY_MODELS.md`
- `docs/MODEL_CATALOG.md`

`make pre-submit-local` already checks these inside the Docker image. To rerun
only the image assertion:

```bash
docker run --rm --entrypoint sh nemoguardian/self-hosted:latest \
  -c 'test -s /opt/nemoguardian/LICENSE && test -s /opt/nemoguardian/NOTICE && test -s /opt/nemoguardian/docs/THIRD_PARTY_MODELS.md && test -s /opt/nemoguardian/docs/MODEL_CATALOG.md'
```

## 4. Discord Bot Readiness

Before presenting the bot as the product surface:

- `docs/BOT_RUNBOOK.md` exists and lists Discord scopes, permissions, intents, and setup commands.
- `DISCORD_BOT_TOKEN` is only supplied through the environment or a secret manager.
- `NEMOGUARDIAN_BOT_CONFIG_PATH` and `NEMOGUARDIAN_BOT_AUDIT_PATH` point outside the repo for live runs.
- Adapter tests cover unsafe delete, action toggles, controversial flag, bot-message ignore, configured exclusions, mod-log output, redacted evidence, case lookup, history, stats, channel drill-down, failure inspection, repeat-offender review, channel hotspot review, policy-rule hotspot review, Twitch action planning, and webhook auth.
- CLI tests cover offline audit case lookup, history, stats, channel drill-down, failure inspection, repeat-offender JSON output, channel hotspot JSON output, policy-rule hotspot JSON output, and `since_hours` filters.
- For a live Discord recording, install with `bot` + `applications.commands`, enable Message Content intent, and run `/nemoguardian setup` plus `/nemoguardian doctor` in a test server first.

## 5. GPU Demo Host

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

## 6. Recording

Use `docs/VIDEO_SCRIPT.md`.

Before recording:

- Hide terminal environment variables and API keys.
- Keep `/health` visible long enough to prove CUDA/model configuration.
- Keep the `/providers/offers` table or demo GPU table visible long enough to prove the commodity GPU cost story.
- Do not record a final take until both real-model smoke commands pass on the GPU host.

## 7. Submission Links

After `demo-evidence.json` and the hosted video URL exist, paste the final video
URL into `docs/SUBMISSION_FORM.md` and run:

```bash
make final-submission-check FINAL_CHECK_FLAGS="--video-url https://<hosted-demo-video-url>"
```

Submit these:

- Repository: https://github.com/claudlos/nemoguardian
- Model/license manifest: `docs/THIRD_PARTY_MODELS.md`
- Form draft: `docs/SUBMISSION_FORM.md`
- Demo video: final hosted recording URL
- Writeup: `SUBMISSION.md`, or a public post adapted from it
