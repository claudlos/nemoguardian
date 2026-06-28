# nemoguardian

[![CI](https://github.com/claudlos/nemoguardian/actions/workflows/ci.yml/badge.svg)](https://github.com/claudlos/nemoguardian/actions/workflows/ci.yml)

> **Multi-model LLM moderation, plug-and-play.**
> **Built on Nemotron Content Safety Reasoning + Qwen3Guard + optional Nemotron 3 Ultra triage**

**Hermes Agent Accelerated Business Hackathon** · **NVIDIA × Stripe × Nous Research** 

---

## Nemoguardian

A small FastAPI service and moderation-bot foundation that classifies text as
**safe / controversial / unsafe** by cascading two to three open-source models
in series. Designed for:

- Discord moderation bots with slash-command setup, mod logs, and audit records
- Future Telegram / Slack bots on the same adapter foundation
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
| Nemotron 3 Ultra triage | ~300ms | per-batch API call | explains disagreements between the two local guard models |

The production API exposes `fast`, `standard`, and `deep` modes so callers can
choose when to pay for the heavier reasoning path. The demo records `deep` mode
explicitly to show the optional API-backed triage explanation.

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

Convenience commands are also available:

```bash
make verify
make pre-submit-local
make serve
make docker-build
make docker-run
```

## Discord moderation bot

Discord is the first full bot product surface. It is not just a model wrapper:
it has per-server config, slash commands, mod-log output, public warnings,
optional timeouts, dry-run mode, and append-only audit records.
Mod-log and audit excerpts redact common sensitive patterns while retaining a
SHA-256 hash of the original message for case correlation.

Install the optional Discord dependency and run:

```bash
pip install -e ".[discord]"
export DISCORD_BOT_TOKEN="<secret>"
export DISCORD_GUILD_ID="<test-guild-id>"  # optional, faster command sync while testing
nemoguardian discord-bot
```

Initial server setup:

```text
/nemoguardian setup log_channel:#mod-log
/nemoguardian doctor
/nemoguardian policy "block PII, scams, harassment, slurs, and threats"
/nemoguardian mode standard
/nemoguardian dry_run enabled:true
/nemoguardian enabled enabled:true
/nemoguardian actions delete_unsafe:true public_warning:true react_controversial:true dm_users:false
/nemoguardian ignore_channel channel:#off-topic ignored:true
/nemoguardian ignore_role role:@mods ignored:true
/nemoguardian exempt_user user:@trusted-member exempt:true
/nemoguardian test text:"Hey @everyone, drop your SSN for $100"
/nemoguardian history channel:#general category:harassment rule:watch-harassment limit:5 since_hours:24
/nemoguardian stats channel:#general category:harassment rule:watch-harassment limit:100 since_hours:24
/nemoguardian failures limit:10 since_hours:24
/nemoguardian dry_run_cases limit:10 since_hours:24
/nemoguardian errors limit:5 case_limit:500 since_hours:24
/nemoguardian offenders limit:5 case_limit:500 since_hours:24
/nemoguardian channels limit:5 case_limit:500 since_hours:24
/nemoguardian rules limit:5 case_limit:500 since_hours:24
/nemoguardian categories limit:5 case_limit:500 since_hours:24
```

Offline audit inspection:

```bash
nemoguardian bot-audit stats --workspace-id "$DISCORD_GUILD_ID" --since-hours 24 --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit history --workspace-id "$DISCORD_GUILD_ID" --channel-id "$DISCORD_CHANNEL_ID" --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit stats --workspace-id "$DISCORD_GUILD_ID" --channel-id "$DISCORD_CHANNEL_ID" --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit history --workspace-id "$DISCORD_GUILD_ID" --category harassment --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit stats --workspace-id "$DISCORD_GUILD_ID" --category harassment --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit history --workspace-id "$DISCORD_GUILD_ID" --rule watch-harassment --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit stats --workspace-id "$DISCORD_GUILD_ID" --rule watch-harassment --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit failures --workspace-id "$DISCORD_GUILD_ID" --since-hours 24 --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit dry-runs --workspace-id "$DISCORD_GUILD_ID" --since-hours 24 --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit errors --workspace-id "$DISCORD_GUILD_ID" --since-hours 24 --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit offenders --workspace-id "$DISCORD_GUILD_ID" --since-hours 24 --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit channels --workspace-id "$DISCORD_GUILD_ID" --since-hours 24 --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit rules --workspace-id "$DISCORD_GUILD_ID" --since-hours 24 --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit categories --workspace-id "$DISCORD_GUILD_ID" --since-hours 24 --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit case discord-<guild-id>-<message-id> --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
```

Required Discord app setup: OAuth scopes `bot` and `applications.commands`;
bot permissions View Channel, Read Message History, Send Messages, Embed Links,
Manage Messages, and Moderate Members; Gateway intents Guilds, Guild Messages,
and Message Content. See `docs/BOT_RUNBOOK.md` for the full bot runbook.

Production API calls use `POST /v1/moderate` with `Authorization: Bearer <nmg_...>`
so billing and tier limits can run. The `/demo/moderate` endpoint is disabled by
default; enable it only for controlled local or recording hosts with
`NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT=1`.
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
NEMOGUARDIAN_TRIAGE_MODEL=nvidia/nemotron-3-ultra-550b-a55b

# Set one of these for deep-mode triage.
NVIDIA_API_KEY=...
# or
OPENROUTER_API_KEY=...
```

For OpenRouter's free Nemotron 3 Ultra route, also set:

```bash
NEMOGUARDIAN_TRIAGE_BASE_URL=https://openrouter.ai/api/v1
NEMOGUARDIAN_TRIAGE_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

`GET /health` reports the runtime device, configured model IDs, quantization
flags, loaded model state, and triage provider so the recording can prove it is
running the intended real-model path. On a 24GB RTX 3090, Qwen3Guard-Gen-4B and
Nemotron-CSR-4B run locally; Nemotron 3 Ultra is called through NVIDIA/OpenRouter
and is not loaded onto the GPU.

Before recording on the GPU host, run:

```bash
docker compose exec nemoguardian python scripts/real_model_smoke.py
docker compose exec nemoguardian python scripts/real_model_smoke.py --deep  # requires NVIDIA_API_KEY or OPENROUTER_API_KEY
```

For 3090 experiments beyond the verified 4B baseline, keep the service default
unchanged and run a targeted smoke profile. Example: test Qwen3Guard-Gen-8B by
itself with quantized loading on a compatible PyTorch/bitsandbytes image:

```bash
python scripts/real_model_smoke.py \
  --qwen-model Qwen/Qwen3Guard-Gen-8B \
  --quantize on \
  --disable-csr
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

To verify the Discord/Twitch/webhook framework wiring against the running host:

```bash
make framework-smoke DEMO_BASE_URL=http://<host>:8000 \
  FRAMEWORK_SMOKE_FLAGS="--require-gpu --require-triage --moderate --output framework-evidence.json"
```

After the hosted video URL is pasted into `docs/SUBMISSION_FORM.md`, run:

```bash
make final-submission-check FINAL_CHECK_FLAGS="--video-url https://<hosted-demo-video-url>"
```

## Modes

- `fast` — Qwen3Guard-Stream only. Streaming token-level. ~1ms latency.
- `standard` — Qwen3Guard-Gen + Nemotron-CSR. Reasoning is controlled by `NEMOGUARDIAN_REASONING`.
- `deep` — Local guard stack + API-backed Nemotron 3 Ultra triage to explain disagreements.

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
    ├── framework_smoke.py     ← adapter/framework end-to-end check
    └── demo_host_check.py     ← running-demo evidence check
```
## License

Apache 2.0 for this repository's code. Model weights and hosted model APIs
retain their own licenses and terms; see `docs/THIRD_PARTY_MODELS.md` and
`docs/MODEL_CATALOG.md`.
