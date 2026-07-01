# Changelog

All notable changes to **nemoguardian** are recorded here. Versions follow
[Semantic Versioning](https://semver.org/). The first public release is **1.0.0**
(the hackathon submission).

## [1.0.0] — 2026-06-30

First public release, submitted to the **Hermes Agent Accelerated Business
Hackathon** (NVIDIA × Stripe × Nous Research).

### Highlights

- **Multi-model moderation cascade** — Qwen3Guard-Stream 0.6B (fast) +
  Qwen3Guard-Gen 4B + NVIDIA Nemotron-Content-Safety-Reasoning 4B
  (standard) + optional Nemotron 3 Ultra 550B triage (deep). Free-tier
  via NVIDIA NIM or OpenRouter for the 550B step.
- **Self-hostable FastAPI service** with CORS allowlist, per-key/IP rate
  limiting, request body-size cap, signed-Slack-secret requirement for
  HTTP mode.
- **Platform adapters** — Discord (full bot, slash commands, mod log,
  audit), Twitch (enforcing chat moderator with persisted escalation
  strikes), Slack, Telegram, Matrix, Reddit adapter skeletons (offline
  smoke-verified, ready for live tokens), Webhook (verdict-only forward).
- **Agent self-guard** — `nemoguardian guard` CLI verb (exit-code gate),
  drop-in Hermes Agent skill (`integrations/hermes-skill/`), and an MCP
  server (`nemoguardian mcp`) for any MCP-compatible client.
- **Stripe-style billing** — Free / Pro / Scale / Self-hosted plans,
  metered usage, webhook handler, customer portal, API-key database,
  self-hosted provisioning endpoints.
- **Multi-cloud GPU provider registry** — Vast.ai (live API client) plus
  8 other providers with honest static catalog offers; the same Docker
  image runs on RTX 3090, A100, H100, Hetzner, on-prem.
- **Operator surface** — `make verify`, `make platform-smoke`,
  `make demo-check`, `make framework-smoke`, `make pre-submit-local`,
  `make final-submission-check`. CI runs lint + 764-test pytest suite
  on every push.

### Notes

- This is a **self-hostable template**, not a managed SaaS. You run it
  on your own hardware and own the audit data.
- Only Discord is a full bot today; the other platforms are adapter
  skeletons with offline-tested parsers and live-token smoke as the next
  validation tier. See [`docs/PLATFORM_COVERAGE.md`](docs/PLATFORM_COVERAGE.md).
- Model weights and hosted model APIs retain their own licenses and
  terms; see [`docs/THIRD_PARTY_MODELS.md`](docs/THIRD_PARTY_MODELS.md).
- Pre-submission evidence bundle:
  `make pre-submit-local` → `build/local-pre-submit-summary.json`
- Final-submission evidence bundle (after recording the demo):
  `make final-submission-check -- --video-url <URL>`

[1.0.0]: https://github.com/claudlos/nemoguardian/releases/tag/v1.0.0
