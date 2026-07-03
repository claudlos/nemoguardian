# Changelog

All notable changes to **nemoguardian** are recorded here. Versions follow
[Semantic Versioning](https://semver.org/). The first public release is **1.0.0**
(the hackathon submission).

## [Unreleased]

Reconciliation of a parallel hardening line onto the v1.0.0 release.

### Security
- **NMG-009 master-key leak fixed:** provisioning endpoints no longer inject the
  caller's raw master API key into rented GPU VMs. A moderation-only
  `instance`-scoped key is minted per VM, tied to the job, and revoked if the
  provision fails or aborts (incl. zero-cost auto-confirmed offers). New
  `POST /billing/jobs/{id}/teardown` destroys the instance and revokes its key,
  closing the credential lifecycle. `require_full_scope` rejects instance-scoped
  keys (403) on key-management, portal, provisioning, and billing-data reads
  (`/billing/usage`, `/billing/gpu-credits`, `/billing/jobs`).
- Default HF model checkpoints pinned by full commit SHA (`revision=`), overridable
  via `NEMOGUARDIAN_*_REVISION`.
- `starlette>=1.3.1` floor (PYSEC-2026-248/249, CVE-2026-48817/48818).
- CI hardened: Python 3.10/3.11/3.12 matrix, Docker build job, gitleaks
  (full history), CodeQL, and pip-audit — alongside the existing offline guards.
- Added SECURITY.md, issue/PR templates, CODEOWNERS, dependabot, CONTRIBUTING.

### Changed
- **Fail-closed aggregation:** when no model produces a usable verdict (all
  errored, or the request disabled every model), the cascade returns UNSAFE by
  default (`NEMOGUARDIAN_FAIL_MODE=open` restores a safe pass-through). Previously
  all-errored returned CONTROVERSIAL and an empty verdict set returned SAFE.
- Qwen3Guard output parsing raises on format drift instead of silently voting Safe.
- `/v1/moderate/stream` bills after tokens are delivered (was up-front) and the
  terminal chunk reports the worst verdict seen (was hardcoded safe).
- Stripe usage push prefers Billing Meter Events with a legacy fallback; webhook
  subscription-period parsing handles item-level `current_period_*` (API 2025-03+).
- SQLite billing DB defaults to `~/.nemoguardian` (0700/0600) outside Docker, with
  WAL + busy_timeout; blocking billing/auth calls moved off the event loop.
- Rate limiting now also covers `/billing/keys`; coverage gate added to `make test`.

### Fixed
- Broken `nemoguardian` console entry point (`[project.scripts]` target didn't exist).
- `/providers/offers` and provisioning reused the memoized provider registry.
- Removed the dead `NEMOGUARDIAN_CASCADE_MODE` env var.

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
