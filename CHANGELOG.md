# Changelog

All notable changes to nemoguardian are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **Rate limiting (NMG-011):** in-process token bucket per API key (client IP on the
  demo route). `NEMOGUARDIAN_RATE_LIMIT_RPM` (default 60, `0` disables); 429 with
  `Retry-After` on exhaustion.
- **Scoped instance tokens (NMG-009):** provisioning now mints a moderation-only
  `instance`-scoped API key for the VM instead of forwarding the caller's master key;
  the key is tied to the provisioning job and revoked if the provision fails.
  Key-management, portal, and provisioning routes reject instance-scoped keys (403).
- **Fail-mode contract:** when *every* model fails (or none ran), the verdict follows
  `NEMOGUARDIAN_FAIL_MODE` — `closed` (default) → `unsafe`, `open` → `safe`. Partial
  failure still degrades gracefully (surviving models aggregate).
- **Pinned model revisions:** default HF checkpoints load with full commit-SHA
  `revision=` pins; override via `NEMOGUARDIAN_QWEN_REVISION` /
  `NEMOGUARDIAN_QWEN_STREAM_REVISION` / `NEMOGUARDIAN_CSR_REVISION`.
- Deep-mode triage API call timeout (`NEMOGUARDIAN_TRIAGE_TIMEOUT_S`, default 30 s).
- Adversarial-eval harness prep for NMG-006 under `eval/` (garak REST config,
  bypass-rate analyzer, GPU-host runbook).
- CI: Python 3.10/3.11/3.12 matrix, Docker build job, coverage gate
  (`--cov-fail-under=82`; measured 87%).
- API reference (`docs/API.md`) and architecture guide (`docs/ARCHITECTURE.md`).

### Changed
- Model-output parsing (Qwen3Guard, Nemotron-CSR) now raises on format drift instead
  of silently defaulting to `safe`; the error verdict is excluded from aggregation.
- `/v1/moderate/stream` bills after tokens are actually delivered (was billed
  up-front) and the terminal chunk reports the worst verdict seen across the stream
  (was hardcoded `safe`).
- Stripe integration updated for current APIs: usage push prefers Billing Meter
  Events (`STRIPE_METER_EVENT_NAME`) with a legacy usage-record fallback; webhook
  subscription-period parsing handles item-level `current_period_*` (API 2025-03+).
- Blocking billing/auth calls (SQLite, Stripe HTTP) moved off the event loop;
  `last_used_at` writes throttled to once per key per 60 s.
- The billing/auth SQLite DB defaults to `~/.nemoguardian/nemoguardian.db`
  (0700/0600 on POSIX) instead of `/tmp` outside Docker.
- Dockerfile installs dependencies from `pyproject.toml` (single source of truth).

### Fixed
- Broken `nemoguardian` console entry point (`[project.scripts]` pointed at a
  nonexistent function).
- `/providers/offers` rebuilt the full 9-provider registry on every request; it now
  reuses the memoized singleton.
- Removed the dead `NEMOGUARDIAN_CASCADE_MODE` env var (documented in three places,
  read nowhere — the per-request `mode` field is the real control).
- README/SUBMISSION provider tables re-synced with the code catalog (Hetzner $2.10).

### Security
- SECURITY.md audit table: NMG-009 and NMG-011 closed; NMG-006 (adversarial eval)
  remains open with the run harness prepared in `eval/`.

## [0.1.0] — 2026-06-27

Hackathon submission baseline.

### Added
- FastAPI moderation service: Qwen3Guard-Gen/Stream + Nemotron-CSR cascade with
  fast/standard/deep modes and API-backed Nemotron 3 Ultra triage adjudication.
- NemoClaw YAML policy gate with Discord/Twitch/generic presets.
- Stripe billing (checkout, webhooks with signature verification + idempotency,
  metered usage, tiers) on SQLite.
- GPU provider catalog + provisioning (live Vast.ai client; 8 stub catalogs),
  cheapest-fit provisioning endpoint.
- Discord/Twitch/webhook adapters, demo console, Docker image (non-root),
  CI (lint, tests, gitleaks, CodeQL, pip-audit).
- Two internal security passes: NMG-001..NMG-013 findings; all Critical/High fixed
  except the ML-layer adversarial eval (NMG-006).
