# Security Policy

> **Status:** two internal security passes (NMG-001 → NMG-013). All
> Critical/High items are fixed except NMG-006 (adversarial eval, ML-layer),
> whose measurement harness is prepared under `eval/`. **Do not run nemoguardian
> on the public internet without working through the hardening checklist.**

## Supported versions

| Version | Supported |
| ------- | --------- |
| `main`  | ✅ Active |
| `v1.1.0` | ✅ Latest release |
| `v1.0.0` | ⚠️ Superseded (contains the NMG-009 provisioning key leak; upgrade to v1.1.0) |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security bugs.**

Use **GitHub private vulnerability reporting**: [Report a vulnerability](https://github.com/claudlos/nemoguardian/security/advisories/new) on this repository. Reports go directly and privately to the maintainer.

When reporting, please include:

- Reproduction steps (curl, Python snippet, screenshot)
- Environment (commit SHA, Python version, Docker image tag, GPU host if relevant)
- Impact assessment (what data is exposed, what an attacker can do)
- Any suggested fix (optional but appreciated)

## Audit status

Canonical source of truth — if you find a regression, link to the `NMG-NNN` ID.

| # | Finding | Sev | Status |
|---|---------|-----|--------|
| NMG-001 | Webhook trust-on-empty-secret | Crit | ✅ Fixed — fail-closed by default, opt-in demo only |
| NMG-002 | Raw API key in webhook response | Crit | ✅ Fixed — returns `api_key_id` only |
| NMG-003 | Demo endpoint on by default | High | ✅ Fixed — defaults `0`, opt-in |
| NMG-004 | No webhook replay protection | High | ✅ Fixed — 300s timestamp tolerance + event-id idempotency (dedup table) |
| NMG-005 | CORS wildcard `["*"]` | High | ✅ Fixed — env allowlist `NEMOGUARDIAN_CORS_ORIGINS`, localhost-only default; `*` is an explicit opt-in |
| NMG-006 | Prompt injection into guards | High | ❌ Open (ML-layer; run harness ready in `eval/`) |
| NMG-007 | SQLite hygiene | Med | ✅ Fixed — WAL + busy_timeout; non-Docker DB defaults to `~/.nemoguardian` (0700/0600), container DB at `/var/lib/nemoguardian` |
| NMG-008 | SSH key interpolated into shell | Med | ✅ Fixed — SSH-key validation at all provisioning endpoints |
| NMG-009 | Raw key forwarded to instances | Med | ✅ Fixed — provisioning mints a moderation-only `instance`-scoped key (tracked on the job, revoked on failed provisions); the caller's master key never leaves the control plane |
| NMG-010 | Verbose errors to clients | Low | ✅ Fixed — full trace logged server-side, generic detail returned |
| NMG-011 | No rate limiting | Med | ✅ Fixed — ASGI fixed-window limiter keyed on validated-API-key hash / trusted client IP, enforced before auth, with a request body-size cap (`NEMOGUARDIAN_RATE_LIMIT`, `NEMOGUARDIAN_MAX_BODY_BYTES`). Fail-open when unset (logs a warning); set a positive limit in production |
| NMG-012 | Container runs as root | Low | ✅ Fixed — runs as non-root `app` user (uid 1001) |
| NMG-013 | No secret-scan/SAST in CI | Info | ✅ Fixed — CI runs gitleaks (full history), CodeQL, and pip-audit, plus offline doc-link/secret/build guards |
| — | Triage model slug 220b-a12b → 550b-a55b | — | ✅ Fixed |

## Hardening checklist before exposing to the internet

Items checked below are enforced in code on `main`; unchecked items are operator steps.

- [x] Demo endpoint defaults to off (`NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT=0`) — NMG-003
- [ ] Replace `NEMOGUARDIAN_API_KEY` with a real `nmg_...` value (placeholder values are rejected)
- [ ] Set `NEMOGUARDIAN_CORS_ORIGINS` to your actual frontend origins — NMG-005 (localhost-only by default)
- [ ] **Set a positive `NEMOGUARDIAN_RATE_LIMIT`** — the limiter is fail-open (logs a warning) until you do — NMG-011
- [ ] Run behind an authenticated reverse proxy (Caddy, nginx, Cloudflare Access); set `NEMOGUARDIAN_TRUST_PROXY` to your proxy hop count
- [ ] Enable structured logging and ship to a SIEM
- [x] Default model weights pinned by full commit SHA (transformers `revision`); set `NEMOGUARDIAN_*_REVISION` when overriding models
- [x] Cascade fails closed on total model failure (`NEMOGUARDIAN_FAIL_MODE=closed`, the default)
- [ ] Verify Stripe webhook signature secret is set (no empty `STRIPE_WEBHOOK_SECRET`) — verification is fail-closed when unset
- [x] Container runs as a non-root user — NMG-012
- [x] CI scans for secrets, SAST, and dependency CVEs — NMG-013
- [ ] Mirror `.env` secrets into a secret manager (Doppler, Vault, AWS SM)

## Out-of-scope

- Adversarial robustness of the moderation models themselves (tracked as NMG-006; harness in `eval/`)
- Denial-of-service protection at the model layer (use the proxy)
- Multi-tenant key isolation (single-tenant self-hosted is the supported deployment)
