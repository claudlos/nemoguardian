# Security Policy

> **Status:** hackathon submission. The codebase has had two internal security passes (NMG-001 → NMG-013 in the audit table); all Critical/High items are fixed except NMG-006 (adversarial eval, ML-layer). **Do not run nemoguardian on the public internet without working through the hardening checklist.**

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| `main`  | ✅ Active          |
| tags    | ❌ Pre-release — no tagged releases yet |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security bugs.**

Use **GitHub private vulnerability reporting**: [Report a vulnerability](https://github.com/claudlos/nemoguardian/security/advisories/new) on this repository. Reports go directly and privately to the maintainer.

Alternatively, ping `@claudlos` on the hackathon Discord. Expect an acknowledgement within 48 hours.

When reporting, please include:

- Reproduction steps (curl, Python snippet, screenshot)
- Environment (commit SHA, Python version, Docker image tag, GPU host if relevant)
- Impact assessment (what data is exposed, what an attacker can do)
- Any suggested fix (optional but appreciated)

## Audit status

This table is the same one in the submission writeup. It is the canonical source of truth — if you find a regression, link to the `NMG-NNN` ID.

| # | Finding | Sev | Status |
|---|---------|-----|--------|
| NMG-001 | Webhook trust-on-empty-secret | Crit | ✅ Fixed — fail-closed by default, opt-in demo only |
| NMG-002 | Raw API key in webhook response | Crit | ✅ Fixed — returns `api_key_id` only |
| NMG-003 | Demo endpoint on by default | High | ✅ Fixed — defaults `0`, opt-in |
| NMG-004 | No webhook replay protection | High | ✅ Fixed — 300s timestamp tolerance + event-id idempotency (dedup table) |
| NMG-005 | CORS wildcard `["*"]` | High | ✅ Fixed — env allowlist `NEMOGUARDIAN_CORS_ORIGINS`, defaults to deny-all cross-origin |
| NMG-006 | Prompt injection into guards | High | ❌ Open (ML-layer; needs adversarial eval) |
| NMG-007 | SQLite hygiene | Med | ✅ Fixed — WAL mode + busy_timeout; container DB moved out of `/tmp` to `/var/lib/nemoguardian` |
| NMG-008 | SSH key interpolated into shell | Med | ✅ Fixed — `validate_ssh_public_key()` at all provisioning endpoints; heredoc rendering |
| NMG-009 | Raw key forwarded to instances | Med | ✅ Fixed — provisioning mints a moderation-only `instance`-scoped key (revoked on failed provisions, tracked on the job); the caller's master key never leaves the control plane |
| NMG-010 | Verbose errors to clients | Low | ✅ Fixed — `_client_error()` logs full trace server-side, returns generic detail + correlation id |
| NMG-011 | No rate limiting | Med | ✅ Fixed — in-process token bucket per API key / client IP (`NEMOGUARDIAN_RATE_LIMIT_RPM`, default 60; 429 + Retry-After). Single-process scope; add proxy-layer limiting for multi-worker deploys |
| NMG-012 | Container runs as root | Low | ✅ Fixed — runs as non-root `app` user (uid 1001) |
| NMG-013 | No secret-scan/SAST in CI | Info | ✅ Fixed — CI runs gitleaks (full history), CodeQL, and pip-audit |
| — | Triage model slug 220b-a12b → 550b-a55b | — | ✅ Fixed |

## Hardening checklist before exposing to the internet

Items checked below are enforced in code on `main`; unchecked items are operator steps.

- [x] Demo endpoint defaults to off (`NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT=0`) — NMG-003
- [ ] Replace `NEMOGUARDIAN_API_KEY` with a real `nmg_...` value (placeholder values are rejected)
- [ ] Set `NEMOGUARDIAN_CORS_ORIGINS` to your actual frontend origins — NMG-005 (defaults to deny-all cross-origin; same-origin `/demo` still works)
- [ ] Run behind an authenticated reverse proxy (Caddy, nginx, Cloudflare Access)
- [ ] Enable structured logging and ship to a SIEM
- [x] App-layer rate limiting (token bucket, `NEMOGUARDIAN_RATE_LIMIT_RPM`) — NMG-011. Still add proxy-layer limiting when running multiple workers/nodes
- [x] Default model weights pinned by full commit SHA (transformers `revision`); set `NEMOGUARDIAN_*_REVISION` when overriding models
- [ ] Verify Stripe webhook signature secret is set (no empty `STRIPE_WEBHOOK_SECRET`) — verification is fail-closed when unset
- [x] Container runs as a non-root user — NMG-012
- [x] CI scans for secrets, SAST, and dependency CVEs — NMG-013
- [ ] Mirror `.env` secrets into a secret manager (Doppler, Vault, AWS SM)

## Out-of-scope for the hackathon

- Adversarial robustness of the moderation models themselves (covered by upstream model safety teams)
- Denial-of-service protection at the model layer (use the proxy)
- Multi-tenant key isolation (single-tenant self-hosted is the supported deployment)

## Acknowledgements

We thank the security review volunteers from the hackathon for the initial audit pass and the community for the responsible disclosure that helped us harden the webhook path.
