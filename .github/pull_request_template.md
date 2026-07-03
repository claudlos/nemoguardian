## Summary

<!-- One-paragraph description of the change. -->

## What changed

<!-- Bullet list of the user-visible changes. Link to issues with `Fixes #123`. -->

-

## How was it tested?

<!--
  - [ ] make verify passes locally
  - [ ] New tests added in tests/ (if behavior change)
  - [ ] docs/ updated (if user-facing change)
-->

- [ ] `make verify` (ruff + pytest) passes locally
- [ ] New/updated tests added (if behavior change)
- [ ] Docs updated under `docs/` (if user-facing)
- [ ] Issue linked: `Fixes #…` or explains why not

## Areas touched

- [ ] `nemoguardian/cascade.py` / `aggregator.py`
- [ ] `nemoguardian/models/`
- [ ] `nemoguardian/policy/` / `presets`
- [ ] `nemoguardian/billing/`
- [ ] `nemoguardian/providers/`
- [ ] `nemoguardian/adapters/`
- [ ] `nemoguardian/server.py` / routes
- [ ] `demo/` / `scripts/`
- [ ] `docs/`
- [ ] CI / `.github/`
- [ ] No production code changed (docs, tests, tooling only)

## Security

- [ ] No new external input is reflected without sanitization
- [ ] No new logging of raw API keys, full message bodies, or PII
- [ ] No new outbound network calls without an env-gated opt-in
- [ ] If this touches billing/webhooks/auth, I requested review from `@claudlos`

## Screenshots / logs

<!-- Optional: paste terminal output, evidence JSON, or a short GIF. -->
