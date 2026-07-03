# nemoguardian API reference

Hand-curated from the generated OpenAPI schema (`GET /openapi.json` on a running
server, or interactive docs at `/docs`). Base URL below assumes a local deploy on
port 8000.

**Authentication:** endpoints marked 🔑 need `Authorization: Bearer nmg_<key>`.
Endpoints marked 🔑*full* additionally reject `instance`-scoped keys (the
moderation-only credentials minted for provisioned VMs) with 403.

**Rate limiting:** an ASGI fixed-window limiter guards `/v1/moderate*`,
`/demo/moderate`, and `/billing/keys`, keyed on the validated-API-key hash (or
trusted client IP), enforced before auth, with a request body-size cap.
Configure with `NEMOGUARDIAN_RATE_LIMIT` (requests per
`NEMOGUARDIAN_RATE_WINDOW_SECONDS`, default 60s) — fail-open when unset (logs a
warning). Exhaustion returns `429` with a `Retry-After` header.

**Note:** this reference covers the core moderation + billing API. v1.0.0 also
ships platform adapters (Discord/Twitch/Slack/Telegram/Matrix/Reddit), a review
queue, GPU-credit endpoints, and an MCP server — see the README and
`GET /openapi.json` for the full surface.

---

## Moderation

### `POST /v1/moderate` 🔑

Run the cascade on one text.

Request body (`ModerateRequest`):

```json
{
  "text": "the content to moderate",
  "mode": "standard",          // "fast" | "standard" | "deep"
  "policy": "no PII",          // optional free-form policy for the models
  "response": null             // optional assistant response for pair moderation
}
```

Query params: `policy_preset` (`discord` | `twitch` | `generic`) **or**
`policy_yaml` (inline NemoClaw YAML).

Response (`ModerateResponse`): `verdict` (`safe` | `controversial` | `unsafe`),
`score` (0..1), `reasons[]`, `categories[]`, `matched_policy_rule`,
`model_verdicts{}` (per-model verdict/score/latency/error), `total_latency_ms`,
`request_id`, `usage_info`.

Modes: `fast` = Qwen3Guard-Stream document summary only; `standard` = Qwen3Guard-Gen
+ Nemotron-CSR; `deep` = standard + Nemotron 3 Ultra triage adjudication (needs
`NVIDIA_API_KEY` or `OPENROUTER_API_KEY`).

Errors: `401` bad/missing key · `402` mode not in tier · `429` rate limit or
monthly allowance · `500` sanitized `{error, correlation_id}`.

### `POST /v1/moderate/stream` 🔑

NDJSON stream of per-token verdicts from Qwen3Guard-Stream. Each chunk:
`token_index`, `partial_text`, `verdict_so_far`, `score_so_far`, `is_terminal`,
`latency_ms`. The terminal chunk carries the **worst** verdict seen across the
stream. Billing occurs only if tokens were actually delivered.

### `POST /demo/moderate`

Same cascade, no auth/billing — **disabled by default**; enable with
`NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT=1` on controlled hosts only. Rate limited by
client IP. `GET /demo` serves the moderation console UI.

### `GET /health`

Model load state, GPU availability/name, runtime model config, triage
configuration. No auth.

---

## Billing

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /billing/plans` | — | All tiers with prices, allowances, features. |
| `POST /billing/checkout` | — | Stripe Checkout session for a tier (demo fallback without `STRIPE_SECRET_KEY`). |
| `POST /billing/webhook` | Stripe signature | Subscription lifecycle receiver (idempotent on event id; fail-closed without `STRIPE_WEBHOOK_SECRET`). |
| `POST /billing/portal` | 🔑*full* | Stripe customer-portal session. |
| `POST /billing/keys` | 🔑*full* | Mint a new API key (raw key shown once; hash-only at rest). |
| `GET /billing/usage` | 🔑 | Current-period usage + overage projection. |

## Provisioning & providers

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /providers` | — | Catalog metadata for all 9 GPU providers. |
| `GET /providers/offers` | — | Cross-provider offers, price-sorted; `only_fits=true` (default) filters to ≥ cascade VRAM. Filters: `gpu_model`, `max_price_usd`. |
| `POST /billing/provision` | 🔑*full* | Simulated self-hosted provisioning job (self_hosted tier). |
| `POST /billing/provision/cheapest` | 🔑*full* | Provision the cheapest fitting offer across providers. |
| `POST /billing/provision/{provider}` | 🔑*full* | Provision on a specific provider. |
| `GET /billing/jobs/{job_id}` | 🔑 | Poll a provisioning job (status, endpoint, ssh). |

Provisioned VMs receive a freshly-minted **instance-scoped** API key in their env —
never the caller's master key (NMG-009). Only the Vast.ai client is a live API
integration; the other providers serve real cached catalog prices but simulated
provisioning (disclosed in SUBMISSION.md).
