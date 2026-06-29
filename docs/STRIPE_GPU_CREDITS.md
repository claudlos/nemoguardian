# Stripe GPU Credits

`nemoguardian` keeps Stripe prominent as the funding layer for self-hosted GPU
moderation.

For test-mode development, Stripe Checkout and webhook testing can run before a
live account is activated. For live processing, publish the GitHub Pages site
from `/docs` and use its public URL as the business website:

```text
https://claudlos.github.io/nemoguardian/
```

The public site includes product/service details, USD pricing, support, payment
security language, terms, privacy, and refund/cancellation pages.

The flow is:

1. A community owner or agent creates a Stripe Checkout session for GPU credits.
2. Stripe sends `checkout.session.completed` to `/billing/webhook`.
3. The webhook credits the customer's local GPU wallet.
4. `/billing/provision/cheapest` or `/billing/provision/vastai` reserves wallet
   credit before renting a GPU.
5. If provider provisioning fails, the reservation is refunded automatically.

Stripe does not directly pay Vast.ai. The operator's Vast.ai account is still
the provider account. Stripe collects funds from the user/customer, then
`nemoguardian` uses that funded balance as the permission gate for renting 3090s
or other GPU boxes.

## Local Secret Setup

Keep Stripe keys outside the repo.

```bash
make stripe-env-setup
source ~/.config/nemoguardian/stripe.env
```

The helper extracts `pk_...` and `sk_...` keys and writes:

```bash
export STRIPE_PUBLISHABLE_KEY="<publishable>"
export STRIPE_SECRET_KEY="<secret>"
export STRIPE_WEBHOOK_SECRET=""
```

After creating a Stripe webhook endpoint, set `STRIPE_WEBHOOK_SECRET` in that
same private env file.

## Create GPU Credit Checkout

Demo/offline mode works without `STRIPE_SECRET_KEY` and immediately records the
credit locally:

```bash
curl -X POST http://localhost:8000/billing/gpu-credits/checkout \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "owner@example.com",
    "amount_cents": 2500,
    "success_url": "http://localhost:8000/demo",
    "cancel_url": "http://localhost:8000/demo"
  }'
```

With `STRIPE_SECRET_KEY` set, this returns a real Stripe Checkout URL. The
payment credits the wallet only after Stripe calls `/billing/webhook`.

## Check Balance

```bash
curl http://localhost:8000/billing/gpu-credits \
  -H "Authorization: Bearer $NEMOGUARDIAN_API_KEY"
```

Response fields:

- `balance_cents` - available GPU credit.
- `events` - recent ledger events such as `stripe_topup`,
  `provision_reserve`, and `provision_refund`.

## Rent A GPU With Funded Credits

Cheapest fitting provider:

```bash
curl -X POST http://localhost:8000/billing/provision/cheapest \
  -H "Authorization: Bearer $NEMOGUARDIAN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "max_price_usd": 0.20,
    "reserve_hours": 3,
    "ssh_public_key": "ssh-ed25519 AAAA...",
    "image": "nemoguardian/self-hosted:latest"
  }'
```

Specific Vast.ai path:

```bash
curl -X POST http://localhost:8000/billing/provision/vastai \
  -H "Authorization: Bearer $NEMOGUARDIAN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "gpu_model": "RTX 3090",
    "max_price_usd": 0.20,
    "reserve_hours": 3,
    "ssh_public_key": "ssh-ed25519 AAAA..."
  }'
```

For a $0.07/hr RTX 3090 and `reserve_hours=3`, the reservation is 21 cents.
The API returns `gpu_credit_reserved_cents` and the remaining
`gpu_credit_balance_cents`.

## Webhook

Point Stripe at:

```text
POST https://<host>/billing/webhook
```

Required event:

```text
checkout.session.completed
```

Existing subscription lifecycle events are still supported for the plan system.
GPU-credit checkout sessions are distinguished with metadata:

```json
{
  "nemoguardian_checkout_kind": "gpu_credit",
  "nemoguardian_customer_id": "123",
  "nemoguardian_gpu_credit_cents": "2500"
}
```

Webhook processing is idempotent for GPU-credit checkout session IDs.
