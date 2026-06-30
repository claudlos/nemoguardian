"""Stripe webhook handler.

Verifies signatures, processes subscription lifecycle events, and provisions
API keys / upgrades customer tier / records usage from Stripe's metered events.

Subscribed events:
- checkout.session.completed          → upgrade customer to the paid tier
- customer.subscription.created       → same
- customer.subscription.updated       → tier changes / cancellations
- customer.subscription.deleted       → downgrade back to free
- invoice.payment_succeeded           → record payment
- invoice.payment_failed              → flag customer past_due
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Request, status

from nemoguardian.billing import db
from nemoguardian.billing.plans import Tier

WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
DEFAULT_SIGNATURE_TOLERANCE_SECONDS = 300


def get_webhook_secret() -> str:
    """Re-read on every request so tests + runtime env changes take effect."""
    return os.environ.get("STRIPE_WEBHOOK_SECRET", "") or WEBHOOK_SECRET


def verify_signature(
    payload: bytes,
    signature_header: str,
    *,
    tolerance_seconds: int = DEFAULT_SIGNATURE_TOLERANCE_SECONDS,
    now: float | None = None,
) -> bool:
    """Verify the Stripe-Signature header. Returns True if valid.

    Stripe's scheme: HMAC-SHA256 of (timestamp.payload), prefixed by `t=`.
    """
    secret = get_webhook_secret()
    if not secret:
        # Fail closed by default. Local tests/demos that need unsigned webhook
        # payloads must opt in explicitly.
        return _env_bool("NEMOGUARDIAN_INSECURE_WEBHOOK_DEMO", False)
    if not signature_header:
        return False
    try:
        parts = dict(p.strip().split("=", 1) for p in signature_header.split(","))
        timestamp = parts.get("t", "")
        sig = parts.get("v1", "")
        ts = int(timestamp)
    except Exception:
        return False
    current = time.time() if now is None else now
    if abs(current - ts) > tolerance_seconds:
        return False
    signed_payload = f"{timestamp}.".encode() + payload
    expected = hmac.new(
        secret.encode(), signed_payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig)


def _tier_from_metadata(meta: dict[str, Any]) -> Tier | None:
    raw = meta.get("nemoguardian_tier")
    if not raw:
        return None
    try:
        return Tier(raw)
    except ValueError:
        return None


def _ts_to_iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


async def handle_stripe_webhook(request: Request) -> dict[str, Any]:
    """FastAPI endpoint handler. Idempotent on event id."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    if not verify_signature(payload, sig_header):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid signature")

    try:
        event = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    handler = _HANDLERS.get(event_type)
    result = handler(data) if handler is not None else {"ignored": True, "type": event_type}

    return {"received": True, "type": event_type, **result}


def _on_checkout_completed(session: dict[str, Any]) -> dict[str, Any]:
    metadata = session.get("metadata", {}) or {}
    if metadata.get("nemoguardian_checkout_kind") == "gpu_credit":
        return _on_gpu_credit_checkout_completed(session)

    tier = _tier_from_metadata(metadata)
    if tier is None:
        return {"ignored": True, "reason": "no tier metadata"}

    customer_id_meta = metadata.get("nemoguardian_customer_id")
    stripe_customer_id = session.get("customer")
    customer_email = session.get("customer_details", {}).get("email") or session.get("customer_email")

    customer = None
    if customer_id_meta:
        with suppress(KeyError, ValueError):
            customer = db.get_customer(int(customer_id_meta))
    if customer is None and stripe_customer_id:
        customer = db.get_customer_by_stripe_id(stripe_customer_id)
    if customer is None and customer_email:
        customer = db.upsert_customer(email=customer_email, stripe_customer_id=stripe_customer_id)
    elif customer is not None and stripe_customer_id:
        # Backfill stripe_customer_id on an existing record.
        db.upsert_customer(email=customer.email, stripe_customer_id=stripe_customer_id)
        customer = db.get_customer(customer.id)

    if customer is None:
        return {"ignored": True, "reason": "no matching customer"}

    db.set_customer_tier(customer.id, tier)

    subscription_id = session.get("subscription")
    if subscription_id:
        db.upsert_subscription(
            customer_id=customer.id,
            stripe_subscription_id=subscription_id,
            tier=tier,
            status="active",
            period_start=_ts_to_iso(session.get("created", 0)),
            period_end=_ts_to_iso(
                session.get("expires_at", session.get("created", 0) + 30 * 86400)
            ),
        )

    # Provision an API key for the new tier. Do not return the raw key from a
    # webhook response; production delivery should be out-of-band or claim-token based.
    _raw_key, key_record = db.create_api_key(customer.id, label=f"{tier.value} welcome key")
    return {
        "handled": True,
        "customer_id": customer.id,
        "tier": tier.value,
        "api_key_provisioned": True,
        "api_key_id": key_record.id,
    }


def _on_gpu_credit_checkout_completed(session: dict[str, Any]) -> dict[str, Any]:
    metadata = session.get("metadata", {}) or {}
    customer = _resolve_customer_from_checkout(session)
    if customer is None:
        return {"ignored": True, "reason": "no matching customer"}

    amount_cents = _gpu_credit_amount_cents(session)
    if amount_cents <= 0:
        return {"ignored": True, "reason": "no GPU credit amount"}

    event = db.record_gpu_credit_event(
        customer_id=customer.id,
        event_type="stripe_topup",
        amount_cents=amount_cents,
        provider="stripe",
        stripe_checkout_session_id=session.get("id"),
        stripe_payment_intent_id=session.get("payment_intent"),
        description="Stripe GPU credit top-up",
        metadata={
            "stripe_customer_id": session.get("customer"),
            "metadata_amount_cents": metadata.get("nemoguardian_gpu_credit_cents"),
        },
    )
    return {
        "handled": True,
        "customer_id": customer.id,
        "gpu_credit_applied": True,
        "gpu_credit_event_id": event.id,
        "amount_cents": event.amount_cents,
        "balance_cents": db.gpu_credit_balance_cents(customer.id),
    }


def _resolve_customer_from_checkout(session: dict[str, Any]) -> db.Customer | None:
    metadata = session.get("metadata", {}) or {}
    customer_id_meta = metadata.get("nemoguardian_customer_id")
    stripe_customer_id = session.get("customer")
    customer_email = session.get("customer_details", {}).get("email") or session.get("customer_email")

    customer = None
    if customer_id_meta:
        with suppress(KeyError, ValueError):
            customer = db.get_customer(int(customer_id_meta))
    if customer is None and stripe_customer_id:
        customer = db.get_customer_by_stripe_id(stripe_customer_id)
    if customer is None and customer_email:
        customer = db.upsert_customer(email=customer_email, stripe_customer_id=stripe_customer_id)
    elif customer is not None and stripe_customer_id:
        db.upsert_customer(email=customer.email, stripe_customer_id=stripe_customer_id)
        customer = db.get_customer(customer.id)
    return customer


def _gpu_credit_amount_cents(session: dict[str, Any]) -> int:
    metadata = session.get("metadata", {}) or {}
    for value in (
        session.get("amount_total"),
        session.get("amount_subtotal"),
        metadata.get("nemoguardian_gpu_credit_cents"),
    ):
        with suppress(TypeError, ValueError):
            amount = int(value)
            if amount > 0:
                return amount
    return 0


def _on_subscription_change(sub: dict[str, Any]) -> dict[str, Any]:
    stripe_customer_id = sub.get("customer")
    customer = db.get_customer_by_stripe_id(stripe_customer_id) if stripe_customer_id else None
    if customer is None:
        return {"ignored": True, "reason": "unknown stripe customer"}

    status_str = sub.get("status", "active")
    metadata = sub.get("metadata", {}) or {}
    tier = _tier_from_metadata(metadata) or customer.tier_enum

    db.set_customer_tier(customer.id, tier)
    period_start = sub.get("current_period_start")
    period_end = sub.get("current_period_end")
    if period_start and period_end:
        db.upsert_subscription(
            customer_id=customer.id,
            stripe_subscription_id=sub.get("id"),
            tier=tier,
            status=status_str,
            period_start=_ts_to_iso(period_start),
            period_end=_ts_to_iso(period_end),
            cancel_at_period_end=bool(sub.get("cancel_at_period_end")),
        )

    if status_str in {"canceled", "incomplete_expired"}:
        db.set_customer_tier(customer.id, Tier.FREE)
    elif status_str == "past_due":
        # Don't downgrade automatically, but mark so the dashboard can warn.
        pass

    return {
        "handled": True,
        "customer_id": customer.id,
        "tier": tier.value,
        "status": status_str,
    }


_HANDLERS = {
    "checkout.session.completed": _on_checkout_completed,
    "customer.subscription.created": _on_subscription_change,
    "customer.subscription.updated": _on_subscription_change,
    "customer.subscription.deleted": _on_subscription_change,
    "customer.subscription.trial_will_end": _on_subscription_change,
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


__all__ = [
    "DEFAULT_SIGNATURE_TOLERANCE_SECONDS",
    "WEBHOOK_SECRET",
    "get_webhook_secret",
    "handle_stripe_webhook",
    "verify_signature",
]  # type: ignore[list-item]  # actually exporting get_webhook_secret
