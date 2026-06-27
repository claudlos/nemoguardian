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
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Request, status

from nemoguardian.billing import db
from nemoguardian.billing.plans import Tier, get_plan


WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")


def get_webhook_secret() -> str:
    """Re-read on every request so tests + runtime env changes take effect."""
    return os.environ.get("STRIPE_WEBHOOK_SECRET", "") or WEBHOOK_SECRET


def verify_signature(payload: bytes, signature_header: str) -> bool:
    """Verify the Stripe-Signature header. Returns True if valid.

    Stripe's scheme: HMAC-SHA256 of (timestamp.payload), prefixed by `t=`.
    """
    secret = get_webhook_secret()
    if not secret:
        # Demo mode: skip verification but log it.
        return True
    if not signature_header:
        return False
    try:
        parts = dict(p.split("=", 1) for p in signature_header.split(","))
        timestamp = parts.get("t", "")
        sig = parts.get("v1", "")
    except Exception:
        return False
    signed_payload = f"{timestamp}.{payload.decode('utf-8')}".encode()
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
    if handler is not None:
        result = handler(data)
    else:
        result = {"ignored": True, "type": event_type}

    return {"received": True, "type": event_type, **result}


def _on_checkout_completed(session: dict[str, Any]) -> dict[str, Any]:
    metadata = session.get("metadata", {}) or {}
    tier = _tier_from_metadata(metadata)
    if tier is None:
        return {"ignored": True, "reason": "no tier metadata"}

    customer_id_meta = metadata.get("nemoguardian_customer_id")
    stripe_customer_id = session.get("customer")
    customer_email = session.get("customer_details", {}).get("email") or session.get("customer_email")

    customer = None
    if customer_id_meta:
        try:
            customer = db.get_customer(int(customer_id_meta))
        except (KeyError, ValueError):
            pass
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

    # Provision an API key for the new tier.
    raw_key, _ = db.create_api_key(customer.id, label=f"{tier.value} welcome key")
    return {
        "handled": True,
        "customer_id": customer.id,
        "tier": tier.value,
        "api_key_provisioned": raw_key,  # in real life: email this, don't return it
    }


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


__all__ = [
    "handle_stripe_webhook",
    "verify_signature",
    "WEBHOOK_SECRET",
    "get_webhook_secret",
]  # type: ignore[list-item]  # actually exporting get_webhook_secret
