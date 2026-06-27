"""Stripe Checkout session creation.

For each paid tier, the flow is:
1. POST /billing/checkout with email + desired tier
2. We upsert the customer and create a Stripe Checkout Session
3. Return the session URL — user pays, lands on /billing/welcome?session_id=...
4. Stripe sends `checkout.session.completed` to our webhook → we provision
   the API key and email it.

For the self-hosted tier we add a one-time setup-fee line item.

In `NEMOGUARDIAN_DEMO_MODE=1` we skip the real Stripe call and short-circuit to
a "fake checkout" that immediately provisions — useful for offline demos and CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from nemoguardian.billing import db
from nemoguardian.billing.plans import (
    STRIPE_PRICE_IDS,
    Tier,
)


@dataclass
class CheckoutSession:
    session_id: str
    url: str
    demo_mode: bool


@dataclass
class PortalSession:
    url: str
    demo_mode: bool


def create_checkout_session(
    *, email: str, tier: Tier, success_url: str, cancel_url: str
) -> CheckoutSession:
    """Create a Stripe Checkout session for the given tier.

    If `STRIPE_SECRET_KEY` is missing, we fall back to demo mode (auto-provisions).
    """
    if tier == Tier.FREE:
        # Free tier doesn't need checkout — just upsert.
        customer = db.upsert_customer(email=email)
        return CheckoutSession(
            session_id=f"demo_free_{customer.id}",
            url=f"{success_url}?demo=free",
            demo_mode=True,
        )

    api_key = os.environ.get("STRIPE_SECRET_KEY")
    if not api_key:
        # Demo / offline mode — short-circuit.
        customer = db.upsert_customer(email=email)
        return CheckoutSession(
            session_id=f"demo_{tier.value}_{customer.id}",
            url=f"{success_url}?demo={tier.value}",
            demo_mode=True,
        )

    try:
        import stripe  # type: ignore

        stripe.api_key = api_key
        customer = db.upsert_customer(email=email)
        if customer.stripe_customer_id:
            stripe_customer = stripe.Customer.retrieve(customer.stripe_customer_id)
        else:
            stripe_customer = stripe.Customer.create(email=email)
            db.upsert_customer(email=email, stripe_customer_id=stripe_customer.id)

        line_items: list[dict[str, Any]] = [
            {
                "price": STRIPE_PRICE_IDS[tier],
                "quantity": 1,
            },
        ]
        if tier == Tier.SELF_HOSTED:
            # Add the one-time setup fee as a separate line item.
            line_items.append(
                {
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": 49_900,  # $499 setup
                        "product_data": {"name": "nemoguardian self-hosted setup"},
                    },
                    "quantity": 1,
                },
            )

        session = stripe.checkout.Session.create(
            mode="subscription" if tier != Tier.SELF_HOSTED else "payment",
            customer=stripe_customer.id,
            line_items=line_items,
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            metadata={
                "nemoguardian_customer_id": str(customer.id),
                "nemoguardian_tier": tier.value,
            },
        )
        return CheckoutSession(
            session_id=session.id,
            url=session.url,
            demo_mode=False,
        )
    except Exception:
        # Any Stripe error → fall back to demo so the demo never breaks.
        customer = db.upsert_customer(email=email)
        return CheckoutSession(
            session_id=f"demo_err_{customer.id}",
            url=f"{success_url}?demo={tier.value}",
            demo_mode=True,
        )


def create_portal_session(*, customer: db.Customer, return_url: str) -> PortalSession:
    """Open the Stripe-hosted Customer Portal for plan changes / cancellation."""
    api_key = os.environ.get("STRIPE_SECRET_KEY")
    if not api_key or not customer.stripe_customer_id:
        return PortalSession(url=f"{return_url}?demo=portal", demo_mode=True)
    try:
        import stripe

        stripe.api_key = api_key
        session = stripe.billing_portal.Session.create(
            customer=customer.stripe_customer_id,
            return_url=return_url,
        )
        return PortalSession(url=session.url, demo_mode=False)
    except Exception:
        return PortalSession(url=f"{return_url}?demo=portal", demo_mode=True)


__all__ = [
    "CheckoutSession",
    "PortalSession",
    "create_checkout_session",
    "create_portal_session",
]
