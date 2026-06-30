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
from uuid import uuid4

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


@dataclass
class GpuCreditCheckoutSession:
    session_id: str
    url: str
    demo_mode: bool
    customer_id: int
    amount_cents: int
    balance_cents: int | None = None


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


def create_gpu_credit_checkout_session(
    *,
    email: str,
    amount_cents: int,
    success_url: str,
    cancel_url: str,
) -> GpuCreditCheckoutSession:
    """Create a one-time Stripe Checkout payment that funds GPU credits.

    Demo mode immediately records the credit locally so the provisioning flow
    can be exercised without live Stripe credentials.
    """
    if amount_cents < 500:
        raise ValueError("GPU credit top-up must be at least $5.00")

    api_key = os.environ.get("STRIPE_SECRET_KEY")
    customer = db.upsert_customer(email=email)
    if not api_key:
        event = db.record_gpu_credit_event(
            customer_id=customer.id,
            event_type="stripe_topup",
            amount_cents=amount_cents,
            stripe_checkout_session_id=f"demo_gpu_credit_{customer.id}_{uuid4().hex[:12]}",
            description="Demo GPU credit top-up",
        )
        return GpuCreditCheckoutSession(
            session_id=event.stripe_checkout_session_id or f"demo_gpu_credit_{customer.id}",
            url=f"{success_url}?demo=gpu_credit&amount_cents={amount_cents}",
            demo_mode=True,
            customer_id=customer.id,
            amount_cents=amount_cents,
            balance_cents=db.gpu_credit_balance_cents(customer.id),
        )

    try:
        import stripe  # type: ignore

        stripe.api_key = api_key
        if customer.stripe_customer_id:
            stripe_customer = stripe.Customer.retrieve(customer.stripe_customer_id)
        else:
            stripe_customer = stripe.Customer.create(email=email)
            db.upsert_customer(email=email, stripe_customer_id=stripe_customer.id)
            customer = db.get_customer(customer.id)

        session = stripe.checkout.Session.create(
            mode="payment",
            customer=stripe_customer.id,
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": amount_cents,
                        "product_data": {
                            "name": "nemoguardian GPU credits",
                            "description": "Credits reserved for rented GPU moderation capacity.",
                        },
                    },
                    "quantity": 1,
                }
            ],
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            metadata={
                "nemoguardian_checkout_kind": "gpu_credit",
                "nemoguardian_customer_id": str(customer.id),
                "nemoguardian_gpu_credit_cents": str(amount_cents),
            },
        )
        return GpuCreditCheckoutSession(
            session_id=session.id,
            url=session.url,
            demo_mode=False,
            customer_id=customer.id,
            amount_cents=amount_cents,
            balance_cents=db.gpu_credit_balance_cents(customer.id),
        )
    except Exception as exc:
        raise RuntimeError("Stripe GPU credit checkout failed") from exc


__all__ = [
    "CheckoutSession",
    "GpuCreditCheckoutSession",
    "PortalSession",
    "create_checkout_session",
    "create_gpu_credit_checkout_session",
    "create_portal_session",
]
