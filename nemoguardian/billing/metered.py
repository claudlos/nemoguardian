"""Metered usage reporting.

In production: after each call, we push a `usage_record` to Stripe so the
customer's invoice at end of month includes overage charges.

In demo mode (no Stripe key): just record locally and let the dashboard show
projected overage.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from nemoguardian.billing import db
from nemoguardian.billing.plans import Tier


def report_usage(customer_id: int, *, call_type: str, units: int = 1, request_id: str | None = None) -> None:
    """Record a usage event. Push to Stripe if overage is met and key is set."""
    for _ in range(units):
        db.record_usage(customer_id, call_type=call_type, request_id=request_id)

    api_key = os.environ.get("STRIPE_SECRET_KEY")
    if not api_key:
        return  # demo mode — recorded locally only

    try:
        import stripe  # type: ignore

        stripe.api_key = api_key
        customer = db.get_customer(customer_id)
        if not customer.stripe_customer_id:
            return
        # Find the subscription item for the metered plan and report usage.
        subs = stripe.Subscription.list(customer=customer.stripe_customer_id, status="active")
        if not subs.data:
            return
        sub = subs.data[0]
        for item in sub.get("items", {}).get("data", []):
            price = item.get("price", {})
            if price.get("recurring", {}).get("usage_type") == "metered":
                stripe.SubscriptionItem.create_usage_record(
                    item["id"],
                    quantity=units,
                    timestamp=int(datetime.now(timezone.utc).timestamp()),
                    action="increment",
                )
                return
    except Exception:
        # Don't let a Stripe hiccup kill the moderation call.
        return


def check_allowance(customer_id: int) -> tuple[bool, dict[str, Any]]:
    """Returns (allowed, info_dict) for the current billing period.

    Always returns allowed=True on free tier if usage < allowance.
    Returns allowed=False if over the cap.
    """
    customer = db.get_customer(customer_id)
    plan = customer.plan
    # Period = current calendar month UTC.
    now = datetime.now(timezone.utc)
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
    next_month = now.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    # Advance to next month
    if next_month.month == 12:
        period_end_dt = next_month.replace(year=next_month.year + 1, month=1)
    else:
        period_end_dt = next_month.replace(month=next_month.month + 1)
    period_end = period_end_dt.isoformat()

    summary = db.usage_for_period(customer_id, period_start, period_end)
    allowed = summary.total_calls < plan.monthly_allowance or plan.tier in {Tier.SCALE, Tier.SELF_HOSTED}
    return allowed, {
        "total_calls": summary.total_calls,
        "allowance": plan.monthly_allowance,
        "overage_calls": summary.overage_calls,
        "overage_cents": summary.overage_cents,
        "period_start": period_start,
        "period_end": period_end,
        "tier": plan.tier.value,
    }


__all__ = ["check_allowance", "report_usage"]
