"""Subscription-lifecycle webhook handlers (_on_subscription_change) — previously 0% covered."""

from __future__ import annotations

import pytest

from nemoguardian.billing import db
from nemoguardian.billing.plans import Tier
from nemoguardian.billing.webhook import _on_subscription_change


@pytest.fixture(autouse=True)
def _isolated_db(isolated_db):
    yield


def _customer(tier: Tier = Tier.PRO, stripe_id: str = "cus_123"):
    c = db.upsert_customer(email="sub@example.com", stripe_customer_id=stripe_id)
    db.set_customer_tier(c.id, tier)
    return db.get_customer(c.id)


def _sub_payload(
    *,
    status: str = "active",
    stripe_customer: str = "cus_123",
    metadata: dict | None = None,
    top_level_period: bool = True,
    item_period: bool = False,
) -> dict:
    payload: dict = {
        "id": "sub_abc",
        "customer": stripe_customer,
        "status": status,
        "metadata": metadata or {},
        "cancel_at_period_end": False,
    }
    if top_level_period:
        payload["current_period_start"] = 1_750_000_000
        payload["current_period_end"] = 1_752_600_000
    if item_period:
        payload["items"] = {
            "data": [
                {
                    "current_period_start": 1_750_000_000,
                    "current_period_end": 1_752_600_000,
                }
            ]
        }
    return payload


def test_unknown_stripe_customer_ignored():
    out = _on_subscription_change(_sub_payload(stripe_customer="cus_nobody"))
    assert out == {"ignored": True, "reason": "unknown stripe customer"}


def test_canceled_downgrades_to_free():
    customer = _customer(tier=Tier.SCALE)
    out = _on_subscription_change(_sub_payload(status="canceled"))
    assert out["handled"] is True
    assert db.get_customer(customer.id).tier == Tier.FREE.value


def test_incomplete_expired_downgrades_to_free():
    customer = _customer(tier=Tier.PRO)
    _on_subscription_change(_sub_payload(status="incomplete_expired"))
    assert db.get_customer(customer.id).tier == Tier.FREE.value


def test_past_due_keeps_tier():
    customer = _customer(tier=Tier.PRO)
    out = _on_subscription_change(_sub_payload(status="past_due"))
    assert out["status"] == "past_due"
    assert db.get_customer(customer.id).tier == Tier.PRO.value


def test_metadata_tier_change_applies():
    customer = _customer(tier=Tier.PRO)
    _on_subscription_change(
        _sub_payload(metadata={"nemoguardian_tier": "scale"})
    )
    assert db.get_customer(customer.id).tier == Tier.SCALE.value


def test_invalid_metadata_tier_falls_back_to_current():
    customer = _customer(tier=Tier.PRO)
    out = _on_subscription_change(
        _sub_payload(metadata={"nemoguardian_tier": "platinum-deluxe"})
    )
    assert out["tier"] == Tier.PRO.value
    assert db.get_customer(customer.id).tier == Tier.PRO.value


def test_top_level_period_upserts_subscription():
    customer = _customer()
    _on_subscription_change(_sub_payload())
    subs = db.init_db().execute(
        "SELECT * FROM subscriptions WHERE customer_id = ?", (customer.id,)
    ).fetchall()
    assert len(subs) == 1
    assert subs[0]["stripe_subscription_id"] == "sub_abc"


def test_item_level_period_fallback_upserts_subscription():
    """Stripe API 2025-03+ moved current_period_* onto subscription items."""
    customer = _customer()
    _on_subscription_change(_sub_payload(top_level_period=False, item_period=True))
    subs = db.init_db().execute(
        "SELECT * FROM subscriptions WHERE customer_id = ?", (customer.id,)
    ).fetchall()
    assert len(subs) == 1


def test_missing_period_everywhere_skips_upsert_but_handles():
    customer = _customer()
    out = _on_subscription_change(_sub_payload(top_level_period=False))
    assert out["handled"] is True  # tier logic still ran
    subs = db.init_db().execute(
        "SELECT * FROM subscriptions WHERE customer_id = ?", (customer.id,)
    ).fetchall()
    assert subs == []
