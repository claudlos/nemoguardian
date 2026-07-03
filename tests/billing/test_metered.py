"""Metered billing — the Stripe usage push (the revenue side effect) and period math."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nemoguardian.billing import db, metered


@pytest.fixture(autouse=True)
def _isolated_db(isolated_db):
    yield


def _stripe_customer():
    c = db.upsert_customer(email="meter@example.com", stripe_customer_id="cus_meter")
    return db.get_customer(c.id)


def _install_stripe_stub(monkeypatch, stub) -> None:
    monkeypatch.setitem(sys.modules, "stripe", stub)


def test_no_stripe_key_records_locally_only(monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    customer = _stripe_customer()
    metered.report_usage(customer.id, call_type="standard")
    _allowed, info = metered.check_allowance(customer.id)
    assert info["total_calls"] == 1


def test_meter_event_path_preferred(monkeypatch):
    """New Stripe API: billing.MeterEvent replaces usage records."""
    customer = _stripe_customer()
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setenv("STRIPE_METER_EVENT_NAME", "nemoguardian_calls")

    created = []
    stub = SimpleNamespace(
        api_key=None,
        billing=SimpleNamespace(MeterEvent=SimpleNamespace(create=lambda **kw: created.append(kw))),
        Subscription=SimpleNamespace(list=lambda **kw: pytest.fail("legacy path used")),
    )
    _install_stripe_stub(monkeypatch, stub)

    metered.report_usage(customer.id, call_type="deep", units=3)
    assert len(created) == 1
    assert created[0]["event_name"] == "nemoguardian_calls"
    assert created[0]["payload"]["stripe_customer_id"] == "cus_meter"
    assert created[0]["payload"]["value"] == "3"


def test_legacy_usage_record_fallback(monkeypatch):
    """Old stripe-python (< v12): SubscriptionItem.create_usage_record still works."""
    customer = _stripe_customer()
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.delenv("STRIPE_METER_EVENT_NAME", raising=False)

    recorded = []

    def create_usage_record(item_id, **kw):
        recorded.append((item_id, kw))

    sub = {
        "items": {
            "data": [
                {"id": "si_flat", "price": {"recurring": {"usage_type": "licensed"}}},
                {"id": "si_metered", "price": {"recurring": {"usage_type": "metered"}}},
            ]
        }
    }
    # dict payload must also support .get like the SDK object — plain dict does.
    stub = SimpleNamespace(
        api_key=None,
        billing=SimpleNamespace(),  # no MeterEvent attr → legacy path
        Subscription=SimpleNamespace(list=lambda **kw: SimpleNamespace(data=[sub])),
        SubscriptionItem=SimpleNamespace(create_usage_record=create_usage_record),
    )
    _install_stripe_stub(monkeypatch, stub)

    metered.report_usage(customer.id, call_type="standard")
    assert recorded and recorded[0][0] == "si_metered"
    assert recorded[0][1]["quantity"] == 1


def test_neither_api_available_is_silent(monkeypatch):
    """A stripe module with no usable usage API must not break moderation."""
    customer = _stripe_customer()
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.delenv("STRIPE_METER_EVENT_NAME", raising=False)
    stub = SimpleNamespace(
        api_key=None,
        billing=SimpleNamespace(),
        SubscriptionItem=SimpleNamespace(),  # no create_usage_record attr
    )
    _install_stripe_stub(monkeypatch, stub)
    metered.report_usage(customer.id, call_type="standard")  # must not raise
    _allowed, info = metered.check_allowance(customer.id)
    assert info["total_calls"] == 1


def test_stripe_exception_swallowed(monkeypatch):
    customer = _stripe_customer()
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    monkeypatch.setenv("STRIPE_METER_EVENT_NAME", "m")

    def boom(**kw):
        raise RuntimeError("stripe down")

    stub = SimpleNamespace(
        api_key=None,
        billing=SimpleNamespace(MeterEvent=SimpleNamespace(create=boom)),
    )
    _install_stripe_stub(monkeypatch, stub)
    metered.report_usage(customer.id, call_type="standard")  # must not raise


def test_check_allowance_december_rollover(monkeypatch):
    """The period-end computation must cross the year boundary correctly."""
    customer = _stripe_customer()

    class FrozenDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 12, 15, 12, 0, 0, tzinfo=tz or timezone.utc)

    monkeypatch.setattr(metered, "datetime", FrozenDT)
    _allowed, info = metered.check_allowance(customer.id)
    assert info["period_start"].startswith("2026-12-01")
    assert info["period_end"].startswith("2027-01-01")


def test_check_allowance_blocks_over_cap(monkeypatch):
    customer = _stripe_customer()  # free tier
    plan = db.get_customer(customer.id).plan
    for _ in range(plan.monthly_allowance):
        db.record_usage(customer.id, call_type="fast")
    allowed, info = metered.check_allowance(customer.id)
    assert not allowed
    assert info["total_calls"] == plan.monthly_allowance
