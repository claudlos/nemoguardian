"""DB layer tests — customers, api_keys, usage, subscriptions, provisioning."""

from __future__ import annotations

import pytest

from nemoguardian.billing import db
from nemoguardian.billing.plans import Tier


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """Use a fresh SQLite DB per test."""
    db_path = tmp_path / "nmg_test.db"
    monkeypatch.setenv("NEMOGUARDIAN_DB_PATH", str(db_path))
    db._conn = None
    db.init_db(str(db_path))
    yield
    db._conn = None


def test_upsert_customer_creates_then_updates():
    a = db.upsert_customer(email="a@example.com")
    assert a.id >= 1
    assert a.tier == "free"
    b = db.upsert_customer(email="a@example.com", stripe_customer_id="cus_X")
    assert b.id == a.id
    assert b.stripe_customer_id == "cus_X"


def test_set_customer_tier():
    a = db.upsert_customer(email="a@example.com")
    db.set_customer_tier(a.id, Tier.PRO)
    got = db.get_customer(a.id)
    assert got.tier == "pro"


def test_create_and_revoke_api_key():
    a = db.upsert_customer(email="a@example.com")
    raw, record = db.create_api_key(a.id, label="laptop")
    assert raw.startswith("nmg_")
    assert record.label == "laptop"

    # Lookup
    found = db.lookup_customer_by_api_key(raw)
    assert found is not None
    assert found.email == "a@example.com"

    # Revoke
    db.revoke_api_key(a.id, record.id)
    assert db.lookup_customer_by_api_key(raw) is None


def test_max_api_keys_enforced():
    a = db.upsert_customer(email="a@example.com")
    # FREE = 1 key
    db.create_api_key(a.id)
    with pytest.raises(ValueError, match="max API keys"):
        db.create_api_key(a.id)


def test_record_and_summarize_usage():
    a = db.upsert_customer(email="a@example.com")
    db.record_usage(a.id, call_type="standard")
    db.record_usage(a.id, call_type="standard")
    db.record_usage(a.id, call_type="deep")
    # Period = current month; use the metered helper for the canonical window
    from nemoguardian.billing.metered import check_allowance

    _allowed, info = check_allowance(a.id)
    assert info["total_calls"] == 3
    assert info["allowance"] == 1_000  # free tier
    assert info["overage_calls"] == 0


def test_subscription_roundtrip():
    a = db.upsert_customer(email="a@example.com")
    sub = db.upsert_subscription(
        customer_id=a.id,
        stripe_subscription_id="sub_X",
        tier=Tier.PRO,
        status="active",
        period_start="2026-06-01T00:00:00+00:00",
        period_end="2026-07-01T00:00:00+00:00",
    )
    got = db.get_subscription(sub.id)
    assert got.tier == "pro"
    assert got.stripe_subscription_id == "sub_X"


def test_provisioning_job_lifecycle():
    a = db.upsert_customer(email="a@example.com")
    job = db.create_provisioning_job(a.id, tier=Tier.SELF_HOSTED, provider="vastai")
    assert job.status == "pending"
    db.update_provisioning_job(job.id, status="live", instance_id="vast-123", endpoint_url="https://x")
    got = db.get_provisioning_job(job.id)
    assert got.status == "live"
    assert got.instance_id == "vast-123"
    assert got.endpoint_url == "https://x"
