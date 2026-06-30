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
    assert db.get_customer_by_stripe_id("cus_X") == b
    assert db.get_customer_by_stripe_id("missing") is None


def test_transaction_rolls_back_on_error():
    conn = db.init_db()
    before = conn.execute("SELECT COUNT(*) AS n FROM customers").fetchone()["n"]

    with pytest.raises(RuntimeError, match="boom"), db._tx(conn):
        conn.execute(
            """INSERT INTO customers (email, stripe_customer_id, tier, created_at, updated_at)
               VALUES ('rollback@example.com', NULL, 'free', 'now', 'now')"""
        )
        raise RuntimeError("boom")

    after = conn.execute("SELECT COUNT(*) AS n FROM customers").fetchone()["n"]
    assert after == before
    assert db.get_customer_by_email("rollback@example.com") is None


def test_missing_customer_raises_key_error():
    with pytest.raises(KeyError, match="no customer with id 999"):
        db.get_customer(999)


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
    assert db.list_api_keys(a.id) == [record]

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


def test_subscription_update_and_missing_lookup():
    first = db.upsert_customer(email="first@example.com")
    second = db.upsert_customer(email="second@example.com")
    sub = db.upsert_subscription(
        customer_id=first.id,
        stripe_subscription_id="sub_X",
        tier=Tier.PRO,
        status="active",
        period_start="2026-06-01T00:00:00+00:00",
        period_end="2026-07-01T00:00:00+00:00",
    )

    updated = db.upsert_subscription(
        customer_id=second.id,
        stripe_subscription_id="sub_X",
        tier=Tier.SCALE,
        status="past_due",
        period_start="2026-07-01T00:00:00+00:00",
        period_end="2026-08-01T00:00:00+00:00",
        cancel_at_period_end=True,
    )

    assert updated.id == sub.id
    assert updated.customer_id == second.id
    assert updated.tier == "scale"
    assert updated.status == "past_due"
    assert updated.cancel_at_period_end == 1
    assert updated.current_period_start == "2026-07-01T00:00:00+00:00"

    with pytest.raises(KeyError, match="no subscription with id 999"):
        db.get_subscription(999)


def test_provisioning_job_lifecycle():
    a = db.upsert_customer(email="a@example.com")
    job = db.create_provisioning_job(a.id, tier=Tier.SELF_HOSTED, provider="vastai")
    assert job.status == "pending"
    db.update_provisioning_job(
        job.id,
        status="live",
        instance_id="vast-123",
        endpoint_url="https://x",
        ssh_command="ssh vast-123",
        error_message="recovered",
    )
    got = db.get_provisioning_job(job.id)
    assert got.status == "live"
    assert got.instance_id == "vast-123"
    assert got.endpoint_url == "https://x"
    assert got.ssh_command == "ssh vast-123"
    assert got.error_message == "recovered"

    db.update_provisioning_job(job.id)
    listed = db.list_provisioning_jobs(a.id)
    assert listed == [got]

    with pytest.raises(KeyError, match="no provisioning job with id 999"):
        db.get_provisioning_job(999)


def test_gpu_credit_ledger_topup_is_idempotent_and_listable():
    customer = db.upsert_customer(email="credits@example.com")

    first = db.record_gpu_credit_event(
        customer_id=customer.id,
        event_type="stripe_topup",
        amount_cents=2500,
        stripe_checkout_session_id="cs_test_123",
        stripe_payment_intent_id="pi_test_123",
        description="Stripe GPU credit top-up",
        metadata={"source": "stripe"},
    )
    second = db.record_gpu_credit_event(
        customer_id=customer.id,
        event_type="stripe_topup",
        amount_cents=2500,
        stripe_checkout_session_id="cs_test_123",
        description="webhook retry",
    )

    assert second.id == first.id
    assert db.gpu_credit_balance_cents(customer.id) == 2500
    assert db.get_gpu_credit_event_by_checkout_session("cs_test_123") == first
    assert db.get_gpu_credit_event_by_checkout_session("missing") is None
    events = db.list_gpu_credit_events(customer.id)
    assert events == [first]
    assert events[0].metadata == {"source": "stripe"}


def test_gpu_credit_ledger_rejects_zero_amount_and_missing_event():
    customer = db.upsert_customer(email="zero@example.com")

    with pytest.raises(ValueError, match="amount cannot be zero"):
        db.record_gpu_credit_event(
            customer_id=customer.id,
            event_type="manual",
            amount_cents=0,
        )

    with pytest.raises(KeyError, match="no GPU credit event with id 999"):
        db.get_gpu_credit_event(999)


def test_reserve_gpu_credits_debits_balance_atomically():
    customer = db.upsert_customer(email="reserve@example.com")
    job = db.create_provisioning_job(customer.id, tier=Tier.SELF_HOSTED, provider="vastai")
    db.record_gpu_credit_event(
        customer_id=customer.id,
        event_type="stripe_topup",
        amount_cents=5000,
        stripe_checkout_session_id="cs_reserve",
    )

    reservation = db.reserve_gpu_credits(
        customer_id=customer.id,
        amount_cents=1200,
        provider="vastai",
        job_id=job.id,
        description="Reserve RTX 3090 test run",
        metadata={"gpu": "RTX 3090"},
    )

    assert reservation.event_type == "provision_reserve"
    assert reservation.amount_cents == -1200
    assert reservation.provider == "vastai"
    assert reservation.job_id == job.id
    assert reservation.metadata == {"gpu": "RTX 3090"}
    assert db.gpu_credit_balance_cents(customer.id) == 3800
    assert [event.id for event in db.list_gpu_credit_events(customer.id, limit=1)] == [
        reservation.id
    ]


def test_reserve_gpu_credits_rejects_invalid_or_insufficient_amounts():
    customer = db.upsert_customer(email="insufficient@example.com")

    with pytest.raises(ValueError, match="reservation amount must be positive"):
        db.reserve_gpu_credits(customer_id=customer.id, amount_cents=0, provider="vastai")

    with pytest.raises(ValueError, match="insufficient GPU credits"):
        db.reserve_gpu_credits(customer_id=customer.id, amount_cents=100, provider="vastai")
