"""Billing helper tests for checkout, metering, and provisioning."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from nemoguardian.billing import auth, checkout, db, metered, provisioning
from nemoguardian.billing.plans import Tier


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    db_path = tmp_path / "nmg_test.db"
    monkeypatch.setenv("NEMOGUARDIAN_DB_PATH", str(db_path))
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    db._conn = None
    db.init_db(str(db_path))
    yield
    db._conn = None


def _install_stripe(monkeypatch, fake_stripe):
    monkeypatch.setitem(sys.modules, "stripe", fake_stripe)
    return fake_stripe


def test_checkout_uses_stripe_create_for_self_hosted(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    calls: dict[str, object] = {}

    class FakeCustomer:
        @staticmethod
        def create(*, email: str):
            calls["created_customer_email"] = email
            return SimpleNamespace(id="cus_created")

        @staticmethod
        def retrieve(customer_id: str):
            raise AssertionError(f"unexpected retrieve: {customer_id}")

    class FakeCheckoutSession:
        @staticmethod
        def create(**kwargs):
            calls["checkout_kwargs"] = kwargs
            return SimpleNamespace(id="cs_self_hosted", url="https://stripe.test/self-hosted")

    fake_stripe = _install_stripe(
        monkeypatch,
        SimpleNamespace(
            Customer=FakeCustomer,
            checkout=SimpleNamespace(Session=FakeCheckoutSession),
        ),
    )

    session = checkout.create_checkout_session(
        email="self@example.com",
        tier=Tier.SELF_HOSTED,
        success_url="https://app.test/success",
        cancel_url="https://app.test/cancel",
    )

    assert fake_stripe.api_key == "sk_test"
    assert session.session_id == "cs_self_hosted"
    assert session.url == "https://stripe.test/self-hosted"
    assert session.demo_mode is False
    assert db.get_customer_by_email("self@example.com").stripe_customer_id == "cus_created"
    checkout_kwargs = calls["checkout_kwargs"]
    assert checkout_kwargs["mode"] == "payment"
    assert checkout_kwargs["customer"] == "cus_created"
    assert checkout_kwargs["success_url"] == "https://app.test/success?session_id={CHECKOUT_SESSION_ID}"
    assert checkout_kwargs["cancel_url"] == "https://app.test/cancel"
    assert checkout_kwargs["metadata"]["nemoguardian_tier"] == "self_hosted"
    assert len(checkout_kwargs["line_items"]) == 2
    assert checkout_kwargs["line_items"][1]["price_data"]["unit_amount"] == 49_900


def test_checkout_reuses_existing_stripe_customer_for_subscription(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    db.upsert_customer(email="pro@example.com", stripe_customer_id="cus_existing")
    calls: dict[str, object] = {}

    class FakeCustomer:
        @staticmethod
        def create(*, email: str):
            raise AssertionError(f"unexpected create: {email}")

        @staticmethod
        def retrieve(customer_id: str):
            calls["retrieved_customer_id"] = customer_id
            return SimpleNamespace(id=customer_id)

    class FakeCheckoutSession:
        @staticmethod
        def create(**kwargs):
            calls["checkout_kwargs"] = kwargs
            return SimpleNamespace(id="cs_pro", url="https://stripe.test/pro")

    _install_stripe(
        monkeypatch,
        SimpleNamespace(
            Customer=FakeCustomer,
            checkout=SimpleNamespace(Session=FakeCheckoutSession),
        ),
    )

    session = checkout.create_checkout_session(
        email="pro@example.com",
        tier=Tier.PRO,
        success_url="https://app.test/success",
        cancel_url="https://app.test/cancel",
    )

    assert session.demo_mode is False
    assert calls["retrieved_customer_id"] == "cus_existing"
    checkout_kwargs = calls["checkout_kwargs"]
    assert checkout_kwargs["mode"] == "subscription"
    assert checkout_kwargs["customer"] == "cus_existing"
    assert checkout_kwargs["line_items"] == [
        {"price": "price_nemoguardian_pro_monthly", "quantity": 1}
    ]


def test_checkout_falls_back_to_demo_on_stripe_error(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")

    class BrokenCustomer:
        @staticmethod
        def create(*, email: str):
            raise RuntimeError("stripe unavailable")

    _install_stripe(monkeypatch, SimpleNamespace(Customer=BrokenCustomer))

    session = checkout.create_checkout_session(
        email="err@example.com",
        tier=Tier.PRO,
        success_url="https://app.test/success",
        cancel_url="https://app.test/cancel",
    )

    assert session.demo_mode is True
    assert session.session_id.startswith("demo_err_")
    assert session.url == "https://app.test/success?demo=pro"
    assert db.get_customer_by_email("err@example.com") is not None


def test_portal_falls_back_to_demo_on_stripe_error(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    customer = db.upsert_customer(email="portal-error@example.com", stripe_customer_id="cus_portal")

    class BrokenPortalSession:
        @staticmethod
        def create(**_kwargs):
            raise RuntimeError("stripe unavailable")

    _install_stripe(monkeypatch, SimpleNamespace(billing_portal=SimpleNamespace(Session=BrokenPortalSession)))

    session = checkout.create_portal_session(customer=customer, return_url="https://app.test/account")

    assert session.demo_mode is True
    assert session.url == "https://app.test/account?demo=portal"


def test_portal_uses_stripe_when_customer_has_stripe_id(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    customer = db.upsert_customer(email="portal@example.com", stripe_customer_id="cus_portal")
    calls = {}

    class FakePortalSession:
        @staticmethod
        def create(**kwargs):
            calls["portal_kwargs"] = kwargs
            return SimpleNamespace(url="https://stripe.test/portal")

    fake_stripe = _install_stripe(
        monkeypatch,
        SimpleNamespace(billing_portal=SimpleNamespace(Session=FakePortalSession)),
    )

    session = checkout.create_portal_session(
        customer=customer,
        return_url="https://app.test/account",
    )

    assert fake_stripe.api_key == "sk_test"
    assert session.demo_mode is False
    assert session.url == "https://stripe.test/portal"
    assert calls["portal_kwargs"] == {
        "customer": "cus_portal",
        "return_url": "https://app.test/account",
    }


def test_report_usage_pushes_metered_usage_record(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    customer = db.upsert_customer(email="usage@example.com", stripe_customer_id="cus_usage")
    calls: dict[str, object] = {}

    class FakeSubscription:
        @staticmethod
        def list(**kwargs):
            calls["subscription_list_kwargs"] = kwargs
            return SimpleNamespace(
                data=[
                    {
                        "items": {
                            "data": [
                                {
                                    "id": "si_flat",
                                    "price": {"recurring": {"usage_type": "licensed"}},
                                },
                                {
                                    "id": "si_metered",
                                    "price": {"recurring": {"usage_type": "metered"}},
                                },
                            ]
                        }
                    }
                ]
            )

    class FakeSubscriptionItem:
        @staticmethod
        def create_usage_record(item_id, **kwargs):
            calls["usage_record_item_id"] = item_id
            calls["usage_record_kwargs"] = kwargs

    fake_stripe = _install_stripe(
        monkeypatch,
        SimpleNamespace(
            Subscription=FakeSubscription,
            SubscriptionItem=FakeSubscriptionItem,
        ),
    )

    metered.report_usage(customer.id, call_type="standard", units=3, request_id="req-1")

    assert fake_stripe.api_key == "sk_test"
    assert calls["subscription_list_kwargs"] == {
        "customer": "cus_usage",
        "status": "active",
    }
    assert calls["usage_record_item_id"] == "si_metered"
    assert calls["usage_record_kwargs"]["quantity"] == 3
    assert calls["usage_record_kwargs"]["action"] == "increment"
    allowed, info = metered.check_allowance(customer.id)
    assert allowed is True
    assert info["total_calls"] == 3


def test_report_usage_skips_stripe_when_customer_or_subscription_is_missing(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    customer = db.upsert_customer(email="local-only@example.com")
    calls: list[dict] = []

    class FakeSubscription:
        @staticmethod
        def list(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[])

    _install_stripe(monkeypatch, SimpleNamespace(Subscription=FakeSubscription))

    metered.report_usage(customer.id, call_type="standard", units=1)
    assert calls == []

    stripe_customer = db.upsert_customer(email="no-subscription@example.com", stripe_customer_id="cus_empty")
    metered.report_usage(stripe_customer.id, call_type="standard", units=1)

    assert calls == [{"customer": "cus_empty", "status": "active"}]


def test_report_usage_ignores_stripe_errors_after_local_record(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    customer = db.upsert_customer(email="usage-error@example.com", stripe_customer_id="cus_usage")

    class BrokenSubscription:
        @staticmethod
        def list(**kwargs):
            raise RuntimeError("stripe timeout")

    _install_stripe(monkeypatch, SimpleNamespace(Subscription=BrokenSubscription))

    metered.report_usage(customer.id, call_type="deep", units=2, request_id="req-err")

    allowed, info = metered.check_allowance(customer.id)
    assert allowed is True
    assert info["total_calls"] == 2


def test_check_allowance_handles_december_period_boundary(monkeypatch):
    class FrozenDateTime:
        @staticmethod
        def now(_tz):
            return datetime(2026, 12, 15, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(metered, "datetime", FrozenDateTime)
    customer = db.upsert_customer(email="december@example.com")

    allowed, info = metered.check_allowance(customer.id)

    assert allowed is True
    assert info["period_start"] == "2026-12-01T00:00:00+00:00"
    assert info["period_end"] == "2027-01-01T00:00:00+00:00"


async def test_require_api_key_rejects_bad_headers_and_uses_env_bootstrap(monkeypatch):
    with pytest.raises(HTTPException) as invalid_scheme:
        await auth.require_api_key("Basic nmg_test")
    assert invalid_scheme.value.status_code == 401
    assert "invalid Authorization header" in invalid_scheme.value.detail

    with pytest.raises(HTTPException) as bad_prefix:
        await auth.require_api_key("Bearer bad-key")
    assert bad_prefix.value.status_code == 401
    assert "must start with 'nmg_'" in bad_prefix.value.detail

    monkeypatch.setenv("NEMOGUARDIAN_API_KEY", "nmg_env_bootstrap")
    monkeypatch.setenv("NEMOGUARDIAN_TIER", "not-a-tier")
    context = await auth.require_api_key("Bearer nmg_env_bootstrap")

    assert context.customer.email == "self-hosted@nemoguardian.local"
    assert context.plan.tier == Tier.SELF_HOSTED
    assert context.raw_key == "nmg_env_bootstrap"
    assert auth._upgrade_target("unknown.feature") == Tier.SCALE


def test_provisioning_rejects_unknown_provider():
    customer = db.upsert_customer(email="provision@example.com")

    with pytest.raises(ValueError, match="unsupported provider"):
        asyncio.run(provisioning.provision_instance(customer_id=customer.id, provider="bogus"))


async def test_run_job_marks_on_prem_job_live(monkeypatch):
    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(provisioning.asyncio, "sleep", fake_sleep)
    customer = db.upsert_customer(email="onprem@example.com")
    job = db.create_provisioning_job(
        customer.id,
        tier=Tier.SELF_HOSTED,
        provider="on_prem",
    )

    await provisioning._run_job(job.id, "on_prem", "ssh-ed25519 AAA test@example")

    got = db.get_provisioning_job(job.id)
    assert got.status == "live"
    assert got.instance_id == f"onprem-{job.id}"
    assert got.endpoint_url is None
    assert "ssh-ed25519 AAA test@example" in got.ssh_command
    assert "docker compose up -d" in got.ssh_command


async def test_run_job_marks_remote_provider_live(monkeypatch):
    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(provisioning.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(provisioning.secrets, "token_hex", lambda _n: "abc123")
    monkeypatch.setenv("NEMOGUARDIAN_VASTAI_API_KEY", "vast-test-key")
    customer = db.upsert_customer(email="vast@example.com")
    job = db.create_provisioning_job(
        customer.id,
        tier=Tier.SELF_HOSTED,
        provider="vastai",
    )

    await provisioning._run_job(job.id, "vastai", None)

    got = db.get_provisioning_job(job.id)
    assert got.status == "live"
    assert got.instance_id == "vastai-abc123"
    assert got.endpoint_url == "https://vastai-abc123.nemoguardian.dev"
    assert got.ssh_command == (
        "ssh -i ~/.ssh/nemoguardian_vastai-abc123 "
        "nemoguardian@vastai-abc123.nemoguardian.dev"
    )


async def test_run_job_marks_job_failed_on_provider_exception(monkeypatch):
    async def fail_sleep(_seconds: float) -> None:
        raise RuntimeError("provider down")

    monkeypatch.setattr(provisioning.asyncio, "sleep", fail_sleep)
    customer = db.upsert_customer(email="failed-provider@example.com")
    job = db.create_provisioning_job(
        customer.id,
        tier=Tier.SELF_HOSTED,
        provider="vastai",
    )

    await provisioning._run_job(job.id, "vastai", None)

    got = db.get_provisioning_job(job.id)
    assert got.status == "failed"
    assert got.error_message == "provider down"


def test_render_onprem_snippet_uses_placeholder_or_public_key():
    placeholder = provisioning._render_onprem_snippet(None)
    custom = provisioning._render_onprem_snippet("ssh-ed25519 AAA test@example")

    assert "<paste your SSH public key>" in placeholder
    assert "ssh-ed25519 AAA test@example" in custom
    assert "docker compose up -d" in custom
