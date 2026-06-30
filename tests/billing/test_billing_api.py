"""End-to-end billing API tests using mocked cascade + real DB."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemoguardian import server as srv
from nemoguardian.billing import db as billing_db
from nemoguardian.billing import webhook as billing_webhook
from nemoguardian.billing.plans import Tier
from nemoguardian.cascade import Cascade
from nemoguardian.policy.presets import get_preset
from nemoguardian.providers.base import Instance, InstanceState, Offer, ProviderName, ProvisionError
from nemoguardian.schemas import ModelVerdict, ModerateResponse, VerdictLabel


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    db_path = tmp_path / "nmg_test.db"
    monkeypatch.setenv("NEMOGUARDIAN_DB_PATH", str(db_path))
    monkeypatch.delenv("NEMOGUARDIAN_API_KEY", raising=False)
    monkeypatch.delenv("NEMOGUARDIAN_TIER", raising=False)
    monkeypatch.delenv("NEMOGUARDIAN_SELF_HOSTED_EMAIL", raising=False)
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("NEMOGUARDIAN_INSECURE_WEBHOOK_DEMO", raising=False)
    billing_db._conn = None
    billing_db.init_db(str(db_path))
    yield


def _mock_moderate(request, policy_engine=None, **kw) -> ModerateResponse:
    csr = ModelVerdict(
        model_id="mock-csr",
        verdict=VerdictLabel.UNSAFE,
        score=0.95,
        latency_ms=20.0,
        reasoning="PII",
        categories=["PII"],
    )
    qwen = ModelVerdict(
        model_id="mock-qwen", verdict=VerdictLabel.SAFE, score=0.1, latency_ms=10.0,
    )
    from nemoguardian.aggregator import aggregate

    agg = aggregate({"qwen3_guard_gen": qwen, "nemotron_csr": csr})
    if policy_engine is not None:
        d = policy_engine.evaluate(
            verdict=agg.verdict, score=agg.score, categories=agg.categories
        )
        if d.final_label is not None:
            agg.verdict = d.final_label
    return ModerateResponse(
        verdict=agg.verdict,
        score=agg.score,
        reasons=agg.reasons,
        categories=agg.categories,
        matched_policy_rule=policy_engine.evaluate(
            verdict=agg.verdict, score=agg.score, categories=agg.categories
        ).matched_rule if policy_engine else None,
        model_verdicts={"qwen3_guard_gen": qwen, "nemotron_csr": csr},
        total_latency_ms=42.0,
        mode=request.mode,
    )


class FakeProvider:
    name = ProviderName.VAST_AI

    def __init__(
        self,
        offers: list[Offer],
        *,
        instance: Instance | None = None,
        provision_error: Exception | None = None,
    ) -> None:
        self.offers = offers
        self.instance = instance
        self.provision_error = provision_error
        self.list_calls: list[dict] = []
        self.provision_calls: list[dict] = []

    async def list_offers(self, *, gpu_model=None, max_price_usd=None):
        self.list_calls.append({"gpu_model": gpu_model, "max_price_usd": max_price_usd})
        offers = self.offers
        if gpu_model:
            offers = [offer for offer in offers if offer.gpu_model == gpu_model]
        if max_price_usd is not None:
            offers = [offer for offer in offers if offer.price_per_hour_usd <= max_price_usd]
        return offers

    async def provision(self, offer, *, ssh_public_key=None, image=None, env=None):
        self.provision_calls.append({
            "offer": offer,
            "ssh_public_key": ssh_public_key,
            "image": image,
            "env": env,
        })
        if self.provision_error:
            raise self.provision_error
        return self.instance or Instance(
            provider=ProviderName.VAST_AI,
            instance_id="fake-instance",
            gpu_model=offer.gpu_model,
            vram_gb=offer.vram_gb,
            region=offer.region,
            state=InstanceState.LIVE,
            endpoint_url="https://gpu.test",
            ssh_command="ssh root@gpu.test",
            hourly_price_usd=offer.price_per_hour_usd,
        )


class BrokenOfferProvider:
    name = ProviderName.RUNPOD

    async def list_offers(self, *, gpu_model=None, max_price_usd=None):
        raise RuntimeError("provider unavailable")


class FakeProviderRegistry:
    def __init__(self, providers: list[object]) -> None:
        self.providers = {provider.name: provider for provider in providers}

    def all(self) -> list[object]:
        return list(self.providers.values())

    def get(self, name: ProviderName):
        return self.providers[name]


@pytest.fixture
def client(monkeypatch):
    fake_cascade = MagicMock(spec=Cascade)
    fake_cascade.moderate.side_effect = _mock_moderate
    fake_cascade.loaded_models.return_value = {
        "qwen3_guard_gen": False,
        "qwen3_guard_stream": False,
        "nemotron_csr": False,
    }
    fake_cascade.model_config_summary.return_value = {
        "qwen_gen_model": "mock-qwen",
        "nemotron_csr_model": "mock-csr",
        "enable_triage": True,
    }
    fake_cascade.triage_status.return_value = {
        "configured": False,
        "provider": None,
        "model": "mock-triage",
        "base_url": "https://example.test/v1",
    }
    policies = {name: get_preset(name) for name in ("discord", "twitch", "generic")}
    monkeypatch.setattr(srv, "get_cascade", lambda: fake_cascade)
    monkeypatch.setattr(srv, "get_policies", lambda: policies)
    from nemoguardian.server import app

    return TestClient(app)


def _auth_headers_for_tier(tier: Tier, *, email: str = "tier@example.com") -> tuple[dict[str, str], billing_db.Customer]:
    customer = billing_db.upsert_customer(email=email)
    billing_db.set_customer_tier(customer.id, tier)
    customer = billing_db.get_customer(customer.id)
    raw, _ = billing_db.create_api_key(customer.id, label=f"{tier.value}-test")
    return {"Authorization": f"Bearer {raw}"}, customer


def _fund_gpu_credits(customer: billing_db.Customer, amount_cents: int = 10_000) -> None:
    billing_db.record_gpu_credit_event(
        customer_id=customer.id,
        event_type="stripe_topup",
        amount_cents=amount_cents,
        stripe_checkout_session_id=f"cs_fund_{customer.id}_{amount_cents}",
        description="Test GPU credit funding",
    )


def test_list_plans(client):
    r = client.get("/billing/plans")
    assert r.status_code == 200
    plans = r.json()["plans"]
    assert {p["tier"] for p in plans} == {"free", "pro", "scale", "self_hosted"}
    pro = next(p for p in plans if p["tier"] == "pro")
    assert pro["monthly_price_cents"] == 1_900
    assert "cascade.standard" in pro["features"]


def test_checkout_creates_customer(client):
    r = client.post(
        "/billing/checkout",
        json={"email": "buyer@example.com", "tier": "pro"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["demo_mode"] is True  # no STRIPE_SECRET_KEY set
    assert body["session_id"].startswith("demo_pro_")
    customer = billing_db.get_customer_by_email("buyer@example.com")
    assert customer is not None


def test_gpu_credit_checkout_demo_credits_customer(client):
    r = client.post(
        "/billing/gpu-credits/checkout",
        json={"email": "gpu-buyer@example.com", "amount_cents": 2_500},
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["demo_mode"] is True
    assert body["amount_cents"] == 2_500
    assert body["balance_cents"] == 2_500
    customer = billing_db.get_customer_by_email("gpu-buyer@example.com")
    assert customer is not None
    assert billing_db.gpu_credit_balance_cents(customer.id) == 2_500


def test_gpu_credit_balance_requires_api_key_and_lists_events(client):
    headers, customer = _auth_headers_for_tier(
        Tier.SELF_HOSTED, email="gpu-balance@example.com"
    )
    _fund_gpu_credits(customer, 2_500)

    missing = client.get("/billing/gpu-credits")
    assert missing.status_code == 401

    r = client.get("/billing/gpu-credits", headers=headers)

    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == customer.id
    assert body["balance_cents"] == 2_500
    assert body["events"][0]["event_type"] == "stripe_topup"


def test_gpu_credit_checkout_status_reports_pending_and_credit(client):
    pending = client.get(
        "/billing/gpu-credits/checkout-status",
        params={"session_id": "cs_test_missing"},
    )

    assert pending.status_code == 200
    assert pending.json() == {
        "session_id": "cs_test_missing",
        "credited": False,
        "amount_cents": None,
        "balance_cents": None,
        "currency": "usd",
        "event_id": None,
        "occurred_at": None,
    }

    customer = billing_db.upsert_customer(email="gpu-status@example.com")
    event = billing_db.record_gpu_credit_event(
        customer_id=customer.id,
        event_type="stripe_topup",
        amount_cents=5_000,
        stripe_checkout_session_id="cs_test_status",
        stripe_payment_intent_id="pi_status",
        description="Stripe GPU credit top-up",
    )

    credited = client.get(
        "/billing/gpu-credits/checkout-status",
        params={"session_id": "cs_test_status"},
    )

    assert credited.status_code == 200
    body = credited.json()
    assert body["session_id"] == "cs_test_status"
    assert body["credited"] is True
    assert body["amount_cents"] == 5_000
    assert body["balance_cents"] == 5_000
    assert body["event_id"] == event.id
    assert body["occurred_at"] == event.occurred_at


def test_webhook_checkout_completed_upgrades_tier(client, monkeypatch):
    # First create the customer via checkout.
    client.post("/billing/checkout", json={"email": "buyer@example.com", "tier": "free"})
    customer = billing_db.get_customer_by_email("buyer@example.com")

    # Now simulate a Stripe webhook for the upgrade.
    payload = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_test_1",
                "customer": "cus_test_X",
                "customer_email": "buyer@example.com",
                "subscription": "sub_test_1",
                "created": 1717200000,
                "expires_at": 1719792000,
                "metadata": {
                    "nemoguardian_customer_id": str(customer.id),
                    "nemoguardian_tier": "pro",
                },
            }
        },
    }
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    body, signature = _signed_payload(payload, "whsec_test")
    r = client.post(
        "/billing/webhook",
        content=body,
        headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["api_key_provisioned"] is True
    assert isinstance(body["api_key_id"], int)
    assert "nmg_" not in r.text
    upgraded = billing_db.get_customer(customer.id)
    assert upgraded.tier == "pro"
    assert upgraded.stripe_customer_id == "cus_test_X"


def test_webhook_gpu_credit_checkout_adds_balance_idempotently(client, monkeypatch):
    customer = billing_db.upsert_customer(
        email="gpu-webhook@example.com", stripe_customer_id="cus_gpu"
    )
    payload = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_gpu_topup",
                "customer": "cus_gpu",
                "payment_intent": "pi_gpu",
                "amount_total": 3_500,
                "metadata": {
                    "nemoguardian_checkout_kind": "gpu_credit",
                    "nemoguardian_customer_id": str(customer.id),
                    "nemoguardian_gpu_credit_cents": "3500",
                },
            }
        },
    }
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    body, signature = _signed_payload(payload, "whsec_test")

    first = client.post(
        "/billing/webhook",
        content=body,
        headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
    )
    second = client.post(
        "/billing/webhook",
        content=body,
        headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["gpu_credit_applied"] is True
    assert second.json()["gpu_credit_event_id"] == first.json()["gpu_credit_event_id"]
    assert billing_db.gpu_credit_balance_cents(customer.id) == 3_500


def test_webhook_signature_parser_rejects_malformed_headers(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")

    assert billing_webhook.verify_signature(b"{}", "not-a-stripe-header") is False
    assert billing_webhook.verify_signature(b"{}", "t=not-an-int,v1=abc") is False
    assert billing_webhook.verify_signature(b"{}", "t=1717200000,v1=abc", now=1717200000) is False


def test_webhook_rejects_invalid_json_after_valid_signature(client, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    body = b"{not-json"
    timestamp = int(time.time())
    digest = hmac.new(
        b"whsec_test",
        f"{timestamp}.".encode() + body,
        hashlib.sha256,
    ).hexdigest()

    r = client.post(
        "/billing/webhook",
        content=body,
        headers={
            "Stripe-Signature": f"t={timestamp},v1={digest}",
            "Content-Type": "application/json",
        },
    )

    assert r.status_code == 400
    assert "invalid JSON" in r.json()["detail"]


def test_webhook_ignores_unknown_event_type(client, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    payload = {"type": "invoice.payment_succeeded", "data": {"object": {"id": "in_1"}}}
    body, signature = _signed_payload(payload, "whsec_test")

    r = client.post(
        "/billing/webhook",
        content=body,
        headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
    )

    assert r.status_code == 200
    assert r.json() == {
        "received": True,
        "type": "invoice.payment_succeeded",
        "ignored": True,
    }


def test_webhook_checkout_ignores_missing_invalid_and_unmatched_metadata(client, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")

    for payload, reason in [
        (
            {
                "type": "checkout.session.completed",
                "data": {"object": {"metadata": {}}},
            },
            "no tier metadata",
        ),
        (
            {
                "type": "checkout.session.completed",
                "data": {"object": {"metadata": {"nemoguardian_tier": "enterprise"}}},
            },
            "no tier metadata",
        ),
        (
            {
                "type": "checkout.session.completed",
                "data": {"object": {"metadata": {"nemoguardian_tier": "pro"}}},
            },
            "no matching customer",
        ),
    ]:
        body, signature = _signed_payload(payload, "whsec_test")
        r = client.post(
            "/billing/webhook",
            content=body,
            headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
        )

        assert r.status_code == 200
        assert r.json()["ignored"] is True
        assert r.json()["reason"] == reason


def test_webhook_checkout_matches_existing_stripe_customer(client, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    customer = billing_db.upsert_customer(
        email="stripe-match@example.com",
        stripe_customer_id="cus_existing",
    )
    payload = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "customer": "cus_existing",
                "metadata": {"nemoguardian_tier": "scale"},
            }
        },
    }
    body, signature = _signed_payload(payload, "whsec_test")

    r = client.post(
        "/billing/webhook",
        content=body,
        headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
    )

    assert r.status_code == 200
    assert r.json()["customer_id"] == customer.id
    assert r.json()["tier"] == "scale"
    assert billing_db.get_customer(customer.id).tier == "scale"


def test_webhook_checkout_creates_customer_from_email_details(client, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    payload = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "customer": "cus_new",
                "customer_details": {"email": "new-checkout@example.com"},
                "metadata": {"nemoguardian_tier": "pro"},
            }
        },
    }
    body, signature = _signed_payload(payload, "whsec_test")

    r = client.post(
        "/billing/webhook",
        content=body,
        headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
    )

    assert r.status_code == 200
    customer = billing_db.get_customer_by_email("new-checkout@example.com")
    assert customer is not None
    assert customer.stripe_customer_id == "cus_new"
    assert customer.tier == "pro"


def test_webhook_subscription_lifecycle_updates_customer_and_subscription(client, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    customer = billing_db.upsert_customer(
        email="subscriber@example.com",
        stripe_customer_id="cus_subscriber",
    )
    payload = {
        "type": "customer.subscription.updated",
        "data": {
            "object": {
                "id": "sub_live",
                "customer": "cus_subscriber",
                "status": "active",
                "metadata": {"nemoguardian_tier": "scale"},
                "current_period_start": 1717200000,
                "current_period_end": 1719792000,
                "cancel_at_period_end": True,
            }
        },
    }
    body, signature = _signed_payload(payload, "whsec_test")

    r = client.post(
        "/billing/webhook",
        content=body,
        headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
    )

    assert r.status_code == 200
    assert r.json()["handled"] is True
    assert r.json()["tier"] == "scale"
    updated = billing_db.get_customer(customer.id)
    assert updated.tier == "scale"
    row = billing_db.init_db().execute(
        "SELECT * FROM subscriptions WHERE stripe_subscription_id = ?",
        ("sub_live",),
    ).fetchone()
    assert row["status"] == "active"
    assert row["tier"] == "scale"
    assert row["cancel_at_period_end"] == 1


def test_webhook_subscription_canceled_downgrades_to_free(client, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    customer = billing_db.upsert_customer(
        email="cancel@example.com",
        stripe_customer_id="cus_cancel",
    )
    billing_db.set_customer_tier(customer.id, Tier.PRO)
    payload = {
        "type": "customer.subscription.deleted",
        "data": {
            "object": {
                "id": "sub_cancel",
                "customer": "cus_cancel",
                "status": "canceled",
                "metadata": {"nemoguardian_tier": "pro"},
            }
        },
    }
    body, signature = _signed_payload(payload, "whsec_test")

    r = client.post(
        "/billing/webhook",
        content=body,
        headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
    )

    assert r.status_code == 200
    assert r.json()["status"] == "canceled"
    assert r.json()["tier"] == "pro"
    assert billing_db.get_customer(customer.id).tier == "free"


def test_webhook_subscription_past_due_keeps_existing_tier(client, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    customer = billing_db.upsert_customer(
        email="pastdue@example.com",
        stripe_customer_id="cus_pastdue",
    )
    billing_db.set_customer_tier(customer.id, Tier.PRO)
    payload = {
        "type": "customer.subscription.updated",
        "data": {
            "object": {
                "id": "sub_pastdue",
                "customer": "cus_pastdue",
                "status": "past_due",
                "metadata": {},
            }
        },
    }
    body, signature = _signed_payload(payload, "whsec_test")

    r = client.post(
        "/billing/webhook",
        content=body,
        headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
    )

    assert r.status_code == 200
    assert r.json()["status"] == "past_due"
    assert r.json()["tier"] == "pro"
    assert billing_db.get_customer(customer.id).tier == "pro"


def test_webhook_signature_required_when_secret_missing(client):
    payload = {"type": "checkout.session.completed", "data": {"object": {}}}
    r = client.post("/billing/webhook", json=payload)
    assert r.status_code == 400


def test_webhook_signature_required_when_secret_set(client, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    payload = {"type": "checkout.session.completed", "data": {"object": {}}}
    r = client.post(
        "/billing/webhook",
        json=payload,
        headers={"Stripe-Signature": ""},
    )
    # No signature but webhook secret set → 400
    assert r.status_code == 400


def test_webhook_rejects_replayed_signature(client, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    payload = {"type": "checkout.session.completed", "data": {"object": {}}}
    body, signature = _signed_payload(payload, "whsec_test", timestamp=1)
    r = client.post(
        "/billing/webhook",
        content=body,
        headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
    )
    assert r.status_code == 400


def test_unsigned_webhook_requires_explicit_insecure_demo_opt_in(client, monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_INSECURE_WEBHOOK_DEMO", "1")
    payload = {"type": "customer.subscription.deleted", "data": {"object": {}}}
    r = client.post("/billing/webhook", json=payload)
    assert r.status_code == 200
    assert r.json()["received"] is True


def test_portal_returns_demo_session_for_authenticated_customer(client):
    customer = billing_db.upsert_customer(email="portal@example.com")
    raw, _ = billing_db.create_api_key(customer.id, label="portal-test")
    r = client.post(
        "/billing/portal",
        json={"return_url": "https://example.test/account"},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["customer_id"] == customer.id
    assert body["demo_mode"] is True
    assert body["url"] == "https://example.test/account?demo=portal"


def test_moderate_requires_api_key(client):
    r = client.post("/v1/moderate", json={"text": "drop your SSN"})
    assert r.status_code == 401


def test_stream_requires_api_key(client):
    r = client.post("/v1/moderate/stream", json={"text": "drop your SSN"})
    assert r.status_code == 401


def test_env_api_key_bootstraps_self_hosted_customer(client, monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_API_KEY", "nmg_env_real_test_key")
    monkeypatch.setenv("NEMOGUARDIAN_TIER", "self_hosted")
    monkeypatch.setenv("NEMOGUARDIAN_SELF_HOSTED_EMAIL", "box@example.com")
    r = client.post(
        "/v1/moderate",
        json={"text": "drop your SSN", "mode": "deep"},
        headers={"Authorization": "Bearer nmg_env_real_test_key"},
        params={"policy_preset": "discord"},
    )
    assert r.status_code == 200, r.text
    customer = billing_db.get_customer_by_email("box@example.com")
    assert customer is not None
    assert customer.tier == "self_hosted"


@pytest.mark.parametrize(
    "placeholder",
    [
        "nmg_change_me",
        "nmg_default_change_me",
        "nmg_paste_your_key_here",
        "nmg_replace_with_demo_key",
    ],
)
def test_placeholder_env_api_key_is_not_accepted(client, monkeypatch, placeholder):
    monkeypatch.setenv("NEMOGUARDIAN_API_KEY", placeholder)
    r = client.post(
        "/v1/moderate",
        json={"text": "hello", "mode": "fast"},
        headers={"Authorization": f"Bearer {placeholder}"},
    )
    assert r.status_code == 401


def test_free_tier_moderate_succeeds(client):
    customer = billing_db.upsert_customer(email="free@example.com")
    raw, _ = billing_db.create_api_key(customer.id, label="t")
    # FREE tier only allows `fast` mode — use mode=fast.
    r = client.post(
        "/v1/moderate",
        json={"text": "drop your SSN", "mode": "fast"},
        headers={"Authorization": f"Bearer {raw}"},
        params={"policy_preset": "discord"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # The mock returns unsafe regardless of input, but cascade uses `fast` mode
    # which doesn't run the qwen/csr models in real life. Mocked here = unsafe.
    assert body["verdict"] == "unsafe"
    # Usage recorded
    _, info = __import__("nemoguardian.billing.metered", fromlist=["check_allowance"]).check_allowance(customer.id)
    assert info["total_calls"] == 1


def test_free_tier_blocks_deep_mode(client):
    customer = billing_db.upsert_customer(email="free@example.com")
    raw, _ = billing_db.create_api_key(customer.id, label="t")
    r = client.post(
        "/v1/moderate",
        json={"text": "hello", "mode": "deep"},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 402
    assert "deep" in r.json()["detail"].lower()


def test_free_tier_blocks_stream(client):
    customer = billing_db.upsert_customer(email="free@example.com")
    raw, _ = billing_db.create_api_key(customer.id, label="t")
    r = client.post(
        "/v1/moderate/stream",
        json={"text": "hello"},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 402
    assert "stream" in r.json()["detail"].lower()


def test_pro_tier_allows_deep_mode(client):
    customer = billing_db.upsert_customer(email="pro@example.com")
    billing_db.set_customer_tier(customer.id, Tier.SCALE)  # SCALE has deep
    raw, _ = billing_db.create_api_key(customer.id)
    r = client.post(
        "/v1/moderate",
        json={"text": "hello", "mode": "deep"},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 200


def test_self_hosted_required_for_provision(client):
    customer = billing_db.upsert_customer(email="pro@example.com")
    billing_db.set_customer_tier(customer.id, Tier.PRO)
    raw, _ = billing_db.create_api_key(customer.id)
    r = client.post(
        "/billing/provision",
        json={"provider": "vastai"},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 402


def test_self_hosted_provisioning_lifecycle(client):
    customer = billing_db.upsert_customer(email="self@example.com")
    billing_db.set_customer_tier(customer.id, Tier.SELF_HOSTED)
    raw, _ = billing_db.create_api_key(customer.id)
    # Create job
    r = client.post(
        "/billing/provision",
        json={"provider": "vastai"},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    job_id = body["job_id"]
    assert body["status"] in {"pending", "provisioning", "live"}
    # Poll status
    r2 = client.get(f"/billing/jobs/{job_id}", headers={"Authorization": f"Bearer {raw}"})
    assert r2.status_code == 200


def test_usage_endpoint_returns_allowance(client):
    customer = billing_db.upsert_customer(email="buyer@example.com")
    raw, _ = billing_db.create_api_key(customer.id)
    r = client.get("/billing/usage", headers={"Authorization": f"Bearer {raw}"})
    assert r.status_code == 200
    body = r.json()
    assert body["allowance"] == 1_000
    assert body["tier"] == "free"


def test_create_key_endpoint_returns_new_key_once(client):
    headers, _customer = _auth_headers_for_tier(Tier.PRO, email="key-create@example.com")

    r = client.post("/billing/keys", headers=headers, json={"label": "bot-prod"})

    assert r.status_code == 200
    body = r.json()
    assert body["api_key"].startswith("nmg_")
    assert body["label"] == "bot-prod"
    assert body["tier"] == "pro"


def test_billing_job_status_rejects_other_customer(client):
    owner_headers, owner = _auth_headers_for_tier(Tier.SELF_HOSTED, email="job-owner@example.com")
    other_headers, _other = _auth_headers_for_tier(Tier.SELF_HOSTED, email="job-other@example.com")
    create = client.post("/billing/provision", headers=owner_headers, json={"provider": "on_prem"})
    assert create.status_code == 200

    r = client.get(f"/billing/jobs/{create.json()['job_id']}", headers=other_headers)

    assert r.status_code == 403
    assert r.json()["detail"] == "not your job"
    assert billing_db.get_customer(owner.id).email == "job-owner@example.com"


def test_providers_catalog_endpoint_lists_known_providers(client):
    r = client.get("/providers")

    assert r.status_code == 200
    providers = r.json()["providers"]
    assert any(provider["name"] == "vastai" for provider in providers)
    assert all("supports_provisioning" in provider for provider in providers)


def test_provider_offers_filters_fits_and_ignores_provider_errors(client, monkeypatch):
    small = Offer(ProviderName.VAST_AI, "RTX 3060", 12, 0.03, "US", offer_id="small")
    fit = Offer(ProviderName.VAST_AI, "RTX 3090", 24, 0.07, "US", offer_id="fit")
    provider = FakeProvider([small, fit])
    registry = FakeProviderRegistry([provider, BrokenOfferProvider()])
    monkeypatch.setattr(srv, "providers_registry", lambda: registry)

    r = client.get("/providers/offers")

    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["offers"][0]["offer_id"] == "fit"
    assert body["cascade_vram_requirement_gb"] == 24.0

    r_all = client.get(
        "/providers/offers",
        params={"only_fits": False, "gpu_model": "RTX 3060", "max_price_usd": 0.04},
    )
    assert r_all.status_code == 200
    assert [offer["offer_id"] for offer in r_all.json()["offers"]] == ["small"]
    assert provider.list_calls[-1] == {"gpu_model": "RTX 3060", "max_price_usd": 0.04}


def test_billing_provision_cheapest_requires_self_hosted(client):
    headers, _customer = _auth_headers_for_tier(Tier.PRO, email="pro-cheapest@example.com")

    r = client.post(
        "/billing/provision/cheapest",
        headers=headers,
        json={"max_price_usd": 0.10},
    )

    assert r.status_code == 402
    assert "self-hosted provisioning requires" in r.json()["detail"]


def test_billing_provision_cheapest_persists_job(client, monkeypatch):
    headers, customer = _auth_headers_for_tier(Tier.SELF_HOSTED, email="cheapest@example.com")
    _fund_gpu_credits(customer, 10_000)
    offer = Offer(ProviderName.VAST_AI, "RTX 3090", 24, 0.07, "US", offer_id="fit")
    instance = Instance(
        provider=ProviderName.VAST_AI,
        instance_id="vast-1",
        gpu_model="RTX 3090",
        vram_gb=24,
        region="US",
        state=InstanceState.LIVE,
        endpoint_url="https://vast.test",
        ssh_command="ssh root@vast.test",
    )
    provider = FakeProvider([offer], instance=instance)
    monkeypatch.setattr(srv, "providers_registry", lambda: FakeProviderRegistry([provider]))

    r = client.post(
        "/billing/provision/cheapest",
        headers=headers,
        json={
            "ssh_public_key": "ssh-ed25519 AAAA test",
            "image": "nemoguardian:test",
            "max_price_usd": 0.10,
        },
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "live"
    assert body["instance_id"] == "vast-1"
    assert body["endpoint_url"] == "https://vast.test"
    assert body["gpu_credit_reserved_cents"] == 21
    assert body["gpu_credit_balance_cents"] == 9_979
    assert provider.provision_calls[0]["ssh_public_key"] == "ssh-ed25519 AAAA test"
    assert provider.provision_calls[0]["image"] == "nemoguardian:test"
    assert provider.provision_calls[0]["env"]["NEMOGUARDIAN_CUSTOMER_ID"] == str(customer.id)
    assert provider.provision_calls[0]["env"]["NEMOGUARDIAN_API_KEY"].startswith("nmg_")
    job = billing_db.get_provisioning_job(body["job_id"])
    assert job.customer_id == customer.id
    assert job.status == "live"
    assert billing_db.gpu_credit_balance_cents(customer.id) == 9_979


def test_billing_provision_cheapest_reports_provider_failure(client, monkeypatch):
    headers, customer = _auth_headers_for_tier(
        Tier.SELF_HOSTED, email="cheapest-fail@example.com"
    )
    _fund_gpu_credits(customer, 10_000)
    offer = Offer(ProviderName.VAST_AI, "RTX 3090", 24, 0.07, "US", offer_id="fit")
    provider = FakeProvider([offer], provision_error=RuntimeError("no capacity"))
    monkeypatch.setattr(srv, "providers_registry", lambda: FakeProviderRegistry([provider]))

    r = client.post("/billing/provision/cheapest", headers=headers, json={})

    assert r.status_code == 502
    assert "provisioning failed: no capacity" in r.json()["detail"]
    assert billing_db.gpu_credit_balance_cents(customer.id) == 10_000


def test_billing_provision_cheapest_requires_gpu_credits(client, monkeypatch):
    headers, _customer = _auth_headers_for_tier(
        Tier.SELF_HOSTED, email="no-credits@example.com"
    )
    offer = Offer(ProviderName.VAST_AI, "RTX 3090", 24, 0.07, "US", offer_id="fit")
    provider = FakeProvider([offer])
    monkeypatch.setattr(srv, "providers_registry", lambda: FakeProviderRegistry([provider]))

    r = client.post("/billing/provision/cheapest", headers=headers, json={})

    assert r.status_code == 402
    assert r.json()["detail"]["error"] == "insufficient_gpu_credits"
    assert provider.provision_calls == []


def test_billing_provision_specific_validates_provider_and_offer_filters(client, monkeypatch):
    headers, _customer = _auth_headers_for_tier(Tier.SELF_HOSTED, email="specific@example.com")
    provider = FakeProvider([Offer(ProviderName.VAST_AI, "RTX 3060", 12, 0.03, "US")])
    monkeypatch.setattr(srv, "providers_registry", lambda: FakeProviderRegistry([provider]))

    unknown = client.post("/billing/provision/not-a-provider", headers=headers, json={})
    assert unknown.status_code == 400

    no_fit = client.post("/billing/provision/vastai", headers=headers, json={})
    assert no_fit.status_code == 404
    assert "no fitting offers" in no_fit.json()["detail"]


def test_billing_provision_specific_requires_self_hosted(client):
    headers, _customer = _auth_headers_for_tier(Tier.PRO, email="specific-pro@example.com")

    r = client.post("/billing/provision/vastai", headers=headers, json={})

    assert r.status_code == 402
    assert r.json()["detail"] == "self-hosted provisioning requires the self_hosted plan"


def test_billing_provision_specific_persists_selected_offer(client, monkeypatch):
    headers, customer = _auth_headers_for_tier(Tier.SELF_HOSTED, email="specific-success@example.com")
    _fund_gpu_credits(customer, 10_000)
    cheap = Offer(ProviderName.VAST_AI, "RTX 3090", 24, 0.07, "US", offer_id="cheap")
    selected = Offer(ProviderName.VAST_AI, "RTX 4090", 24, 0.12, "US", offer_id="selected")
    provider = FakeProvider([cheap, selected])
    monkeypatch.setattr(srv, "providers_registry", lambda: FakeProviderRegistry([provider]))

    r = client.post(
        "/billing/provision/vastai",
        headers=headers,
        json={
            "offer_id": "selected",
            "gpu_model": "RTX 4090",
            "ssh_public_key": "ssh-ed25519 AAAA test",
            "image": "nemoguardian:test",
        },
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "live"
    assert body["instance_id"] == "fake-instance"
    assert body["gpu_credit_reserved_cents"] == 36
    assert body["gpu_credit_balance_cents"] == 9_964
    assert provider.provision_calls[0]["offer"].offer_id == "selected"
    assert provider.provision_calls[0]["ssh_public_key"] == "ssh-ed25519 AAAA test"
    assert provider.provision_calls[0]["image"] == "nemoguardian:test"
    assert provider.provision_calls[0]["env"]["NEMOGUARDIAN_CUSTOMER_ID"] == str(customer.id)


def test_billing_provision_specific_reports_provider_error(client, monkeypatch):
    headers, customer = _auth_headers_for_tier(
        Tier.SELF_HOSTED, email="specific-error@example.com"
    )
    _fund_gpu_credits(customer, 10_000)
    provider = FakeProvider(
        [Offer(ProviderName.VAST_AI, "RTX 3090", 24, 0.07, "US", offer_id="fit")],
        provision_error=ProvisionError("denied"),
    )
    monkeypatch.setattr(srv, "providers_registry", lambda: FakeProviderRegistry([provider]))

    r = client.post("/billing/provision/vastai", headers=headers, json={})

    assert r.status_code == 502
    assert "vastai provisioning failed: denied" in r.json()["detail"]
    assert billing_db.gpu_credit_balance_cents(customer.id) == 10_000


def _signed_payload(payload: dict, secret: str, *, timestamp: int | None = None) -> tuple[bytes, str]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ts = int(time.time()) if timestamp is None else timestamp
    digest = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return body, f"t={ts},v1={digest}"
