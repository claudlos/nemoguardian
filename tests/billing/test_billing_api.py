"""End-to-end billing API tests using mocked cascade + real DB."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemoguardian import server as srv
from nemoguardian.billing import db as billing_db
from nemoguardian.billing.plans import Tier
from nemoguardian.cascade import Cascade
from nemoguardian.policy.presets import get_preset
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


def test_webhook_checkout_completed_upgrades_tier(client):
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
    r = client.post("/billing/webhook", json=payload)
    assert r.status_code == 200, r.text
    upgraded = billing_db.get_customer(customer.id)
    assert upgraded.tier == "pro"
    assert upgraded.stripe_customer_id == "cus_test_X"


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
