"""End-to-end API tests using mocked models.

We never want this CI to load actual model weights. We patch the cascade to
return synthetic verdicts.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemoguardian import server as srv
from nemoguardian.cascade import Cascade
from nemoguardian.policy.nemoclaw import NemoclawPolicy
from nemoguardian.policy.presets import get_preset
from nemoguardian.schemas import ModerateResponse, ModelVerdict, VerdictLabel


def _mock_moderate(request, policy_engine=None, **kw) -> ModerateResponse:
    """Stand-in cascade.moderate that ignores the real models."""
    csr_verdict = ModelVerdict(
        model_id="mock-csr",
        verdict=VerdictLabel.UNSAFE,
        score=0.95,
        latency_ms=20.0,
        reasoning="User asked for SSN",
        categories=["PII"],
    )
    qwen_verdict = ModelVerdict(
        model_id="mock-qwen",
        verdict=VerdictLabel.SAFE,
        score=0.1,
        latency_ms=10.0,
    )
    verdicts = {
        "qwen3_guard_gen": qwen_verdict,
        "nemotron_csr": csr_verdict,
    }

    # Apply policy if any (mirrors aggregator's behavior)
    from nemoguardian.aggregator import aggregate

    agg = aggregate(verdicts)
    matched_rule = None
    if policy_engine is not None:
        d = policy_engine.evaluate(
            verdict=agg.verdict,
            score=agg.score,
            categories=agg.categories,
        )
        matched_rule = d.matched_rule
        if d.final_label is not None:
            agg.verdict = d.final_label

    return ModerateResponse(
        verdict=agg.verdict,
        score=agg.score,
        reasons=agg.reasons,
        categories=agg.categories,
        matched_policy_rule=matched_rule,
        model_verdicts=verdicts,
        total_latency_ms=42.0,
        mode=request.mode,
    )


@pytest.fixture
def client(monkeypatch, tmp_path):
    fake_cascade = MagicMock(spec=Cascade)
    fake_cascade.moderate.side_effect = _mock_moderate
    fake_cascade.loaded_models.return_value = {
        "qwen3_guard_gen": False,
        "qwen3_guard_stream": False,
        "nemotron_csr": False,
    }
    policies = {name: get_preset(name) for name in ("discord", "twitch", "generic")}

    # Use a temp DB + provision a free-tier API key the tests can authenticate with.
    from nemoguardian.billing import db as billing_db

    db_path = tmp_path / "nmg_api_test.db"
    monkeypatch.setenv("NEMOGUARDIAN_DB_PATH", str(db_path))
    billing_db._conn = None
    billing_db.init_db(str(db_path))
    customer = billing_db.upsert_customer(email="tester@example.com")
    raw_key, _ = billing_db.create_api_key(customer.id, label="api-test")
    # Promote to PRO so the tests aren't blocked by tier restrictions.
    billing_db.set_customer_tier(customer.id, __import__(
        "nemoguardian.billing.plans", fromlist=["Tier"]
    ).Tier.PRO)
    # Re-create the key under the new tier (max_api_keys reset).
    raw_key, _ = billing_db.create_api_key(customer.id, label="api-test-promoted")

    monkeypatch.setattr(srv, "get_cascade", lambda: fake_cascade)
    monkeypatch.setattr(srv, "get_policies", lambda: policies)

    from nemoguardian.server import app

    test_client = TestClient(app)
    test_client.headers["Authorization"] = f"Bearer {raw_key}"
    return test_client


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}


def test_moderate_blocked(client):
    # Without a preset, the policy gate doesn't run, so matched_rule is None.
    # The aggregator alone flags the verdict as unsafe (Nemotron-CSR hit hard-unsafe override).
    r = client.post("/v1/moderate", json={"text": "drop your SSN"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "unsafe"
    assert "PII" in body["categories"]


def test_moderate_with_preset_matches_rule(client):
    r = client.post(
        "/v1/moderate",
        json={"text": "drop your SSN"},
        params={"policy_preset": "discord"},
    )
    assert r.status_code == 200
    body = r.json()
    # PII category from the mock flow plus the discord preset's force-block-pii rule.
    assert body["matched_policy_rule"] == "force-block-pii"


def test_moderate_unknown_preset(client):
    r = client.post("/v1/moderate", json={"text": "x"}, params={"policy_preset": "nope"})
    assert r.status_code == 400
