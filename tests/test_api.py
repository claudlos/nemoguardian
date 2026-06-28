"""End-to-end API tests using mocked models.

We never want this CI to load actual model weights. We patch the cascade to
return synthetic verdicts.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemoguardian import server as srv
from nemoguardian.cascade import Cascade
from nemoguardian.policy.presets import get_preset
from nemoguardian.schemas import ModelVerdict, ModerateResponse, VerdictLabel


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
    fake_cascade.stream_token_verdicts.return_value = [
        (VerdictLabel.SAFE, 0.05, 0),
        (VerdictLabel.UNSAFE, 0.80, 3),
    ]
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
        "configured": True,
        "provider": "nvidia",
        "model": "mock-triage",
        "base_url": "https://example.test/v1",
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
    assert body["model_config"]["qwen_gen_model"] == "mock-qwen"
    assert body["triage_configured"] is True
    assert body["triage_provider"] == "nvidia"
    assert "runtime_device" in body


def test_health_falls_back_to_cpu_when_torch_probe_fails(client, monkeypatch):
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(
        is_available=lambda: (_ for _ in ()).throw(RuntimeError("cuda probe failed"))
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    r = client.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["gpu_available"] is False
    assert body["gpu_name"] is None
    assert body["runtime_device"] == "cpu"


def test_state_bootstrap_builds_cascade_and_policy_cache(monkeypatch):
    created: list[object] = []
    fake_cascade = SimpleNamespace()

    def fake_cascade_factory(config):
        created.append(config)
        return fake_cascade

    monkeypatch.setattr(srv._State, "cascade", None)
    monkeypatch.setattr(srv._State, "policies", {})
    monkeypatch.setattr(srv, "Cascade", fake_cascade_factory)
    monkeypatch.setattr(srv.CascadeConfig, "from_env", staticmethod(lambda: "config-from-env"))
    monkeypatch.setattr(srv, "PRESETS", {"discord": object(), "twitch": object()})
    monkeypatch.setattr(srv, "get_preset", lambda name: f"policy-{name}")

    assert srv.get_cascade() is fake_cascade
    assert srv.get_policies() == {"discord": "policy-discord", "twitch": "policy-twitch"}
    assert created == ["config-from-env"]


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


def test_moderate_accepts_inline_policy_yaml(client):
    policy_yaml = """
name: inline-test
rules:
  - id: inline-block-pii
    when:
      categories_include: ["PII"]
    then:
      final_label: unsafe
"""

    r = client.post(
        "/v1/moderate",
        json={"text": "drop your SSN"},
        params={"policy_yaml": policy_yaml},
    )

    assert r.status_code == 200
    assert r.json()["matched_policy_rule"] == "inline-block-pii"


def test_moderate_returns_500_on_cascade_error(client):
    srv.get_cascade().moderate.side_effect = RuntimeError("cascade down")

    r = client.post("/v1/moderate", json={"text": "drop your SSN"})

    assert r.status_code == 500
    assert "moderation failed: cascade down" in r.json()["detail"]


def test_moderate_returns_429_when_allowance_exceeded(client, monkeypatch):
    monkeypatch.setattr(
        srv.billing_metered,
        "check_allowance",
        lambda _customer_id: (
            False,
            {
                "total_calls": 50_000,
                "allowance": 50_000,
                "overage_calls": 1,
                "overage_cents": 0.1,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-07-01T00:00:00+00:00",
            },
        ),
    )

    r = client.post("/v1/moderate", json={"text": "drop your SSN"})

    assert r.status_code == 429
    assert r.json()["detail"]["error"] == "monthly allowance exceeded"
    assert r.json()["detail"]["upgrade_url"] == "/billing/checkout?plan=scale"


def test_demo_moderate_disabled_by_default(client):
    client.headers.pop("Authorization", None)
    r = client.post("/demo/moderate", json={"text": "drop your SSN"})
    assert r.status_code == 404


def test_demo_moderate_does_not_require_api_key_when_enabled(client, monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT", "1")
    client.headers.pop("Authorization", None)
    r = client.post(
        "/demo/moderate",
        json={"text": "drop your SSN", "policy": "block PII"},
        params={"policy_preset": "discord"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "unsafe"
    assert body["matched_policy_rule"] == "force-block-pii"


def test_demo_moderate_can_be_disabled(client, monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT", "0")
    client.headers.pop("Authorization", None)
    r = client.post("/demo/moderate", json={"text": "hello"})
    assert r.status_code == 404


def test_demo_moderate_returns_500_on_cascade_error(client, monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT", "yes")
    client.headers.pop("Authorization", None)
    srv.get_cascade().moderate.side_effect = RuntimeError("demo down")

    r = client.post("/demo/moderate", json={"text": "hello"})

    assert r.status_code == 500
    assert "demo moderation failed: demo down" in r.json()["detail"]


def test_demo_endpoint_enabled_accepts_truthy_values(monkeypatch):
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT", value)
        assert srv._demo_endpoint_enabled() is True


def test_demo_ui_served(client):
    r = client.get("/demo")
    assert r.status_code == 200
    assert "Moderation Console" in r.text


def test_stream_moderation_is_authenticated_and_metered(client):
    r = client.post("/v1/moderate/stream", json={"text": "drop your SSN"})
    assert r.status_code == 200

    lines = [json.loads(line) for line in r.text.strip().splitlines()]
    assert lines[0]["verdict_so_far"] == "safe"
    assert lines[1]["verdict_so_far"] == "unsafe"
    assert lines[-1]["is_terminal"] is True

    from nemoguardian.billing import db as billing_db
    from nemoguardian.billing import metered as billing_metered

    customer = billing_db.get_customer_by_email("tester@example.com")
    assert customer is not None
    _allowed, usage = billing_metered.check_allowance(customer.id)
    assert usage["total_calls"] == 1


def test_stream_moderation_returns_terminal_safe_chunk_on_error(client):
    srv.get_cascade().stream_token_verdicts.side_effect = RuntimeError("stream down")

    r = client.post("/v1/moderate/stream", json={"text": "drop your SSN"})

    assert r.status_code == 200
    lines = [json.loads(line) for line in r.text.strip().splitlines()]
    assert lines == [
        {
            "token_index": 0,
            "partial_text": "drop your SSN",
            "verdict_so_far": "safe",
            "score_so_far": 0.0,
            "is_terminal": True,
            "latency_ms": 0.0,
        }
    ]


def test_stream_moderation_returns_429_when_allowance_exceeded(client, monkeypatch):
    monkeypatch.setattr(
        srv.billing_metered,
        "check_allowance",
        lambda _customer_id: (
            False,
            {
                "total_calls": 50_000,
                "allowance": 50_000,
                "overage_calls": 1,
                "overage_cents": 0.1,
                "period_start": "2026-06-01T00:00:00+00:00",
                "period_end": "2026-07-01T00:00:00+00:00",
            },
        ),
    )

    r = client.post("/v1/moderate/stream", json={"text": "drop your SSN"})

    assert r.status_code == 429
    assert r.json()["detail"]["error"] == "monthly allowance exceeded"
