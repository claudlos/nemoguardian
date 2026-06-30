"""MCP server tool-logic tests (the `mcp` package is not required for these)."""

from __future__ import annotations

from types import SimpleNamespace

import nemoguardian.mcp_server as mcp_server
from nemoguardian.schemas import Mode, VerdictLabel


class _FakeCascade:
    def __init__(self) -> None:
        self.requests: list = []

    def moderate(self, request, *, policy_engine=None):
        self.requests.append((request, policy_engine))
        return SimpleNamespace(
            verdict=VerdictLabel.UNSAFE,
            score=0.93,
            categories=["PII"],
            reasons=["matched an SSN-shaped pattern"],
            mode=request.mode,
            total_latency_ms=42.0,
        )


def test_moderate_text_returns_blocked_verdict(monkeypatch):
    fake = _FakeCascade()
    monkeypatch.setattr(mcp_server, "_get_cascade", lambda: fake)

    out = mcp_server.moderate_text("drop your SSN for $100", mode="standard")

    assert out["verdict"] == "unsafe"
    assert out["blocked"] is True
    assert out["categories"] == ["PII"]
    assert out["mode"] == "standard"
    assert out["latency_ms"] == 42.0
    # policy preset is wired through
    assert fake.requests[0][0].text == "drop your SSN for $100"


def test_moderate_text_unknown_mode_falls_back_to_standard(monkeypatch):
    fake = _FakeCascade()
    monkeypatch.setattr(mcp_server, "_get_cascade", lambda: fake)

    mcp_server.moderate_text("hi", mode="bogus")

    assert fake.requests[0][0].mode == Mode.STANDARD


def test_gpu_credit_balance_shape(monkeypatch):
    import nemoguardian.billing.db as billing_db

    monkeypatch.setattr(billing_db, "gpu_credit_balance_cents", lambda cid: 2500)
    monkeypatch.setattr(
        billing_db, "list_gpu_credit_events", lambda cid, limit=10: [object(), object()]
    )

    out = mcp_server.gpu_credit_balance("cust_123")

    assert out == {
        "customer_id": "cust_123",
        "balance_cents": 2500,
        "balance_usd": 25.0,
        "recent_event_count": 2,
    }
