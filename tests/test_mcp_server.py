"""MCP server tool-logic tests (the `mcp` package is not required for these)."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

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


def test_get_cascade_builds_and_caches_singleton(monkeypatch):
    import nemoguardian.cascade as cascade_mod

    built: list[object] = []

    class FakeConfig:
        @classmethod
        def from_env(cls):
            return "config-from-env"

    class FakeCascade:
        def __init__(self, config):
            self.config = config
            built.append(self)

    monkeypatch.setattr(mcp_server, "_CASCADE", None)
    monkeypatch.setattr(cascade_mod, "CascadeConfig", FakeConfig)
    monkeypatch.setattr(cascade_mod, "Cascade", FakeCascade)

    first = mcp_server._get_cascade()
    second = mcp_server._get_cascade()

    assert first is second
    assert len(built) == 1
    assert first.config == "config-from-env"


def test_gpu_credit_balance_shape(monkeypatch):
    import nemoguardian.billing.db as billing_db

    monkeypatch.setattr(billing_db, "gpu_credit_balance_cents", lambda cid: 2500, raising=False)
    monkeypatch.setattr(
        billing_db,
        "list_gpu_credit_events",
        lambda cid, limit=10: [object(), object()],
        raising=False,
    )

    out = mcp_server.gpu_credit_balance("cust_123")

    assert out == {
        "customer_id": "cust_123",
        "available": True,
        "balance_cents": 2500,
        "balance_usd": 25.0,
        "recent_event_count": 2,
    }


def test_gpu_credit_balance_tolerates_recent_event_errors(monkeypatch):
    import nemoguardian.billing.db as billing_db

    def boom(_customer_id, *, limit=10):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(billing_db, "gpu_credit_balance_cents", lambda cid: 1250, raising=False)
    monkeypatch.setattr(billing_db, "list_gpu_credit_events", boom, raising=False)

    out = mcp_server.gpu_credit_balance("cust_123")

    assert out["balance_cents"] == 1250
    assert out["balance_usd"] == 12.5
    assert out["recent_event_count"] == 0


def test_gpu_credit_balance_reports_unavailable_without_ledger(monkeypatch):
    import nemoguardian.billing.db as billing_db

    monkeypatch.delattr(billing_db, "gpu_credit_balance_cents", raising=False)
    monkeypatch.delattr(billing_db, "list_gpu_credit_events", raising=False)

    out = mcp_server.gpu_credit_balance("cust_123")

    assert out == {
        "customer_id": "cust_123",
        "available": False,
        "balance_cents": None,
        "balance_usd": None,
        "recent_event_count": 0,
        "error": "GPU-credit ledger helpers are not available in this build",
    }


def test_build_server_registers_tools(monkeypatch):
    fastmcp_module = ModuleType("mcp.server.fastmcp")
    registered: list[object] = []

    class FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name

        def tool(self):
            def decorator(func):
                registered.append(func)
                return func

            return decorator

    fastmcp_module.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp", ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)

    server = mcp_server.build_server()

    assert server.name == mcp_server.SERVER_NAME
    assert registered == [mcp_server.moderate_text, mcp_server.gpu_credit_balance]


def test_build_server_import_error_has_install_hint(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mcp.server.fastmcp":
            raise ImportError("missing mcp")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        mcp_server.build_server()
    except ImportError as exc:
        assert "pip install -e" in str(exc)
    else:
        raise AssertionError("build_server should raise when mcp is unavailable")
