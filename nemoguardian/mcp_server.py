"""Model Context Protocol (MCP) server exposing the cascade as agent tools.

Any MCP client — Hermes, Claude, Codex, or your own agent — can call:

- ``moderate_text(text, mode, policy)`` — screen content through the NemoGuardian
  cascade and get back a verdict, score, reasons, and categories.
- ``gpu_credit_balance(customer_id)`` — read a self-hosted GPU-credit wallet.

Run it with ``nemoguardian mcp`` (stdio transport).

The ``mcp`` package is an optional dependency (``pip install -e ".[mcp]"``). The
tool *logic* lives in plain module functions so it stays importable and testable
without the package; only ``build_server()`` / ``run()`` need ``mcp`` installed.
"""

from __future__ import annotations

from typing import Any

from nemoguardian.schemas import Mode, VerdictLabel

SERVER_NAME = "nemoguardian"

_CASCADE: Any = None


def _get_cascade() -> Any:
    """Lazily build and cache the cascade singleton (models load on first call)."""
    global _CASCADE
    if _CASCADE is None:
        from nemoguardian.cascade import Cascade, CascadeConfig

        _CASCADE = Cascade(CascadeConfig.from_env())
    return _CASCADE


def moderate_text(
    text: str,
    mode: str = "standard",
    policy: str | None = None,
) -> dict[str, Any]:
    """Screen one piece of text through the NemoGuardian moderation cascade.

    Args:
        text: The content to classify.
        mode: ``fast`` | ``standard`` | ``deep`` (unknown values fall back to standard).
        policy: Optional custom policy, e.g. ``"no financial advice"``.

    Returns a dict with the final ``verdict`` (safe/controversial/unsafe), the
    aggregate ``score``, ``categories``, ``reasons``, and a ``blocked`` flag
    (true when the verdict is ``unsafe``). Agents should not act on or post
    content when ``blocked`` is true.
    """
    from nemoguardian.policy.presets import get_preset
    from nemoguardian.schemas import ModerateRequest

    try:
        mode_enum = Mode(mode)
    except ValueError:
        mode_enum = Mode.STANDARD

    result = _get_cascade().moderate(
        ModerateRequest(text=text, policy=policy, mode=mode_enum),
        policy_engine=get_preset("generic"),
    )
    return {
        "verdict": result.verdict.value,
        "score": result.score,
        "blocked": result.verdict == VerdictLabel.UNSAFE,
        "categories": list(result.categories),
        "reasons": list(result.reasons),
        "mode": result.mode.value,
        "latency_ms": result.total_latency_ms,
    }


def gpu_credit_balance(customer_id: str) -> dict[str, Any]:
    """Return the GPU-credit wallet balance for a self-hosted customer id.

    Reads the Stripe-funded GPU-credit ledger so an autonomous operator can check
    its own balance before reserving rented GPU capacity.
    """
    from nemoguardian.billing import db as billing_db

    balance_fn = getattr(billing_db, "gpu_credit_balance_cents", None)
    if not callable(balance_fn):
        return {
            "customer_id": customer_id,
            "available": False,
            "balance_cents": None,
            "balance_usd": None,
            "recent_event_count": 0,
            "error": "GPU-credit ledger helpers are not available in this build",
        }

    balance = int(balance_fn(customer_id))
    try:
        events_fn = getattr(billing_db, "list_gpu_credit_events", None)
        recent = events_fn(customer_id, limit=10) if callable(events_fn) else []
        recent_count = len(list(recent))
    except Exception:
        recent_count = 0
    return {
        "customer_id": customer_id,
        "available": True,
        "balance_cents": balance,
        "balance_usd": round(balance / 100.0, 2),
        "recent_event_count": recent_count,
    }


def build_server() -> Any:
    """Construct the FastMCP server with the cascade tools registered.

    Requires the ``mcp`` extra; raises ImportError with a clear hint otherwise.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "MCP support requires the 'mcp' extra: pip install -e \".[mcp]\""
        ) from exc

    server = FastMCP(SERVER_NAME)
    server.tool()(moderate_text)
    server.tool()(gpu_credit_balance)
    return server


def run(transport: str = "stdio") -> None:  # pragma: no cover - process entrypoint
    """Run the MCP server (default stdio transport)."""
    build_server().run(transport=transport)


__all__ = ["SERVER_NAME", "build_server", "gpu_credit_balance", "moderate_text", "run"]
