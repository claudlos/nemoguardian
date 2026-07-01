"""Tests for production API hardening: CORS allowlist, rate limiting, body cap.

These build small isolated FastAPI apps (with a stub ``/v1/moderate`` route) so
they exercise the real middleware without loading models, hitting the billing
DB, or depending on the module-level singleton app's import-time CORS config.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemoguardian import middleware as mw


def _make_app(limiter: mw.FixedWindowRateLimiter | None = None) -> FastAPI:
    app = FastAPI()
    mw.install_hardening(app, limiter=limiter or mw.FixedWindowRateLimiter())

    @app.post("/v1/moderate")
    async def _moderate(payload: dict) -> dict:  # pragma: no cover - trivial
        return {"ok": True}

    @app.get("/health")
    async def _health() -> dict:  # pragma: no cover - trivial
        return {"status": "ok"}

    return app


# --- CORS -----------------------------------------------------------------


def test_cors_defaults_to_localhost_allowlist(monkeypatch):
    monkeypatch.delenv("NEMOGUARDIAN_CORS_ORIGINS", raising=False)
    assert mw.cors_origins() == list(mw.DEFAULT_CORS_ORIGINS)
    assert "*" not in mw.cors_origins()


def test_cors_env_allowlist_is_parsed(monkeypatch):
    monkeypatch.setenv(
        "NEMOGUARDIAN_CORS_ORIGINS",
        "https://app.example.com, https://admin.example.com",
    )
    assert mw.cors_origins() == [
        "https://app.example.com",
        "https://admin.example.com",
    ]


def test_cors_echoes_configured_origin_not_wildcard(monkeypatch):
    origin = "https://app.example.com"
    monkeypatch.setenv("NEMOGUARDIAN_CORS_ORIGINS", origin)
    client = TestClient(_make_app())

    r = client.post("/v1/moderate", json={"text": "hi"}, headers={"Origin": origin})

    assert r.status_code == 200
    allow_origin = r.headers["access-control-allow-origin"]
    assert allow_origin == origin
    assert allow_origin != "*"


def test_cors_rejects_unlisted_origin(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_CORS_ORIGINS", "https://app.example.com")
    client = TestClient(_make_app())

    r = client.post(
        "/v1/moderate", json={"text": "hi"}, headers={"Origin": "https://evil.example.com"}
    )

    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_cors_wildcard_only_when_explicit(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_CORS_ORIGINS", "*")
    assert mw.cors_origins() == ["*"]


# --- Rate limiting --------------------------------------------------------


def test_rate_limiter_returns_429_past_limit_then_resets(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_RATE_LIMIT", "2")
    monkeypatch.setenv("NEMOGUARDIAN_RATE_WINDOW_SECONDS", "60")
    limiter = mw.FixedWindowRateLimiter()
    client = TestClient(_make_app(limiter))

    assert client.post("/v1/moderate", json={"text": "a"}).status_code == 200
    assert client.post("/v1/moderate", json={"text": "b"}).status_code == 200

    blocked = client.post("/v1/moderate", json={"text": "c"})
    assert blocked.status_code == 429
    body = blocked.json()
    assert body["detail"] == "rate limit exceeded"
    assert body["limit"] == 2
    assert "retry-after" in {k.lower() for k in blocked.headers}

    # Resetting the window state lets traffic flow again.
    limiter.reset()
    assert client.post("/v1/moderate", json={"text": "d"}).status_code == 200


def test_rate_limiter_fails_open_when_unconfigured(monkeypatch):
    monkeypatch.delenv("NEMOGUARDIAN_RATE_LIMIT", raising=False)
    client = TestClient(_make_app())
    for _ in range(20):
        assert client.post("/v1/moderate", json={"text": "x"}).status_code == 200


def test_rate_limiter_only_guards_moderation_paths(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_RATE_LIMIT", "1")
    client = TestClient(_make_app())
    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_fixed_window_resets_after_window_elapses():
    limiter = mw.FixedWindowRateLimiter()
    allowed, _ = limiter.check("k", limit=1, window_seconds=10, now=100.0)
    assert allowed is True
    allowed, retry = limiter.check("k", limit=1, window_seconds=10, now=101.0)
    assert allowed is False
    assert retry > 0
    # After the window elapses the counter resets.
    allowed, _ = limiter.check("k", limit=1, window_seconds=10, now=200.0)
    assert allowed is True


def test_fixed_window_keys_are_independent():
    limiter = mw.FixedWindowRateLimiter()
    assert limiter.check("a", limit=1, window_seconds=10, now=0.0)[0] is True
    # Different key is unaffected by "a" being at its limit.
    assert limiter.check("b", limit=1, window_seconds=10, now=0.0)[0] is True
    assert limiter.check("a", limit=1, window_seconds=10, now=0.0)[0] is False


# --- Body-size limit ------------------------------------------------------


def test_oversized_body_returns_413(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_MAX_BODY_BYTES", "10")
    client = TestClient(_make_app())

    r = client.post("/v1/moderate", json={"text": "x" * 500})

    assert r.status_code == 413
    body = r.json()
    assert body["detail"] == "request body too large"
    assert body["max_bytes"] == 10


def test_body_under_limit_passes(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_MAX_BODY_BYTES", "10000")
    client = TestClient(_make_app())
    r = client.post("/v1/moderate", json={"text": "hi"})
    assert r.status_code == 200


def test_body_limit_disabled_when_zero(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_MAX_BODY_BYTES", "0")
    assert mw.max_body_bytes() is None
    client = TestClient(_make_app())
    r = client.post("/v1/moderate", json={"text": "x" * 5000})
    assert r.status_code == 200


def test_body_size_only_guards_moderation_paths(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_MAX_BODY_BYTES", "1")
    client = TestClient(_make_app())
    # /health is unprotected, so a large query is irrelevant; it stays 200.
    assert client.get("/health").status_code == 200


# --- Config parsing edge cases --------------------------------------------


@pytest.mark.parametrize("raw", ["notanint", "-5", "0", ""])
def test_rate_limit_invalid_or_nonpositive_disables(monkeypatch, raw):
    monkeypatch.setenv("NEMOGUARDIAN_RATE_LIMIT", raw)
    assert mw.rate_limit() is None


def test_rate_window_falls_back_on_bad_value(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_RATE_WINDOW_SECONDS", "nope")
    assert mw.rate_window_seconds() == mw.DEFAULT_RATE_WINDOW_SECONDS


def test_is_protected_path():
    assert mw.is_protected_path("/v1/moderate")
    assert mw.is_protected_path("/v1/moderate/stream")
    assert mw.is_protected_path("/demo/moderate")
    assert not mw.is_protected_path("/health")
    assert not mw.is_protected_path("/billing/usage")
