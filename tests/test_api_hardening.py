"""Tests for production API hardening: CORS allowlist, rate limiting, body cap.

These build small isolated FastAPI apps (with a stub ``/v1/moderate`` route) so
they exercise the real middleware without loading models, hitting the billing
DB, or depending on the module-level singleton app's import-time CORS config.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from nemoguardian import middleware as mw


def _make_app(
    limiter: mw.FixedWindowRateLimiter | None = None,
    *,
    key_validator: Callable[[str], bool] | None = None,
) -> FastAPI:
    app = FastAPI()
    mw.install_hardening(
        app,
        limiter=limiter or mw.FixedWindowRateLimiter(),
        key_validator=key_validator,
    )

    @app.post("/v1/moderate")
    async def _moderate(payload: dict) -> dict:  # pragma: no cover - trivial
        return {"ok": True}

    @app.get("/health")
    async def _health() -> dict:  # pragma: no cover - trivial
        return {"status": "ok"}

    return app


def _headers(**pairs: str) -> Headers:
    return Headers(raw=[(k.encode(), v.encode()) for k, v in pairs.items()])


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


# --- Rate-limit key: spoof / bypass regressions ---------------------------


def test_rotating_random_bearer_does_not_bypass_limit(monkeypatch):
    """A random/unvalidated Authorization header must NOT mint a fresh bucket.

    Regression for the header-spoof bypass: keying on the raw Authorization
    value let an attacker rotate the Bearer every request so the limit never
    tripped. Unvalidated keys now fall back to the (constant) client IP.
    """
    monkeypatch.setenv("NEMOGUARDIAN_RATE_LIMIT", "2")
    monkeypatch.setenv("NEMOGUARDIAN_RATE_WINDOW_SECONDS", "60")
    monkeypatch.delenv("NEMOGUARDIAN_TRUST_PROXY", raising=False)
    client = TestClient(_make_app(mw.FixedWindowRateLimiter()))

    codes = [
        client.post(
            "/v1/moderate",
            json={"text": "x"},
            headers={"Authorization": f"Bearer random-{i}"},
        ).status_code
        for i in range(5)
    ]
    assert codes[:2] == [200, 200]
    assert 429 in codes[2:]


def test_rotating_xff_does_not_bypass_limit(monkeypatch):
    """Untrusted X-Forwarded-For must be ignored for keying (no proxy configured)."""
    monkeypatch.setenv("NEMOGUARDIAN_RATE_LIMIT", "2")
    monkeypatch.setenv("NEMOGUARDIAN_RATE_WINDOW_SECONDS", "60")
    monkeypatch.delenv("NEMOGUARDIAN_TRUST_PROXY", raising=False)
    client = TestClient(_make_app(mw.FixedWindowRateLimiter()))

    codes = [
        client.post(
            "/v1/moderate",
            json={"text": "x"},
            headers={"X-Forwarded-For": f"10.0.0.{i}"},
        ).status_code
        for i in range(5)
    ]
    assert codes[:2] == [200, 200]
    assert 429 in codes[2:]


def test_untrusted_xff_ignored_uses_peer_ip(monkeypatch):
    monkeypatch.delenv("NEMOGUARDIAN_TRUST_PROXY", raising=False)
    scope = {"type": "http", "client": ("203.0.113.7", 5555)}
    headers = _headers(**{"x-forwarded-for": "1.2.3.4, 5.6.7.8"})
    assert mw._client_ip(scope, headers) == "203.0.113.7"


def test_trusted_proxy_honors_configured_hop(monkeypatch):
    # One trusted hop: the real client is the rightmost XFF entry (added by our
    # own proxy); attacker-supplied prefix entries are ignored.
    monkeypatch.setenv("NEMOGUARDIAN_TRUST_PROXY", "1")
    scope = {"type": "http", "client": ("203.0.113.7", 5555)}
    headers = _headers(**{"x-forwarded-for": "9.9.9.9, 203.0.113.42"})
    assert mw._client_ip(scope, headers) == "203.0.113.42"


def test_trusted_proxy_falls_back_when_too_few_hops(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_TRUST_PROXY", "2")
    scope = {"type": "http", "client": ("203.0.113.7", 5555)}
    headers = _headers(**{"x-forwarded-for": "203.0.113.42"})
    # Fewer entries than trusted hops -> untrusted -> peer IP.
    assert mw._client_ip(scope, headers) == "203.0.113.7"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", 1), ("2", 2), ("true", 1), ("on", 1), ("0", 0), ("false", 0), ("nope", 0), ("-3", 0)],
)
def test_trust_proxy_hops_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("NEMOGUARDIAN_TRUST_PROXY", raw)
    assert mw.trust_proxy_hops() == expected


def test_trust_proxy_hops_unset(monkeypatch):
    monkeypatch.delenv("NEMOGUARDIAN_TRUST_PROXY", raising=False)
    assert mw.trust_proxy_hops() == 0


def test_valid_key_is_limited_per_identity_regardless_of_xff(monkeypatch):
    """A validated API key is keyed alone: same key + rotating XFF share a bucket."""
    monkeypatch.setenv("NEMOGUARDIAN_RATE_LIMIT", "2")
    monkeypatch.setenv("NEMOGUARDIAN_RATE_WINDOW_SECONDS", "60")
    monkeypatch.delenv("NEMOGUARDIAN_TRUST_PROXY", raising=False)
    client = TestClient(
        _make_app(mw.FixedWindowRateLimiter(), key_validator=lambda k: k == "nmg_valid")
    )

    codes = [
        client.post(
            "/v1/moderate",
            json={"text": "x"},
            headers={
                "Authorization": "Bearer nmg_valid",
                "X-Forwarded-For": f"10.0.0.{i}",
            },
        ).status_code
        for i in range(5)
    ]
    assert codes[:2] == [200, 200]
    assert 429 in codes[2:]


def test_distinct_valid_keys_get_independent_buckets(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_RATE_LIMIT", "1")
    monkeypatch.setenv("NEMOGUARDIAN_RATE_WINDOW_SECONDS", "60")
    client = TestClient(
        _make_app(
            mw.FixedWindowRateLimiter(),
            key_validator=lambda k: k in {"nmg_a", "nmg_b"},
        )
    )
    ok_a = client.post("/v1/moderate", json={"t": "1"}, headers={"Authorization": "Bearer nmg_a"})
    ok_b = client.post("/v1/moderate", json={"t": "1"}, headers={"Authorization": "Bearer nmg_b"})
    blocked_a = client.post(
        "/v1/moderate", json={"t": "2"}, headers={"Authorization": "Bearer nmg_a"}
    )
    assert ok_a.status_code == 200
    assert ok_b.status_code == 200  # independent key, unaffected by nmg_a's limit
    assert blocked_a.status_code == 429


def test_key_validator_failure_is_fail_safe(monkeypatch):
    """A validator that raises must not break moderation; treat as unauthenticated."""
    monkeypatch.setenv("NEMOGUARDIAN_RATE_LIMIT", "2")

    def _boom(_key: str) -> bool:
        raise RuntimeError("db down")

    client = TestClient(_make_app(mw.FixedWindowRateLimiter(), key_validator=_boom))
    codes = [
        client.post(
            "/v1/moderate", json={"t": "x"}, headers={"Authorization": f"Bearer nmg_{i}"}
        ).status_code
        for i in range(5)
    ]
    # Never a 500; falls back to IP keying so the limit still trips.
    assert 500 not in codes
    assert codes[:2] == [200, 200]
    assert 429 in codes[2:]


# --- Limiter memory safety (eviction / cap) -------------------------------


def test_expired_entries_are_evicted():
    limiter = mw.FixedWindowRateLimiter()
    limiter.check("k", limit=5, window_seconds=10, now=0.0)
    assert "k" in limiter._hits
    # A later check past the window evicts the stale "k" entry.
    limiter.check("other", limit=5, window_seconds=10, now=100.0)
    assert "k" not in limiter._hits
    assert set(limiter._hits) == {"other"}


def test_distinct_key_flood_is_capped():
    limiter = mw.FixedWindowRateLimiter(max_keys=50)
    # All within one window: eviction-by-expiry can't help, so the cap must.
    for i in range(500):
        limiter.check(f"key-{i}", limit=1, window_seconds=3600, now=1.0)
    assert len(limiter._hits) <= 50


# --- Body-size: chunked / missing Content-Length --------------------------


def _run_asgi(app, scope, body_chunks):
    """Drive a raw-ASGI app with a chunked body and collect the response."""
    import asyncio

    messages = [
        {"type": "http.request", "body": chunk, "more_body": i < len(body_chunks) - 1}
        for i, chunk in enumerate(body_chunks)
    ]
    sent: list[dict] = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


def test_chunked_oversized_body_returns_413(monkeypatch):
    """A body with no Content-Length (chunked) must still be capped on real bytes."""
    monkeypatch.setenv("NEMOGUARDIAN_MAX_BODY_BYTES", "10")
    monkeypatch.delenv("NEMOGUARDIAN_RATE_LIMIT", raising=False)

    async def _app(scope, receive, send):  # pragma: no cover - should not run
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    middleware = mw.ModerationHardeningMiddleware(_app)
    scope = {
        "type": "http",
        "path": "/v1/moderate",
        "method": "POST",
        "headers": [],  # note: NO content-length header
        "client": ("203.0.113.7", 5555),
    }
    # 12 bytes across two chunks, no Content-Length -> must 413.
    sent = _run_asgi(middleware, scope, [b"xxxxxx", b"xxxxxx"])
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413


def test_chunked_body_under_limit_passes(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_MAX_BODY_BYTES", "1000")
    monkeypatch.delenv("NEMOGUARDIAN_RATE_LIMIT", raising=False)
    seen: dict[str, bytes] = {}

    async def _app(scope, receive, send):
        body = b""
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        seen["body"] = body
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    middleware = mw.ModerationHardeningMiddleware(_app)
    scope = {
        "type": "http",
        "path": "/v1/moderate",
        "method": "POST",
        "headers": [],
        "client": ("203.0.113.7", 5555),
    }
    sent = _run_asgi(middleware, scope, [b"abc", b"def"])
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 200
    # The downstream app still receives the full replayed body.
    assert seen["body"] == b"abcdef"


# --- Fail-open default visibility -----------------------------------------


def test_fail_open_default_warns_and_stays_unlimited(monkeypatch, caplog):
    monkeypatch.delenv("NEMOGUARDIAN_RATE_LIMIT", raising=False)
    with caplog.at_level("WARNING", logger="nemoguardian.middleware"):
        warned = mw.warn_if_unlimited()
    assert warned is True
    assert any("not rate limited" in r.message.lower() for r in caplog.records)
    # Behavior is still fail-open (unlimited) as specified.
    client = TestClient(_make_app())
    for _ in range(20):
        assert client.post("/v1/moderate", json={"text": "x"}).status_code == 200


def test_warn_if_unlimited_silent_when_configured(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_RATE_LIMIT", "5")
    assert mw.warn_if_unlimited() is False
