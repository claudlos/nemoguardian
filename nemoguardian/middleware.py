"""Production hardening middleware for the FastAPI surface.

Self-host templates expose the moderation endpoints straight to the internet,
so this module adds three cheap, dependency-free guards:

* **CORS allowlist** (``configure_cors``) — replaces a wildcard ``*`` origin with
  an env-configurable allowlist. See ``configure_cors`` for the trade-off.
* **Rate limiting** — a fixed-window in-memory limiter keyed by API key + client
  IP on the moderation endpoints. Fail-open only when unconfigured.
* **Request body-size limit** — rejects oversized moderation payloads with 413.

Everything here is pure Python/Starlette: no new third-party dependency (no
``slowapi``/``redis``). The limiter state lives in-process, which is the right
default for a single-node self-host template; horizontal deployments should
front this with a shared store (documented in ``FixedWindowRateLimiter``).
"""

from __future__ import annotations

import os
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# --- Configuration knobs ---------------------------------------------------

#: Safe localhost origins used when ``NEMOGUARDIAN_CORS_ORIGINS`` is unset.
DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
)

#: Default max moderation request body (bytes) when the env knob is unset.
DEFAULT_MAX_BODY_BYTES = 1_048_576  # 1 MiB

#: Default rate-limit window (seconds) when only a limit is configured.
DEFAULT_RATE_WINDOW_SECONDS = 60.0

#: Moderation paths guarded by the limiter + body-size check. Matched as
#: prefixes so ``/v1/moderate/stream`` is covered by ``/v1/moderate``.
PROTECTED_PREFIXES: tuple[str, ...] = ("/v1/moderate", "/demo/moderate")


def _env(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def cors_origins() -> list[str]:
    """Resolve the configured CORS allowlist.

    ``NEMOGUARDIAN_CORS_ORIGINS`` is a comma-separated list of origins. When
    unset/blank we fall back to a localhost-only allowlist so a fresh deploy is
    safe by default. ``*`` is only honored when explicitly present in the list —
    the operator opts into wildcard CORS knowingly.
    """
    raw = _env("NEMOGUARDIAN_CORS_ORIGINS")
    if raw is None:
        return list(DEFAULT_CORS_ORIGINS)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or list(DEFAULT_CORS_ORIGINS)


def max_body_bytes() -> int | None:
    """Max moderation body size in bytes; ``None`` disables the check.

    ``NEMOGUARDIAN_MAX_BODY_BYTES=0`` (or negative) disables it explicitly.
    """
    raw = _env("NEMOGUARDIAN_MAX_BODY_BYTES")
    if raw is None:
        return DEFAULT_MAX_BODY_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_BODY_BYTES
    return value if value > 0 else None


def rate_limit() -> int | None:
    """Max requests per window; ``None`` (unset/invalid/<=0) disables limiting.

    Fail-open: when no positive limit is configured, the moderation endpoints
    are not rate limited. Any positive value enables the limiter (else 429).
    """
    raw = _env("NEMOGUARDIAN_RATE_LIMIT")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def rate_window_seconds() -> float:
    raw = _env("NEMOGUARDIAN_RATE_WINDOW_SECONDS")
    if raw is None:
        return DEFAULT_RATE_WINDOW_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_RATE_WINDOW_SECONDS
    return value if value > 0 else DEFAULT_RATE_WINDOW_SECONDS


def is_protected_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in PROTECTED_PREFIXES)


# --- Fixed-window rate limiter --------------------------------------------


class FixedWindowRateLimiter:
    """Thread-safe in-memory fixed-window counter.

    Keys map to ``(window_start, count)``. Each ``check`` increments the count
    for the current window and reports whether the caller is within ``limit``.
    State is per-process; a multi-node deployment needs a shared backing store
    (Redis/memcached) instead — out of scope for this single-node template.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, tuple[float, int]] = {}

    def check(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: float,
        now: float | None = None,
    ) -> tuple[bool, float]:
        """Register a hit; return ``(allowed, retry_after_seconds)``."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            start, count = self._hits.get(key, (now, 0))
            if now - start >= window_seconds:
                start, count = now, 0
            count += 1
            self._hits[key] = (start, count)
            allowed = count <= limit
            retry_after = 0.0 if allowed else max(0.0, window_seconds - (now - start))
            return allowed, retry_after

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


#: Process-wide limiter shared by the app's middleware instance. Exposed so
#: tests (and operators poking a REPL) can ``reset()`` between runs.
default_limiter = FixedWindowRateLimiter()


def _client_ip(scope: Scope, headers: Headers) -> str:
    # Honor a single-hop reverse proxy if it sets X-Forwarded-For; otherwise
    # use the direct peer. This is best-effort keying, not auth.
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = scope.get("client")
    if client:
        return str(client[0])
    return "unknown"


def _rate_key(scope: Scope, headers: Headers) -> str:
    api_key = ""
    authorization = headers.get("authorization")
    if authorization:
        parts = authorization.split(" ", 1)
        api_key = parts[1].strip() if len(parts) == 2 else authorization.strip()
    return f"{api_key or 'anon'}|{_client_ip(scope, headers)}"


class ModerationHardeningMiddleware:
    """ASGI middleware enforcing body-size + rate limits on moderation paths.

    Implemented as raw ASGI (not ``BaseHTTPMiddleware``) so it never consumes
    the request body — the downstream handler still reads it normally. The
    body-size check uses the ``Content-Length`` header, which FastAPI/Starlette
    clients always set for JSON payloads; requests without a length are passed
    through (the app's own parsing still bounds memory for JSON bodies).
    """

    def __init__(self, app: ASGIApp, *, limiter: FixedWindowRateLimiter | None = None) -> None:
        self.app = app
        self.limiter = limiter or default_limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_protected_path(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)

        rejection = self._body_too_large(headers) or self._rate_limited(scope, headers)
        if rejection is not None:
            await rejection(scope, receive, send)
            return

        await self.app(scope, receive, send)

    @staticmethod
    def _body_too_large(headers: Headers) -> JSONResponse | None:
        limit = max_body_bytes()
        if limit is None:
            return None
        content_length = headers.get("content-length")
        if content_length is None:
            return None
        try:
            declared = int(content_length)
        except ValueError:
            return None
        if declared <= limit:
            return None
        return JSONResponse(
            status_code=413,
            content={
                "detail": "request body too large",
                "max_bytes": limit,
                "received_bytes": declared,
            },
        )

    def _rate_limited(self, scope: Scope, headers: Headers) -> JSONResponse | None:
        limit = rate_limit()
        if limit is None:  # fail-open: unconfigured
            return None
        window = rate_window_seconds()
        allowed, retry_after = self.limiter.check(
            _rate_key(scope, headers), limit=limit, window_seconds=window
        )
        if allowed:
            return None
        retry_seconds = int(retry_after) + 1
        return JSONResponse(
            status_code=429,
            content={
                "detail": "rate limit exceeded",
                "limit": limit,
                "window_seconds": window,
                "retry_after_seconds": retry_seconds,
            },
            headers={"Retry-After": str(retry_seconds)},
        )


# --- App wiring ------------------------------------------------------------


def configure_cors(app: FastAPI) -> None:
    """Install a CORS allowlist instead of a blanket wildcard.

    Trade-off: ``allow_origins=["*"]`` is convenient but lets any website drive
    the browser-side API on a user's behalf. We default to a localhost allowlist
    and require the operator to opt into additional origins (or ``*``) via
    ``NEMOGUARDIAN_CORS_ORIGINS``. Credentials are intentionally left disabled;
    the API authenticates via ``Authorization: Bearer`` headers, not cookies,
    so wildcard-vs-credentials incompatibility never bites us.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )


def install_hardening(app: FastAPI, *, limiter: FixedWindowRateLimiter | None = None) -> None:
    """Attach body-size + rate-limit guards, then the CORS allowlist.

    CORS is added last so it wraps outermost and can short-circuit preflight
    ``OPTIONS`` before the limiter counts them.
    """
    app.add_middleware(ModerationHardeningMiddleware, limiter=limiter or default_limiter)
    configure_cors(app)


__all__ = [
    "DEFAULT_CORS_ORIGINS",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_RATE_WINDOW_SECONDS",
    "FixedWindowRateLimiter",
    "ModerationHardeningMiddleware",
    "configure_cors",
    "cors_origins",
    "default_limiter",
    "install_hardening",
    "is_protected_path",
    "max_body_bytes",
    "rate_limit",
    "rate_window_seconds",
]
