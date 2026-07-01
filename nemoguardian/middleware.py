"""Production hardening middleware for the FastAPI surface.

Self-host templates expose the moderation endpoints straight to the internet,
so this module adds three cheap, dependency-free guards:

* **CORS allowlist** (``configure_cors``) — replaces a wildcard ``*`` origin with
  an env-configurable allowlist. See ``configure_cors`` for the trade-off.
* **Rate limiting** — a fixed-window in-memory limiter keyed on a *non-spoofable*
  identity: a validated API key for authenticated callers, otherwise the trusted
  client IP (``scope['client']`` unless an explicit reverse-proxy hop count is
  configured via ``NEMOGUARDIAN_TRUST_PROXY``). Fail-open only when unconfigured,
  and that fail-open state is logged loudly at install time.
* **Request body-size limit** — rejects oversized moderation payloads with 413,
  enforced on the *actual* received bytes (chunked / missing ``Content-Length``
  cannot bypass it) in addition to a fast ``Content-Length`` reject.

Everything here is pure Python/Starlette: no new third-party dependency (no
``slowapi``/``redis``). The limiter state lives in-process, which is the right
default for a single-node self-host template; horizontal deployments should
front this with a shared store (documented in ``FixedWindowRateLimiter``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

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


def trust_proxy_hops() -> int:
    """Number of trusted reverse-proxy hops in front of the app.

    ``X-Forwarded-For`` is attacker-controlled and MUST NOT be trusted by
    default: anyone can send ``X-Forwarded-For: 1.2.3.4`` to mint a fresh
    rate-limit bucket. We only honor it when the operator explicitly declares
    how many proxy hops sit in front of the app via ``NEMOGUARDIAN_TRUST_PROXY``.

    The value is the hop count (``"1"`` for a single reverse proxy). Truthy
    aliases (``true``/``yes``/``on``) mean one hop; anything unset, blank,
    invalid, or ``<= 0`` means *do not trust the header at all* (0 hops).
    """
    raw = _env("NEMOGUARDIAN_TRUST_PROXY")
    if raw is None:
        return 0
    low = raw.lower()
    if low in {"true", "yes", "on"}:
        return 1
    if low in {"false", "no", "off"}:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return value if value > 0 else 0


def is_protected_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in PROTECTED_PREFIXES)


# --- Fixed-window rate limiter --------------------------------------------


#: Hard cap on distinct keys held in memory, so a distinct-key flood (e.g. one
#: request per forged IP) can't grow the map without bound even inside a window.
DEFAULT_MAX_KEYS = 100_000


class FixedWindowRateLimiter:
    """Thread-safe in-memory fixed-window counter.

    Keys map to ``(window_start, count)``. Each ``check`` increments the count
    for the current window and reports whether the caller is within ``limit``.

    Memory safety: every ``check`` first evicts entries whose window has fully
    expired, so the map only retains *active* windows. A ``max_keys`` cap is a
    backstop against a burst of distinct keys within a single window — the
    oldest windows are dropped first once the cap is exceeded. Without this a
    distinct-key flood would grow ``_hits`` until the process OOMs.

    State is per-process; a multi-node deployment needs a shared backing store
    (Redis/memcached) instead — out of scope for this single-node template.
    """

    def __init__(self, *, max_keys: int = DEFAULT_MAX_KEYS) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, tuple[float, int]] = {}
        self._max_keys = max(1, max_keys)

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
            self._evict_expired(now, window_seconds)
            start, count = self._hits.get(key, (now, 0))
            if now - start >= window_seconds:
                start, count = now, 0
            count += 1
            self._hits[key] = (start, count)
            if len(self._hits) > self._max_keys:
                self._evict_to_cap()
            allowed = count <= limit
            retry_after = 0.0 if allowed else max(0.0, window_seconds - (now - start))
            return allowed, retry_after

    def _evict_expired(self, now: float, window_seconds: float) -> None:
        """Drop entries whose window has fully elapsed. Caller holds the lock."""
        expired = [k for k, (start, _) in self._hits.items() if now - start >= window_seconds]
        for k in expired:
            del self._hits[k]

    def _evict_to_cap(self) -> None:
        """Backstop: drop the oldest windows until under ``max_keys``. Locked."""
        overflow = len(self._hits) - self._max_keys
        if overflow <= 0:
            return
        oldest = sorted(self._hits.items(), key=lambda kv: kv[1][0])[:overflow]
        for k, _ in oldest:
            del self._hits[k]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


#: Process-wide limiter shared by the app's middleware instance. Exposed so
#: tests (and operators poking a REPL) can ``reset()`` between runs.
default_limiter = FixedWindowRateLimiter()


def _peer_ip(scope: Scope) -> str:
    client = scope.get("client")
    if client:
        return str(client[0])
    return "unknown"


def _client_ip(scope: Scope, headers: Headers) -> str:
    """Best-effort *non-spoofable* client IP for rate-limit keying.

    ``X-Forwarded-For`` is only consulted when ``NEMOGUARDIAN_TRUST_PROXY``
    declares one or more trusted hops. With ``N`` trusted hops the real client
    is the ``N``-th entry counted from the right (each trusted proxy appends the
    address it received the request from, so the rightmost entries are the ones
    our own infrastructure added and cannot be forged by the client). Any
    attacker-supplied prefix entries are ignored. If the header is missing or
    has fewer entries than trusted hops, we fall back to the ASGI peer IP.
    """
    hops = trust_proxy_hops()
    if hops > 0:
        forwarded = headers.get("x-forwarded-for")
        if forwarded:
            parts = [p.strip() for p in forwarded.split(",") if p.strip()]
            if len(parts) >= hops:
                return parts[-hops]
    return _peer_ip(scope)


def _extract_api_key(headers: Headers) -> str:
    authorization = headers.get("authorization")
    if not authorization:
        return ""
    parts = authorization.split(" ", 1)
    return parts[1].strip() if len(parts) == 2 else authorization.strip()


def _hash_key(api_key: str) -> str:
    # Avoid holding raw secrets as dict keys in memory; the digest is a stable
    # per-key identity that is enough for bucketing.
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _too_large_response(limit: int, received: int) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={
            "detail": "request body too large",
            "max_bytes": limit,
            "received_bytes": received,
        },
    )


def _declared_length(headers: Headers) -> int | None:
    content_length = headers.get("content-length")
    if content_length is None:
        return None
    try:
        return int(content_length)
    except ValueError:
        return None


def _replay_receive(messages: list[Message], original: Receive) -> Receive:
    """Return a ``receive`` that replays already-consumed ASGI messages.

    Once the buffered messages are exhausted it delegates to the ``original``
    receive so long-lived handlers (e.g. streaming responses that poll for
    client disconnect) still block on the real transport instead of seeing a
    fabricated ``http.disconnect``.
    """
    iterator = iter(messages)

    async def receive() -> Message:
        try:
            return next(iterator)
        except StopIteration:
            return await original()

    return receive


class ModerationHardeningMiddleware:
    """ASGI middleware enforcing body-size + rate limits on moderation paths.

    Implemented as raw ASGI (not ``BaseHTTPMiddleware``). For protected paths it
    buffers the request body up to the configured cap and replays it to the app,
    so the size limit is enforced on the *actual* bytes received — a chunked or
    ``Content-Length``-less request cannot bypass the cap. Memory is bounded by
    the cap: as soon as the running total exceeds it we return 413 and stop
    reading. A fast ``Content-Length`` reject short-circuits the obvious case
    without reading any body.

    Rate limiting is keyed on a non-spoofable identity: a *validated* API key
    (via the injected ``key_validator``) for authenticated callers, otherwise
    the trusted client IP. Unvalidated / random ``Authorization`` values fall
    back to IP keying so they cannot mint fresh buckets.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: FixedWindowRateLimiter | None = None,
        key_validator: Callable[[str], bool] | None = None,
    ) -> None:
        self.app = app
        self.limiter = limiter or default_limiter
        self._key_validator = key_validator

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not is_protected_path(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)

        rejection = self._rate_limited(scope, headers)
        if rejection is not None:
            await rejection(scope, receive, send)
            return

        limit = max_body_bytes()
        if limit is not None:
            # Fast path: a declared Content-Length over the cap is rejected
            # without reading the body at all.
            declared = _declared_length(headers)
            if declared is not None and declared > limit:
                await _too_large_response(limit, declared)(scope, receive, send)
                return
            # Enforce on actual received bytes (covers chunked / missing CL).
            receive, overflow = await self._buffer_capped_body(receive, limit)
            if overflow is not None:
                await _too_large_response(limit, overflow)(scope, receive, send)
                return

        await self.app(scope, receive, send)

    @staticmethod
    async def _buffer_capped_body(receive: Receive, limit: int) -> tuple[Receive, int | None]:
        """Read the request body, capping at ``limit`` bytes.

        Returns ``(replay_receive, overflow)`` where ``overflow`` is the running
        byte total once it exceeds ``limit`` (or ``None`` if the body fit). The
        returned ``receive`` replays the buffered messages to the app so the
        downstream handler still sees the full body.
        """
        messages: list[Message] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                messages.append(message)
                if message["type"] == "http.disconnect":
                    break
                continue
            total += len(message.get("body", b""))
            messages.append(message)
            if total > limit:
                return _replay_receive(messages, receive), total
            if not message.get("more_body", False):
                break
        return _replay_receive(messages, receive), None

    def _rate_key(self, scope: Scope, headers: Headers) -> str:
        api_key = _extract_api_key(headers)
        if api_key and self._is_valid_key(api_key):
            return f"key:{_hash_key(api_key)}"
        return f"ip:{_client_ip(scope, headers)}"

    def _is_valid_key(self, api_key: str) -> bool:
        if self._key_validator is None:
            return False
        try:
            return bool(self._key_validator(api_key))
        except Exception:  # fail-safe: never raise on the moderation path
            logger.exception("rate-limit key validator raised; treating as unauthenticated")
            return False

    def _rate_limited(self, scope: Scope, headers: Headers) -> JSONResponse | None:
        limit = rate_limit()
        if limit is None:  # fail-open: unconfigured
            return None
        window = rate_window_seconds()
        allowed, retry_after = self.limiter.check(
            self._rate_key(scope, headers), limit=limit, window_seconds=window
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


def warn_if_unlimited() -> bool:
    """Log a loud warning when rate limiting is unconfigured (fail-open).

    Returns ``True`` when a warning was emitted, so callers/tests can assert on
    the visible fail-open state. We deliberately keep the fail-open *default*
    (no limit) rather than silently throttling operators, but make it visible.
    """
    if rate_limit() is None:
        logger.warning(
            "NEMOGUARDIAN_RATE_LIMIT is not set: moderation endpoints are NOT "
            "rate limited (fail-open). Set NEMOGUARDIAN_RATE_LIMIT to a positive "
            "integer to enable per-identity limiting before exposing this "
            "service to untrusted traffic."
        )
        return True
    return False


def install_hardening(
    app: FastAPI,
    *,
    limiter: FixedWindowRateLimiter | None = None,
    key_validator: Callable[[str], bool] | None = None,
) -> None:
    """Attach body-size + rate-limit guards, then the CORS allowlist.

    CORS is added last so it wraps outermost and can short-circuit preflight
    ``OPTIONS`` before the limiter counts them. Emits a startup warning when the
    rate limiter is unconfigured so the fail-open state is never silent.
    """
    warn_if_unlimited()
    app.add_middleware(
        ModerationHardeningMiddleware,
        limiter=limiter or default_limiter,
        key_validator=key_validator,
    )
    configure_cors(app)


__all__ = [
    "DEFAULT_CORS_ORIGINS",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_MAX_KEYS",
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
    "trust_proxy_hops",
    "warn_if_unlimited",
]
