"""In-process token-bucket rate limiting (NMG-011).

A deliberately small, dependency-free limiter: one bucket per caller key
(API-key id, or client IP for unauthenticated demo traffic), refilled
continuously at the configured requests-per-minute rate.

Scope: single process. This matches the deployment story (one uvicorn worker,
one shared SQLite connection). Multi-process/multi-node deployments should add
proxy-layer limiting as well — see SECURITY.md's hardening checklist.

Configuration: ``NEMOGUARDIAN_RATE_LIMIT_RPM`` (default 60; ``0`` disables).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_RPM = 60.0


def _rpm_from_env() -> float:
    raw = os.environ.get("NEMOGUARDIAN_RATE_LIMIT_RPM", "").strip()
    if not raw:
        return DEFAULT_RPM
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_RPM


@dataclass
class _Bucket:
    capacity: float
    refill_per_s: float
    tokens: float
    updated: float


class RateLimiter:
    """Token bucket per key. Thread-safe; clock injectable for tests."""

    def __init__(
        self,
        *,
        rpm: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._rpm_override = rpm
        self._clock = clock or time.monotonic
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    @property
    def rpm(self) -> float:
        if self._rpm_override is not None:
            return self._rpm_override
        return _rpm_from_env()

    def check(self, key: str) -> tuple[bool, float]:
        """Consume one token for ``key``. Returns (allowed, retry_after_seconds)."""
        rpm = self.rpm
        if rpm <= 0:
            return True, 0.0
        now = self._clock()
        refill_per_s = rpm / 60.0
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(capacity=rpm, refill_per_s=refill_per_s, tokens=rpm, updated=now)
                self._buckets[key] = bucket
            else:
                if bucket.capacity != rpm:  # env changed between requests
                    bucket.capacity = rpm
                    bucket.refill_per_s = refill_per_s
                    bucket.tokens = min(bucket.tokens, rpm)
                bucket.tokens = min(
                    bucket.capacity,
                    bucket.tokens + (now - bucket.updated) * bucket.refill_per_s,
                )
                bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0.0
            return False, (1.0 - bucket.tokens) / bucket.refill_per_s

    def reset(self) -> None:
        """Drop all buckets (tests)."""
        with self._lock:
            self._buckets.clear()


__all__ = ["DEFAULT_RPM", "RateLimiter"]
