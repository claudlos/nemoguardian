"""Test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Reset module-level singletons between tests so we don't carry model state."""
    from nemoguardian import server as srv

    srv._State.cascade = None
    srv._State.policies = {}
    # Rate-limit buckets are keyed by API-key id, which repeats across tests'
    # fresh DBs — clear them so one test can't exhaust another's bucket.
    srv._rate_limiter.reset()
    yield
