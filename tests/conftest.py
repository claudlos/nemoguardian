"""Test fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Reset module-level singletons between tests so we don't carry model state."""
    from nemoguardian import server as srv

    srv._State.cascade = None
    srv._State.policies = {}
    yield
