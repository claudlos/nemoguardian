"""Shared test fixtures."""

from unittest.mock import MagicMock

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


@pytest.fixture
def isolated_db(monkeypatch, tmp_path):
    """Fresh SQLite DB per test, with billing-related env cleared.

    Yields the ``nemoguardian.billing.db`` module for convenience.
    """
    from nemoguardian.billing import db

    db_path = tmp_path / "nmg_test.db"
    monkeypatch.setenv("NEMOGUARDIAN_DB_PATH", str(db_path))
    for var in (
        "NEMOGUARDIAN_API_KEY",
        "NEMOGUARDIAN_TIER",
        "NEMOGUARDIAN_SELF_HOSTED_EMAIL",
        "STRIPE_SECRET_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_METER_EVENT_NAME",
        "NEMOGUARDIAN_INSECURE_WEBHOOK_DEMO",
    ):
        monkeypatch.delenv(var, raising=False)
    db._conn = None
    db.init_db(str(db_path))
    yield db
    db._conn = None


def make_mock_cascade(moderate_side_effect=None, *, triage_configured: bool = False) -> MagicMock:
    """A MagicMock(spec=Cascade) with the attributes /health and moderate need."""
    from nemoguardian.cascade import Cascade

    fake = MagicMock(spec=Cascade)
    if moderate_side_effect is not None:
        fake.moderate.side_effect = moderate_side_effect
    fake.loaded_models.return_value = {
        "qwen3_guard_gen": False,
        "qwen3_guard_stream": False,
        "nemotron_csr": False,
    }
    fake.model_config_summary.return_value = {
        "qwen_gen_model": "mock-qwen",
        "nemotron_csr_model": "mock-csr",
        "enable_triage": True,
    }
    fake.triage_status.return_value = {
        "configured": triage_configured,
        "provider": "nvidia" if triage_configured else None,
        "model": "mock-triage",
        "base_url": "https://example.test/v1",
    }
    return fake
