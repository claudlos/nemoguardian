"""Instance-scoped API keys (NMG-009) — DB layer, auth gating, provisioning flow."""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from nemoguardian import server as srv
from nemoguardian.billing import db
from nemoguardian.billing.plans import Tier


@pytest.fixture(autouse=True)
def _isolated_db(isolated_db):
    """Fresh SQLite DB per test (shared fixture from conftest)."""
    yield


def _self_hosted_customer():
    customer = db.upsert_customer(email="scoped@example.com")
    db.set_customer_tier(customer.id, Tier.SELF_HOSTED)
    return db.get_customer(customer.id)


# --- DB layer ---------------------------------------------------------------

def test_create_api_key_scope_persisted():
    customer = _self_hosted_customer()
    raw, record = db.create_api_key(customer.id, label="vm", scope="instance")
    assert record.scope == "instance"
    found = db.lookup_api_key(raw)
    assert found is not None
    _, key = found
    assert key.scope == "instance"
    assert key.id == record.id


def test_create_api_key_rejects_unknown_scope():
    customer = _self_hosted_customer()
    with pytest.raises(ValueError, match="unknown API key scope"):
        db.create_api_key(customer.id, scope="admin")


def test_instance_keys_exempt_from_quota():
    customer = db.upsert_customer(email="freequota@example.com")  # free tier: 1 key max
    db.create_api_key(customer.id, label="the one full key")
    with pytest.raises(ValueError, match="max API keys"):
        db.create_api_key(customer.id, label="second full key")
    # Instance keys don't consume the quota.
    db.create_api_key(customer.id, label="vm-1", scope="instance")
    db.create_api_key(customer.id, label="vm-2", scope="instance")


def test_migration_adds_columns_to_old_db(tmp_path, monkeypatch):
    """A DB created before scope/api_key_id existed gains them idempotently."""
    old = tmp_path / "old.db"
    conn = sqlite3.connect(str(old))
    conn.executescript(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
            stripe_customer_id TEXT, tier TEXT NOT NULL DEFAULT 'free',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL REFERENCES customers(id),
            key_hash TEXT UNIQUE NOT NULL, label TEXT, created_at TEXT NOT NULL,
            last_used_at TEXT, revoked INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE provisioning_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL REFERENCES customers(id),
            tier TEXT NOT NULL, provider TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', instance_id TEXT,
            endpoint_url TEXT, ssh_command TEXT, error_message TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        """
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("NEMOGUARDIAN_DB_PATH", str(old))
    db._conn = None
    db.init_db(str(old))  # runs SCHEMA (IF NOT EXISTS) + _migrate
    customer = db.upsert_customer(email="old@example.com")
    _raw, record = db.create_api_key(customer.id, scope="instance")
    assert record.scope == "instance"

    # Idempotent: re-running init on the migrated file must not error.
    db._conn = None
    db.init_db(str(old))
    assert db.lookup_api_key(_raw) is not None


# --- Auth gating over the API -------------------------------------------------

@pytest.fixture
def scoped_client(monkeypatch):
    from unittest.mock import MagicMock

    from nemoguardian.cascade import Cascade
    from nemoguardian.schemas import ModelVerdict, ModerateResponse, VerdictLabel

    def _mock_moderate(request, policy_engine=None, **kw):
        v = ModelVerdict(model_id="m", verdict=VerdictLabel.SAFE, score=0.0, latency_ms=1.0)
        return ModerateResponse(
            verdict=VerdictLabel.SAFE, score=0.0, reasons=[], categories=[],
            model_verdicts={"qwen3_guard_gen": v}, total_latency_ms=1.0, mode=request.mode,
        )

    fake_cascade = MagicMock(spec=Cascade)
    fake_cascade.moderate.side_effect = _mock_moderate
    monkeypatch.setattr(srv, "get_cascade", lambda: fake_cascade)
    return TestClient(srv.app)


def test_instance_key_can_moderate_but_not_manage(scoped_client):
    customer = _self_hosted_customer()
    raw_instance, _ = db.create_api_key(customer.id, label="vm", scope="instance")
    headers = {"Authorization": f"Bearer {raw_instance}"}

    r = scoped_client.post("/v1/moderate", json={"text": "hello"}, headers=headers)
    assert r.status_code == 200

    r = scoped_client.post("/billing/keys", json={"label": "sneaky"}, headers=headers)
    assert r.status_code == 403
    assert "instance-scoped" in r.json()["detail"]

    r = scoped_client.post(
        "/billing/provision/cheapest", json={"ssh_public_key": None}, headers=headers
    )
    assert r.status_code == 403


def test_full_key_still_manages(scoped_client):
    customer = _self_hosted_customer()
    raw_full, _ = db.create_api_key(customer.id, label="admin")
    r = scoped_client.post(
        "/billing/keys", json={"label": "new"}, headers={"Authorization": f"Bearer {raw_full}"}
    )
    assert r.status_code == 200
    assert r.json()["api_key"].startswith("nmg_")


# --- Provisioning injects the scoped key, not the caller's -----------------------

def test_provision_cheapest_injects_instance_key(scoped_client, monkeypatch):
    customer = _self_hosted_customer()
    raw_full, _ = db.create_api_key(customer.id, label="admin")

    captured_env = {}

    class FakeState:
        value = "live"

    class FakeInstance:
        instance_id = "fake-1"
        endpoint_url = "https://fake-1.example"
        ssh_command = "ssh fake"
        state = FakeState()

    class FakeOffer:
        class provider:
            value = "vastai"

    async def fake_cheapest_fit(**kwargs):
        captured_env.update(kwargs.get("env") or {})
        return FakeOffer(), FakeInstance()

    monkeypatch.setattr(srv, "provision_cheapest_fit", fake_cheapest_fit)

    r = scoped_client.post(
        "/billing/provision/cheapest",
        json={},
        headers={"Authorization": f"Bearer {raw_full}"},
    )
    assert r.status_code == 200
    injected = captured_env["NEMOGUARDIAN_API_KEY"]
    assert injected.startswith("nmg_")
    assert injected != raw_full  # the master key never leaves the control plane

    # The injected key is instance-scoped and tied to the job.
    found = db.lookup_api_key(injected)
    assert found is not None
    _, key = found
    assert key.scope == "instance"
    job = db.get_provisioning_job(r.json()["job_id"])
    assert job.api_key_id == key.id


def test_provision_failure_revokes_minted_key(scoped_client, monkeypatch):
    customer = _self_hosted_customer()
    raw_full, _ = db.create_api_key(customer.id, label="admin")

    captured_env = {}

    async def exploding_cheapest_fit(**kwargs):
        captured_env.update(kwargs.get("env") or {})
        raise RuntimeError("provider down")

    monkeypatch.setattr(srv, "provision_cheapest_fit", exploding_cheapest_fit)

    r = scoped_client.post(
        "/billing/provision/cheapest",
        json={},
        headers={"Authorization": f"Bearer {raw_full}"},
    )
    assert r.status_code == 502
    # The minted credential must not survive the failed provision.
    assert db.lookup_api_key(captured_env["NEMOGUARDIAN_API_KEY"]) is None
