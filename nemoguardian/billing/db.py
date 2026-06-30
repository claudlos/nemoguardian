"""SQLite-backed customers / api_keys / usage tables.

We deliberately use stdlib sqlite3 so the demo has zero external dependencies.
The same schema works on Postgres in production.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nemoguardian.billing.plans import Tier, get_plan

DEFAULT_DB_PATH = Path(
    os.environ.get("NEMOGUARDIAN_DB_PATH", "/tmp/nemoguardian.db")
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email           TEXT UNIQUE NOT NULL,
    stripe_customer_id  TEXT,
    tier            TEXT NOT NULL DEFAULT 'free',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    key_hash        TEXT UNIQUE NOT NULL,
    label           TEXT,
    created_at      TEXT NOT NULL,
    last_used_at    TEXT,
    revoked         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);

CREATE TABLE IF NOT EXISTS usage_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    api_key_id      INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
    occurred_at     TEXT NOT NULL,
    call_type       TEXT NOT NULL,    -- 'fast' | 'standard' | 'deep'
    units           INTEGER NOT NULL DEFAULT 1,
    request_id      TEXT
);

CREATE INDEX IF NOT EXISTS idx_usage_customer_month ON usage_events(customer_id, occurred_at);

CREATE TABLE IF NOT EXISTS subscriptions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    stripe_subscription_id  TEXT UNIQUE,
    tier            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',  -- active | trialing | past_due | canceled
    current_period_start TEXT NOT NULL,
    current_period_end   TEXT NOT NULL,
    cancel_at_period_end  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS provisioning_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    tier            TEXT NOT NULL,
    provider        TEXT NOT NULL,           -- 'vastai' | 'digitalocean' | 'lambda' | 'on_prem'
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | provisioning | live | failed
    instance_id     TEXT,
    endpoint_url    TEXT,
    ssh_command     TEXT,
    error_message   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gpu_credit_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    occurred_at     TEXT NOT NULL,
    event_type      TEXT NOT NULL,           -- stripe_topup | provision_reserve | provision_refund | manual
    amount_cents    INTEGER NOT NULL,        -- positive = credit, negative = debit/reserve
    currency        TEXT NOT NULL DEFAULT 'usd',
    provider        TEXT,
    job_id          INTEGER REFERENCES provisioning_jobs(id) ON DELETE SET NULL,
    stripe_checkout_session_id TEXT,
    stripe_payment_intent_id   TEXT,
    description     TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    UNIQUE(event_type, stripe_checkout_session_id)
);

CREATE INDEX IF NOT EXISTS idx_gpu_credit_customer_time
ON gpu_credit_events(customer_id, occurred_at);
"""


_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def init_db(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initialise the database (idempotent). Returns a connection."""
    global _conn
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        if _conn is None:
            _conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
            _conn.row_factory = sqlite3.Row
            _conn.executescript("PRAGMA foreign_keys = ON;")
            _conn.executescript(SCHEMA)
        return _conn


@contextmanager
def _tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    with _lock:
        cur = conn.cursor()
        cur.execute("BEGIN")
        try:
            yield conn
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# --- Customer --------------------------------------------------------------

@dataclass
class Customer:
    id: int
    email: str
    stripe_customer_id: str | None
    tier: str
    created_at: str
    updated_at: str

    @property
    def tier_enum(self) -> Tier:
        return Tier(self.tier)

    @property
    def plan(self):
        return get_plan(self.tier_enum)


def upsert_customer(email: str, *, stripe_customer_id: str | None = None) -> Customer:
    conn = init_db()
    now = _now()
    with _tx(conn):
        existing = conn.execute(
            "SELECT * FROM customers WHERE email = ?", (email,)
        ).fetchone()
        if existing is None:
            cur = conn.execute(
                """INSERT INTO customers (email, stripe_customer_id, tier, created_at, updated_at)
                   VALUES (?, ?, 'free', ?, ?)""",
                (email, stripe_customer_id, now, now),
            )
            customer_id = cur.lastrowid
        else:
            customer_id = existing["id"]
            if stripe_customer_id and not existing["stripe_customer_id"]:
                conn.execute(
                    "UPDATE customers SET stripe_customer_id = ?, updated_at = ? WHERE id = ?",
                    (stripe_customer_id, now, customer_id),
                )
    return get_customer(customer_id)


def get_customer(customer_id: int) -> Customer:
    conn = init_db()
    row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    if row is None:
        raise KeyError(f"no customer with id {customer_id}")
    return Customer(**dict(row))


def get_customer_by_email(email: str) -> Customer | None:
    conn = init_db()
    row = conn.execute("SELECT * FROM customers WHERE email = ?", (email,)).fetchone()
    return Customer(**dict(row)) if row else None


def get_customer_by_stripe_id(stripe_customer_id: str) -> Customer | None:
    conn = init_db()
    row = conn.execute(
        "SELECT * FROM customers WHERE stripe_customer_id = ?", (stripe_customer_id,)
    ).fetchone()
    return Customer(**dict(row)) if row else None


def set_customer_tier(customer_id: int, tier: Tier) -> None:
    conn = init_db()
    conn.execute(
        "UPDATE customers SET tier = ?, updated_at = ? WHERE id = ?",
        (tier.value, _now(), customer_id),
    )


# --- API Keys --------------------------------------------------------------

@dataclass
class ApiKey:
    id: int
    customer_id: int
    label: str | None
    created_at: str
    last_used_at: str | None
    revoked: bool


def _generate_raw_key() -> tuple[str, str]:
    """Generate a fresh API key. Returns (raw_key, hash)."""
    raw = "nmg_" + secrets.token_urlsafe(32)
    return raw, _hash_key(raw)


def create_api_key(customer_id: int, label: str | None = None) -> tuple[str, ApiKey]:
    """Create a new API key for the customer. Returns (raw_key, record).

    The raw key is shown to the user ONCE. Only the hash is stored.
    """
    conn = init_db()
    customer = get_customer(customer_id)
    plan = customer.plan
    active_count = conn.execute(
        "SELECT COUNT(*) AS n FROM api_keys WHERE customer_id = ? AND revoked = 0",
        (customer_id,),
    ).fetchone()["n"]
    if active_count >= plan.max_api_keys:
        raise ValueError(
            f"max API keys for tier {plan.tier.value} reached ({plan.max_api_keys})"
        )
    raw, h = _generate_raw_key()
    now = _now()
    with _tx(conn):
        cur = conn.execute(
            """INSERT INTO api_keys (customer_id, key_hash, label, created_at)
               VALUES (?, ?, ?, ?)""",
            (customer_id, h, label, now),
        )
        key_id = cur.lastrowid
    return raw, ApiKey(
        id=key_id,
        customer_id=customer_id,
        label=label,
        created_at=now,
        last_used_at=None,
        revoked=False,
    )


def lookup_customer_by_api_key(raw_key: str) -> Customer | None:
    """Resolve a raw API key to a customer, or None if invalid/revoked."""
    conn = init_db()
    h = _hash_key(raw_key)
    row = conn.execute(
        """SELECT c.* FROM customers c
           JOIN api_keys k ON k.customer_id = c.id
           WHERE k.key_hash = ? AND k.revoked = 0""",
        (h,),
    ).fetchone()
    if row is None:
        return None
    # Touch last_used_at (best-effort, no tx needed)
    conn.execute("UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?", (_now(), h))
    return Customer(**dict(row))


def revoke_api_key(customer_id: int, key_id: int) -> None:
    conn = init_db()
    conn.execute(
        "UPDATE api_keys SET revoked = 1 WHERE id = ? AND customer_id = ?",
        (key_id, customer_id),
    )


def _api_key_from_row(row: sqlite3.Row) -> ApiKey:
    data = dict(row)
    data.pop("key_hash", None)
    data["revoked"] = bool(data["revoked"])
    return ApiKey(**data)


def list_api_keys(customer_id: int) -> list[ApiKey]:
    conn = init_db()
    rows = conn.execute(
        "SELECT * FROM api_keys WHERE customer_id = ? ORDER BY id DESC",
        (customer_id,),
    ).fetchall()
    return [_api_key_from_row(r) for r in rows]


# --- Usage ----------------------------------------------------------------

@dataclass
class UsageSummary:
    customer_id: int
    period_start: str
    period_end: str
    total_calls: int
    allowance: int
    overage_calls: int
    overage_cents: float


def record_usage(customer_id: int, *, call_type: str, request_id: str | None = None) -> None:
    conn = init_db()
    conn.execute(
        """INSERT INTO usage_events (customer_id, call_type, occurred_at, request_id)
           VALUES (?, ?, ?, ?)""",
        (customer_id, call_type, _now(), request_id),
    )


def usage_for_period(customer_id: int, period_start: str, period_end: str) -> UsageSummary:
    conn = init_db()
    row = conn.execute(
        """SELECT COUNT(*) AS n FROM usage_events
           WHERE customer_id = ? AND occurred_at >= ? AND occurred_at < ?""",
        (customer_id, period_start, period_end),
    ).fetchone()
    total = int(row["n"])
    customer = get_customer(customer_id)
    plan = customer.plan
    overage = max(0, total - plan.monthly_allowance)
    return UsageSummary(
        customer_id=customer_id,
        period_start=period_start,
        period_end=period_end,
        total_calls=total,
        allowance=plan.monthly_allowance,
        overage_calls=overage,
        overage_cents=round(overage * plan.overage_per_call_cents, 4),
    )


# --- Subscriptions --------------------------------------------------------

@dataclass
class Subscription:
    id: int
    customer_id: int
    stripe_subscription_id: str | None
    tier: str
    status: str
    current_period_start: str
    current_period_end: str
    cancel_at_period_end: bool
    created_at: str
    updated_at: str


def upsert_subscription(
    *,
    customer_id: int,
    stripe_subscription_id: str | None,
    tier: Tier,
    status: str,
    period_start: str,
    period_end: str,
    cancel_at_period_end: bool = False,
) -> Subscription:
    conn = init_db()
    now = _now()
    with _tx(conn):
        existing = None
        if stripe_subscription_id:
            existing = conn.execute(
                "SELECT * FROM subscriptions WHERE stripe_subscription_id = ?",
                (stripe_subscription_id,),
            ).fetchone()
        if existing is None:
            cur = conn.execute(
                """INSERT INTO subscriptions
                   (customer_id, stripe_subscription_id, tier, status,
                    current_period_start, current_period_end, cancel_at_period_end,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (customer_id, stripe_subscription_id, tier.value, status,
                 period_start, period_end, int(cancel_at_period_end), now, now),
            )
            sub_id = cur.lastrowid
        else:
            conn.execute(
                """UPDATE subscriptions SET
                   tier = ?, status = ?, current_period_start = ?,
                   current_period_end = ?, cancel_at_period_end = ?,
                   customer_id = ?, updated_at = ?
                   WHERE id = ?""",
                (tier.value, status, period_start, period_end,
                 int(cancel_at_period_end), customer_id, now, existing["id"]),
            )
            sub_id = existing["id"]
    return get_subscription(sub_id)


def get_subscription(sub_id: int) -> Subscription:
    conn = init_db()
    row = conn.execute("SELECT * FROM subscriptions WHERE id = ?", (sub_id,)).fetchone()
    if row is None:
        raise KeyError(f"no subscription with id {sub_id}")
    return Subscription(**dict(row))


# --- Provisioning ---------------------------------------------------------

@dataclass
class ProvisioningJob:
    id: int
    customer_id: int
    tier: str
    provider: str
    status: str
    instance_id: str | None
    endpoint_url: str | None
    ssh_command: str | None
    error_message: str | None
    created_at: str
    updated_at: str


def create_provisioning_job(
    customer_id: int, *, tier: Tier, provider: str
) -> ProvisioningJob:
    conn = init_db()
    now = _now()
    cur = conn.execute(
        """INSERT INTO provisioning_jobs
           (customer_id, tier, provider, status, created_at, updated_at)
           VALUES (?, ?, ?, 'pending', ?, ?)""",
        (customer_id, tier.value, provider, now, now),
    )
    return get_provisioning_job(cur.lastrowid)


def get_provisioning_job(job_id: int) -> ProvisioningJob:
    conn = init_db()
    row = conn.execute("SELECT * FROM provisioning_jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(f"no provisioning job with id {job_id}")
    return ProvisioningJob(**dict(row))


def update_provisioning_job(
    job_id: int,
    *,
    status: str | None = None,
    instance_id: str | None = None,
    endpoint_url: str | None = None,
    ssh_command: str | None = None,
    error_message: str | None = None,
) -> None:
    conn = init_db()
    fields, values = [], []
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if instance_id is not None:
        fields.append("instance_id = ?")
        values.append(instance_id)
    if endpoint_url is not None:
        fields.append("endpoint_url = ?")
        values.append(endpoint_url)
    if ssh_command is not None:
        fields.append("ssh_command = ?")
        values.append(ssh_command)
    if error_message is not None:
        fields.append("error_message = ?")
        values.append(error_message)
    if not fields:
        return
    fields.append("updated_at = ?")
    values.append(_now())
    values.append(job_id)
    conn.execute(f"UPDATE provisioning_jobs SET {', '.join(fields)} WHERE id = ?", values)


def list_provisioning_jobs(customer_id: int) -> list[ProvisioningJob]:
    conn = init_db()
    rows = conn.execute(
        "SELECT * FROM provisioning_jobs WHERE customer_id = ? ORDER BY id DESC",
        (customer_id,),
    ).fetchall()
    return [ProvisioningJob(**dict(r)) for r in rows]


# --- GPU Credits ----------------------------------------------------------

@dataclass
class GpuCreditEvent:
    id: int
    customer_id: int
    occurred_at: str
    event_type: str
    amount_cents: int
    currency: str
    provider: str | None
    job_id: int | None
    stripe_checkout_session_id: str | None
    stripe_payment_intent_id: str | None
    description: str | None
    metadata_json: str

    @property
    def metadata(self) -> dict:
        try:
            data = json.loads(self.metadata_json)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


def record_gpu_credit_event(
    *,
    customer_id: int,
    event_type: str,
    amount_cents: int,
    currency: str = "usd",
    provider: str | None = None,
    job_id: int | None = None,
    stripe_checkout_session_id: str | None = None,
    stripe_payment_intent_id: str | None = None,
    description: str | None = None,
    metadata: dict | None = None,
) -> GpuCreditEvent:
    """Append a GPU-credit ledger event.

    Stripe checkout top-ups are idempotent on ``(event_type,
    stripe_checkout_session_id)`` so webhook retries do not double-credit a
    wallet.
    """
    if amount_cents == 0:
        raise ValueError("GPU credit event amount cannot be zero")

    conn = init_db()
    if stripe_checkout_session_id:
        existing = conn.execute(
            """SELECT * FROM gpu_credit_events
               WHERE event_type = ? AND stripe_checkout_session_id = ?""",
            (event_type, stripe_checkout_session_id),
        ).fetchone()
        if existing is not None:
            return GpuCreditEvent(**dict(existing))

    now = _now()
    metadata_json = json.dumps(metadata or {}, sort_keys=True)
    with _tx(conn):
        cur = conn.execute(
            """INSERT INTO gpu_credit_events (
                   customer_id, occurred_at, event_type, amount_cents, currency,
                   provider, job_id, stripe_checkout_session_id,
                   stripe_payment_intent_id, description, metadata_json
               )
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                customer_id,
                now,
                event_type,
                int(amount_cents),
                currency,
                provider,
                job_id,
                stripe_checkout_session_id,
                stripe_payment_intent_id,
                description,
                metadata_json,
            ),
        )
        event_id = cur.lastrowid
    return get_gpu_credit_event(event_id)


def get_gpu_credit_event(event_id: int) -> GpuCreditEvent:
    conn = init_db()
    row = conn.execute("SELECT * FROM gpu_credit_events WHERE id = ?", (event_id,)).fetchone()
    if row is None:
        raise KeyError(f"no GPU credit event with id {event_id}")
    return GpuCreditEvent(**dict(row))


def get_gpu_credit_event_by_checkout_session(
    stripe_checkout_session_id: str,
) -> GpuCreditEvent | None:
    conn = init_db()
    row = conn.execute(
        """SELECT * FROM gpu_credit_events
           WHERE event_type = 'stripe_topup'
             AND stripe_checkout_session_id = ?
           ORDER BY id DESC
           LIMIT 1""",
        (stripe_checkout_session_id,),
    ).fetchone()
    return GpuCreditEvent(**dict(row)) if row else None


def gpu_credit_balance_cents(customer_id: int | str) -> int:
    conn = init_db()
    row = conn.execute(
        """SELECT COALESCE(SUM(amount_cents), 0) AS balance
           FROM gpu_credit_events
           WHERE customer_id = ?""",
        (customer_id,),
    ).fetchone()
    return int(row["balance"])


def list_gpu_credit_events(customer_id: int | str, *, limit: int = 50) -> list[GpuCreditEvent]:
    conn = init_db()
    rows = conn.execute(
        """SELECT * FROM gpu_credit_events
           WHERE customer_id = ?
           ORDER BY id DESC
           LIMIT ?""",
        (customer_id, max(0, limit)),
    ).fetchall()
    return [GpuCreditEvent(**dict(row)) for row in rows]


def reserve_gpu_credits(
    *,
    customer_id: int,
    amount_cents: int,
    provider: str,
    job_id: int | None = None,
    description: str | None = None,
    metadata: dict | None = None,
) -> GpuCreditEvent:
    """Atomically debit GPU credits before provisioning rented capacity."""
    if amount_cents <= 0:
        raise ValueError("GPU credit reservation amount must be positive")

    conn = init_db()
    now = _now()
    metadata_json = json.dumps(metadata or {}, sort_keys=True)
    with _tx(conn):
        row = conn.execute(
            """SELECT COALESCE(SUM(amount_cents), 0) AS balance
               FROM gpu_credit_events
               WHERE customer_id = ?""",
            (customer_id,),
        ).fetchone()
        balance = int(row["balance"])
        if balance < amount_cents:
            raise ValueError(
                f"insufficient GPU credits: need {amount_cents} cents, have {balance} cents"
            )
        cur = conn.execute(
            """INSERT INTO gpu_credit_events (
                   customer_id, occurred_at, event_type, amount_cents, currency,
                   provider, job_id, description, metadata_json
               )
               VALUES (?, ?, 'provision_reserve', ?, 'usd', ?, ?, ?, ?)""",
            (
                customer_id,
                now,
                -int(amount_cents),
                provider,
                job_id,
                description,
                metadata_json,
            ),
        )
        event_id = cur.lastrowid
    return get_gpu_credit_event(event_id)


__all__ = [
    "ApiKey",
    "Customer",
    "GpuCreditEvent",
    "ProvisioningJob",
    "Subscription",
    "UsageSummary",
    "create_api_key",
    "create_provisioning_job",
    "get_customer",
    "get_customer_by_email",
    "get_customer_by_stripe_id",
    "get_gpu_credit_event",
    "get_gpu_credit_event_by_checkout_session",
    "get_provisioning_job",
    "get_subscription",
    "gpu_credit_balance_cents",
    "init_db",
    "list_api_keys",
    "list_gpu_credit_events",
    "list_provisioning_jobs",
    "lookup_customer_by_api_key",
    "record_gpu_credit_event",
    "record_usage",
    "reserve_gpu_credits",
    "revoke_api_key",
    "set_customer_tier",
    "update_provisioning_job",
    "upsert_customer",
    "upsert_subscription",
    "usage_for_period",
]
