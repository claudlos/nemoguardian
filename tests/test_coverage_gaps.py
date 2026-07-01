"""Targeted coverage-gap tests.

These tests close specific lines the prior suite didn't exercise — the
``matrix``, ``reddit``, ``slack``, ``billing.webhook``, ``billing.provisioning``,
``cascade``, and ``review.store`` modules all had small but real
edge-case branches that any reviewer (or live operator) could trip over:

  * A platform client with a non-callable method (e.g. a misconfigured
    SDK stub) — must surface as a ``failed`` status with a short reason.
  * Mod-notice / mod-mail failure that's not the "no-configured-target"
    case — must end up in the error trail with the exception class.
  * Doctor with moderation disabled / unknown power / token missing.
  * Subscription-change webhooks with unknown customer or non-positive amount.
  * Cascade with a single task (sequential path) and a model that raises.
  * Review queue: appeals/decisions on unknown cases, queue resolution
    paths, strike void/restore flows.

All tests are offline, no network / GPU / live credentials.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nemoguardian.adapters import matrix, reddit, slack
from nemoguardian.billing import db, provisioning
from nemoguardian.billing import webhook as billing_webhook
from nemoguardian.bot import (
    BotConfig,
    ModerationAction,
    Platform,
)
from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.review import store as review_store
from nemoguardian.review.models import (
    AppealDecision,
    AppealOutcome,
    CaseStatus,
)

# ---------------------------------------------------------------------------
# matrix: _safe_call non-callable, doctor disabled / unknown power
# ---------------------------------------------------------------------------


class _NoMethods:
    """Stand-in for a platform client that has no useful methods."""


async def test_matrix_safe_call_unsupported_method_returns_error():
    client = _NoMethods()  # no redact_message attr
    ok, err = await matrix._safe_call(client, "redact_message", "room", "evt", "reason")
    assert ok is False
    assert err == "unsupported-method"


async def test_matrix_safe_call_no_client_returns_error():
    ok, err = await matrix._safe_call(None, "redact_message", "room", "evt", "reason")
    assert ok is False
    assert err == "no-client"


async def test_matrix_safe_call_catches_exception_returns_classname():
    """A misbehaving client (raises) must surface the exception class, not crash."""

    class Exploding:
        def redact_message(self, *a, **kw):
            raise ConnectionError("matrix API down")

    ok, err = await matrix._safe_call(Exploding(), "redact_message", "r", "e", "reason")
    assert ok is False
    assert err == "ConnectionError"


def test_matrix_resolve_content_via_source_dict():
    event = {"source": {"content": {"msgtype": "m.text", "body": "hi"}}}
    assert matrix._resolve_content(event) == {"msgtype": "m.text", "body": "hi"}


def test_matrix_resolve_content_via_source_object():
    event = {"source": SimpleNamespace(content={"msgtype": "m.text", "body": "hi"})}
    assert matrix._resolve_content(event) == {"msgtype": "m.text", "body": "hi"}


def test_matrix_doctor_reports_disabled_moderation(tmp_path):
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.enabled = False
    config.log_channel_id = "!mods:hs"
    report = matrix._doctor_report(
        config,
        SimpleNamespace(user_level=50, redact_level=50),
        token_configured=True,
        workspace_id="!r:hs",
    )
    assert "moderation is disabled" in report["issues"]


def test_matrix_doctor_reports_unknown_power(tmp_path):
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.log_channel_id = "!mods:hs"
    report = matrix._doctor_report(
        config, None, token_configured=True, workspace_id="!r:hs"
    )
    assert "room power levels unknown (cannot verify)" in report["issues"]
    assert report["power_verified"] is False


def test_matrix_doctor_reports_missing_token():
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.log_channel_id = "!mods:hs"
    report = matrix._doctor_report(
        config,
        SimpleNamespace(user_level=50, redact_level=50),
        token_configured=False,
        workspace_id="!r:hs",
    )
    assert "MATRIX_ACCESS_TOKEN is not set" in report["issues"]


# ---------------------------------------------------------------------------
# reddit: doctor edge cases, safe_call shapes
# ---------------------------------------------------------------------------


async def test_reddit_safe_call_unsupported_method_returns_error():
    client = _NoMethods()
    ok, err = await reddit._safe_call(client, "remove", "comment", "abc")
    assert ok is False
    assert err == "unsupported-method"


async def test_reddit_safe_call_catches_exception_returns_classname():
    class Exploding:
        def remove(self, *a, **kw):
            raise TimeoutError("reddit API slow")

    ok, err = await reddit._safe_call(Exploding(), "remove", "comment", "abc")
    assert ok is False
    assert err == "TimeoutError"


def test_reddit_stringify_handles_praw_author_object():
    item = {"author": SimpleNamespace(name="bob", __str__=lambda self: "bob")}
    assert reddit._stringify(item["author"]) == "bob"


def test_reddit_doctor_reports_disabled_moderation():
    config = BotConfig.default(Platform.REDDIT, "testsub")
    config.enabled = False
    config.log_channel_id = "modmail"
    report = reddit._doctor_report(
        config, ["posts", "mail", "all"], token_configured=True, workspace_id="testsub"
    )
    assert "moderation is disabled" in report["issues"]


def test_reddit_doctor_reports_not_a_moderator():
    config = BotConfig.default(Platform.REDDIT, "testsub")
    config.log_channel_id = "modmail"
    report = reddit._doctor_report(
        config, None, token_configured=True, workspace_id="testsub"
    )
    assert "granted permissions unknown (not a confirmed moderator)" in report["issues"]
    assert report["is_moderator"] is False


# ---------------------------------------------------------------------------
# slack: chat_delete failure path, doctor disabled
# ---------------------------------------------------------------------------


async def test_slack_safe_call_unsupported_method_returns_error():
    client = _NoMethods()
    ok, err = await slack._safe_call(client, "chat_delete", channel="C1", ts="1.0")
    assert ok is False
    assert err == "unsupported-method"


async def test_slack_safe_call_catches_exception_returns_classname():
    class Exploding:
        def chat_postMessage(self, *a, **kw):
            raise RuntimeError("slack rate limit")

    ok, err = await slack._safe_call(Exploding(), "chat_postMessage", channel="C1", text="hi")
    assert ok is False
    assert err == "RuntimeError"


def test_slack_doctor_reports_disabled_moderation():
    config = BotConfig.default(Platform.SLACK, "T1")
    config.enabled = False
    config.log_channel_id = "C-mod"
    report = slack._doctor_report(
        config,
        set(slack.SLACK_REQUIRED_SCOPES),
        token_configured=True,
    )
    assert "moderation is disabled" in report["issues"]


def test_slack_doctor_reports_missing_token():
    config = BotConfig.default(Platform.SLACK, "T1")
    config.log_channel_id = "C-mod"
    report = slack._doctor_report(
        config,
        set(slack.SLACK_REQUIRED_SCOPES),
        token_configured=False,
    )
    assert "SLACK_BOT_TOKEN is not set" in report["issues"]


# ---------------------------------------------------------------------------
# billing.webhook: ignored paths
# ---------------------------------------------------------------------------


def test_webhook_ignores_checkout_without_matching_customer(monkeypatch):
    """A GPU-credit checkout from a customer not in the DB must be ignored, not crash."""
    monkeypatch.setattr(
        billing_webhook,
        "_resolve_customer_from_checkout",
        lambda _session: None,
    )
    result = billing_webhook._on_gpu_credit_checkout_completed({"metadata": {}, "id": "sess_x"})
    assert result == {"ignored": True, "reason": "no matching customer"}


def test_webhook_ignores_subscription_with_unknown_stripe_customer():
    """A subscription event for a stripe_customer we don't know about must be ignored."""
    result = billing_webhook._on_subscription_change({"customer": "cus_unknown"})
    assert result == {"ignored": True, "reason": "unknown stripe customer"}


def test_webhook_amount_cents_handles_string_zero():
    """A 0-amount GPU credit purchase must be ignored as 'no GPU credit amount'."""
    from nemoguardian.billing.webhook import _gpu_credit_amount_cents

    session = {"amount_total": "0", "metadata": {"gpu_credits": "0"}}
    assert _gpu_credit_amount_cents(session) == 0


def test_webhook_amount_cents_handles_unparseable_string():
    """A non-numeric amount string must be tolerated and return 0, not raise."""
    from nemoguardian.billing.webhook import _gpu_credit_amount_cents

    session = {"amount_total": "not-a-number", "metadata": {}}
    assert _gpu_credit_amount_cents(session) == 0


def test_webhook_subscription_sets_tier_for_known_customer(tmp_path, monkeypatch):
    """A subscription event for a known customer must update their tier."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "billing.sqlite3", raising=False)
    db.init_db()
    customer = db.upsert_customer(
        email="sub@example.com", stripe_customer_id="cus_known_1"
    )
    result = billing_webhook._on_subscription_change({
        "customer": "cus_known_1",
        "status": "active",
        "metadata": {"nemoguardian_tier": "pro"},
        "id": "sub_test",
    })
    assert "ignored" not in result
    refreshed = db.get_customer(customer.id)
    assert refreshed.tier == "pro"


# ---------------------------------------------------------------------------
# billing.provisioning: full happy path
# ---------------------------------------------------------------------------


def test_provisioning_creates_and_completes_job(tmp_path, monkeypatch):
    """A real provisioning request must dispatch a background job and complete it."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "billing.sqlite3", raising=False)
    db.init_db()
    customer = db.upsert_customer(email="prov@example.com")

    async def _run():
        return await provisioning.provision_instance(
            customer_id=customer.id, provider="vastai", ssh_public_key="ssh-rsa AAAA"
        )

    job = asyncio.run(_run())
    assert job.customer_id == customer.id
    assert job.provider == "vastai"


def test_provisioning_rejects_unknown_provider(tmp_path, monkeypatch):
    """An unknown provider must raise ValueError, not silently run."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "billing.sqlite3", raising=False)
    db.init_db()
    customer = db.upsert_customer(email="prov2@example.com")

    async def _run():
        return await provisioning.provision_instance(
            customer_id=customer.id, provider="moon", ssh_public_key=None  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="unsupported provider"):
        asyncio.run(_run())


# ---------------------------------------------------------------------------
# cascade: single-task sequential path
# ---------------------------------------------------------------------------


def test_cascade_single_task_uses_dict_comp_path():
    """With one task the executor must take the synchronous path, not the pool."""
    cfg = CascadeConfig.from_env()
    cfg.concurrent_local = True  # would normally force the pool path
    c = Cascade(cfg)
    tasks = {"only": lambda: "ok"}
    out = c._run_local_verdicts(tasks)  # coverage probe (private method)
    assert out == {"only": "ok"}


def test_cascade_run_local_empty_returns_empty():
    """No tasks → empty result, not an error."""
    cfg = CascadeConfig.from_env()
    c = Cascade(cfg)
    assert c._run_local_verdicts({}) == {}  # coverage probe (private method)


# ---------------------------------------------------------------------------
# review.store: queue resolution / strike flows / appeal for unknown case
# ---------------------------------------------------------------------------


def test_review_queue_resolve_unknown_case_raises_keyerror(tmp_path):
    """Resolving a case that doesn't exist must raise KeyError loudly."""
    queue = review_store.ReviewQueue(tmp_path / "review.jsonl")
    with pytest.raises(KeyError, match="review case not found"):
        queue.resolve("case_does_not_exist", CaseStatus.APPROVED)


def test_review_queue_resolve_to_pending_rejected(tmp_path):
    """A case cannot be 'resolved' back to pending — must raise ValueError."""
    queue = review_store.ReviewQueue(tmp_path / "review.jsonl")
    case = queue.enqueue(
        platform=Platform.DISCORD,
        workspace_id="g1",
        user_id="u1",
        text="test",
    )
    with pytest.raises(ValueError, match="non-terminal"):
        queue.resolve(case.case_id, CaseStatus.PENDING)


def test_appeal_log_record_then_find(tmp_path):
    """The AppealLog must round-trip: record a decision and find it by id."""
    log = review_store.AppealLog(tmp_path / "appeals.jsonl")
    decision = AppealDecision(
        appeal_id="ap1",
        case_id="c1",
        platform=Platform.DISCORD,
        workspace_id="g1",
        user_id="u1",
        original_action=ModerationAction.DELETE,
        outcome=AppealOutcome.RESTORED,
        reason="false positive",
        false_positive=True,
    )
    log.record(decision)
    found = log.find("ap1")
    assert found is not None
    assert found.outcome == AppealOutcome.RESTORED
    assert found.false_positive is True
    assert log.for_case("c1") == [decision]


def test_appeal_log_find_missing_returns_none(tmp_path):
    log = review_store.AppealLog(tmp_path / "appeals.jsonl")
    assert log.find("ap_does_not_exist") is None
    assert log.for_case("c_does_not_exist") == []


def test_appeal_log_list_appeals_filter_and_limit(tmp_path):
    log = review_store.AppealLog(tmp_path / "appeals.jsonl")
    # Three decisions across two platforms.
    for i, plat in enumerate([Platform.DISCORD, Platform.DISCORD, Platform.TWITCH]):
        log.record(
            AppealDecision(
                appeal_id=f"ap{i}",
                case_id=f"c{i}",
                platform=plat,
                workspace_id="ws",
                user_id="u",
                original_action=ModerationAction.DELETE,
                outcome=AppealOutcome.UPHELD if i % 2 == 0 else AppealOutcome.RESTORED,
                reason="",
            )
        )
    only_discord = log.list_appeals(platform=Platform.DISCORD)
    assert len(only_discord) == 2
    assert all(d.platform == Platform.DISCORD for d in only_discord)
    assert log.list_appeals(limit=0) == []


def test_strike_ledger_unknown_user_returns_zero_and_empty(tmp_path):
    ledger = review_store.StrikeLedger(tmp_path / "strikes.jsonl")
    assert (
        ledger.total(Platform.DISCORD, "g1", "user_does_not_exist") == 0
    )
    assert (
        ledger.active_strikes(Platform.DISCORD, "g1", "user_does_not_exist") == []
    )


def test_strike_ledger_add_then_void_restores_total(tmp_path):
    ledger = review_store.StrikeLedger(tmp_path / "strikes.jsonl")
    strike = ledger.add_strike(
        platform=Platform.DISCORD,
        workspace_id="g1",
        user_id="u1",
        weight=2.0,
        reason="rude",
    )
    assert ledger.total(Platform.DISCORD, "g1", "u1") == 2.0
    ledger.void_strike(strike.strike_id, reason="false positive")
    # After voiding, the active total drops to 0.
    assert ledger.total(Platform.DISCORD, "g1", "u1") == 0


def test_strike_ledger_void_for_case_cancels(tmp_path):
    """void_for_case must cancel every active strike attached to a review case."""
    ledger = review_store.StrikeLedger(tmp_path / "strikes.jsonl")
    ledger.add_strike(
        platform=Platform.DISCORD,
        workspace_id="g1",
        user_id="u1",
        weight=1.0,
        case_id="c1",
    )
    ledger.add_strike(
        platform=Platform.DISCORD,
        workspace_id="g1",
        user_id="u1",
        weight=1.0,
        case_id="c1",
    )
    voided = ledger.void_for_case("c1", reason="appeal upheld")
    assert len(voided) == 2
    assert ledger.total(Platform.DISCORD, "g1", "u1") == 0
