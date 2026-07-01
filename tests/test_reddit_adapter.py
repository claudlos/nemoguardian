"""Tests for the Reddit platform adapter.

No live Reddit: the mod-action client is faked (sync and async variants) and the
cascade is stubbed, so these tests need no GPU, network, token, or praw.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from nemoguardian.adapters import reddit
from nemoguardian.adapters.base import PlatformAdapter
from nemoguardian.bot import (
    AuditLog,
    BotConfig,
    ConfigStore,
    ModerationAction,
    ModerationEvaluation,
    ModerationPlan,
    Platform,
)
from nemoguardian.bot.audit import text_hash
from nemoguardian.schemas import ModerateResponse, VerdictLabel


class FakeCascade:
    def __init__(
        self,
        verdict: VerdictLabel,
        *,
        categories: list[str] | None = None,
        matched_policy_rule: str | None = None,
    ) -> None:
        self.verdict = verdict
        self.categories = categories or []
        self.matched_policy_rule = matched_policy_rule
        self.calls: list[dict[str, Any]] = []

    def moderate(self, request, *, policy_engine=None):
        self.calls.append({"text": request.text, "mode": request.mode.value})
        return ModerateResponse(
            verdict=self.verdict,
            score=0.9,
            reasons=["fake"],
            categories=self.categories,
            matched_policy_rule=self.matched_policy_rule or ("fake-rule" if self.categories else None),
            model_verdicts={},
            total_latency_ms=1.0,
            mode=request.mode,
            request_id="req-test",
        )


class FakeRedditClient:
    """Synchronous mod-action client stand-in recording every call."""

    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fail = fail or set()

    def _call(self, name: str, *args: Any) -> None:
        if name in self.fail:
            raise RuntimeError(f"{name} failed")
        self.calls.append((name, args))

    def remove(self, kind: str, item_id: str) -> None:
        self._call("remove", kind, item_id)

    def report(self, kind: str, item_id: str, reason: str) -> None:
        self._call("report", kind, item_id, reason)

    def send_modmail(self, subreddit: str, subject: str, body: str) -> None:
        self._call("send_modmail", subreddit, subject, body)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def args_for(self, name: str) -> tuple[Any, ...]:
        for call_name, args in self.calls:
            if call_name == name:
                return args
        raise AssertionError(f"no call to {name}")


class FakeAsyncRedditClient(FakeRedditClient):
    """Async mod-action client stand-in (coroutine methods)."""

    async def remove(self, kind: str, item_id: str) -> None:  # type: ignore[override]
        self._call("remove", kind, item_id)

    async def report(self, kind: str, item_id: str, reason: str) -> None:  # type: ignore[override]
        self._call("report", kind, item_id, reason)

    async def send_modmail(self, subreddit: str, subject: str, body: str) -> None:  # type: ignore[override]
        self._call("send_modmail", subreddit, subject, body)


def _stores(tmp_path):
    return (
        ConfigStore(tmp_path / "reddit-config.json"),
        AuditLog(tmp_path / "reddit-audit.jsonl"),
    )


def _comment(text: str, *, sub: str = "testsub", author: str = "bob", item_id: str = "c1"):
    return {
        "kind": "comment",
        "id": item_id,
        "subreddit": sub,
        "author": author,
        "body": text,
        "permalink": "/r/testsub/comments/x/_/c1/",
    }


def _submission(title: str, selftext: str = "", *, sub: str = "testsub", author: str = "bob", item_id: str = "s1"):
    return {
        "kind": "submission",
        "id": item_id,
        "subreddit": sub,
        "author": author,
        "title": title,
        "selftext": selftext,
    }


def _evaluation(
    *,
    config: BotConfig | None = None,
    action: ModerationAction,
    categories: list[str] | None = None,
    text: str = "drop your SSN",
    **plan_changes: Any,
):
    config = config or BotConfig.default(Platform.REDDIT, "testsub")
    item = reddit.RedditItem(
        kind="comment",
        item_id="c1",
        subreddit="testsub",
        author="bob",
        text=text,
    )
    context = reddit._context_from_item(item)
    result = ModerateResponse(
        verdict=VerdictLabel.UNSAFE,
        score=0.9,
        reasons=["fake"],
        categories=categories or ["PII"],
        matched_policy_rule="fake-rule",
        model_verdicts={},
        total_latency_ms=12.0,
        mode=config.mode,
        request_id="req-test",
    )
    plan = ModerationPlan(action=action, reason=", ".join(categories or ["PII"]))
    for key, value in plan_changes.items():
        setattr(plan, key, value)
    return item, ModerationEvaluation(context=context, config=config, result=result, plan=plan)


# --- parsing -------------------------------------------------------------


def test_parse_comment_and_submission():
    parsed_comment = reddit.parse_reddit_item(_comment("hi there"))
    assert parsed_comment == reddit.RedditItem(
        kind="comment",
        item_id="c1",
        subreddit="testsub",
        author="bob",
        text="hi there",
        permalink="/r/testsub/comments/x/_/c1/",
    )
    parsed_sub = reddit.parse_reddit_item(_submission("A title", "body text"))
    assert parsed_sub.kind == "submission"
    assert parsed_sub.text == "A title\nbody text"


def test_parse_praw_like_objects_stringify_author_and_subreddit():
    praw_comment = SimpleNamespace(
        id="c9",
        body="from praw",
        author=SimpleNamespace(name="alice"),
        subreddit=SimpleNamespace(display_name="pics"),
        permalink="/r/pics/_/c9/",
    )
    parsed = reddit.parse_reddit_item(praw_comment)
    assert parsed.kind == "comment"
    assert parsed.author == "alice"
    assert parsed.subreddit == "pics"


@pytest.mark.parametrize(
    "item",
    [
        {"kind": "comment", "subreddit": "testsub", "author": "bob", "body": "no id"},
        {"kind": "comment", "id": "c1", "subreddit": "testsub", "author": "bob", "body": "   "},
        None,
    ],
)
def test_parse_skips_incomplete_items(item):
    assert reddit.parse_reddit_item(item) is None


# --- capabilities & planning --------------------------------------------


def test_capabilities_are_honest():
    assert reddit.capabilities() == {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
        ModerationAction.DELETE,
        ModerationAction.NOTIFY_MODS,
    }


def test_reddit_decision_degrades_unsupported_actions():
    decision = reddit.reddit_decision(ModerationAction.TIMEOUT)
    assert decision.degraded is True
    assert decision.action == ModerationAction.FLAG
    assert "timeout" in decision.reason
    assert "degraded to flag" in decision.reason
    assert reddit.reddit_decision(ModerationAction.DELETE).degraded is False


# --- handler flow --------------------------------------------------------


async def test_handler_removes_unsafe_and_audits_redacted(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.REDDIT, "testsub")
    config.log_channel_id = "modmail"
    config_store.save(config)
    client = FakeRedditClient()
    handler = reddit.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )

    await handler(_comment("email jane@example.com SSN 123-45-6789"), client=client)

    assert "remove" in client.names()
    assert "send_modmail" in client.names()
    record = audit_log.recent()[0]
    assert record["action"] == "delete"
    assert record["execution_status"] == "remove+modmail"
    assert record["error"] is None
    assert "jane@example.com" not in record["text_excerpt"]
    assert "[email]" in record["text_excerpt"]
    assert record["text_sha256"] == text_hash("email jane@example.com SSN 123-45-6789")
    for _name, args in client.calls:
        assert all("jane@example.com" not in str(arg) for arg in args)


async def test_handler_moderates_submission(tmp_path):
    # Submissions (not just comments) are moderated end to end.
    config_store, audit_log = _stores(tmp_path)
    client = FakeRedditClient()

    result = await reddit.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )(_submission("A bad title", "with SSN 123-45-6789"), client=client)

    assert result is not None
    assert "remove" in client.names()
    assert client.args_for("remove")[0] == "submission"
    assert audit_log.recent()[0]["action"] == "delete"


async def test_handler_reports_controversial(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    client = FakeRedditClient()

    await reddit.make_handler(
        FakeCascade(VerdictLabel.CONTROVERSIAL, categories=["harassment"]),
        config_store=config_store,
        audit_log=audit_log,
    )(_comment("borderline"), client=client)

    assert "report" in client.names()
    assert "remove" not in client.names()
    assert audit_log.recent()[0]["action"] == "flag"


async def test_handler_skips_ignored_subreddit(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.REDDIT, "testsub")
    config.ignored_channel_ids.add("testsub")
    config_store.save(config)
    client = FakeRedditClient()
    cascade = FakeCascade(VerdictLabel.UNSAFE, categories=["PII"])

    await reddit.make_handler(cascade, config_store=config_store, audit_log=audit_log)(
        _comment("drop your SSN"), client=client
    )

    assert cascade.calls == []
    assert client.calls == []
    assert audit_log.recent() == []


async def test_handler_dry_run_plans_without_enforcing(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.REDDIT, "testsub")
    config.dry_run = True
    config.log_channel_id = "modmail"
    config_store.save(config)
    client = FakeRedditClient()

    await reddit.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )(_comment("drop your SSN"), client=client)

    assert client.calls == []
    record = audit_log.recent()[0]
    assert record["execution_status"] == "dry-run"
    assert record["dry_run"] is True


async def test_handler_async_client_is_awaited(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    client = FakeAsyncRedditClient()

    await reddit.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )(_comment("drop your SSN"), client=client)

    assert "remove" in client.names()
    assert audit_log.recent()[0]["action"] == "delete"


# --- apply / degradation -------------------------------------------------


async def test_apply_escalates_unsupported_enforcement_to_remove(tmp_path):
    # An UNSAFE verdict asking for BAN (unsupported) must not under-enforce to a
    # bare report: it falls back to the strongest supported enforcement (remove).
    config = BotConfig.default(Platform.REDDIT, "testsub")
    config.log_channel_id = "modmail"
    client = FakeRedditClient()
    item, evaluation = _evaluation(config=config, action=ModerationAction.BAN)

    status, error = await reddit.apply_reddit_actions(client, item, evaluation)

    assert error is None
    assert status == "remove+modmail"
    assert evaluation.plan.action == ModerationAction.DELETE
    assert "report" not in client.names()
    assert "remove" in client.names()
    note = client.args_for("send_modmail")[2]
    assert "escalated to delete" in note


async def test_apply_escalates_timeout_to_remove_not_report(tmp_path):
    config = BotConfig.default(Platform.REDDIT, "testsub")
    config.log_channel_id = "modmail"
    client = FakeRedditClient()
    item, evaluation = _evaluation(config=config, action=ModerationAction.TIMEOUT)

    status, error = await reddit.apply_reddit_actions(client, item, evaluation)

    assert error is None
    assert status == "remove+modmail"
    assert evaluation.plan.action == ModerationAction.DELETE
    assert "remove" in client.names()


async def test_apply_degrades_to_report_when_verdict_not_unsafe(tmp_path):
    # Escalation is gated on an UNSAFE verdict; a controversial item still reports
    # for a genuinely unsupported action.
    config = BotConfig.default(Platform.REDDIT, "testsub")
    config.log_channel_id = "modmail"
    client = FakeRedditClient()
    item, evaluation = _evaluation(config=config, action=ModerationAction.TIMEOUT)
    evaluation.result.verdict = VerdictLabel.CONTROVERSIAL

    status, _error = await reddit.apply_reddit_actions(client, item, evaluation)

    assert status == "report+modmail"
    assert evaluation.plan.action == ModerationAction.FLAG
    assert "remove" not in client.names()


async def test_apply_allows_without_side_effects():
    item, evaluation = _evaluation(action=ModerationAction.ALLOW, categories=[])
    client = FakeRedditClient()

    status, error = await reddit.apply_reddit_actions(client, item, evaluation)

    assert (status, error) == ("allowed", None)
    assert client.calls == []


async def test_apply_reports_partial_when_remove_fails():
    config = BotConfig.default(Platform.REDDIT, "testsub")
    config.log_channel_id = "modmail"
    client = FakeRedditClient(fail={"remove"})
    item, evaluation = _evaluation(config=config, action=ModerationAction.DELETE)

    status, error = await reddit.apply_reddit_actions(client, item, evaluation)

    assert status == "partial"
    assert error == "remove:RuntimeError"
    assert "send_modmail" in client.names()


async def test_apply_without_client():
    config = BotConfig.default(Platform.REDDIT, "testsub")
    item, evaluation = _evaluation(config=config, action=ModerationAction.DELETE)

    status, error = await reddit.apply_reddit_actions(None, item, evaluation)

    assert status == "failed"
    assert error == "remove:no-client"


# --- adapter surface -----------------------------------------------------


def test_reddit_adapter_satisfies_protocol(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = reddit.RedditAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    assert isinstance(adapter, PlatformAdapter)
    assert adapter.platform == Platform.REDDIT
    assert adapter.capabilities() == reddit.capabilities()


def test_adapter_configure_round_trips(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = reddit.RedditAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    updated = adapter.configure("testsub", log_channel_id="modmail", dry_run=True)
    assert updated.log_channel_id == "modmail"
    assert adapter.configure("testsub").dry_run is True


async def test_adapter_handle_event_end_to_end(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = reddit.RedditAdapter(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )
    client = FakeRedditClient()

    await adapter.handle_event(_submission("A bad title", "with SSN"), client=client)

    assert "remove" in client.names()
    assert client.args_for("remove")[0] == "submission"
    assert audit_log.recent()[0]["action"] == "delete"


# --- doctor --------------------------------------------------------------


def test_doctor_reports_not_moderator(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = reddit.RedditAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )

    report = adapter.doctor("testsub")

    assert report["platform"] == "reddit"
    assert report["readiness"] == "needs attention"
    assert report["is_moderator"] is False
    assert "bot is not a subreddit moderator" in report["missing_required"]
    assert report["capabilities"] == sorted(a.value for a in reddit.capabilities())


def test_doctor_ready_when_full_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    config_store, audit_log = _stores(tmp_path)
    adapter = reddit.RedditAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )

    report = adapter.doctor("testsub", moderator=SimpleNamespace(mod_permissions=["all"]))

    assert report["readiness"] == "ready"
    assert report["issues"] == []
    assert report["can_remove"] is True
    assert report["can_modmail"] is True


def test_doctor_flags_missing_posts_permission(tmp_path, monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    config_store, audit_log = _stores(tmp_path)
    adapter = reddit.RedditAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )

    report = adapter.doctor("testsub", moderator=["mail"])

    assert report["is_moderator"] is True
    assert report["can_remove"] is False
    assert "posts" in report["missing_required"]
    assert report["can_modmail"] is True


# --- entry point & lazy import ------------------------------------------


def test_run_bot_requires_credentials(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET"):
        reddit.run_bot()


def test_run_bot_requires_subreddit(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.delenv("REDDIT_SUBREDDIT", raising=False)
    with pytest.raises(RuntimeError, match="REDDIT_SUBREDDIT env var required"):
        reddit.run_bot()


def test_import_nemoguardian_does_not_require_praw():
    assert "praw" not in sys.modules
