"""Tests for the Slack platform adapter.

No live Slack: the web client is faked (sync and async variants) and the cascade
is stubbed, so these tests need no GPU, network, or token.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from nemoguardian.adapters import slack
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


class FakeSlackClient:
    """Synchronous Slack ``WebClient`` stand-in recording every call."""

    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail = fail or set()

    def _call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        if name in self.fail:
            raise RuntimeError(f"{name} failed")
        self.calls.append((name, kwargs))
        return {"ok": True}

    def chat_delete(self, **kwargs: Any):
        return self._call("chat_delete", **kwargs)

    def chat_postMessage(self, **kwargs: Any):
        return self._call("chat_postMessage", **kwargs)

    def chat_postEphemeral(self, **kwargs: Any):
        return self._call("chat_postEphemeral", **kwargs)

    def reactions_add(self, **kwargs: Any):
        return self._call("reactions_add", **kwargs)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def kwargs_for(self, name: str) -> dict[str, Any]:
        for call_name, kwargs in self.calls:
            if call_name == name:
                return kwargs
        raise AssertionError(f"no call to {name}")


class FakeAsyncSlackClient(FakeSlackClient):
    """Async ``AsyncWebClient`` stand-in (coroutine methods)."""

    async def chat_delete(self, **kwargs: Any):  # type: ignore[override]
        return self._call("chat_delete", **kwargs)

    async def chat_postMessage(self, **kwargs: Any):  # type: ignore[override]
        return self._call("chat_postMessage", **kwargs)

    async def chat_postEphemeral(self, **kwargs: Any):  # type: ignore[override]
        return self._call("chat_postEphemeral", **kwargs)

    async def reactions_add(self, **kwargs: Any):  # type: ignore[override]
        return self._call("reactions_add", **kwargs)


def _stores(tmp_path):
    return (
        ConfigStore(tmp_path / "slack-config.json"),
        AuditLog(tmp_path / "slack-audit.jsonl"),
    )


def _event(text: str, *, team: str = "T1", channel: str = "C1", user: str = "U1", ts: str = "1700000000.000100"):
    return {
        "type": "event_callback",
        "team_id": team,
        "event": {
            "type": "message",
            "user": user,
            "channel": channel,
            "text": text,
            "ts": ts,
            "channel_type": "channel",
        },
    }


def _evaluation(
    *,
    config: BotConfig | None = None,
    action: ModerationAction,
    categories: list[str] | None = None,
    text: str = "drop your SSN",
    **plan_changes: Any,
):
    config = config or BotConfig.default(Platform.SLACK, "T1")
    message = slack.SlackMessage(team_id="T1", channel_id="C1", user_id="U1", text=text, ts="1.2")
    context = slack._context_from_message(message)
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
    return message, ModerationEvaluation(context=context, config=config, result=result, plan=plan)


# --- parsing -------------------------------------------------------------


def test_parse_envelope_and_bare_event_match():
    parsed_env = slack.parse_slack_event(_event("hi"))
    parsed_bare = slack.parse_slack_event(_event("hi")["event"] | {"team": "T1"})
    assert parsed_env == slack.SlackMessage(
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        text="hi",
        ts="1700000000.000100",
        channel_type="channel",
    )
    assert parsed_bare.team_id == "T1"
    assert parsed_bare.text == "hi"


@pytest.mark.parametrize(
    "event",
    [
        {"type": "event_callback", "event": {"type": "reaction_added"}},
        {"type": "event_callback", "event": {"type": "message", "subtype": "message_changed"}},
        {"type": "event_callback", "event": {"type": "message", "bot_id": "B1", "user": "U1", "channel": "C1", "ts": "1"}},
        {"type": "event_callback", "event": {"type": "message", "channel": "C1", "ts": "1"}},  # no user
        "not-a-dict",
    ],
)
def test_parse_skips_non_user_messages(event):
    assert slack.parse_slack_event(event) is None


# --- capabilities & planning --------------------------------------------


def test_capabilities_are_honest():
    assert slack.capabilities() == {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
        ModerationAction.NOTIFY_MODS,
        ModerationAction.NOTIFY_USER,
    }


def test_slack_decision_degrades_unsupported_actions():
    delete_decision = slack.slack_decision(ModerationAction.DELETE)
    assert delete_decision.degraded is True
    assert delete_decision.action == ModerationAction.FLAG
    decision = slack.slack_decision(ModerationAction.TIMEOUT)
    assert decision.degraded is True
    assert decision.action == ModerationAction.FLAG
    assert "timeout" in decision.reason
    assert "degraded to flag" in decision.reason


# --- handler flow --------------------------------------------------------


async def test_handler_flags_unsafe_and_audits_redacted(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.SLACK, "T1")
    config.log_channel_id = "C-mod"
    config_store.save(config)
    client = FakeSlackClient()
    handler = slack.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )

    await handler(
        _event("email jane@example.com SSN 123-45-6789"),
        client=client,
    )

    assert "chat_delete" not in client.names()
    assert "chat_postMessage" in client.names()  # warning + mod-log

    record = audit_log.recent()[0]
    assert record["action"] == "flag"
    assert record["execution_status"] == "warning+flag"
    assert record["error"] is None
    assert "jane@example.com" not in record["text_excerpt"]
    assert "123-45-6789" not in record["text_excerpt"]
    assert "[email]" in record["text_excerpt"]
    assert record["text_sha256"] == text_hash("email jane@example.com SSN 123-45-6789")
    # raw text never appears in any posted Slack message
    for _name, kwargs in client.calls:
        assert "jane@example.com" not in kwargs.get("text", "")


async def test_handler_preserves_bolt_body_team_id_for_bare_events(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.SLACK, "T-body")
    config.log_channel_id = "C-mod"
    config_store.save(config)
    client = FakeSlackClient()
    handler = slack.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )

    await handler(
        _event("drop your SSN", team="T-body")["event"],
        client=client,
        body={"team_id": "T-body"},
    )

    record = audit_log.recent()[0]
    assert record["workspace_id"] == "T-body"
    assert "slack-T-body-" in record["case_id"]


async def test_handler_skips_ignored_channel(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.SLACK, "T1")
    config.ignored_channel_ids.add("C1")
    config_store.save(config)
    client = FakeSlackClient()
    cascade = FakeCascade(VerdictLabel.UNSAFE, categories=["PII"])

    await slack.make_handler(cascade, config_store=config_store, audit_log=audit_log)(
        _event("drop your SSN"), client=client
    )

    assert cascade.calls == []
    assert client.calls == []
    assert audit_log.recent() == []


async def test_handler_dry_run_plans_without_enforcing(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.SLACK, "T1")
    config.dry_run = True
    config.log_channel_id = "C-mod"
    config_store.save(config)
    client = FakeSlackClient()

    await slack.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )(_event("drop your SSN"), client=client)

    assert "chat_delete" not in client.names()
    # only the mod-log post happens in dry-run
    assert client.names() == ["chat_postMessage"]
    assert "applied: `dry-run`" in client.kwargs_for("chat_postMessage")["text"]
    record = audit_log.recent()[0]
    assert record["execution_status"] == "dry-run"
    assert record["dry_run"] is True


async def test_handler_async_client_is_awaited(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    client = FakeAsyncSlackClient()

    await slack.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )(_event("drop your SSN"), client=client)

    assert "chat_delete" not in client.names()
    assert audit_log.recent()[0]["action"] == "flag"


# --- apply / degradation -------------------------------------------------


async def test_apply_degrades_unsupported_timeout_to_flag(tmp_path):
    config = BotConfig.default(Platform.SLACK, "T1")
    config.log_channel_id = "C-mod"
    client = FakeSlackClient()
    message, evaluation = _evaluation(
        config=config,
        action=ModerationAction.TIMEOUT,
        timeout_user=True,
    )

    status, error = await slack.apply_slack_actions(client, message, evaluation)

    assert error is None
    assert status == "flag"
    # planned action is degraded to flag for an honest audit trail
    assert evaluation.plan.action == ModerationAction.FLAG
    # no Slack timeout API was attempted; only the mod-log carries the reason
    assert client.names() == ["chat_postMessage"]
    mod_log = client.kwargs_for("chat_postMessage")["text"]
    assert "degraded to flag" in mod_log
    assert "action: `flag`" in mod_log


async def test_apply_allows_without_side_effects():
    message, evaluation = _evaluation(action=ModerationAction.ALLOW, categories=[])
    client = FakeSlackClient()

    status, error = await slack.apply_slack_actions(client, message, evaluation)

    assert (status, error) == ("allowed", None)
    assert client.calls == []


async def test_apply_delete_degrades_to_flag_without_calling_delete():
    config = BotConfig.default(Platform.SLACK, "T1")
    client = FakeSlackClient(fail={"chat_delete"})
    message, evaluation = _evaluation(
        config=config,
        action=ModerationAction.DELETE,
        delete_message=True,
        public_warning=True,
    )

    status, error = await slack.apply_slack_actions(client, message, evaluation)

    assert status == "warning+flag"
    assert error is None
    assert evaluation.plan.action == ModerationAction.FLAG
    assert "chat_delete" not in client.names()
    assert "chat_postMessage" in client.names()  # warning still posted


async def test_apply_delete_degrades_to_flag_without_client():
    config = BotConfig.default(Platform.SLACK, "T1")
    message, evaluation = _evaluation(
        config=config,
        action=ModerationAction.DELETE,
        delete_message=True,
    )

    status, error = await slack.apply_slack_actions(None, message, evaluation)

    assert status == "flag"
    assert error is None
    assert evaluation.plan.action == ModerationAction.FLAG


# --- adapter surface -----------------------------------------------------


def test_slack_adapter_satisfies_protocol(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = slack.SlackAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    assert isinstance(adapter, PlatformAdapter)
    assert adapter.platform == Platform.SLACK
    assert adapter.capabilities() == slack.capabilities()


def test_adapter_configure_round_trips(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = slack.SlackAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    updated = adapter.configure("T1", log_channel_id="C-mod", dry_run=True)
    assert updated.log_channel_id == "C-mod"
    assert adapter.configure("T1").dry_run is True


async def test_adapter_handle_event_end_to_end(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = slack.SlackAdapter(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )
    client = FakeSlackClient()

    await adapter.handle_event(_event("drop your SSN"), client=client)

    assert "chat_delete" not in client.names()
    assert audit_log.recent()[0]["action"] == "flag"


# --- doctor --------------------------------------------------------------


def test_doctor_reports_missing_scopes_and_setup(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = slack.SlackAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )

    report = adapter.doctor("T1", granted_scopes=["channels:history"], token_configured=False)

    assert report["platform"] == "slack"
    assert report["readiness"] == "needs attention"
    assert "chat:write" in report["missing_required_scopes"]
    assert "SLACK_BOT_TOKEN is not set" in report["issues"]
    assert "mod-log channel is not set" in report["issues"]
    assert report["capabilities"] == sorted(a.value for a in slack.capabilities())


def test_doctor_ready_when_configured(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = slack.SlackAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    adapter.configure("T1", log_channel_id="C-mod")

    report = adapter.doctor(
        "T1",
        granted_scopes=list(slack.SLACK_REQUIRED_SCOPES),
        token_configured=True,
    )

    assert report["readiness"] == "ready"
    assert report["issues"] == []
    assert report["missing_required_scopes"] == []
    assert report["scopes_verified"] is True


def test_doctor_flags_unknown_scopes(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = slack.SlackAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    adapter.configure("T1", log_channel_id="C-mod")

    report = adapter.doctor("T1", token_configured=True)

    assert report["scopes_verified"] is False
    assert "granted scopes unknown (cannot verify)" in report["issues"]
    assert report["missing_required_scopes"] == list(slack.SLACK_REQUIRED_SCOPES)


# --- entry point ---------------------------------------------------------


def test_run_bot_requires_token(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="SLACK_BOT_TOKEN env var required"):
        slack.run_bot()


def test_run_bot_requires_signing_secret_for_http_mode(monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="SLACK_SIGNING_SECRET env var required"):
        slack.run_bot()


def test_run_bot_starts_http_when_signing_secret_present(monkeypatch):
    class FakeApp:
        def __init__(self) -> None:
            self.started_ports: list[int] = []

        def start(self, *, port: int) -> None:
            self.started_ports.append(port)

    app = FakeApp()
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing-test")
    monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
    monkeypatch.setenv("PORT", "3123")
    monkeypatch.setattr(slack, "build_app", lambda: app)

    slack.run_bot()

    assert app.started_ports == [3123]


def test_run_bot_allows_socket_mode_without_signing_secret(monkeypatch):
    socket_mode_module = ModuleType("slack_bolt.adapter.socket_mode")
    starts: list[tuple[Any, str]] = []

    class FakeSocketModeHandler:
        def __init__(self, app: Any, app_token: str) -> None:
            self.app = app
            self.app_token = app_token

        def start(self) -> None:
            starts.append((self.app, self.app_token))

    app = object()
    socket_mode_module.SocketModeHandler = FakeSocketModeHandler
    monkeypatch.setitem(sys.modules, "slack_bolt.adapter.socket_mode", socket_mode_module)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-test")
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)
    monkeypatch.setattr(slack, "build_app", lambda: app)

    slack.run_bot()

    assert starts == [(app, "xapp-test")]


def test_import_nemoguardian_does_not_require_slack_bolt():
    assert "slack_bolt" not in sys.modules


def test_build_app_passes_bolt_body_to_handler(monkeypatch):
    slack_bolt_module = ModuleType("slack_bolt")
    apps: list[Any] = []

    class FakeApp:
        def __init__(self, *, token: str | None, signing_secret: str | None) -> None:
            self.token = token
            self.signing_secret = signing_secret
            self.listeners: dict[str, Any] = {}
            apps.append(self)

        def event(self, name: str):
            def decorator(func):
                self.listeners[name] = func
                return func

            return decorator

    captured: list[dict[str, Any]] = []

    async def fake_handler(event: Any, *, client: Any = None, body: Any = None) -> None:
        captured.append({"event": event, "client": client, "body": body})

    slack_bolt_module.App = FakeApp
    monkeypatch.setitem(sys.modules, "slack_bolt", slack_bolt_module)
    monkeypatch.setattr(slack, "make_handler", lambda **_kwargs: fake_handler)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "signing-test")

    app = slack.build_app()
    event = _event("hi", team="T-live")["event"]
    body = {"team_id": "T-live", "event": event}
    app.listeners["message"](event, object(), body)

    assert captured == [{"event": event, "client": captured[0]["client"], "body": body}]
