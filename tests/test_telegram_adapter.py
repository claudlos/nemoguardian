"""Tests for the Telegram group-moderation adapter.

No live Telegram: the Bot API is a fake recording client and the cascade is a
stub. Covers update parsing, admin-permission doctor checks, action planning,
repeat-offender escalation, dry-run, ignored chats and redacted audit records.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from nemoguardian.adapters import telegram
from nemoguardian.adapters.base import PlatformAdapter
from nemoguardian.bot import (
    AuditLog,
    BotConfig,
    ConfigStore,
    ModerationAction,
    ModerationContext,
    ModerationEvaluation,
    ModerationPlan,
    Platform,
)
from nemoguardian.schemas import ModerateResponse, VerdictLabel


class FakeCascade:
    def __init__(
        self,
        verdict: VerdictLabel,
        *,
        categories: list[str] | None = None,
    ) -> None:
        self.verdict = verdict
        self.categories = categories or []
        self.calls: list[str] = []

    def moderate(self, request, *, policy_engine=None):
        self.calls.append(request.text)
        return ModerateResponse(
            verdict=self.verdict,
            score=0.91,
            reasons=["fake"],
            categories=self.categories,
            matched_policy_rule="fake-rule" if self.categories else None,
            model_verdicts={},
            total_latency_ms=1.0,
            mode=request.mode,
            request_id="req-test",
        )


class FakeApi:
    """Records Bot API calls; optionally raises to exercise the error path."""

    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fail = fail or set()

    async def delete_message(self, chat_id, message_id):
        self.calls.append(("delete", (chat_id, message_id)))
        if "delete" in self.fail:
            raise RuntimeError("boom")

    async def ban_chat_member(self, chat_id, user_id):
        self.calls.append(("ban", (chat_id, user_id)))
        if "ban" in self.fail:
            raise RuntimeError("boom")

    async def restrict_chat_member(self, chat_id, user_id):
        self.calls.append(("mute", (chat_id, user_id)))
        if "mute" in self.fail:
            raise RuntimeError("boom")

    async def send_message(self, chat_id, text):
        self.calls.append(("send", (chat_id, text)))
        if "send" in self.fail:
            raise RuntimeError("boom")

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]


def _stores(tmp_path):
    return (
        ConfigStore(tmp_path / "bot-config.json"),
        AuditLog(tmp_path / "bot-audit.jsonl"),
    )


def _update(
    text: str = "drop your SSN",
    *,
    chat_id: int = 100,
    chat_type: str = "supergroup",
    user_id: int = 7,
    username: str = "viewer",
    message_id: int = 555,
    is_bot: bool = False,
) -> dict[str, Any]:
    """A raw Telegram webhook-shaped update (dict)."""
    return {
        "update_id": 1,
        "message": {
            "message_id": message_id,
            "chat": {"id": chat_id, "type": chat_type},
            "from": {"id": user_id, "username": username, "is_bot": is_bot},
            "text": text,
        },
    }


def _manual_evaluation(
    parsed: telegram.ParsedMessage,
    *,
    config: BotConfig | None = None,
    result: ModerateResponse | None = None,
    action: ModerationAction = ModerationAction.DELETE,
) -> ModerationEvaluation:
    config = config or BotConfig.default(Platform.TELEGRAM, parsed.chat_id)
    context = ModerationContext(
        platform=Platform.TELEGRAM,
        workspace_id=parsed.chat_id,
        channel_id=parsed.chat_id,
        message_id=parsed.message_id,
        user_id=parsed.user_id,
        username=parsed.username,
        text=parsed.text,
    )
    if result is None:
        result = ModerateResponse(
            verdict=VerdictLabel.UNSAFE,
            score=0.9,
            reasons=["fake"],
            categories=["pii"],
            matched_policy_rule="fake-rule",
            model_verdicts={},
            total_latency_ms=1.0,
            mode=config.mode,
            request_id="req-test",
        )
    return ModerationEvaluation(
        context=context,
        config=config,
        result=result,
        plan=ModerationPlan(action=action, reason="pii"),
    )


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parse_update_dict_supergroup():
    parsed = telegram.parse_update(_update())
    assert parsed is not None
    assert parsed.chat_id == "100"
    assert parsed.chat_type == "supergroup"
    assert parsed.message_id == "555"
    assert parsed.user_id == "7"
    assert parsed.username == "viewer"
    assert parsed.text == "drop your SSN"


def test_parse_update_sdk_object():
    """A python-telegram-bot-style Update object parses the same way."""
    update = SimpleNamespace(
        message=SimpleNamespace(
            message_id=9,
            chat=SimpleNamespace(id=42, type="group"),
            from_user=SimpleNamespace(id=3, username="bob", is_bot=False, first_name="Bob"),
            text="hello",
        )
    )
    parsed = telegram.parse_update(update)
    assert parsed is not None
    assert (parsed.chat_id, parsed.message_id, parsed.user_id, parsed.text) == ("42", "9", "3", "hello")


def test_parse_update_ignores_private_and_non_message():
    assert telegram.parse_update(_update(chat_type="private")) is None
    assert telegram.parse_update(_update(chat_type="channel")) is None
    assert telegram.parse_update({"message": {"chat": None}}) is None
    assert telegram.parse_update({"update_id": 1}) is None
    assert telegram.parse_update({}) is None


def test_parse_update_falls_back_to_first_name_and_caption():
    update = {
        "message": {
            "message_id": 1,
            "chat": {"id": 5, "type": "group"},
            "from": {"id": 2, "first_name": "Alice"},
            "caption": "see attachment",
        }
    }
    parsed = telegram.parse_update(update)
    assert parsed is not None
    assert parsed.username == "Alice"
    assert parsed.text == "see attachment"


# --------------------------------------------------------------------------- #
# capabilities / protocol conformance
# --------------------------------------------------------------------------- #
def test_capabilities_are_honest():
    caps = telegram.capabilities()
    assert caps == {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
        ModerationAction.DELETE,
        ModerationAction.BAN,
        ModerationAction.MUTE,
        ModerationAction.NOTIFY_MODS,
    }
    # timeout is NOT something a Telegram bot does -> excluded.
    assert ModerationAction.TIMEOUT not in caps


def test_adapter_satisfies_protocol(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(config_store=config_store, audit_log=audit_log)
    assert isinstance(adapter, PlatformAdapter)
    assert adapter.platform == Platform.TELEGRAM


# --------------------------------------------------------------------------- #
# action planning + enforcement
# --------------------------------------------------------------------------- #
async def test_unsafe_message_is_deleted_and_audited(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(
        FakeCascade(VerdictLabel.UNSAFE, categories=["pii"]),
        config_store=config_store,
        audit_log=audit_log,
    )
    api = FakeApi()

    evaluation = await adapter.handle_event(_update(), api=api)

    assert evaluation is not None
    assert "delete" in api.kinds()
    record = audit_log.recent()[0]
    assert record["platform"] == "telegram"
    assert record["action"] == "delete"
    assert record["execution_status"].startswith("delete")


async def test_mute_when_timeout_unsafe_configured(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.TELEGRAM, "100")
    config.timeout_unsafe = True
    config_store.save(config)
    adapter = telegram.TelegramAdapter(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    api = FakeApi()

    await adapter.handle_event(_update(), api=api)

    kinds = api.kinds()
    assert "mute" in kinds
    assert "delete" in kinds  # offending message still removed
    assert audit_log.recent()[0]["action"] == "mute"


async def test_repeat_offender_escalates_to_ban(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
        ban_after=2,
    )

    # First offense: delete only.
    api1 = FakeApi()
    await adapter.handle_event(_update(message_id=1), api=api1)
    assert "ban" not in api1.kinds()

    # Second offense by the same user in the same chat -> ban.
    api2 = FakeApi()
    await adapter.handle_event(_update(message_id=2), api=api2)
    assert "ban" in api2.kinds()
    assert audit_log.recent()[-1]["action"] == "ban"


async def test_controversial_message_flags_without_destructive_calls(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(
        FakeCascade(VerdictLabel.CONTROVERSIAL),
        config_store=config_store,
        audit_log=audit_log,
    )
    api = FakeApi()

    await adapter.handle_event(_update(), api=api)

    assert "delete" not in api.kinds()
    assert "ban" not in api.kinds()
    assert audit_log.recent()[0]["action"] == "flag"


async def test_safe_message_takes_no_action(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    api = FakeApi()

    await adapter.handle_event(_update(text="hello friends"), api=api)

    assert api.calls == []
    assert audit_log.recent()[0]["execution_status"] == "allowed"


async def test_notify_mods_posts_to_log_channel(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.TELEGRAM, "100")
    config.log_channel_id = "999"
    config_store.save(config)
    adapter = telegram.TelegramAdapter(
        FakeCascade(VerdictLabel.UNSAFE, categories=["pii"]),
        config_store=config_store,
        audit_log=audit_log,
    )
    api = FakeApi()

    await adapter.handle_event(_update(text="my ssn is 123-45-6789"), api=api)

    sends = [args for kind, args in api.calls if kind == "send"]
    assert sends and sends[0][0] == "999"
    # mod-log message must be redacted (no raw SSN).
    assert "123-45-6789" not in sends[0][1]
    assert "[ssn]" in sends[0][1]


# --------------------------------------------------------------------------- #
# dry-run + ignored chats + bots
# --------------------------------------------------------------------------- #
async def test_dry_run_makes_no_api_calls(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.TELEGRAM, "100")
    config.dry_run = True
    config_store.save(config)
    adapter = telegram.TelegramAdapter(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    api = FakeApi()

    await adapter.handle_event(_update(), api=api)

    assert api.calls == []
    record = audit_log.recent()[0]
    assert record["execution_status"] == "dry-run"
    assert record["dry_run"] is True


async def test_ignored_chat_is_skipped(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.TELEGRAM, "100")
    config.ignored_channel_ids = {"100"}
    config_store.save(config)
    cascade = FakeCascade(VerdictLabel.UNSAFE)
    adapter = telegram.TelegramAdapter(cascade, config_store=config_store, audit_log=audit_log)
    api = FakeApi()

    evaluation = await adapter.handle_event(_update(), api=api)

    assert evaluation is not None and evaluation.skipped
    assert evaluation.skip_reason == "ignored-channel"
    assert cascade.calls == []
    assert api.calls == []
    assert audit_log.recent() == []


async def test_bot_authored_message_is_ignored(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    cascade = FakeCascade(VerdictLabel.UNSAFE)
    adapter = telegram.TelegramAdapter(cascade, config_store=config_store, audit_log=audit_log)
    api = FakeApi()

    result = await adapter.handle_event(_update(is_bot=True), api=api)

    assert result is None
    assert cascade.calls == []
    assert api.calls == []


# --------------------------------------------------------------------------- #
# fail-safe action path
# --------------------------------------------------------------------------- #
async def test_api_failure_is_surfaced_not_raised(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    api = FakeApi(fail={"delete"})

    # Must not raise on the moderation path.
    await adapter.handle_event(_update(), api=api)

    record = audit_log.recent()[0]
    assert record["execution_status"] == "failed"
    assert "delete:RuntimeError" in record["error"]


async def test_ban_failure_is_partial_when_delete_succeeds(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
        ban_after=1,
    )
    api = FakeApi(fail={"ban"})

    await adapter.handle_event(_update(), api=api)

    record = audit_log.recent()[0]
    assert record["action"] == "ban"
    assert record["execution_status"] == "partial"
    assert "ban:RuntimeError" in record["error"]


async def test_mute_failure_is_partial_when_delete_succeeds(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.TELEGRAM, "100")
    config.timeout_unsafe = True
    config_store.save(config)
    adapter = telegram.TelegramAdapter(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    api = FakeApi(fail={"mute"})

    await adapter.handle_event(_update(), api=api)

    record = audit_log.recent()[0]
    assert record["action"] == "mute"
    assert record["execution_status"] == "partial"
    assert "mute:RuntimeError" in record["error"]


async def test_notify_mods_failure_is_partial_when_delete_succeeds(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.TELEGRAM, "100")
    config.log_channel_id = "999"
    config_store.save(config)
    adapter = telegram.TelegramAdapter(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    api = FakeApi(fail={"send"})

    await adapter.handle_event(_update(), api=api)

    record = audit_log.recent()[0]
    assert record["execution_status"] == "partial"
    assert "notify_mods:RuntimeError" in record["error"]


async def test_delete_disabled_unsafe_degrades_to_planned_flag():
    parsed = telegram.parse_update(_update())
    assert parsed is not None
    config = BotConfig.default(Platform.TELEGRAM, "100")
    config.delete_unsafe = False
    evaluation = _manual_evaluation(parsed, config=config)
    api = FakeApi()

    status, error = await telegram.apply_telegram_actions(api, parsed, evaluation)

    assert (status, error) == ("planned", None)
    assert evaluation.plan.action is ModerationAction.FLAG
    assert api.calls == []


async def test_apply_result_none_flags_without_api_calls():
    parsed = telegram.parse_update(_update())
    assert parsed is not None
    config = BotConfig.default(Platform.TELEGRAM, "100")
    context = ModerationContext(
        platform=Platform.TELEGRAM,
        workspace_id="100",
        channel_id="100",
        message_id=parsed.message_id,
        user_id=parsed.user_id,
        username=parsed.username,
        text=parsed.text,
    )
    evaluation = ModerationEvaluation(
        context=context,
        config=config,
        result=None,
        plan=ModerationPlan(action=ModerationAction.DELETE, reason="skipped"),
    )
    api = FakeApi()

    status, error = await telegram.apply_telegram_actions(api, parsed, evaluation)

    assert (status, error) == ("planned", None)
    assert evaluation.plan.action is ModerationAction.FLAG
    assert api.calls == []


# --------------------------------------------------------------------------- #
# doctor (offline admin-permission check)
# --------------------------------------------------------------------------- #
def test_doctor_reports_ready_with_full_admin_rights(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-test-token")
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(config_store=config_store, audit_log=audit_log)
    member = {
        "status": "administrator",
        "can_delete_messages": True,
        "can_restrict_members": True,
    }

    report = adapter.doctor("100", member)

    assert report["readiness"] == "ready"
    assert report["can_delete_messages"] is True
    assert report["can_restrict_members"] is True
    assert report["missing_required"] == []


def test_doctor_flags_missing_delete_permission(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-test-token")
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(config_store=config_store, audit_log=audit_log)
    member = {
        "status": "administrator",
        "can_delete_messages": False,
        "can_restrict_members": False,
    }

    report = adapter.doctor("100", member)

    assert report["readiness"] == "needs attention"
    assert "can_delete_messages" in report["missing_required"]
    assert "can_restrict_members" in report["missing_recommended"]


def test_doctor_handles_non_admin_and_missing_member(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(config_store=config_store, audit_log=audit_log)

    not_admin = adapter.doctor("100", {"status": "member"})
    assert "bot is not a chat administrator" in not_admin["missing_required"]
    assert "TELEGRAM_BOT_TOKEN is not set" in not_admin["issues"]

    unknown = adapter.doctor("100", None)
    assert "bot chat-member status unknown" in unknown["missing_required"]


def test_doctor_creator_has_implicit_rights(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(config_store=config_store, audit_log=audit_log)

    report = adapter.doctor("100", {"status": "creator"})

    assert report["can_delete_messages"] is True
    assert report["can_restrict_members"] is True
    assert report["readiness"] == "ready"


# --------------------------------------------------------------------------- #
# configure + degradation guard
# --------------------------------------------------------------------------- #
def test_configure_round_trips(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(config_store=config_store, audit_log=audit_log)

    updated = adapter.configure("100", dry_run=True)
    assert updated.dry_run is True
    assert adapter.configure("100").dry_run is True


async def test_adapter_apply_action_and_record_audit(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = telegram.TelegramAdapter(config_store=config_store, audit_log=audit_log)
    parsed = telegram.parse_update(_update())
    assert parsed is not None
    evaluation = _manual_evaluation(parsed)
    api = FakeApi()

    status, error = await adapter.apply_action(parsed, evaluation, api=api)
    adapter.record_audit(evaluation, execution_status=status, error=error)

    assert status == "delete"
    assert audit_log.recent()[0]["action"] == "delete"


async def test_bot_api_adapter_converts_ids_and_permissions(monkeypatch):
    telegram_module = ModuleType("telegram")
    calls: list[tuple[str, dict[str, Any]]] = []

    class ChatPermissions:
        def __init__(self, *, can_send_messages: bool) -> None:
            self.can_send_messages = can_send_messages

    class FakeBot:
        async def delete_message(self, **kwargs: Any) -> None:
            calls.append(("delete", kwargs))

        async def ban_chat_member(self, **kwargs: Any) -> None:
            calls.append(("ban", kwargs))

        async def restrict_chat_member(self, **kwargs: Any) -> None:
            calls.append(("restrict", kwargs))

        async def send_message(self, **kwargs: Any) -> None:
            calls.append(("send", kwargs))

    telegram_module.ChatPermissions = ChatPermissions
    monkeypatch.setitem(sys.modules, "telegram", telegram_module)
    api = telegram._BotApi(FakeBot())

    await api.delete_message("100", "5")
    await api.ban_chat_member("100", "7")
    await api.restrict_chat_member("100", "7")
    await api.send_message("100", "hello")

    assert calls[0] == ("delete", {"chat_id": 100, "message_id": 5})
    assert calls[1] == ("ban", {"chat_id": 100, "user_id": 7})
    assert calls[2][0] == "restrict"
    assert calls[2][1]["permissions"].can_send_messages is False
    assert calls[3] == ("send", {"chat_id": 100, "text": "hello"})


def test_unsupported_action_degrades_to_flag():
    decision = telegram.degrade_action(
        ModerationAction.TIMEOUT,
        telegram.capabilities(),
        Platform.TELEGRAM,
    )
    assert decision.degraded is True
    assert decision.action == ModerationAction.FLAG
    assert "telegram" in decision.reason


def test_run_bot_requires_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN env var required"):
        telegram.run_bot()


def test_build_application_wires_message_handler(monkeypatch):
    telegram_parent = ModuleType("telegram")
    telegram_ext = ModuleType("telegram.ext")
    handlers: list[Any] = []

    class FakeApplication:
        @staticmethod
        def builder():
            return FakeBuilder()

    class FakeBuilder:
        def token(self, token: str):
            self.seen_token = token
            return self

        def build(self):
            return self

        def add_handler(self, handler: Any) -> None:
            handlers.append(handler)

    class FakeMessageHandler:
        def __init__(self, filters_obj: Any, callback: Any) -> None:
            self.filters_obj = filters_obj
            self.callback = callback

    telegram_ext.Application = FakeApplication
    telegram_ext.MessageHandler = FakeMessageHandler
    telegram_ext.filters = SimpleNamespace(ALL="all-filters")
    monkeypatch.setitem(sys.modules, "telegram", telegram_parent)
    monkeypatch.setitem(sys.modules, "telegram.ext", telegram_ext)

    application = telegram.build_application("token-123")

    assert application.seen_token == "token-123"
    assert len(handlers) == 1
    assert handlers[0].filters_obj == "all-filters"


def test_run_bot_starts_polling(monkeypatch):
    class FakeApplication:
        def __init__(self) -> None:
            self.ran = False

        def run_polling(self) -> None:
            self.ran = True

    app = FakeApplication()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token-123")
    monkeypatch.setattr(telegram, "build_application", lambda token: app)

    telegram.run_bot()

    assert app.ran is True
