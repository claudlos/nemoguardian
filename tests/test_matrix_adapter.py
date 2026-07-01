"""Tests for the Matrix platform adapter.

No live Matrix: the redaction client is faked (sync and async variants) and the
cascade is stubbed, so these tests need no GPU, network, token, or matrix-nio.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from nemoguardian.adapters import matrix
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


class FakeMatrixClient:
    """Synchronous redaction-client stand-in recording every call."""

    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fail = fail or set()

    def _call(self, name: str, *args: Any) -> None:
        if name in self.fail:
            raise RuntimeError(f"{name} failed")
        self.calls.append((name, args))

    def redact_message(self, room_id: str, event_id: str, reason: str) -> None:
        self._call("redact_message", room_id, event_id, reason)

    def send_notice(self, room_id: str, text: str) -> None:
        self._call("send_notice", room_id, text)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def args_for(self, name: str) -> tuple[Any, ...]:
        for call_name, args in self.calls:
            if call_name == name:
                return args
        raise AssertionError(f"no call to {name}")


class FakeAsyncMatrixClient(FakeMatrixClient):
    """Async redaction-client stand-in (coroutine methods)."""

    async def redact_message(self, room_id: str, event_id: str, reason: str) -> None:  # type: ignore[override]
        self._call("redact_message", room_id, event_id, reason)

    async def send_notice(self, room_id: str, text: str) -> None:  # type: ignore[override]
        self._call("send_notice", room_id, text)


def _stores(tmp_path):
    return (
        ConfigStore(tmp_path / "matrix-config.json"),
        AuditLog(tmp_path / "matrix-audit.jsonl"),
    )


def _event(text: str, *, room: str = "!r:hs", sender: str = "@u:hs", event_id: str = "$e1", msgtype: str = "m.text"):
    return {
        "type": "m.room.message",
        "room_id": room,
        "sender": sender,
        "event_id": event_id,
        "content": {"msgtype": msgtype, "body": text},
    }


def _evaluation(
    *,
    config: BotConfig | None = None,
    action: ModerationAction,
    categories: list[str] | None = None,
    text: str = "drop your SSN",
    **plan_changes: Any,
):
    config = config or BotConfig.default(Platform.MATRIX, "!r:hs")
    event = matrix.MatrixEvent(
        room_id="!r:hs",
        event_id="$e1",
        sender="@u:hs",
        body=text,
        msgtype="m.text",
    )
    context = matrix._context_from_event(event)
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
    return event, ModerationEvaluation(context=context, config=config, result=result, plan=plan)


# --- parsing -------------------------------------------------------------


def test_parse_dict_and_nio_object_match():
    parsed_dict = matrix.parse_matrix_event(_event("hi"))
    nio_like = SimpleNamespace(
        event_id="$e1",
        sender="@u:hs",
        body="hi",
        source={"content": {"msgtype": "m.text"}},
    )
    room = SimpleNamespace(room_id="!r:hs")
    parsed_obj = matrix.parse_matrix_event(nio_like, room=room)
    assert parsed_dict == matrix.MatrixEvent(
        room_id="!r:hs", event_id="$e1", sender="@u:hs", body="hi", msgtype="m.text"
    )
    assert parsed_obj.room_id == "!r:hs"
    assert parsed_obj.body == "hi"


def test_parse_room_id_override_from_room():
    event = _event("hi")
    del event["room_id"]
    parsed = matrix.parse_matrix_event(event, room="!other:hs")
    assert parsed is not None
    assert parsed.room_id == "!other:hs"


@pytest.mark.parametrize(
    "event",
    [
        {"type": "m.room.redaction", "room_id": "!r:hs", "sender": "@u:hs", "event_id": "$e"},
        _event("weird type", msgtype="m.location"),
        {"type": "m.room.message", "sender": "@u:hs", "event_id": "$e", "content": {"body": "no room"}},
        {"type": "m.room.message", "room_id": "!r:hs", "event_id": "$e", "content": {"body": "no sender"}},
        None,
    ],
)
def test_parse_skips_non_moderatable_events(event):
    assert matrix.parse_matrix_event(event) is None


def test_parse_moderates_notice_from_other_user():
    # m.notice from another user is real content, not a bot loop — moderate it.
    parsed = matrix.parse_matrix_event(_event("evasion attempt", msgtype="m.notice"))
    assert parsed is not None
    assert parsed.msgtype == "m.notice"
    assert parsed.body == "evasion attempt"


def test_parse_skips_bot_own_event_by_sender_id():
    # The self-loop guard is by identity: the bot's own event is skipped even
    # when it is a normal m.text message (not just m.notice).
    event = _event("bot posted this", sender="@bot:hs", msgtype="m.text")
    assert matrix.parse_matrix_event(event, bot_user_id="@bot:hs") is None
    # a different sender with the same content is still moderated
    assert matrix.parse_matrix_event(event, bot_user_id="@someoneelse:hs") is not None


def test_parse_reads_notice_msgtype_from_nio_source():
    # nio event objects keep msgtype under source['content']; a bot notice there
    # is still skipped by sender identity, while another user's notice is kept.
    nio_notice = SimpleNamespace(
        event_id="$e1",
        sender="@bot:hs",
        body="enforcement notice",
        source={"content": {"msgtype": "m.notice", "body": "enforcement notice"}},
    )
    room = SimpleNamespace(room_id="!r:hs")
    assert matrix.parse_matrix_event(nio_notice, room=room, bot_user_id="@bot:hs") is None
    other = SimpleNamespace(
        event_id="$e2",
        sender="@u:hs",
        body="sneaky notice",
        source={"content": {"msgtype": "m.notice", "body": "sneaky notice"}},
    )
    parsed = matrix.parse_matrix_event(other, room=room, bot_user_id="@bot:hs")
    assert parsed is not None
    assert parsed.msgtype == "m.notice"


# --- capabilities & planning --------------------------------------------


def test_capabilities_are_honest():
    assert matrix.capabilities() == {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
        ModerationAction.DELETE,
        ModerationAction.NOTIFY_MODS,
    }


def test_matrix_decision_degrades_unsupported_actions():
    decision = matrix.matrix_decision(ModerationAction.TIMEOUT)
    assert decision.degraded is True
    assert decision.action == ModerationAction.FLAG
    assert "timeout" in decision.reason
    assert "degraded to flag" in decision.reason
    # redact (delete) is supported and passes through
    assert matrix.matrix_decision(ModerationAction.DELETE).degraded is False


# --- handler flow --------------------------------------------------------


async def test_handler_redacts_unsafe_and_audits_redacted(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.log_channel_id = "!mods:hs"
    config_store.save(config)
    client = FakeMatrixClient()
    handler = matrix.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )

    await handler(_event("email jane@example.com SSN 123-45-6789"), client=client)

    assert "redact_message" in client.names()
    assert "send_notice" in client.names()
    record = audit_log.recent()[0]
    assert record["action"] == "delete"
    assert record["execution_status"] == "redact+notify_mods"
    assert record["error"] is None
    assert "jane@example.com" not in record["text_excerpt"]
    assert "[email]" in record["text_excerpt"]
    assert record["text_sha256"] == text_hash("email jane@example.com SSN 123-45-6789")
    # raw text never leaks into any posted notice
    for _name, args in client.calls:
        assert all("jane@example.com" not in str(arg) for arg in args)


async def test_handler_skips_ignored_room(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.ignored_channel_ids.add("!r:hs")
    config_store.save(config)
    client = FakeMatrixClient()
    cascade = FakeCascade(VerdictLabel.UNSAFE, categories=["PII"])

    await matrix.make_handler(cascade, config_store=config_store, audit_log=audit_log)(
        _event("drop your SSN"), client=client
    )

    assert cascade.calls == []
    assert client.calls == []
    assert audit_log.recent() == []


async def test_handler_dry_run_plans_without_enforcing(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.dry_run = True
    config.log_channel_id = "!mods:hs"
    config_store.save(config)
    client = FakeMatrixClient()

    await matrix.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )(_event("drop your SSN"), client=client)

    # dry-run means no external effect: not even the mod-room notice is sent.
    assert client.calls == []
    record = audit_log.recent()[0]
    assert record["execution_status"] == "dry-run"
    assert record["dry_run"] is True


async def test_handler_skips_bot_own_event_but_moderates_others(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.log_channel_id = "!mods:hs"
    config_store.save(config)
    cascade = FakeCascade(VerdictLabel.UNSAFE, categories=["PII"])
    handler = matrix.make_handler(
        cascade,
        config_store=config_store,
        audit_log=audit_log,
        bot_user_id="@bot:hs",
    )

    # bot's own event (even m.notice) is skipped before the cascade runs
    client = FakeMatrixClient()
    assert await handler(_event("my own notice", sender="@bot:hs", msgtype="m.notice"), client=client) is None
    assert cascade.calls == []
    assert client.calls == []

    # another user's m.notice IS moderated
    result = await handler(_event("evasion", sender="@evil:hs", msgtype="m.notice"), client=client)
    assert result is not None
    assert "redact_message" in client.names()


async def test_handler_async_client_is_awaited(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.log_channel_id = "!mods:hs"
    config_store.save(config)
    client = FakeAsyncMatrixClient()

    await matrix.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )(_event("drop your SSN"), client=client)

    assert "redact_message" in client.names()
    assert audit_log.recent()[0]["action"] == "delete"


# --- apply / degradation -------------------------------------------------


async def test_apply_escalates_unsupported_enforcement_to_delete(tmp_path):
    # An UNSAFE verdict asking for BAN (unsupported) must not under-enforce to a
    # bare flag: it falls back to the strongest supported enforcement (redact).
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.log_channel_id = "!mods:hs"
    client = FakeMatrixClient()
    event, evaluation = _evaluation(config=config, action=ModerationAction.BAN)

    status, error = await matrix.apply_matrix_actions(client, event, evaluation)

    assert error is None
    assert status == "redact+notify_mods"
    assert evaluation.plan.action == ModerationAction.DELETE
    assert "redact_message" in client.names()
    notice = client.args_for("send_notice")[1]
    assert "escalated to delete" in notice
    assert "action: delete" in notice


async def test_apply_escalates_timeout_to_delete_not_flag(tmp_path):
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.log_channel_id = "!mods:hs"
    client = FakeMatrixClient()
    event, evaluation = _evaluation(config=config, action=ModerationAction.TIMEOUT)

    status, error = await matrix.apply_matrix_actions(client, event, evaluation)

    assert error is None
    assert status == "redact+notify_mods"
    assert evaluation.plan.action == ModerationAction.DELETE
    assert "redact_message" in client.names()


async def test_apply_degrades_to_flag_when_verdict_not_unsafe(tmp_path):
    # Escalation is gated on an UNSAFE verdict; a controversial item still just
    # flags for a genuinely unsupported action.
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.log_channel_id = "!mods:hs"
    client = FakeMatrixClient()
    event, evaluation = _evaluation(config=config, action=ModerationAction.TIMEOUT)
    evaluation.result.verdict = VerdictLabel.CONTROVERSIAL

    status, _error = await matrix.apply_matrix_actions(client, event, evaluation)

    assert status == "flag+notify_mods"
    assert evaluation.plan.action == ModerationAction.FLAG
    assert "redact_message" not in client.names()


async def test_apply_allows_without_side_effects():
    event, evaluation = _evaluation(action=ModerationAction.ALLOW, categories=[])
    client = FakeMatrixClient()

    status, error = await matrix.apply_matrix_actions(client, event, evaluation)

    assert (status, error) == ("allowed", None)
    assert client.calls == []


async def test_apply_reports_partial_when_redact_fails():
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    config.log_channel_id = "!mods:hs"
    client = FakeMatrixClient(fail={"redact_message"})
    event, evaluation = _evaluation(config=config, action=ModerationAction.DELETE)

    status, error = await matrix.apply_matrix_actions(client, event, evaluation)

    assert status == "partial"
    assert error == "redact:RuntimeError"
    # mod-room still notified despite the redaction failure
    assert "send_notice" in client.names()


async def test_apply_without_mod_room_or_client():
    config = BotConfig.default(Platform.MATRIX, "!r:hs")
    event, evaluation = _evaluation(config=config, action=ModerationAction.DELETE)

    status, error = await matrix.apply_matrix_actions(None, event, evaluation)

    assert status == "failed"
    assert error == "redact:no-client"


# --- adapter surface -----------------------------------------------------


def test_matrix_adapter_satisfies_protocol(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = matrix.MatrixAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    assert isinstance(adapter, PlatformAdapter)
    assert adapter.platform == Platform.MATRIX
    assert adapter.capabilities() == matrix.capabilities()


def test_adapter_configure_round_trips(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = matrix.MatrixAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    updated = adapter.configure("!r:hs", log_channel_id="!mods:hs", dry_run=True)
    assert updated.log_channel_id == "!mods:hs"
    assert adapter.configure("!r:hs").dry_run is True


async def test_adapter_handle_event_end_to_end(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = matrix.MatrixAdapter(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )
    client = FakeMatrixClient()

    await adapter.handle_event(_event("drop your SSN"), client=client, room="!r:hs")

    assert "redact_message" in client.names()
    assert audit_log.recent()[0]["action"] == "delete"


# --- doctor --------------------------------------------------------------


def test_doctor_reports_insufficient_power(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    adapter = matrix.MatrixAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )

    report = adapter.doctor("!r:hs", power_levels={"user_level": 0, "redact_level": 50})

    assert report["platform"] == "matrix"
    assert report["readiness"] == "needs attention"
    assert report["can_redact"] is False
    assert "mod-room is not set" in report["issues"]
    assert report["capabilities"] == sorted(a.value for a in matrix.capabilities())


def test_doctor_ready_when_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "tok")
    config_store, audit_log = _stores(tmp_path)
    adapter = matrix.MatrixAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    adapter.configure("!r:hs", log_channel_id="!mods:hs")

    report = adapter.doctor("!r:hs", power_levels=SimpleNamespace(user_level=100, redact_level=50))

    assert report["readiness"] == "ready"
    assert report["issues"] == []
    assert report["can_redact"] is True
    assert report["power_verified"] is True


def test_doctor_flags_unknown_power(tmp_path, monkeypatch):
    monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "tok")
    config_store, audit_log = _stores(tmp_path)
    adapter = matrix.MatrixAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
    )
    adapter.configure("!r:hs", log_channel_id="!mods:hs")

    report = adapter.doctor("!r:hs")

    assert report["power_verified"] is False
    assert "room power levels unknown (cannot verify)" in report["issues"]


# --- entry point & lazy import ------------------------------------------


def test_run_bot_requires_token(monkeypatch):
    monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="MATRIX_ACCESS_TOKEN env var required"):
        matrix.run_bot()


def test_run_bot_requires_homeserver(monkeypatch):
    monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "tok")
    monkeypatch.delenv("MATRIX_HOMESERVER", raising=False)
    with pytest.raises(RuntimeError, match="MATRIX_HOMESERVER env var required"):
        matrix.run_bot()


def test_import_nemoguardian_does_not_require_matrix_nio():
    assert "nio" not in sys.modules
