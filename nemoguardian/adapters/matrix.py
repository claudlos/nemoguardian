"""Matrix room-moderation adapter (matrix-nio).

Parses a Matrix ``m.room.message`` event, runs the shared moderation cascade,
and enforces the result with the realistic Matrix action set: redact the
offending event (``m.room.redaction`` via ``room_redact``) and/or notify the
moderators in a configured mod room (``m.notice``). Anything Matrix cannot do
(timeout / ban / mute / queue) degrades to ``flag`` with an auditable reason via
:func:`~nemoguardian.adapters.base.degrade_action`.

Design
------
* The same async handler accepts BOTH ingestion shapes: a raw event ``dict``
  (the JSON body of an ``m.room.message``) *and* a ``matrix-nio`` event object
  (what nio callbacks yield alongside a ``MatrixRoom``). Parsing is duck-typed
  via :func:`_attr` so neither shape requires the SDK.
* The network side is injectable: every enforcement call goes through a small
  ``client`` object (``redact_message`` / ``send_notice``). Tests pass a fake;
  production wraps a real ``nio.AsyncClient`` via :class:`_NioClient`.
  ``matrix-nio`` is imported lazily so ``import nemoguardian`` never needs the
  ``[matrix]`` extra and the moderation path never raises.
* Redaction requires the bot's power level to meet the room's ``redact`` power
  level. :meth:`MatrixAdapter.doctor` checks this offline against a passed-in
  power-levels object (no live call).

Run with::

    MATRIX_HOMESERVER=https://matrix.org MATRIX_USER_ID=@bot:matrix.org \
        MATRIX_ACCESS_TOKEN=xxx python -m nemoguardian.adapters.matrix
"""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass
from typing import Any

from nemoguardian.adapters.base import ActionDecision, degrade_action
from nemoguardian.bot import (
    AuditLog,
    BotConfig,
    ConfigStore,
    ModerationContext,
    ModerationEngine,
    ModerationEvaluation,
    Platform,
    redacted_excerpt,
)
from nemoguardian.bot.engine import case_id
from nemoguardian.bot.types import ModerationAction
from nemoguardian.cascade import Cascade
from nemoguardian.review.service import ReviewService

#: Matrix message ``msgtype`` values this adapter treats as fresh user content.
#: ``m.notice`` is intentionally excluded so we never moderate our own
#: enforcement notices (bot loop guard).
MODERATED_MSGTYPES: frozenset[str] = frozenset(
    {"m.text", "m.emote", "m.image", "m.file", "m.audio", "m.video"}
)

#: Default power level required to redact another user's event when a room's
#: ``m.room.power_levels`` does not pin ``redact`` explicitly (Matrix spec).
DEFAULT_REDACT_POWER: int = 50

#: Normalized actions a Matrix bot can carry out with room-moderator power.
#: ``redact`` maps to :data:`ModerationAction.DELETE`; there is no native
#: timeout/ban/mute primitive here, so those degrade to ``flag``.
MATRIX_CAPABILITIES: frozenset[ModerationAction] = frozenset(
    {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
        ModerationAction.DELETE,
        ModerationAction.NOTIFY_MODS,
    }
)


def capabilities() -> set[ModerationAction]:
    """Return the normalized actions the Matrix adapter can carry out."""
    return set(MATRIX_CAPABILITIES)


@dataclass(frozen=True)
class MatrixEvent:
    """The moderation-relevant slice of a Matrix ``m.room.message`` event."""

    room_id: str
    event_id: str
    sender: str
    body: str
    msgtype: str


def _attr(obj: Any, *names: str) -> Any:
    """Read the first present, non-``None`` field across dict keys / attributes.

    Matrix events arrive either as raw JSON ``dict`` or as nio event objects;
    this reads both without importing the SDK.
    """
    if obj is None:
        return None
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _resolve_room_id(event: Any, room: Any) -> str | None:
    if room is not None:
        if isinstance(room, str):
            return room
        room_id = _attr(room, "room_id")
        if room_id:
            return str(room_id)
    room_id = _attr(event, "room_id")
    return str(room_id) if room_id else None


def parse_matrix_event(event: Any, *, room: Any = None) -> MatrixEvent | None:
    """Parse a Matrix message event into a :class:`MatrixEvent`.

    ``room`` may be a ``MatrixRoom`` (nio), a bare room-id string, or ``None``
    (the room id is then read from the event body). Returns ``None`` for anything
    that is not a fresh, moderatable ``m.room.message`` (wrong type, an
    ``m.notice`` we posted, or missing required fields). Never raises.
    """
    if event is None:
        return None
    # Raw dicts carry a top-level ``type``; nio message objects do not (nio
    # dispatches per event class), so only enforce it when the field is present.
    event_type = _attr(event, "type")
    if event_type is not None and event_type != "m.room.message":
        return None

    content = _attr(event, "content")
    msgtype = _attr(content, "msgtype") or _attr(event, "msgtype")
    body = _attr(content, "body") if content is not None else None
    if body is None:
        body = _attr(event, "body")
    if msgtype is not None and str(msgtype) not in MODERATED_MSGTYPES:
        return None

    room_id = _resolve_room_id(event, room)
    event_id = _attr(event, "event_id")
    sender = _attr(event, "sender")
    if not room_id or not event_id or not sender or body is None:
        return None
    return MatrixEvent(
        room_id=str(room_id),
        event_id=str(event_id),
        sender=str(sender),
        body=str(body),
        msgtype=str(msgtype or "m.text"),
    )


def matrix_decision(action: ModerationAction | str) -> ActionDecision:
    """Resolve ``action`` against Matrix capabilities (unsupported -> ``flag``)."""
    return degrade_action(action, capabilities(), Platform.MATRIX)


def _context_from_event(event: MatrixEvent) -> ModerationContext:
    return ModerationContext(
        platform=Platform.MATRIX,
        workspace_id=event.room_id,
        channel_id=event.room_id,
        message_id=event.event_id,
        user_id=event.sender,
        username=event.sender,
        text=event.body,
    )


async def _safe_call(client: Any, method_name: str, *args: Any, **kwargs: Any) -> tuple[bool, str | None]:
    """Call a client method, awaiting if needed, never raising.

    Returns ``(ok, error)`` where ``error`` is a short exception class name on
    failure so the moderation path stays fail-safe.
    """
    if client is None:
        return False, "no-client"
    method = getattr(client, method_name, None)
    if not callable(method):
        return False, "unsupported-method"
    try:
        result = method(*args, **kwargs)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # fail-safe: surface, never raise on the mod path
        return False, type(exc).__name__
    return True, None


async def apply_matrix_actions(
    client: Any,
    event: MatrixEvent,
    evaluation: ModerationEvaluation,
) -> tuple[str, str | None]:
    """Apply a moderation plan to a Matrix event via the injected ``client``.

    Returns ``(execution_status, error)``. Never raises: every call is guarded.
    Honors dry-run and degrades unsupported actions (timeout/ban/mute) to
    ``flag`` with the reason surfaced in the mod-room notice.
    """
    config = evaluation.config
    plan = evaluation.plan
    if plan.action == ModerationAction.ALLOW:
        return "allowed", None

    decision = degrade_action(plan.action, capabilities(), Platform.MATRIX)
    notes: list[str] = []
    if decision.degraded:
        notes.append(decision.reason or "degraded")
        plan.action = decision.action

    if config.dry_run:
        await _send_mod_notice(client, event, evaluation, applied=["dry-run"], errors=[], notes=notes)
        return "dry-run", decision.reason if decision.degraded else None

    applied: list[str] = []
    errors: list[str] = []

    if decision.action == ModerationAction.DELETE:
        ok, err = await _safe_call(
            client,
            "redact_message",
            event.room_id,
            event.event_id,
            _reason(evaluation),
        )
        if ok:
            applied.append("redact")
        else:
            errors.append(f"redact:{err}")
    elif decision.action == ModerationAction.FLAG:
        applied.append("flag")

    ok, err = await _send_mod_notice(client, event, evaluation, applied=applied, errors=errors, notes=notes)
    if ok:
        applied.append("notify_mods")
    elif err and err != "no-mod-room":
        errors.append(f"notify_mods:{err}")

    if errors:
        return ("partial" if applied else "failed"), "; ".join(errors)
    return ("+".join(applied) if applied else "planned"), None


def make_handler(
    cascade: Cascade | None = None,
    *,
    config_store: ConfigStore | None = None,
    audit_log: AuditLog | None = None,
    review_service: ReviewService | None = None,
    engine: ModerationEngine | None = None,
):
    """Build an async event handler that runs the full Matrix moderation flow.

    The returned coroutine accepts ``(event, *, client, room=None)`` where
    ``event`` is a raw ``dict`` or nio event object, ``room`` is the optional
    ``MatrixRoom``/room-id, and ``client`` is the injectable redaction client. It
    returns the :class:`ModerationEvaluation` (or ``None`` when the event is not a
    moderatable message) so callers/tests can introspect. Never raises.
    """
    engine = engine or ModerationEngine(
        Platform.MATRIX,
        cascade=cascade,
        config_store=config_store,
        audit_log=audit_log,
        review_service=review_service,
    )

    async def on_event(event: Any, *, client: Any = None, room: Any = None) -> ModerationEvaluation | None:
        parsed = parse_matrix_event(event, room=room)
        if parsed is None:
            return None
        config = engine.config_for(parsed.room_id)
        context = _context_from_event(parsed)
        evaluation = await asyncio.to_thread(engine.evaluate, context, config)
        if evaluation.skipped:
            return evaluation
        status, error = await apply_matrix_actions(client, parsed, evaluation)
        engine.record(evaluation, execution_status=status, error=error)
        return evaluation

    return on_event


async def _send_mod_notice(
    client: Any,
    event: MatrixEvent,
    evaluation: ModerationEvaluation,
    *,
    applied: list[str],
    errors: list[str],
    notes: list[str],
) -> tuple[bool, str | None]:
    mod_room = evaluation.config.log_channel_id
    if not mod_room:
        return False, "no-mod-room"
    return await _safe_call(
        client,
        "send_notice",
        mod_room,
        _mod_log_text(event, evaluation, applied=applied, errors=errors, notes=notes),
    )


def _mod_log_text(
    event: MatrixEvent,
    evaluation: ModerationEvaluation,
    *,
    applied: list[str],
    errors: list[str],
    notes: list[str],
) -> str:
    result = evaluation.result
    verdict = result.verdict.value if result is not None else "n/a"
    score = f"{result.score:.2f}" if result is not None else "n/a"
    categories = ", ".join(result.categories) if result is not None and result.categories else "none"
    return (
        "nemoguardian moderation\n"
        f"case: {case_id(evaluation.context)}\n"
        f"room: {event.room_id} sender: {event.sender}\n"
        f"verdict: {verdict} score: {score} action: {evaluation.plan.action.value}\n"
        f"applied: {', '.join(applied) or 'none'} errors: {', '.join(errors) or 'none'}\n"
        f"categories: {categories} notes: {'; '.join(notes) or 'none'}\n"
        f"message: {redacted_excerpt(evaluation.context.text)}"
    )


def _reason(evaluation: ModerationEvaluation) -> str:
    return evaluation.plan.reason or "policy violation"


def _doctor_report(
    config: BotConfig,
    power_levels: Any,
    *,
    token_configured: bool,
    workspace_id: str,
) -> dict[str, Any]:
    """Offline redaction-readiness snapshot from a passed-in power-levels object.

    ``power_levels`` exposes the bot's power (``user_level`` / ``user_power``)
    and the room's redact threshold (``redact_level`` / ``redact``), as a dict or
    object. ``None`` means the levels are unknown and cannot be verified. No live
    Matrix call is made.
    """
    levels_known = power_levels is not None
    user_level = _to_int(_attr(power_levels, "user_level", "user_power"), default=0)
    redact_level = _to_int(_attr(power_levels, "redact_level", "redact"), default=DEFAULT_REDACT_POWER)
    can_redact = levels_known and user_level >= redact_level

    missing_required: list[str] = []
    if not levels_known:
        missing_required.append("room power levels unknown")
    elif not can_redact:
        missing_required.append(f"redact power {user_level} < required {redact_level}")

    issues: list[str] = []
    if not config.enabled:
        issues.append("moderation is disabled")
    if not token_configured:
        issues.append("MATRIX_ACCESS_TOKEN is not set")
    if not config.log_channel_id:
        issues.append("mod-room is not set")
    if not levels_known:
        issues.append("room power levels unknown (cannot verify)")
    elif missing_required:
        issues.append(f"insufficient power: {', '.join(missing_required)}")

    return {
        "platform": Platform.MATRIX.value,
        "workspace_id": workspace_id,
        "enabled": config.enabled,
        "dry_run": config.dry_run,
        "token_configured": token_configured,
        "mod_room": config.log_channel_id,
        "power_verified": levels_known,
        "user_power_level": user_level if levels_known else None,
        "required_redact_level": redact_level,
        "can_redact": can_redact,
        "capabilities": sorted(action.value for action in capabilities()),
        "missing_required": missing_required,
        "readiness": "ready" if not issues else "needs attention",
        "issues": issues,
    }


def _to_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class MatrixAdapter:
    """Thin :class:`~nemoguardian.adapters.base.PlatformAdapter` over Matrix.

    Delegates to the module-level helpers so behavior is shared with the live nio
    client while exposing the normalized interface. Heavy ``matrix-nio`` imports
    stay inside :func:`build_client` / :func:`run_bot`.
    """

    platform = Platform.MATRIX

    def __init__(
        self,
        cascade: Cascade | None = None,
        *,
        config_store: ConfigStore | None = None,
        audit_log: AuditLog | None = None,
        review_service: ReviewService | None = None,
    ) -> None:
        self.engine = ModerationEngine(
            Platform.MATRIX,
            cascade=cascade,
            config_store=config_store,
            audit_log=audit_log,
            review_service=review_service,
        )
        self._handler = make_handler(engine=self.engine)

    def capabilities(self) -> set[ModerationAction]:
        return capabilities()

    def doctor(self, workspace_id: str, power_levels: Any = None) -> dict[str, Any]:
        """Return an offline redaction-readiness report for ``workspace_id``."""
        workspace = str(workspace_id)
        config = self.engine.config_for(workspace)
        token_configured = bool(os.environ.get("MATRIX_ACCESS_TOKEN"))
        return _doctor_report(
            config,
            power_levels,
            token_configured=token_configured,
            workspace_id=workspace,
        )

    def configure(self, workspace_id: str, **changes: Any) -> BotConfig:
        """Read (no ``changes``) or update the per-room config."""
        if changes:
            return self.engine.config_store.update(Platform.MATRIX, str(workspace_id), **changes)
        return self.engine.config_for(str(workspace_id))

    async def handle_event(
        self,
        event: Any,
        *,
        client: Any = None,
        room: Any = None,
    ) -> ModerationEvaluation | None:
        """Moderate one Matrix message event end to end."""
        return await self._handler(event, client=client, room=room)

    async def apply_action(
        self,
        client: Any,
        event: MatrixEvent,
        evaluation: ModerationEvaluation,
    ) -> tuple[str, str | None]:
        """Apply the planned action to ``event`` (delegates to the action flow)."""
        return await apply_matrix_actions(client, event, evaluation)

    def record_audit(
        self,
        evaluation: ModerationEvaluation,
        *,
        execution_status: str,
        error: str | None = None,
    ) -> None:
        self.engine.record(evaluation, execution_status=execution_status, error=error)


class _NioClient:
    """Adapts a live ``nio.AsyncClient`` to the injectable ``client`` surface."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def redact_message(self, room_id: str, event_id: str, reason: str) -> None:
        await self._client.room_redact(room_id, event_id, reason=reason)

    async def send_notice(self, room_id: str, text: str) -> None:
        await self._client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": "m.notice", "body": text},
        )


def build_client(token: str, *, homeserver: str, user_id: str):  # pragma: no cover - requires matrix-nio
    """Create a matrix-nio ``AsyncClient`` wired for moderation.

    ``matrix-nio`` is imported here (lazily) so importing this module never
    requires the ``[matrix]`` extra.
    """
    from nio import AsyncClient

    client = AsyncClient(homeserver, user_id)
    client.access_token = token
    return client


def run_bot() -> None:  # pragma: no cover - manual entry point
    """Entry point: start the Matrix moderation bot via a sync loop."""
    token = os.environ.get("MATRIX_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("MATRIX_ACCESS_TOKEN env var required")
    homeserver = os.environ.get("MATRIX_HOMESERVER")
    if not homeserver:
        raise RuntimeError("MATRIX_HOMESERVER env var required")
    user_id = os.environ.get("MATRIX_USER_ID", "")

    from nio import RoomMessage

    client = build_client(token, homeserver=homeserver, user_id=user_id)
    handler = make_handler(review_service=ReviewService())
    nio_client = _NioClient(client)

    async def on_message(room: Any, event: Any) -> None:
        await handler(event, client=nio_client, room=room)

    client.add_event_callback(on_message, RoomMessage)
    asyncio.run(client.sync_forever(timeout=30000))


if __name__ == "__main__":  # pragma: no cover - manual entry point
    run_bot()


__all__ = [
    "DEFAULT_REDACT_POWER",
    "MATRIX_CAPABILITIES",
    "MODERATED_MSGTYPES",
    "MatrixAdapter",
    "MatrixEvent",
    "apply_matrix_actions",
    "build_client",
    "capabilities",
    "make_handler",
    "matrix_decision",
    "parse_matrix_event",
    "run_bot",
]
