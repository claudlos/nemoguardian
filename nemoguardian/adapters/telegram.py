"""Telegram group-moderation adapter (Bot API).

Parses a Telegram ``Update`` (group / supergroup messages), runs the moderation
cascade, and enforces the result with native Bot API admin actions:
``deleteMessage`` for unsafe content plus an optional ``restrictChatMember``
(mute) / ``banChatMember`` (ban) escalation for repeat offenders. Per-chat
config, ignored chats, dry-run mode and append-only redacted audit records all
flow through the shared :mod:`nemoguardian.bot` foundation.

Design
------
* The same async handler accepts BOTH ingestion shapes: a raw webhook JSON
  ``dict`` (what Telegram POSTs to a webhook) *and* a ``python-telegram-bot``
  ``Update`` object (what long-poll ``getUpdates`` yields). Parsing is duck-typed
  via :func:`_attr` so neither shape requires the SDK.
* The network side is injectable: every enforcement call goes through a small
  ``api`` object (``delete_message`` / ``ban_chat_member`` /
  ``restrict_chat_member`` / ``send_message``). Tests pass a fake; production
  wraps a real ``telegram.Bot`` via :class:`_BotApi`. ``python-telegram-bot`` is
  imported lazily so ``import nemoguardian`` never needs the ``[telegram]`` extra.
* :func:`degrade_action` guards the action path: any action Telegram cannot
  perform degrades to ``flag`` with an auditable reason rather than silently
  no-op. Admin readiness is checked offline by :meth:`TelegramAdapter.doctor`
  against a passed-in chat-member object (no live call).

Run with:
    TELEGRAM_BOT_TOKEN=xxx python -m nemoguardian.adapters.telegram
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from nemoguardian.adapters.base import degrade_action
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
from nemoguardian.schemas import VerdictLabel

#: Chat types this adapter moderates. Private chats and broadcast channels are
#: ignored (a bot cannot meaningfully moderate a 1:1 DM or a channel feed).
MODERATED_CHAT_TYPES: frozenset[str] = frozenset({"group", "supergroup"})

#: After this many *prior* unsafe cases for a user in a chat, the next unsafe
#: message escalates to a ban instead of a mute/delete. Configurable per adapter.
DEFAULT_BAN_AFTER: int = 3

#: Normalized actions a Telegram bot can carry out as a chat administrator.
#: ``timeout`` (a Discord-style fixed window) is intentionally absent — Telegram
#: expresses the same intent as ``mute`` (restrict) and it degrades accordingly.
TELEGRAM_CAPABILITIES: frozenset[ModerationAction] = frozenset(
    {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
        ModerationAction.DELETE,
        ModerationAction.BAN,
        ModerationAction.MUTE,
        ModerationAction.NOTIFY_MODS,
    }
)


def capabilities() -> set[ModerationAction]:
    """Return the normalized actions the Telegram adapter can carry out."""
    return set(TELEGRAM_CAPABILITIES)


@dataclass(frozen=True)
class ParsedMessage:
    """The moderation-relevant slice of a Telegram message."""

    chat_id: str
    chat_type: str
    message_id: str
    user_id: str
    username: str
    text: str
    is_bot: bool = False


def _attr(obj: Any, *names: str) -> Any:
    """Read the first present, non-``None`` field across dict keys / attributes.

    Telegram updates arrive either as raw JSON ``dict`` (webhook) or as SDK
    objects (long-poll); this reads both. Multiple ``names`` cover key drift
    between the wire format (``from``) and the SDK (``from_user``).
    """
    if obj is None:
        return None
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def parse_update(update: Any) -> ParsedMessage | None:
    """Parse a Telegram ``Update`` into a :class:`ParsedMessage`.

    Returns ``None`` for updates this adapter does not moderate (non-message
    updates, private chats, broadcast channels). Never raises on malformed input.
    """
    message = _attr(update, "message", "edited_message")
    if message is None:
        return None
    chat = _attr(message, "chat")
    if chat is None:
        return None
    chat_type = str(_attr(chat, "type") or "")
    if chat_type not in MODERATED_CHAT_TYPES:
        return None

    user = _attr(message, "from_user", "from")
    username = _attr(user, "username") or _attr(user, "first_name") or "unknown"
    return ParsedMessage(
        chat_id=str(_attr(chat, "id") or ""),
        chat_type=chat_type,
        message_id=str(_attr(message, "message_id", "id") or ""),
        user_id=str(_attr(user, "id") or "unknown"),
        username=str(username),
        text=str(_attr(message, "text", "caption") or ""),
        is_bot=bool(_attr(user, "is_bot")),
    )


def _context_from_parsed(parsed: ParsedMessage) -> ModerationContext:
    return ModerationContext(
        platform=Platform.TELEGRAM,
        workspace_id=parsed.chat_id,
        channel_id=parsed.chat_id,
        message_id=parsed.message_id,
        user_id=parsed.user_id,
        username=parsed.username,
        text=parsed.text,
    )


def _resolve_target(
    evaluation: ModerationEvaluation,
    prior_offenses: int,
    ban_after: int,
) -> ModerationAction:
    """Map the platform-neutral plan onto a Telegram enforcement action.

    Unsafe content escalates ban > mute > delete based on repeat-offense count
    and the per-chat config; controversial content (and anything we cannot
    enforce) surfaces as ``flag``.
    """
    result = evaluation.result
    config = evaluation.config
    if result is None:
        return ModerationAction.FLAG
    if result.verdict == VerdictLabel.UNSAFE:
        if ban_after > 0 and prior_offenses + 1 >= ban_after:
            return ModerationAction.BAN
        if config.timeout_unsafe:
            return ModerationAction.MUTE
        if config.delete_unsafe:
            return ModerationAction.DELETE
        return ModerationAction.FLAG
    return ModerationAction.FLAG


async def apply_telegram_actions(
    api: Any,
    parsed: ParsedMessage,
    evaluation: ModerationEvaluation,
    *,
    prior_offenses: int = 0,
    ban_after: int = DEFAULT_BAN_AFTER,
) -> tuple[str, str | None]:
    """Apply a moderation plan to a Telegram message via the injected ``api``.

    Returns ``(execution_status, error)``. Never raises: every Bot API call is
    guarded and surfaced through the status/error pair so the moderation path is
    fail-safe. Honors dry-run and degrades unsupported actions to ``flag``.
    """
    config = evaluation.config
    plan = evaluation.plan
    result = evaluation.result
    if plan.action == ModerationAction.ALLOW:
        return "allowed", None

    target = _resolve_target(evaluation, prior_offenses, ban_after)
    decision = degrade_action(target, capabilities(), Platform.TELEGRAM)
    degrade_reason = decision.reason if decision.degraded else None
    # Record the action Telegram will actually enforce (delete / mute / ban /
    # flag) rather than the platform-neutral plan action (e.g. ``timeout``).
    plan.action = decision.action
    if config.dry_run:
        return "dry-run", degrade_reason

    act = decision.action
    applied: list[str] = []
    errors: list[str] = []

    want_delete = act == ModerationAction.DELETE or (
        config.delete_unsafe and result is not None and result.verdict == VerdictLabel.UNSAFE
    )
    if want_delete:
        try:
            await api.delete_message(parsed.chat_id, parsed.message_id)
            applied.append("delete")
        except Exception as exc:
            errors.append(f"delete:{type(exc).__name__}")

    if act == ModerationAction.BAN:
        try:
            await api.ban_chat_member(parsed.chat_id, parsed.user_id)
            applied.append("ban")
        except Exception as exc:
            errors.append(f"ban:{type(exc).__name__}")
    elif act == ModerationAction.MUTE:
        try:
            await api.restrict_chat_member(parsed.chat_id, parsed.user_id)
            applied.append("mute")
        except Exception as exc:
            errors.append(f"mute:{type(exc).__name__}")

    if config.log_channel_id:
        try:
            await api.send_message(
                config.log_channel_id,
                _mod_log_text(parsed, evaluation, applied=applied, errors=errors),
            )
            applied.append("notify_mods")
        except Exception as exc:
            errors.append(f"notify_mods:{type(exc).__name__}")

    if errors:
        status = "partial" if applied else "failed"
        return status, "; ".join(errors)
    status = "+".join(applied) if applied else "planned"
    return status, degrade_reason


def make_handler(
    cascade: Cascade | None = None,
    *,
    config_store: ConfigStore | None = None,
    audit_log: AuditLog | None = None,
    engine: ModerationEngine | None = None,
    ban_after: int = DEFAULT_BAN_AFTER,
):
    """Build an async update handler that runs the full Telegram moderation flow.

    The returned coroutine accepts ``(update, *, api)`` where ``update`` is a raw
    webhook ``dict`` or an SDK ``Update`` and ``api`` is the injectable Bot API
    client. It returns the :class:`ModerationEvaluation` (or ``None`` when the
    update is not a moderatable group message) so callers/tests can introspect.
    """
    engine = engine or ModerationEngine(
        Platform.TELEGRAM,
        cascade=cascade,
        config_store=config_store,
        audit_log=audit_log,
    )

    async def on_update(update: Any, *, api: Any) -> ModerationEvaluation | None:
        parsed = parse_update(update)
        if parsed is None or parsed.is_bot:
            return None

        config = engine.config_for(parsed.chat_id)
        context = _context_from_parsed(parsed)
        evaluation = await asyncio.to_thread(engine.evaluate, context, config)
        if evaluation.skipped:
            return evaluation

        prior_offenses = _count_prior_offenses(engine.audit_log, parsed)
        status, error = await apply_telegram_actions(
            api,
            parsed,
            evaluation,
            prior_offenses=prior_offenses,
            ban_after=ban_after,
        )
        engine.record(evaluation, execution_status=status, error=error)
        return evaluation

    return on_update


def _count_prior_offenses(audit_log: AuditLog, parsed: ParsedMessage) -> int:
    """Count prior *unsafe* audit cases for this user in this chat (for escalation)."""
    records = audit_log.history(
        Platform.TELEGRAM,
        parsed.chat_id,
        user_id=parsed.user_id,
        verdict=VerdictLabel.UNSAFE,
        limit=100,
    )
    return len(records)


def _mod_log_text(
    parsed: ParsedMessage,
    evaluation: ModerationEvaluation,
    *,
    applied: list[str],
    errors: list[str],
) -> str:
    result = evaluation.result
    verdict = result.verdict.value if result is not None else "n/a"
    score = f"{result.score:.2f}" if result is not None else "n/a"
    categories = ", ".join(result.categories) if result is not None and result.categories else "none"
    return (
        "nemoguardian moderation\n"
        f"case: {case_id(evaluation.context)}\n"
        f"chat: {parsed.chat_id} user: {parsed.user_id} (@{parsed.username})\n"
        f"verdict: {verdict} score: {score} action: {evaluation.plan.action.value}\n"
        f"applied: {', '.join(applied) or 'none'} errors: {', '.join(errors) or 'none'}\n"
        f"categories: {categories}\n"
        f"message: {redacted_excerpt(evaluation.context.text)}"
    )


def _doctor_report(
    config: BotConfig,
    member: Any,
    *,
    token_configured: bool,
    workspace_id: str,
) -> dict[str, object]:
    """Offline admin-readiness snapshot from a passed-in chat-member object.

    ``member`` is the bot's ``ChatMember`` (dict or SDK object). No live call is
    made — the caller fetches it once (e.g. ``getChatMember``) and passes it in.
    """
    status = str(_attr(member, "status") or "") if member is not None else ""
    is_creator = status == "creator"
    is_admin = status in {"administrator", "creator"}
    can_delete = is_creator or bool(_attr(member, "can_delete_messages"))
    can_restrict = is_creator or bool(_attr(member, "can_restrict_members"))

    missing_required: list[str] = []
    missing_recommended: list[str] = []
    if member is None:
        missing_required.append("bot chat-member status unknown")
    elif not is_admin:
        missing_required.append("bot is not a chat administrator")
    if not can_delete:
        missing_required.append("can_delete_messages")
    if not can_restrict:
        missing_recommended.append("can_restrict_members")

    issues: list[str] = []
    if not config.enabled:
        issues.append("moderation is disabled")
    if not token_configured:
        issues.append("TELEGRAM_BOT_TOKEN is not set")
    if missing_required:
        issues.append(f"missing required admin rights: {', '.join(missing_required)}")

    return {
        "platform": Platform.TELEGRAM.value,
        "workspace_id": workspace_id,
        "enabled": config.enabled,
        "is_admin": is_admin,
        "can_delete_messages": can_delete,
        "can_restrict_members": can_restrict,
        "token_configured": token_configured,
        "readiness": "ready" if not issues else "needs attention",
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "issues": issues,
    }


class TelegramAdapter:
    """Thin :class:`~nemoguardian.adapters.base.PlatformAdapter` over Telegram.

    Wraps :func:`make_handler` so behavior is unchanged while exposing the
    normalized interface. Enforcement happens in :meth:`handle_event`
    (parse + moderate + apply + record); :meth:`apply_action` runs only the
    action flow against an injected ``api``.
    """

    platform = Platform.TELEGRAM

    def __init__(
        self,
        cascade: Cascade | None = None,
        *,
        config_store: ConfigStore | None = None,
        audit_log: AuditLog | None = None,
        ban_after: int = DEFAULT_BAN_AFTER,
    ) -> None:
        self.ban_after = ban_after
        self.engine = ModerationEngine(
            Platform.TELEGRAM,
            cascade=cascade,
            config_store=config_store,
            audit_log=audit_log,
        )
        self._handler = make_handler(engine=self.engine, ban_after=ban_after)

    def capabilities(self) -> set[ModerationAction]:
        return capabilities()

    def doctor(self, workspace_id: str, member: Any = None) -> dict[str, object]:
        """Return an offline admin-readiness report for ``workspace_id``."""
        workspace = str(workspace_id)
        config = self.engine.config_for(workspace)
        token_configured = bool(os.environ.get("TELEGRAM_BOT_TOKEN"))
        return _doctor_report(
            config,
            member,
            token_configured=token_configured,
            workspace_id=workspace,
        )

    def configure(self, workspace_id: str, **changes: Any) -> BotConfig:
        """Read (no ``changes``) or update the per-chat config."""
        if changes:
            return self.engine.config_store.update(Platform.TELEGRAM, str(workspace_id), **changes)
        return self.engine.config_for(str(workspace_id))

    async def handle_event(self, update: Any, *, api: Any) -> ModerationEvaluation | None:
        """Moderate one Telegram update end to end."""
        return await self._handler(update, api=api)

    async def apply_action(
        self,
        parsed: ParsedMessage,
        evaluation: ModerationEvaluation,
        *,
        api: Any,
        prior_offenses: int = 0,
    ) -> tuple[str, str | None]:
        """Apply the planned action to ``parsed`` via ``api`` (no recording)."""
        return await apply_telegram_actions(
            api,
            parsed,
            evaluation,
            prior_offenses=prior_offenses,
            ban_after=self.ban_after,
        )

    def record_audit(
        self,
        evaluation: ModerationEvaluation,
        *,
        execution_status: str,
        error: str | None = None,
    ) -> None:
        self.engine.record(evaluation, execution_status=execution_status, error=error)


class _BotApi:
    """Adapts a live ``telegram.Bot`` to the injectable ``api`` surface."""

    def __init__(self, bot: Any) -> None:
        self._bot = bot

    async def delete_message(self, chat_id: str, message_id: str) -> None:
        await self._bot.delete_message(chat_id=int(chat_id), message_id=int(message_id))

    async def ban_chat_member(self, chat_id: str, user_id: str) -> None:
        await self._bot.ban_chat_member(chat_id=int(chat_id), user_id=int(user_id))

    async def restrict_chat_member(self, chat_id: str, user_id: str) -> None:
        from telegram import ChatPermissions

        await self._bot.restrict_chat_member(
            chat_id=int(chat_id),
            user_id=int(user_id),
            permissions=ChatPermissions(can_send_messages=False),
        )

    async def send_message(self, chat_id: str, text: str) -> None:
        await self._bot.send_message(chat_id=int(chat_id), text=text)


def build_application(token: str):
    """Create a python-telegram-bot ``Application`` wired to the moderation flow."""
    from telegram.ext import Application, MessageHandler, filters

    config_store = ConfigStore()
    audit_log = AuditLog()
    handler = make_handler(config_store=config_store, audit_log=audit_log)

    application = Application.builder().token(token).build()

    async def on_message(update: Any, context: Any) -> None:
        await handler(update, api=_BotApi(context.bot))

    application.add_handler(MessageHandler(filters.ALL, on_message))
    return application


def run_bot() -> None:
    """Entry point: start the Telegram moderation bot via long-poll."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var required")
    application = build_application(token)
    application.run_polling()


if __name__ == "__main__":
    run_bot()


__all__ = [
    "DEFAULT_BAN_AFTER",
    "MODERATED_CHAT_TYPES",
    "TELEGRAM_CAPABILITIES",
    "ParsedMessage",
    "TelegramAdapter",
    "apply_telegram_actions",
    "build_application",
    "capabilities",
    "make_handler",
    "parse_update",
    "run_bot",
]
