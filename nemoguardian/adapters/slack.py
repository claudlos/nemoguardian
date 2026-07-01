"""Slack moderation adapter (Slack Bolt).

Ingests Slack ``message`` events, moderates them with the shared cascade/engine,
and enforces the realistic Slack action set: delete the offending message
(``chat.delete``) and/or notify (channel warning + ephemeral notice + mod-log).
Anything Slack cannot do (timeout / ban / mute / queue) degrades to ``flag`` with
an auditable reason via :func:`~nemoguardian.adapters.base.degrade_action`.

Run with (the ``[slack]`` extra installs ``slack-bolt``)::

    SLACK_BOT_TOKEN=xoxb-... SLACK_SIGNING_SECRET=... python -m nemoguardian.adapters.slack

Socket Mode is used automatically when ``SLACK_APP_TOKEN`` is set; otherwise the
Bolt HTTP server starts on ``PORT`` (default ``3000``).

``slack_bolt`` is imported lazily (only inside :func:`build_app` / :func:`run_bot`)
so ``import nemoguardian`` never requires the Slack SDK, and the moderation path
never raises — failures surface as a planned action / execution status instead.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass
from typing import Any

from nemoguardian.adapters.base import ActionDecision, degrade_action, supports
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
from nemoguardian.bot.types import ModerationAction
from nemoguardian.cascade import Cascade
from nemoguardian.review.service import ReviewService

#: Slack emoji name (no surrounding colons) used to flag a message via
#: ``reactions.add``.
WARNING_REACTION_NAME = "warning"

#: OAuth scopes the adapter needs to read + enforce. Reported by :func:`doctor`
#: so an operator can verify the app install without a live API call.
SLACK_REQUIRED_SCOPES: tuple[str, ...] = (
    "channels:history",
    "groups:history",
    "chat:write",
)
SLACK_RECOMMENDED_SCOPES: tuple[str, ...] = (
    "chat:write.public",
    "reactions:write",
    "channels:read",
)

#: Slack message subtypes that are not fresh user messages (edits, deletes,
#: joins, bot posts, ...). They are skipped so we never moderate an edit echo or
#: our own enforcement notices.
_SKIP_SUBTYPES: frozenset[str] = frozenset(
    {
        "message_changed",
        "message_deleted",
        "message_replied",
        "bot_message",
        "channel_join",
        "channel_leave",
        "channel_topic",
        "channel_purpose",
        "channel_name",
        "channel_archive",
        "channel_unarchive",
        "thread_broadcast",
        "tombstone",
    }
)

#: Normalized actions this Slack bot can reliably carry out with its documented
#: bot-token setup. Slack user-message deletion requires elevated/admin install
#: capabilities beyond the default app scopes, so delete/timeout/ban/mute all
#: degrade to ``flag`` rather than failing live after being advertised.
SLACK_CAPABILITIES: frozenset[ModerationAction] = frozenset(
    {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
        ModerationAction.NOTIFY_MODS,
        ModerationAction.NOTIFY_USER,
    }
)


def capabilities() -> set[ModerationAction]:
    """Return the normalized actions the Slack adapter can carry out."""
    return set(SLACK_CAPABILITIES)


@dataclass(frozen=True)
class SlackMessage:
    """A parsed, platform-shaped Slack message event."""

    team_id: str
    channel_id: str
    user_id: str
    text: str
    ts: str
    channel_type: str | None = None
    thread_ts: str | None = None


def parse_slack_event(payload: Any, *, team_id: str | None = None) -> SlackMessage | None:
    """Parse a Slack ``message`` event into a :class:`SlackMessage`.

    Accepts both the full Events API envelope
    (``{"type": "event_callback", "team_id": ..., "event": {...}}``) and the
    bare inner ``event`` dict that Bolt hands listeners. Returns ``None`` for
    anything that is not a fresh user message (wrong type, edit/delete/join
    subtype, bot post, or missing required fields) so callers can simply skip it.
    """
    if not isinstance(payload, dict):
        return None
    inner = payload.get("event")
    if not isinstance(inner, dict):
        inner = payload
    if inner.get("type") != "message":
        return None
    if inner.get("subtype") in _SKIP_SUBTYPES:
        return None
    if inner.get("bot_id"):
        return None
    user_id = inner.get("user")
    channel_id = inner.get("channel")
    ts = inner.get("ts")
    if not user_id or not channel_id or not ts:
        return None
    resolved_team_id = team_id or payload.get("team_id") or inner.get("team") or payload.get("team") or "unknown"
    return SlackMessage(
        team_id=str(resolved_team_id),
        channel_id=str(channel_id),
        user_id=str(user_id),
        text=inner.get("text") or "",
        ts=str(ts),
        channel_type=inner.get("channel_type"),
        thread_ts=inner.get("thread_ts"),
    )


def slack_decision(action: ModerationAction | str) -> ActionDecision:
    """Resolve ``action`` against Slack capabilities (unsupported -> ``flag``)."""
    return degrade_action(action, capabilities(), Platform.SLACK)


def make_handler(
    cascade: Cascade | None = None,
    *,
    config_store: ConfigStore | None = None,
    audit_log: AuditLog | None = None,
    review_service: ReviewService | None = None,
    engine: ModerationEngine | None = None,
):
    """Build an async Slack message handler that runs the moderation flow.

    Mirrors :func:`nemoguardian.adapters.discord.make_handler`: pass a prebuilt
    ``engine`` to share one :class:`ModerationEngine` with :class:`SlackAdapter`,
    otherwise one is built from ``cascade``/``config_store``/``audit_log``.

    The returned coroutine takes a raw Slack event ``dict`` plus an optional
    Slack web ``client`` (sync ``WebClient`` or async ``AsyncWebClient``); it
    never raises on the moderation path.
    """
    engine = engine or ModerationEngine(
        Platform.SLACK,
        cascade=cascade,
        config_store=config_store,
        audit_log=audit_log,
        review_service=review_service,
    )

    async def on_message(event: Any, *, client: Any = None, body: Any = None) -> None:
        body_team_id = body.get("team_id") if isinstance(body, dict) else None
        message = parse_slack_event(event, team_id=str(body_team_id) if body_team_id else None)
        if message is None:
            return
        config = engine.config_for(message.team_id)
        context = _context_from_message(message)
        evaluation = await asyncio.to_thread(engine.evaluate, context, config)
        if evaluation.skipped:
            return
        status, error = await apply_slack_actions(client, message, evaluation)
        engine.record(evaluation, execution_status=status, error=error)

    return on_message


async def apply_slack_actions(
    client: Any,
    message: SlackMessage,
    evaluation: ModerationEvaluation,
) -> tuple[str, str | None]:
    """Apply a moderation plan to a Slack message.

    Returns an execution status string and optional error. Any action Slack
    cannot perform (timeout/ban/...) is degraded to ``flag`` via
    :func:`degrade_action`; the planned action on ``evaluation`` is updated to
    the degraded action so the audit record is honest, and the reason is surfaced
    in the mod-log. Kept separate from the live Bolt app so tests can drive the
    flow with a fake client.
    """
    config = evaluation.config
    plan = evaluation.plan
    if plan.action == ModerationAction.ALLOW:
        return "allowed", None

    decision = degrade_action(plan.action, capabilities(), Platform.SLACK)
    notes: list[str] = []
    if decision.degraded:
        notes.append(decision.reason or "degraded")
        plan.action = decision.action

    if config.dry_run:
        await _send_mod_log(client, message, evaluation, applied=["dry-run"], errors=[], notes=notes)
        return "dry-run", None

    applied: list[str] = []
    errors: list[str] = []

    if plan.delete_message and supports(capabilities(), ModerationAction.DELETE):
        ok, err = await _safe_call(client, "chat_delete", channel=message.channel_id, ts=message.ts)
        if ok:
            applied.append("delete")
        elif err:
            errors.append(f"delete:{err}")

    if plan.add_reaction:
        ok, err = await _safe_call(
            client,
            "reactions_add",
            channel=message.channel_id,
            timestamp=message.ts,
            name=WARNING_REACTION_NAME,
        )
        if ok:
            applied.append("reaction")
        elif err:
            errors.append(f"reaction:{err}")

    if plan.public_warning:
        ok, err = await _safe_call(
            client,
            "chat_postMessage",
            channel=message.channel_id,
            text=_warning_text(message, evaluation),
        )
        if ok:
            applied.append("warning")
        elif err:
            errors.append(f"warning:{err}")

    if plan.notify_user:
        ok, err = await _safe_call(
            client,
            "chat_postEphemeral",
            channel=message.channel_id,
            user=message.user_id,
            text=_notify_text(message, evaluation),
        )
        if ok:
            applied.append("notify-user")
        elif err:
            errors.append(f"notify-user:{err}")

    if decision.degraded:
        applied.append("flag")

    await _send_mod_log(client, message, evaluation, applied=applied, errors=errors, notes=notes)

    if errors:
        return ("partial" if applied else "failed"), ";".join(errors)
    return ("+".join(applied) if applied else "planned"), None


class SlackAdapter:
    """Thin :class:`~nemoguardian.adapters.base.PlatformAdapter` over Slack.

    Delegates to the module-level helpers so behavior is shared with the live
    Bolt app while exposing the normalized interface the capability layer relies
    on. Heavy ``slack_bolt`` imports stay inside :func:`build_app`/:func:`run_bot`.
    """

    platform = Platform.SLACK

    def __init__(
        self,
        cascade: Cascade | None = None,
        *,
        config_store: ConfigStore | None = None,
        audit_log: AuditLog | None = None,
        review_service: ReviewService | None = None,
    ) -> None:
        self.engine = ModerationEngine(
            Platform.SLACK,
            cascade=cascade,
            config_store=config_store,
            audit_log=audit_log,
            review_service=review_service,
        )
        self._handler = make_handler(engine=self.engine)

    def capabilities(self) -> set[ModerationAction]:
        return capabilities()

    def doctor(
        self,
        workspace_id: str,
        granted_scopes: Any = None,
        *,
        token_configured: bool | None = None,
    ) -> dict[str, Any]:
        """Return a readiness report for ``workspace_id`` (never raises).

        ``granted_scopes`` is the set of OAuth scopes the install actually has
        (e.g. from ``auth.test``/``apps.permissions``); pass ``None`` when it is
        unknown and the report lists every required scope as "to grant". No live
        Slack call is made.
        """
        config = self.engine.config_for(str(workspace_id))
        return _doctor_report(config, granted_scopes, token_configured=token_configured)

    def configure(self, workspace_id: str, **changes: Any) -> BotConfig:
        """Read (no ``changes``) or update the per-workspace config."""
        if changes:
            return self.engine.config_store.update(Platform.SLACK, str(workspace_id), **changes)
        return self.engine.config_for(str(workspace_id))

    async def handle_event(self, event: Any, *, client: Any = None) -> None:
        """Moderate one Slack message event end to end."""
        await self._handler(event, client=client)

    async def apply_action(
        self,
        client: Any,
        message: SlackMessage,
        evaluation: ModerationEvaluation,
    ) -> tuple[str, str | None]:
        """Apply the planned action to ``message`` (delegates to the action flow)."""
        return await apply_slack_actions(client, message, evaluation)

    def record_audit(
        self,
        evaluation: ModerationEvaluation,
        *,
        execution_status: str,
        error: str | None = None,
    ) -> None:
        self.engine.record(evaluation, execution_status=execution_status, error=error)


def build_app(
    *,
    config_store: ConfigStore | None = None,
    audit_log: AuditLog | None = None,
    review_service: ReviewService | None = None,
):
    """Create a Slack Bolt ``App`` wired to the moderation handler.

    ``slack_bolt`` is imported here (lazily) so importing this module never
    requires the ``[slack]`` extra.
    """
    from slack_bolt import App

    app = App(
        token=os.environ.get("SLACK_BOT_TOKEN"),
        signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
    )
    handler = make_handler(
        config_store=config_store,
        audit_log=audit_log,
        review_service=review_service or ReviewService(),
    )

    @app.event("message")
    def _on_message(event, client, body) -> None:  # pragma: no cover - requires slack_bolt
        asyncio.run(handler(event, client=client, body=body))

    return app


def run_bot() -> None:
    """Entry point: start the Slack moderation bot."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN env var required")

    app = build_app()
    app_token = os.environ.get("SLACK_APP_TOKEN")
    if app_token:  # pragma: no cover - requires slack_bolt
        from slack_bolt.adapter.socket_mode import SocketModeHandler

        SocketModeHandler(app, app_token).start()
    else:  # pragma: no cover - requires slack_bolt
        app.start(port=int(os.environ.get("PORT", "3000")))


def _context_from_message(message: SlackMessage) -> ModerationContext:
    # ``ts`` is the stable message id; the workspace is the Slack team. We use the
    # user id as the username to avoid a ``users.info`` network call on the hot
    # path (a live bot can enrich this later).
    return ModerationContext(
        platform=Platform.SLACK,
        workspace_id=message.team_id,
        channel_id=message.channel_id,
        message_id=message.ts,
        user_id=message.user_id,
        username=message.user_id,
        text=message.text,
        permalink=None,
    )


async def _safe_call(client: Any, method_name: str, **kwargs: Any) -> tuple[bool, str | None]:
    """Call a Slack web-client method, never raising.

    Works with both the sync ``WebClient`` and the async ``AsyncWebClient``
    (awaiting the result when it is awaitable). Returns ``(ok, error)`` where
    ``error`` is a short exception class name on failure.
    """
    if client is None:
        return False, "no-client"
    method = getattr(client, method_name, None)
    if not callable(method):
        return False, "unsupported-method"
    try:
        result = method(**kwargs)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:  # fail-safe: surface, never raise on the mod path
        return False, type(exc).__name__
    return True, None


async def _send_mod_log(
    client: Any,
    message: SlackMessage,
    evaluation: ModerationEvaluation,
    *,
    applied: list[str],
    errors: list[str],
    notes: list[str],
) -> None:
    channel_id = evaluation.config.log_channel_id
    if not channel_id or client is None:
        return
    await _safe_call(
        client,
        "chat_postMessage",
        channel=channel_id,
        text=_mod_log_text(message, evaluation, applied=applied, errors=errors, notes=notes),
    )


def _mod_log_text(
    message: SlackMessage,
    evaluation: ModerationEvaluation,
    *,
    applied: list[str],
    errors: list[str],
    notes: list[str],
) -> str:
    result = evaluation.result
    if result is None:
        return "nemoguardian skipped a message."
    category_text = ", ".join(result.categories) or "none"
    applied_text = ", ".join(applied) or "none"
    error_text = ", ".join(errors) or "none"
    note_text = "; ".join(notes) or "none"
    return (
        "*nemoguardian moderation*\n"
        f"case: `slack-{message.team_id}-{message.ts}`\n"
        f"user: <@{message.user_id}> channel: <#{message.channel_id}>\n"
        f"verdict: `{result.verdict.value}` score: `{result.score:.2f}` mode: `{result.mode.value}`\n"
        f"action: `{evaluation.plan.action.value}` applied: `{applied_text}` errors: `{error_text}`\n"
        f"categories: `{category_text}` notes: `{note_text}`\n"
        f"rule: `{result.matched_policy_rule or 'none'}` request: `{result.request_id or 'none'}`\n"
        f"message: {redacted_excerpt(evaluation.context.text)}"
    )


def _warning_text(message: SlackMessage, evaluation: ModerationEvaluation) -> str:
    return (
        f":warning: <@{message.user_id}>, that message was blocked by nemoguardian: "
        f"{_reason(evaluation)}"
    )


def _notify_text(message: SlackMessage, evaluation: ModerationEvaluation) -> str:
    return f"Your message was moderated by nemoguardian: {_reason(evaluation)}"


def _reason(evaluation: ModerationEvaluation) -> str:
    return evaluation.plan.reason or "policy violation"


def _doctor_report(
    config: BotConfig,
    granted_scopes: Any,
    *,
    token_configured: bool | None = None,
) -> dict[str, Any]:
    if token_configured is None:
        token_configured = bool(os.environ.get("SLACK_BOT_TOKEN"))

    scopes_known = granted_scopes is not None
    granted = {str(scope) for scope in (granted_scopes or [])}
    if scopes_known:
        missing_required = [scope for scope in SLACK_REQUIRED_SCOPES if scope not in granted]
        missing_recommended = [scope for scope in SLACK_RECOMMENDED_SCOPES if scope not in granted]
    else:
        missing_required = list(SLACK_REQUIRED_SCOPES)
        missing_recommended = list(SLACK_RECOMMENDED_SCOPES)

    issues: list[str] = []
    if not config.enabled:
        issues.append("moderation is disabled")
    if not token_configured:
        issues.append("SLACK_BOT_TOKEN is not set")
    if not config.log_channel_id:
        issues.append("mod-log channel is not set")
    if not scopes_known:
        issues.append("granted scopes unknown (cannot verify)")
    elif missing_required:
        issues.append(f"missing required scopes: {', '.join(missing_required)}")

    return {
        "platform": Platform.SLACK.value,
        "workspace_id": str(config.workspace_id),
        "enabled": config.enabled,
        "dry_run": config.dry_run,
        "token_configured": token_configured,
        "log_channel_id": config.log_channel_id,
        "scopes_verified": scopes_known,
        "required_scopes": list(SLACK_REQUIRED_SCOPES),
        "recommended_scopes": list(SLACK_RECOMMENDED_SCOPES),
        "missing_required_scopes": missing_required,
        "missing_recommended_scopes": missing_recommended,
        "capabilities": sorted(action.value for action in capabilities()),
        "readiness": "ready" if not issues else "needs attention",
        "issues": issues,
    }


if __name__ == "__main__":  # pragma: no cover - manual entry point
    run_bot()


__all__ = [
    "SLACK_CAPABILITIES",
    "SLACK_RECOMMENDED_SCOPES",
    "SLACK_REQUIRED_SCOPES",
    "WARNING_REACTION_NAME",
    "SlackAdapter",
    "SlackMessage",
    "apply_slack_actions",
    "build_app",
    "capabilities",
    "make_handler",
    "parse_slack_event",
    "run_bot",
    "slack_decision",
]
