"""Twitch chat moderation adapter.

Connects to a Twitch channel via ``twitchio``, moderates every chat message with
the cascade, and — unlike the original evaluate-only version — carries out **real
moderation actions** through an *injectable* chat/API client: delete a message
(``/delete`` / ``clearmsg``), timeout a user (``/timeout``), or ban a user
(``/ban``). The client is a small :class:`TwitchClient` protocol so tests drive
the full action flow with a fake and never need a live Twitch connection.

Repeat unsafe offenders escalate delete -> timeout -> ban (see
:func:`escalate_action` + :class:`OffenseTracker`). The tracker persists through
the shared strike ledger when one is configured, with an in-memory fallback for
standalone/offline use. Anything Twitch chat cannot do (mute / queue /
notify_* ...) degrades to ``flag`` with an auditable reason via
:func:`~nemoguardian.adapters.base.degrade_action`. When no client is wired in
the adapter cannot reach Twitch, so enforcement degrades to ``flag`` for review
rather than being silently dropped.

Broadcasters and moderators can configure the bot from chat with ``!nemo``
commands (``status`` / ``dryrun`` / ``policy`` / ``mode``), mirroring the Discord
slash-command intent; authorization is checked offline against the sender's
badges/role (see :func:`is_authorized`).

Run with::

    TWITCH_TOKEN=xxx python -m nemoguardian.adapters.twitch channel_name

``twitchio`` is imported lazily inside :func:`run_bot` so importing this module
never requires the Twitch SDK, and the moderation path never raises — failures
surface as a planned action / execution status instead.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from nemoguardian.adapters.base import ActionDecision, degrade_action
from nemoguardian.bot import (
    AuditLog,
    BotConfig,
    ConfigStore,
    ModerationContext,
    ModerationEngine,
    ModerationEvaluation,
    Platform,
)
from nemoguardian.bot.engine import REVIEW_DIR_ENV, case_id
from nemoguardian.bot.types import ModerationAction
from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.review.service import ReviewService
from nemoguardian.review.store import StrikeLedger
from nemoguardian.schemas import Mode, VerdictLabel

#: Normalized actions the Twitch adapter can carry out with an injected chat/API
#: client. Twitch chat exposes message deletion (``/delete``), user timeouts
#: (``/timeout``) and bans (``/ban``); anything else (mute / queue / notify_*)
#: degrades to ``flag`` rather than being advertised and then failing live.
TWITCH_CAPABILITIES: frozenset[ModerationAction] = frozenset(
    {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
        ModerationAction.DELETE,
        ModerationAction.TIMEOUT,
        ModerationAction.BAN,
    }
)

#: What the adapter can do with *no* live client wired in: it can still evaluate
#: and surface a case for review, but it cannot reach Twitch to enforce, so every
#: enforcement action degrades to ``flag``.
_OFFLINE_CAPABILITIES: frozenset[ModerationAction] = frozenset(
    {ModerationAction.ALLOW, ModerationAction.FLAG}
)

#: Actions that escalate for repeat offenders (delete -> timeout -> ban).
_ESCALATING_ACTIONS: frozenset[ModerationAction] = frozenset(
    {ModerationAction.DELETE, ModerationAction.TIMEOUT}
)

#: How many recorded unsafe offenses (inclusive) escalate a user to a ban.
DEFAULT_BAN_THRESHOLD = 3

#: Chat-command prefixes an authorized broadcaster/mod can use.
COMMAND_PREFIXES: tuple[str, ...] = ("!nemo", "!ng", "!guardian")

#: Command words :func:`handle_command` actually recognizes. Only an authorized
#: sender using one of these is routed to command handling; anything else (e.g.
#: an unauthorized ``!ng <harmful payload>``) falls through to moderation.
RECOGNIZED_COMMANDS: frozenset[str] = frozenset(
    {
        "status",
        "config",
        "help",
        "commands",
        "dryrun",
        "dry_run",
        "dry-run",
        "enabled",
        "enable",
        "moderation",
        "policy",
        "mode",
    }
)

#: Max length of an operator/model-derived value embedded in an IRC command
#: line before truncation.
_MAX_IRC_FIELD_LEN = 400


def _sanitize_irc(value: object) -> str:
    """Neutralize a value before embedding it in an IRC command string.

    Replaces CR/LF and other control characters with spaces so a crafted
    ``reason`` (which can carry operator/model text) cannot inject a second IRC
    line, then truncates to a bounded length. Fail-safe: never raises.
    """
    text = str(value)
    cleaned = "".join(" " if ch in "\r\n" or ord(ch) < 32 else ch for ch in text)
    cleaned = cleaned.strip()
    if len(cleaned) > _MAX_IRC_FIELD_LEN:
        cleaned = cleaned[:_MAX_IRC_FIELD_LEN].rstrip()
    return cleaned


def capabilities() -> set[ModerationAction]:
    """Return the normalized actions the Twitch adapter can carry out."""
    return set(TWITCH_CAPABILITIES)


def twitch_decision(action: ModerationAction | str) -> ActionDecision:
    """Resolve ``action`` against Twitch capabilities (unsupported -> ``flag``)."""
    return degrade_action(action, capabilities(), Platform.TWITCH)


@dataclass(frozen=True)
class TwitchMessage:
    """A parsed, platform-shaped Twitch chat message.

    ``badges`` carries whatever role/badge object the caller has (a twitchio
    ``Chatter``, a badge dict, or a set of role strings); it is only used for
    chat-command authorization, never for moderation.
    """

    channel: str
    user_id: str
    username: str
    message_id: str
    text: str
    badges: Any = None


@runtime_checkable
class TwitchClient(Protocol):
    """Minimal injectable Twitch chat/API surface used to enforce actions.

    A live implementation wraps ``twitchio`` (sending ``/delete``/``/timeout``/
    ``/ban`` IRC commands or calling Helix moderation endpoints); tests pass a
    fake. Methods may be sync or async — :func:`apply_twitch_actions` awaits the
    result when it is awaitable and never lets an exception escape the mod path.
    """

    def delete_message(self, message: TwitchMessage) -> Any: ...

    def timeout_user(self, message: TwitchMessage, *, seconds: int, reason: str) -> Any: ...

    def ban_user(self, message: TwitchMessage, *, reason: str) -> Any: ...

    def send_message(self, channel: str, text: str) -> Any: ...


@dataclass
class OffenseTracker:
    """Per-(channel, user) unsafe-offense counter for escalation.

    When a :class:`~nemoguardian.review.store.StrikeLedger` is available, each
    unsafe Twitch action appends a strike and escalation uses the active ledger
    total. That lets repeat-offender escalation survive reconnects/restarts and
    lets restore/appeal flows void false-positive strikes. If the ledger is not
    configured or fails, the tracker falls back to the old process-local counter
    so moderation remains fail-safe.
    """

    strike_ledger: StrikeLedger | None = None
    _counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def record(
        self,
        channel: str,
        user_id: str,
        *,
        username: str = "",
        reason: str = "",
        categories: list[str] | None = None,
        case_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> int:
        """Increment and return the offense count for ``(channel, user_id)``."""
        if self.strike_ledger is not None:
            try:
                self.strike_ledger.add_strike(
                    platform=Platform.TWITCH,
                    workspace_id=channel,
                    user_id=user_id,
                    username=username,
                    reason=reason,
                    categories=categories or [],
                    case_id=case_id,
                    details={"source": "twitch_adapter", **(details or {})},
                )
                return self.count(channel, user_id)
            except Exception:
                # Strike persistence is bookkeeping; never let it break or skip
                # live moderation.
                pass
        key = (channel, user_id)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def count(self, channel: str, user_id: str) -> int:
        fallback_count = self._counts.get((channel, user_id), 0)
        if self.strike_ledger is not None:
            try:
                ledger_count = int(
                    self.strike_ledger.total(Platform.TWITCH, channel, user_id)
                )
                return max(ledger_count, fallback_count)
            except Exception:
                pass
        return fallback_count

    def reset(self, channel: str, user_id: str) -> None:
        if self.strike_ledger is not None:
            try:
                for strike in self.strike_ledger.active_strikes(
                    Platform.TWITCH, channel, user_id
                ):
                    self.strike_ledger.void_strike(
                        strike.strike_id, reason="twitch offense counter reset"
                    )
            except Exception:
                pass
        self._counts.pop((channel, user_id), None)


def _strike_ledger_for(review_service: ReviewService | None) -> StrikeLedger | None:
    if review_service is not None:
        return review_service.strikes
    review_dir = os.environ.get(REVIEW_DIR_ENV)
    if not review_dir:
        return None
    try:
        return ReviewService.from_dir(review_dir).strikes
    except Exception:
        return None


def escalate_action(
    base_action: ModerationAction,
    offenses: int,
    *,
    ban_threshold: int = DEFAULT_BAN_THRESHOLD,
) -> ModerationAction:
    """Escalate a repeat unsafe offender: delete -> timeout -> ban.

    ``base_action`` is the per-message plan (``delete`` or ``timeout``).
    ``offenses`` is the running unsafe count for the user (1 = first offense).
    The first offense keeps ``base_action`` (a configured ``timeout`` stays a
    timeout); a second offense escalates a delete to a timeout; reaching
    ``ban_threshold`` escalates to a ban. Non-escalating actions pass through.
    """
    if base_action not in _ESCALATING_ACTIONS:
        return base_action
    if offenses >= max(2, ban_threshold):
        return ModerationAction.BAN
    if offenses >= 2 or base_action == ModerationAction.TIMEOUT:
        return ModerationAction.TIMEOUT
    return ModerationAction.DELETE


# --------------------------------------------------------------------------- #
# Chat commands (#22)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TwitchCommand:
    """A parsed ``!nemo <name> <args>`` chat command."""

    name: str
    args: str = ""


def parse_command(text: str) -> TwitchCommand | None:
    """Parse a chat message into a :class:`TwitchCommand`, or ``None``.

    Recognizes any of :data:`COMMAND_PREFIXES` followed by a command word and
    optional argument string (case-insensitive on the prefix + command word).
    Returns ``None`` for ordinary chat so callers can fall through to moderation.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in COMMAND_PREFIXES:
        if lowered == prefix or lowered.startswith(prefix + " "):
            rest = stripped[len(prefix):].strip()
            if not rest:
                return TwitchCommand("status")
            parts = rest.split(None, 1)
            return TwitchCommand(parts[0].lower(), parts[1].strip() if len(parts) > 1 else "")
    return None


def is_authorized(badges: Any) -> bool:
    """Return ``True`` if ``badges`` denotes a broadcaster or moderator.

    Offline authorization check (no live Twitch call). Accepts the shapes a
    caller realistically has: a twitchio ``Chatter``-like object exposing
    ``is_mod``/``is_broadcaster`` (or ``is_moderator``), a badge ``dict`` such as
    ``{"broadcaster": "1"}``, a single role string, or any iterable of role
    strings. Unknown / viewer-only inputs return ``False``.
    """
    if badges is None:
        return False
    # Object with twitchio-style boolean role attributes.
    for attr in ("is_broadcaster", "is_mod", "is_moderator", "broadcaster", "moderator"):
        if bool(getattr(badges, attr, False)):
            return True
    if isinstance(badges, str):
        return badges.strip().lower() in _AUTHORIZED_ROLES
    if isinstance(badges, dict):
        keys = {str(key).strip().lower() for key in badges}
        return bool(keys & _AUTHORIZED_ROLES)
    try:
        roles = {str(role).strip().lower() for role in badges}
    except TypeError:
        return False
    return bool(roles & _AUTHORIZED_ROLES)


_AUTHORIZED_ROLES: frozenset[str] = frozenset({"broadcaster", "moderator", "mod"})


def authorized_command(text: str, badges: Any) -> TwitchCommand | None:
    """Return a command to run ONLY for an authorized, recognized command.

    A message is routed to command handling only when the sender is authorized
    (:func:`is_authorized`) *and* the parsed command word is one the bot actually
    recognizes (:data:`RECOGNIZED_COMMANDS`). In every other case — plain chat,
    an unauthorized user, or a command-prefixed message whose "command" is really
    a payload (e.g. ``!ng <harmful text>``) — this returns ``None`` so the caller
    falls through and still moderates the raw message. This is what stops a
    moderation bypass via the command prefix.
    """
    command = parse_command(text)
    if command is None:
        return None
    if not is_authorized(badges):
        return None
    if command.name not in RECOGNIZED_COMMANDS:
        return None
    return command


def handle_command(
    command: TwitchCommand,
    *,
    config_store: ConfigStore,
    channel_id: str,
    badges: Any,
) -> str:
    """Execute an authorized chat command and return a chat-ready response.

    Unauthorized senders are rejected without mutating any config. Supported:
    ``status``, ``dryrun on|off``, ``policy [text]``, ``mode fast|standard|deep``,
    plus ``help``. Never raises: bad arguments produce a usage hint.
    """
    if not is_authorized(badges):
        return "nemoguardian: only the broadcaster and moderators can configure moderation."

    name = command.name
    args = command.args.strip()

    if name in ("status", "config"):
        return _status_text(config_store.get(Platform.TWITCH, channel_id))

    if name in ("help", "commands"):
        return (
            "nemoguardian commands: !nemo status | !nemo dryrun on|off | "
            "!nemo policy <text> | !nemo mode fast|standard|deep"
        )

    if name in ("dryrun", "dry_run", "dry-run"):
        value = _parse_bool(args)
        if value is None:
            return "nemoguardian: usage !nemo dryrun on|off"
        config = config_store.update(Platform.TWITCH, channel_id, dry_run=value)
        return f"nemoguardian: dry-run {'on' if config.dry_run else 'off'}."

    if name in ("enabled", "enable", "moderation"):
        value = _parse_bool(args)
        if value is None:
            return "nemoguardian: usage !nemo enabled on|off"
        config = config_store.update(Platform.TWITCH, channel_id, enabled=value)
        return f"nemoguardian: moderation {'enabled' if config.enabled else 'disabled'}."

    if name == "policy":
        if not args:
            return f"nemoguardian policy: {config_store.get(Platform.TWITCH, channel_id).policy_text}"
        config = config_store.update(Platform.TWITCH, channel_id, policy_text=args)
        return f"nemoguardian: policy updated -> {config.policy_text}"

    if name == "mode":
        mode = _parse_mode(args)
        if mode is None:
            return "nemoguardian: usage !nemo mode fast|standard|deep"
        config = config_store.update(Platform.TWITCH, channel_id, mode=mode)
        return f"nemoguardian: mode set to {config.mode.value}."

    return f"nemoguardian: unknown command '{name}'. Try !nemo help."


def _parse_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in ("on", "true", "yes", "enable", "enabled", "1"):
        return True
    if lowered in ("off", "false", "no", "disable", "disabled", "0"):
        return False
    return None


def _parse_mode(value: str) -> Mode | None:
    try:
        return Mode(value.strip().lower())
    except ValueError:
        return None


def _status_text(config: BotConfig) -> str:
    return (
        "nemoguardian status: "
        f"enabled={config.enabled} mode={config.mode.value} "
        f"dry_run={config.dry_run} policy_preset={config.policy_preset} "
        f"delete_unsafe={config.delete_unsafe} timeout_unsafe={config.timeout_unsafe} "
        f"({config.timeout_seconds}s)"
    )


# --------------------------------------------------------------------------- #
# Real action dispatch (#21)
# --------------------------------------------------------------------------- #
async def apply_twitch_actions(
    client: TwitchClient | None,
    message: TwitchMessage,
    evaluation: ModerationEvaluation,
    *,
    offenses: int = 1,
    ban_threshold: int = DEFAULT_BAN_THRESHOLD,
) -> tuple[str, str | None]:
    """Apply a moderation plan to a Twitch chat message.

    Returns ``(execution_status, error)``. Repeat unsafe offenders escalate
    (delete -> timeout -> ban). Any action Twitch cannot perform degrades to
    ``flag`` via :func:`degrade_action`, and ``evaluation.plan.action`` is
    updated to the action actually carried out so the audit record is honest.
    Kept separate from the live ``twitchio`` client so tests drive the flow with
    a fake; never raises on the moderation path.
    """
    config = evaluation.config
    plan = evaluation.plan
    if plan.action == ModerationAction.ALLOW:
        return "allowed", None

    desired = plan.action
    if desired in _ESCALATING_ACTIONS:
        desired = escalate_action(desired, offenses, ban_threshold=ban_threshold)

    decision = degrade_action(desired, capabilities(), Platform.TWITCH)
    plan.action = decision.action
    degrade_note = decision.reason if decision.degraded else None

    if config.dry_run:
        return "dry-run", None

    if client is None:
        # No live connection to Twitch: surface for review instead of silently
        # dropping enforcement.
        return _plan_without_client(evaluation)

    applied: list[str] = []
    errors: list[str] = []
    reason = _reason(evaluation)
    action = plan.action
    primary_ok = False

    if action == ModerationAction.DELETE:
        ok, err = await _safe_invoke(client, "delete_message", message)
        _record(applied, errors, "delete", ok, err)
        primary_ok = ok
    elif action == ModerationAction.TIMEOUT:
        ok, err = await _safe_invoke(
            client, "timeout_user", message, seconds=config.timeout_seconds, reason=reason
        )
        _record(applied, errors, "timeout", ok, err)
        primary_ok = ok
    elif action == ModerationAction.BAN:
        ok, err = await _safe_invoke(client, "ban_user", message, reason=reason)
        _record(applied, errors, "ban", ok, err)
        primary_ok = ok
    elif action == ModerationAction.FLAG:
        # Degraded / controversial: nothing to enforce, but the case is surfaced.
        applied.append("flag")

    # Only warn the user when the primary enforcement actually succeeded; a
    # failed delete/timeout/ban should not advertise a moderation that did not
    # happen.
    if (
        config.public_warning
        and primary_ok
        and action
        in (ModerationAction.DELETE, ModerationAction.TIMEOUT, ModerationAction.BAN)
    ):
        ok, err = await _safe_invoke(
            client, "send_message", message.channel, _warning_text(message, evaluation)
        )
        _record(applied, errors, "warning", ok, err)

    if degrade_note:
        errors.append(degrade_note)

    if errors:
        return ("partial" if applied else "failed"), "; ".join(errors)
    return ("+".join(applied) if applied else "planned"), None


def _plan_without_client(evaluation: ModerationEvaluation) -> tuple[str, str | None]:
    plan = evaluation.plan
    if plan.action in _OFFLINE_CAPABILITIES:
        return ("allowed" if plan.action == ModerationAction.ALLOW else "flag"), None
    decision = degrade_action(plan.action, _OFFLINE_CAPABILITIES, Platform.TWITCH)
    plan.action = decision.action
    return "flag", decision.reason


def _record(applied: list[str], errors: list[str], label: str, ok: bool, err: str | None) -> None:
    if ok:
        applied.append(label)
    elif err:
        errors.append(f"{label}:{err}")


async def _safe_invoke(
    client: Any,
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> tuple[bool, str | None]:
    """Call a client method, never raising; awaits awaitable results."""
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


def _reason(evaluation: ModerationEvaluation) -> str:
    return evaluation.plan.reason or "policy violation"


def _warning_text(message: TwitchMessage, evaluation: ModerationEvaluation) -> str:
    return f"@{message.username} your message was moderated by nemoguardian: {_reason(evaluation)}"


def make_moderator(
    cascade: Cascade | None = None,
    *,
    config_store: ConfigStore | None = None,
    audit_log: AuditLog | None = None,
    review_service: ReviewService | None = None,
    channel_id: str = "twitch",
    emit: Callable[[str], None] = print,
    engine: ModerationEngine | None = None,
    client: TwitchClient | None = None,
    offense_tracker: OffenseTracker | None = None,
    ban_threshold: int = DEFAULT_BAN_THRESHOLD,
):
    """Build an async moderator for one Twitch channel.

    Pass a :class:`TwitchClient` to carry out real enforcement (delete / timeout
    / ban) with per-user escalation; without one the flow degrades enforcement to
    ``flag`` for review. Pass a prebuilt ``engine`` to share one
    :class:`ModerationEngine` with :class:`TwitchAdapter`.
    """
    if engine is None:
        cascade = cascade or Cascade(CascadeConfig.from_env())
        engine = ModerationEngine(
            Platform.TWITCH,
            cascade=cascade,
            config_store=config_store,
            audit_log=audit_log,
            review_service=review_service,
        )
    review_for_strikes = review_service if review_service is not None else engine.review_service
    tracker = (
        offense_tracker
        if offense_tracker is not None
        else OffenseTracker(_strike_ledger_for(review_for_strikes))
    )

    async def moderate(
        text: str,
        *,
        user_id: str = "unknown",
        username: str = "unknown",
        message_id: str | None = None,
        badges: Any = None,
    ) -> str:
        config = engine.config_for(channel_id)
        mid = str(message_id) if message_id else f"twitch-{abs(hash((channel_id, user_id, text))) % 10_000_000}"
        context = ModerationContext(
            platform=Platform.TWITCH,
            workspace_id=channel_id,
            channel_id=channel_id,
            message_id=mid,
            user_id=user_id,
            username=username,
            text=text,
        )
        evaluation = await asyncio.to_thread(engine.evaluate, context, config)
        if evaluation.result is None:
            emit(f"[twitch] allow: {text[:60]} (skipped={evaluation.skip_reason})")
            return ModerationAction.ALLOW.value

        if client is None:
            status, error = await apply_twitch_actions(
                None, _message(channel_id, user_id, username, mid, text, badges), evaluation
            )
        else:
            # Only count an offense when enforcement actually runs; a dry-run
            # must not climb the escalation counter (delete -> timeout -> ban).
            offenses = (
                tracker.record(
                    channel_id,
                    user_id,
                    username=username,
                    reason=evaluation.plan.reason,
                    categories=evaluation.result.categories,
                    case_id=case_id(evaluation.context),
                    details={"message_id": mid},
                )
                if evaluation.result.verdict == VerdictLabel.UNSAFE and not config.dry_run
                else 0
            )
            status, error = await apply_twitch_actions(
                client,
                _message(channel_id, user_id, username, mid, text, badges),
                evaluation,
                offenses=offenses,
                ban_threshold=ban_threshold,
            )

        engine.record(evaluation, execution_status=status, error=error)
        action = evaluation.plan.action.value
        emit(
            f"[twitch] {action}: {text[:60]} "
            f"({evaluation.result.verdict.value}, score={evaluation.result.score})"
        )
        return action

    return moderate


def _message(
    channel_id: str,
    user_id: str,
    username: str,
    message_id: str,
    text: str,
    badges: Any,
) -> TwitchMessage:
    return TwitchMessage(
        channel=channel_id,
        user_id=user_id,
        username=username,
        message_id=message_id,
        text=text,
        badges=badges,
    )


class TwitchAdapter:
    """Thin :class:`~nemoguardian.adapters.base.PlatformAdapter` over Twitch chat.

    Wraps :func:`make_moderator`; enforcement happens inside
    :meth:`handle_event` when a :class:`TwitchClient` is injected. :meth:`apply_action`
    surfaces the capability-degraded decision for a planned action.
    """

    platform = Platform.TWITCH

    def __init__(
        self,
        cascade: Cascade | None = None,
        *,
        config_store: ConfigStore | None = None,
        audit_log: AuditLog | None = None,
        review_service: ReviewService | None = None,
        channel_id: str = "twitch",
        client: TwitchClient | None = None,
        emit: Callable[[str], None] = print,
    ) -> None:
        self.channel_id = channel_id
        self.client = client
        self.offense_tracker = OffenseTracker(_strike_ledger_for(review_service))
        self.engine = ModerationEngine(
            Platform.TWITCH,
            cascade=cascade,
            config_store=config_store,
            audit_log=audit_log,
            review_service=review_service,
        )
        self._moderate = make_moderator(
            channel_id=channel_id,
            emit=emit,
            engine=self.engine,
            client=client,
            offense_tracker=self.offense_tracker,
        )

    def capabilities(self) -> set[ModerationAction]:
        return capabilities()

    def doctor(self, workspace_id: str | None = None) -> dict[str, object]:
        """Return a lightweight readiness snapshot (never raises)."""
        workspace = str(workspace_id) if workspace_id is not None else self.channel_id
        config = self.engine.config_for(workspace)
        token_configured = bool(os.environ.get("TWITCH_TOKEN"))
        client_configured = self.client is not None
        issues: list[str] = []
        if not config.enabled:
            issues.append("moderation is disabled")
        if not token_configured:
            issues.append("TWITCH_TOKEN is not set")
        if not client_configured:
            issues.append("no chat client wired in (enforcement degrades to flag)")
        return {
            "platform": Platform.TWITCH.value,
            "workspace_id": workspace,
            "enabled": config.enabled,
            "dry_run": config.dry_run,
            "token_configured": token_configured,
            "client_configured": client_configured,
            "capabilities": sorted(action.value for action in capabilities()),
            "readiness": "ready" if not issues else "needs attention",
            "issues": issues,
        }

    def configure(self, workspace_id: str | None = None, **changes: object) -> BotConfig:
        workspace = str(workspace_id) if workspace_id is not None else self.channel_id
        if changes:
            return self.engine.config_store.update(Platform.TWITCH, workspace, **changes)
        return self.engine.config_for(workspace)

    async def handle_event(
        self,
        text: str,
        *,
        user_id: str = "unknown",
        username: str = "unknown",
        message_id: str | None = None,
        badges: Any = None,
    ) -> str:
        return await self._moderate(
            text, user_id=user_id, username=username, message_id=message_id, badges=badges
        )

    def handle_command(self, text: str, *, badges: Any = None) -> str | None:
        """Parse + execute a chat command; ``None`` when ``text`` isn't a command."""
        command = parse_command(text)
        if command is None:
            return None
        return handle_command(
            command,
            config_store=self.engine.config_store,
            channel_id=self.channel_id,
            badges=badges,
        )

    def apply_action(self, evaluation: ModerationEvaluation) -> ActionDecision:
        """Resolve the planned action against Twitch capabilities."""
        return degrade_action(evaluation.plan.action, self.capabilities(), Platform.TWITCH)

    async def enforce(
        self,
        evaluation: ModerationEvaluation,
        message: TwitchMessage,
        *,
        offenses: int = 1,
    ) -> tuple[str, str | None]:
        """Carry out the planned action via the injected client (delegates)."""
        return await apply_twitch_actions(self.client, message, evaluation, offenses=offenses)

    def record_audit(
        self,
        evaluation: ModerationEvaluation,
        *,
        execution_status: str,
        error: str | None = None,
    ) -> None:
        self.engine.record(evaluation, execution_status=execution_status, error=error)


class _TwitchioClient:  # pragma: no cover - requires a live twitchio connection
    """Live :class:`TwitchClient` backed by ``twitchio`` chat/IRC commands.

    Sends moderation commands through the channel (``/delete``, ``/timeout``,
    ``/ban``). Kept out of the unit-tested path (no live SDK in tests).
    """

    def __init__(self, bot: Any) -> None:
        self._bot = bot

    async def _send(self, channel: str, text: str) -> None:
        chan = self._bot.get_channel(channel)
        if chan is not None:
            await chan.send(text)

    async def delete_message(self, message: TwitchMessage) -> None:
        await self._send(message.channel, f"/delete {message.message_id}")

    async def timeout_user(self, message: TwitchMessage, *, seconds: int, reason: str) -> None:
        safe_reason = _sanitize_irc(reason)
        await self._send(message.channel, f"/timeout {message.username} {seconds} {safe_reason}")

    async def ban_user(self, message: TwitchMessage, *, reason: str) -> None:
        await self._send(message.channel, f"/ban {message.username} {_sanitize_irc(reason)}")

    async def send_message(self, channel: str, text: str) -> None:
        await self._send(channel, text)


async def dispatch_chat(
    content: str,
    author: Any,
    *,
    moderate: Callable[..., Any],
    client: TwitchClient | None,
    config_store: ConfigStore,
    channel: str,
    message_id: str | None = None,
) -> str | None:
    """Route one incoming chat message.

    Authorized + recognized command -> execute it and return its response text.
    Everything else (plain chat, an unauthorized sender, or an unrecognized
    ``!ng <payload>``) -> moderate the raw message and return ``None`` so harmful
    content is always evaluated/enforced. Never raises on the moderation path.
    """
    command = authorized_command(content, author)
    if command is not None:
        response = handle_command(
            command, config_store=config_store, channel_id=channel, badges=author
        )
        await _safe_invoke(client, "send_message", channel, response)
        return response
    await moderate(
        content,
        user_id=str(getattr(author, "id", "unknown")),
        username=str(getattr(author, "name", "unknown")),
        message_id=message_id,
        badges=author,
    )
    return None


def run_bot(channel: str) -> None:
    token = os.environ.get("TWITCH_TOKEN")
    if not token:
        raise RuntimeError("TWITCH_TOKEN env var required")

    from twitchio.ext import commands

    config_store = ConfigStore()
    audit_log = AuditLog()
    review_service = ReviewService()

    bot = commands.Bot(token=token, prefix="!", initial_channels=[channel])
    client = _TwitchioClient(bot)
    moderate = make_moderator(
        channel_id=channel,
        config_store=config_store,
        audit_log=audit_log,
        review_service=review_service,
        client=client,
    )

    @bot.event
    async def event_message(message) -> None:  # pragma: no cover - requires live twitchio
        if getattr(message, "echo", False):
            return
        await dispatch_chat(
            getattr(message, "content", "") or "",
            getattr(message, "author", None),
            moderate=moderate,
            client=client,
            config_store=config_store,
            channel=channel,
            message_id=getattr(message, "id", None),
        )

    bot.run()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m nemoguardian.adapters.twitch <channel>")
        sys.exit(1)
    run_bot(sys.argv[1])


__all__ = [
    "COMMAND_PREFIXES",
    "DEFAULT_BAN_THRESHOLD",
    "RECOGNIZED_COMMANDS",
    "TWITCH_CAPABILITIES",
    "OffenseTracker",
    "TwitchAdapter",
    "TwitchClient",
    "TwitchCommand",
    "TwitchMessage",
    "apply_twitch_actions",
    "authorized_command",
    "capabilities",
    "dispatch_chat",
    "escalate_action",
    "handle_command",
    "is_authorized",
    "make_moderator",
    "parse_command",
    "run_bot",
    "twitch_decision",
]
