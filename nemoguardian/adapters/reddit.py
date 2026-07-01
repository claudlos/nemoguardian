"""Reddit subreddit-moderator adapter (PRAW).

A subreddit moderator helper: parses a comment or submission event, runs the
shared moderation cascade, and enforces the result with the realistic Reddit
mod-action set: ``remove`` unsafe content, ``report`` borderline content, and
optionally drop a moderator note into modmail. Anything Reddit's mod API cannot
do (timeout / ban / mute — handled via native bans elsewhere, not here) degrades
to ``flag`` (a report) with an auditable reason via
:func:`~nemoguardian.adapters.base.degrade_action`.

Design
------
* The same async handler accepts BOTH ingestion shapes: a raw item ``dict`` and
  a PRAW ``Comment`` / ``Submission`` object. Parsing is duck-typed via
  :func:`_attr` so neither shape requires the SDK.
* The network side is injectable: every enforcement call goes through a small
  ``client`` object (``remove`` / ``report`` / ``send_modmail``). Tests pass a
  fake; production wraps a live ``praw.Reddit`` via :class:`_PrawClient`.
  ``praw`` is imported lazily so ``import nemoguardian`` never needs the
  ``[reddit]`` extra and the moderation path never raises.
* Removal / reporting requires the bot to be a subreddit moderator with the
  right permissions. :meth:`RedditAdapter.doctor` checks this offline against a
  passed-in moderator object (no live call).

Run with::

    REDDIT_CLIENT_ID=xxx REDDIT_CLIENT_SECRET=xxx REDDIT_USERNAME=bot \
        REDDIT_PASSWORD=xxx REDDIT_SUBREDDIT=mysub python -m nemoguardian.adapters.reddit
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

#: Reddit mod permissions this adapter needs. ``all`` (full mod) grants both.
REDDIT_REQUIRED_PERMISSIONS: tuple[str, ...] = ("posts",)
REDDIT_RECOMMENDED_PERMISSIONS: tuple[str, ...] = ("mail",)

#: Normalized actions a subreddit-moderator bot can carry out. ``remove`` maps to
#: :data:`ModerationAction.DELETE` and ``report`` to :data:`ModerationAction.FLAG`;
#: there is no per-item timeout/mute here, so those degrade to ``flag``.
REDDIT_CAPABILITIES: frozenset[ModerationAction] = frozenset(
    {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
        ModerationAction.DELETE,
        ModerationAction.NOTIFY_MODS,
    }
)


def capabilities() -> set[ModerationAction]:
    """Return the normalized actions the Reddit adapter can carry out."""
    return set(REDDIT_CAPABILITIES)


@dataclass(frozen=True)
class RedditItem:
    """The moderation-relevant slice of a Reddit comment or submission."""

    kind: str  # "comment" or "submission"
    item_id: str
    subreddit: str
    author: str
    text: str
    permalink: str | None = None


def _attr(obj: Any, *names: str) -> Any:
    """Read the first present, non-``None`` field across dict keys / attributes.

    Reddit items arrive either as raw JSON ``dict`` or as PRAW objects; this reads
    both without importing the SDK.
    """
    if obj is None:
        return None
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _stringify(value: Any) -> str | None:
    """Render a PRAW ``Redditor`` / ``Subreddit`` (or dict/str) to a plain name."""
    if value is None:
        return None
    name = _attr(value, "display_name", "name")
    if name is not None:
        return str(name)
    return str(value)


def parse_reddit_item(item: Any) -> RedditItem | None:
    """Parse a Reddit comment / submission into a :class:`RedditItem`.

    The kind is taken from an explicit ``kind`` field when present, otherwise
    inferred: an item with a ``title`` is a submission, one with a ``body`` is a
    comment. Submission text is ``title`` + ``selftext``. Returns ``None`` when
    the item has no id or no text. Never raises on malformed input.
    """
    if item is None:
        return None

    kind = _attr(item, "kind")
    title = _attr(item, "title")
    body = _attr(item, "body")
    if kind is None:
        kind = "submission" if title is not None else "comment"
    kind = str(kind)

    if kind == "submission":
        selftext = _attr(item, "selftext") or ""
        text = "\n".join(part for part in (str(title or ""), str(selftext)) if part).strip()
    else:
        text = str(body or "").strip()

    item_id = _attr(item, "id", "item_id")
    subreddit = _stringify(_attr(item, "subreddit")) or "unknown"
    author = _stringify(_attr(item, "author")) or "unknown"
    permalink = _attr(item, "permalink")

    if not item_id or not text:
        return None
    return RedditItem(
        kind=kind,
        item_id=str(item_id),
        subreddit=str(subreddit),
        author=str(author),
        text=text,
        permalink=str(permalink) if permalink is not None else None,
    )


def reddit_decision(action: ModerationAction | str) -> ActionDecision:
    """Resolve ``action`` against Reddit capabilities (unsupported -> ``flag``)."""
    return degrade_action(action, capabilities(), Platform.REDDIT)


def _context_from_item(item: RedditItem) -> ModerationContext:
    return ModerationContext(
        platform=Platform.REDDIT,
        workspace_id=item.subreddit,
        channel_id=item.subreddit,
        message_id=item.item_id,
        user_id=item.author,
        username=item.author,
        text=item.text,
        permalink=item.permalink,
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


async def apply_reddit_actions(
    client: Any,
    item: RedditItem,
    evaluation: ModerationEvaluation,
) -> tuple[str, str | None]:
    """Apply a moderation plan to a Reddit item via the injected ``client``.

    Returns ``(execution_status, error)``. Never raises: every call is guarded.
    Honors dry-run and degrades unsupported actions (timeout/ban/mute) to
    ``flag`` (a report) with the reason surfaced in the optional modmail note.
    """
    config = evaluation.config
    plan = evaluation.plan
    if plan.action == ModerationAction.ALLOW:
        return "allowed", None

    decision = degrade_action(plan.action, capabilities(), Platform.REDDIT)
    notes: list[str] = []
    if decision.degraded:
        notes.append(decision.reason or "degraded")
        plan.action = decision.action

    if config.dry_run:
        return "dry-run", decision.reason if decision.degraded else None

    applied: list[str] = []
    errors: list[str] = []

    if decision.action == ModerationAction.DELETE:
        ok, err = await _safe_call(client, "remove", item.kind, item.item_id)
        if ok:
            applied.append("remove")
        else:
            errors.append(f"remove:{err}")
    elif decision.action == ModerationAction.FLAG:
        ok, err = await _safe_call(client, "report", item.kind, item.item_id, _reason(evaluation))
        if ok:
            applied.append("report")
        else:
            errors.append(f"report:{err}")

    if config.log_channel_id:
        ok, err = await _safe_call(
            client,
            "send_modmail",
            item.subreddit,
            f"nemoguardian: {case_id(evaluation.context)}",
            _mod_log_text(item, evaluation, applied=applied, errors=errors, notes=notes),
        )
        if ok:
            applied.append("modmail")
        else:
            errors.append(f"modmail:{err}")

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
    """Build an async item handler that runs the full Reddit moderation flow.

    The returned coroutine accepts ``(item, *, client)`` where ``item`` is a raw
    ``dict`` or PRAW object and ``client`` is the injectable mod-action client. It
    returns the :class:`ModerationEvaluation` (or ``None`` when the item is not
    moderatable) so callers/tests can introspect. Never raises.
    """
    engine = engine or ModerationEngine(
        Platform.REDDIT,
        cascade=cascade,
        config_store=config_store,
        audit_log=audit_log,
        review_service=review_service,
    )

    async def on_item(item: Any, *, client: Any = None) -> ModerationEvaluation | None:
        parsed = parse_reddit_item(item)
        if parsed is None:
            return None
        config = engine.config_for(parsed.subreddit)
        context = _context_from_item(parsed)
        evaluation = await asyncio.to_thread(engine.evaluate, context, config)
        if evaluation.skipped:
            return evaluation
        status, error = await apply_reddit_actions(client, parsed, evaluation)
        engine.record(evaluation, execution_status=status, error=error)
        return evaluation

    return on_item


def _mod_log_text(
    item: RedditItem,
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
        f"subreddit: r/{item.subreddit} {item.kind}: {item.item_id} author: u/{item.author}\n"
        f"verdict: {verdict} score: {score} action: {evaluation.plan.action.value}\n"
        f"applied: {', '.join(applied) or 'none'} errors: {', '.join(errors) or 'none'}\n"
        f"categories: {categories} notes: {'; '.join(notes) or 'none'}\n"
        f"content: {redacted_excerpt(evaluation.context.text)}"
    )


def _reason(evaluation: ModerationEvaluation) -> str:
    return evaluation.plan.reason or "policy violation"


def _doctor_report(
    config: BotConfig,
    moderator: Any,
    *,
    token_configured: bool,
    workspace_id: str,
) -> dict[str, Any]:
    """Offline moderator-readiness snapshot from a passed-in moderator object.

    ``moderator`` is the bot's entry in the subreddit's moderator relationship
    (a PRAW ``Redditor`` exposing ``mod_permissions``), or a dict/list of perms.
    ``None`` means the bot is not a confirmed moderator. No live Reddit call is
    made.
    """
    is_moderator = moderator is not None
    permissions = _permissions(moderator)
    has_all = "all" in permissions
    can_remove = has_all or "posts" in permissions
    can_modmail = has_all or "mail" in permissions

    missing_required: list[str] = []
    missing_recommended: list[str] = []
    if not is_moderator:
        missing_required.append("bot is not a subreddit moderator")
    elif not can_remove:
        missing_required.append("posts")
    if is_moderator and not can_modmail:
        missing_recommended.append("mail")

    issues: list[str] = []
    if not config.enabled:
        issues.append("moderation is disabled")
    if not token_configured:
        issues.append("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set")
    if not is_moderator:
        issues.append("granted permissions unknown (not a confirmed moderator)")
    elif missing_required:
        issues.append(f"missing required mod permissions: {', '.join(missing_required)}")

    return {
        "platform": Platform.REDDIT.value,
        "workspace_id": workspace_id,
        "enabled": config.enabled,
        "dry_run": config.dry_run,
        "token_configured": token_configured,
        "is_moderator": is_moderator,
        "permissions": sorted(permissions),
        "can_remove": can_remove,
        "can_modmail": can_modmail,
        "required_permissions": list(REDDIT_REQUIRED_PERMISSIONS),
        "recommended_permissions": list(REDDIT_RECOMMENDED_PERMISSIONS),
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "capabilities": sorted(action.value for action in capabilities()),
        "readiness": "ready" if not issues else "needs attention",
        "issues": issues,
    }


def _permissions(moderator: Any) -> set[str]:
    if moderator is None:
        return set()
    if isinstance(moderator, (list, tuple, set)):
        return {str(perm) for perm in moderator}
    perms = _attr(moderator, "mod_permissions", "permissions")
    if perms is None:
        return set()
    return {str(perm) for perm in perms}


class RedditAdapter:
    """Thin :class:`~nemoguardian.adapters.base.PlatformAdapter` over Reddit.

    Delegates to the module-level helpers so behavior is shared with the live
    PRAW client while exposing the normalized interface. Heavy ``praw`` imports
    stay inside :func:`build_reddit` / :func:`run_bot`.
    """

    platform = Platform.REDDIT

    def __init__(
        self,
        cascade: Cascade | None = None,
        *,
        config_store: ConfigStore | None = None,
        audit_log: AuditLog | None = None,
        review_service: ReviewService | None = None,
    ) -> None:
        self.engine = ModerationEngine(
            Platform.REDDIT,
            cascade=cascade,
            config_store=config_store,
            audit_log=audit_log,
            review_service=review_service,
        )
        self._handler = make_handler(engine=self.engine)

    def capabilities(self) -> set[ModerationAction]:
        return capabilities()

    def doctor(self, workspace_id: str, moderator: Any = None) -> dict[str, Any]:
        """Return an offline moderator-readiness report for ``workspace_id``."""
        workspace = str(workspace_id)
        config = self.engine.config_for(workspace)
        token_configured = bool(
            os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET")
        )
        return _doctor_report(
            config,
            moderator,
            token_configured=token_configured,
            workspace_id=workspace,
        )

    def configure(self, workspace_id: str, **changes: Any) -> BotConfig:
        """Read (no ``changes``) or update the per-subreddit config."""
        if changes:
            return self.engine.config_store.update(Platform.REDDIT, str(workspace_id), **changes)
        return self.engine.config_for(str(workspace_id))

    async def handle_event(self, item: Any, *, client: Any = None) -> ModerationEvaluation | None:
        """Moderate one Reddit comment / submission end to end."""
        return await self._handler(item, client=client)

    async def apply_action(
        self,
        client: Any,
        item: RedditItem,
        evaluation: ModerationEvaluation,
    ) -> tuple[str, str | None]:
        """Apply the planned action to ``item`` (delegates to the action flow)."""
        return await apply_reddit_actions(client, item, evaluation)

    def record_audit(
        self,
        evaluation: ModerationEvaluation,
        *,
        execution_status: str,
        error: str | None = None,
    ) -> None:
        self.engine.record(evaluation, execution_status=execution_status, error=error)


class _PrawClient:
    """Adapts a live ``praw.Reddit`` to the injectable ``client`` surface."""

    def __init__(self, reddit: Any) -> None:
        self._reddit = reddit

    def _resolve(self, kind: str, item_id: str) -> Any:
        if kind == "submission":
            return self._reddit.submission(id=item_id)
        return self._reddit.comment(id=item_id)

    def remove(self, kind: str, item_id: str) -> None:
        self._resolve(kind, item_id).mod.remove()

    def report(self, kind: str, item_id: str, reason: str) -> None:
        self._resolve(kind, item_id).report(reason)

    def send_modmail(self, subreddit: str, subject: str, body: str) -> None:
        self._reddit.subreddit(subreddit).message(subject=subject, message=body)


def build_reddit():  # pragma: no cover - requires praw
    """Create a live ``praw.Reddit`` from the standard Reddit env vars.

    ``praw`` is imported here (lazily) so importing this module never requires the
    ``[reddit]`` extra.
    """
    import praw

    return praw.Reddit(
        client_id=os.environ.get("REDDIT_CLIENT_ID"),
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
        username=os.environ.get("REDDIT_USERNAME"),
        password=os.environ.get("REDDIT_PASSWORD"),
        user_agent=os.environ.get("REDDIT_USER_AGENT", "nemoguardian-moderator/0.1"),
    )


def run_bot() -> None:  # pragma: no cover - manual entry point
    """Entry point: stream a subreddit's new comments/submissions and moderate."""
    if not (os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET")):
        raise RuntimeError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars required")
    subreddit_name = os.environ.get("REDDIT_SUBREDDIT")
    if not subreddit_name:
        raise RuntimeError("REDDIT_SUBREDDIT env var required")

    reddit = build_reddit()
    handler = make_handler(review_service=ReviewService())
    client = _PrawClient(reddit)
    subreddit = reddit.subreddit(subreddit_name)

    for comment in subreddit.stream.comments(skip_existing=True):
        asyncio.run(handler(comment, client=client))


if __name__ == "__main__":  # pragma: no cover - manual entry point
    run_bot()


__all__ = [
    "REDDIT_CAPABILITIES",
    "REDDIT_RECOMMENDED_PERMISSIONS",
    "REDDIT_REQUIRED_PERMISSIONS",
    "RedditAdapter",
    "RedditItem",
    "apply_reddit_actions",
    "build_reddit",
    "capabilities",
    "make_handler",
    "parse_reddit_item",
    "reddit_decision",
    "run_bot",
]
