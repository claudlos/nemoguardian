"""Platform-neutral moderation bot engine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nemoguardian.bot.audit import AuditLog, AuditRecord, redacted_excerpt, text_hash
from nemoguardian.bot.config import BotConfig, ConfigStore
from nemoguardian.bot.types import ModerationAction, Platform
from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.policy.presets import get_preset
from nemoguardian.schemas import ModerateRequest, ModerateResponse, VerdictLabel

if TYPE_CHECKING:
    from nemoguardian.review.models import ReviewCase
    from nemoguardian.review.service import ReviewService

# Plan actions that should land a case in the human-review queue.
_REVIEW_ACTIONS = frozenset({ModerationAction.FLAG, ModerationAction.QUEUE})

# Env var pointing at the directory that backs the default review store. When
# ``review_queue`` is enabled but no ReviewService was injected, the engine
# lazily builds one from this path (see ``ModerationEngine._default_review_service``).
REVIEW_DIR_ENV = "NEMOGUARDIAN_REVIEW_DIR"


@dataclass
class ModerationContext:
    platform: Platform
    workspace_id: str
    channel_id: str
    message_id: str
    user_id: str
    username: str
    text: str
    user_role_ids: set[str] | None = None
    permalink: str | None = None


@dataclass
class ModerationPlan:
    action: ModerationAction
    delete_message: bool = False
    add_reaction: bool = False
    timeout_user: bool = False
    log_to_mods: bool = False
    notify_user: bool = False
    public_warning: bool = False
    reason: str = "allowed"


@dataclass
class ModerationEvaluation:
    context: ModerationContext
    config: BotConfig
    result: ModerateResponse | None
    plan: ModerationPlan
    skipped: bool = False
    skip_reason: str | None = None


class ModerationEngine:
    def __init__(
        self,
        platform: Platform | str,
        *,
        cascade: Cascade | None = None,
        config_store: ConfigStore | None = None,
        audit_log: AuditLog | None = None,
        review_service: ReviewService | None = None,
    ) -> None:
        self.platform = Platform(platform)
        self.cascade = cascade or Cascade(CascadeConfig.from_env())
        self.config_store = config_store or ConfigStore()
        self.audit_log = audit_log or AuditLog()
        # Opt-in: only enqueue cases for human review when a ReviewService is
        # injected. Left as None (the default) the engine behaves exactly as
        # before — no extra writes, fully backward-compatible.
        self.review_service = review_service

    def config_for(self, workspace_id: str) -> BotConfig:
        return self.config_store.get(self.platform, workspace_id)

    def evaluate(
        self,
        context: ModerationContext,
        config: BotConfig | None = None,
    ) -> ModerationEvaluation:
        config = config or self.config_for(context.workspace_id)
        skip_reason = self._skip_reason(context, config)
        if skip_reason:
            return ModerationEvaluation(
                context=context,
                config=config,
                result=None,
                plan=ModerationPlan(action=ModerationAction.ALLOW, reason=skip_reason),
                skipped=True,
                skip_reason=skip_reason,
            )

        request = ModerateRequest(text=context.text, policy=config.policy_text, mode=config.mode)
        result = self.cascade.moderate(request, policy_engine=get_preset(config.policy_preset))
        plan = plan_for_result(result, config)
        return ModerationEvaluation(context=context, config=config, result=result, plan=plan)

    def record(self, evaluation: ModerationEvaluation, *, execution_status: str, error: str | None = None) -> None:
        if evaluation.result is None:
            return
        record = AuditRecord(
            case_id=case_id(evaluation.context),
            platform=evaluation.context.platform,
            workspace_id=evaluation.context.workspace_id,
            channel_id=evaluation.context.channel_id,
            message_id=evaluation.context.message_id,
            user_id=evaluation.context.user_id,
            username=evaluation.context.username,
            action=evaluation.plan.action,
            verdict=evaluation.result.verdict,
            score=evaluation.result.score,
            mode=evaluation.result.mode,
            categories=evaluation.result.categories,
            matched_policy_rule=evaluation.result.matched_policy_rule,
            request_id=evaluation.result.request_id,
            latency_ms=evaluation.result.total_latency_ms,
            text_sha256=text_hash(evaluation.context.text),
            text_excerpt=redacted_excerpt(evaluation.context.text),
            dry_run=evaluation.config.dry_run,
            execution_status=execution_status,
            error=error,
            details={"permalink": evaluation.context.permalink, "text_redacted": True},
        )
        self.audit_log.append(record)
        self.enqueue_review(evaluation)

    def enqueue_review(self, evaluation: ModerationEvaluation) -> ReviewCase | None:
        """Enqueue a controversial / flagged case for human review (audit #26).

        No-op unless the workspace config opts in (``review_queue``) *and* a review
        store is available. The store is either the :class:`ReviewService` injected
        at construction or — when ``review_queue`` is on and none was injected — a
        default one built lazily from ``NEMOGUARDIAN_REVIEW_DIR`` (see
        :meth:`_default_review_service`). This lets the queue fire on the live
        adapter paths without every adapter having to wire a service in. When the
        config opts out *or* no directory is configured the method stays a no-op,
        so existing deployments are unaffected.

        Only cases that actually need a human — a CONTROVERSIAL verdict or a
        FLAG/QUEUE action — are enqueued; safe and clear-unsafe verdicts
        (auto-actioned) are not.

        The raw message is handed to the review store, which redacts it into an
        excerpt + hash — raw text is never persisted here. Fail-safe: any error is
        swallowed so review bookkeeping can never break the moderation path. The
        review case id mirrors the audit ``case_id`` so re-recording is idempotent.
        """
        if not getattr(evaluation.config, "review_queue", False):
            return None
        service = self.review_service or self._default_review_service()
        if service is None:
            return None
        if not needs_review(evaluation):
            return None
        cid = case_id(evaluation.context)
        try:
            return service.enqueue(
                platform=evaluation.context.platform,
                workspace_id=evaluation.context.workspace_id,
                user_id=evaluation.context.user_id,
                username=evaluation.context.username,
                channel_id=evaluation.context.channel_id,
                message_id=evaluation.context.message_id,
                text=evaluation.context.text,  # redacted inside enqueue; raw never stored
                case_id=cid,
                verdict=evaluation.result.verdict if evaluation.result else None,
                score=evaluation.result.score if evaluation.result else 0.0,
                reason=evaluation.plan.reason,
                categories=list(evaluation.result.categories) if evaluation.result else [],
                action=evaluation.plan.action,
                source_case_id=cid,
            )
        except Exception:
            # Review enqueue is best-effort; never let it break moderation.
            return None

    @staticmethod
    def _default_review_service() -> ReviewService | None:
        """Lazily build the default review store from ``NEMOGUARDIAN_REVIEW_DIR``.

        Returns ``None`` (a no-op) when the env var is unset or the store cannot be
        constructed, so an unconfigured or misconfigured review directory can never
        break moderation. Constructed fresh each call: the underlying append-only
        JSONL stores do no I/O until a case is written, so this stays cheap and
        stays in sync with the current environment.
        """
        review_dir = os.environ.get(REVIEW_DIR_ENV)
        if not review_dir:
            return None
        try:
            from nemoguardian.review.service import ReviewService

            return ReviewService.from_dir(review_dir)
        except Exception:
            return None

    @staticmethod
    def _skip_reason(context: ModerationContext, config: BotConfig) -> str | None:
        if not config.enabled:
            return "disabled"
        if not context.text.strip():
            return "empty-message"
        if context.channel_id in config.ignored_channel_ids:
            return "ignored-channel"
        if context.user_id in config.exempt_user_ids:
            return "exempt-user"
        if config.ignored_role_ids.intersection(context.user_role_ids or set()):
            return "ignored-role"
        return None


def plan_for_result(result: ModerateResponse, config: BotConfig) -> ModerationPlan:
    if result.verdict == VerdictLabel.UNSAFE:
        return ModerationPlan(
            action=ModerationAction.TIMEOUT if config.timeout_unsafe else ModerationAction.DELETE,
            delete_message=config.delete_unsafe,
            timeout_user=config.timeout_unsafe,
            log_to_mods=True,
            notify_user=config.dm_users,
            public_warning=config.public_warning,
            reason=", ".join(result.categories) or result.matched_policy_rule or "policy violation",
        )
    if result.verdict == VerdictLabel.CONTROVERSIAL:
        return ModerationPlan(
            action=ModerationAction.FLAG,
            add_reaction=config.react_controversial,
            log_to_mods=True,
            reason=", ".join(result.categories) or "needs moderator review",
        )
    return ModerationPlan(action=ModerationAction.ALLOW)


def case_id(context: ModerationContext) -> str:
    return f"{context.platform.value}-{context.workspace_id}-{context.message_id}"


def needs_review(evaluation: ModerationEvaluation) -> bool:
    """Whether an evaluation should be surfaced to a human moderator.

    True for a CONTROVERSIAL verdict (the cascade was unsure) or any plan that
    flags / queues the message for review. Safe verdicts (ALLOW) and clear-unsafe
    verdicts (auto DELETE/TIMEOUT) do not need a human and return False.
    """
    if evaluation.result is None:
        return False
    if evaluation.result.verdict == VerdictLabel.CONTROVERSIAL:
        return True
    return evaluation.plan.action in _REVIEW_ACTIONS


__all__ = [
    "ModerationContext",
    "ModerationEngine",
    "ModerationEvaluation",
    "ModerationPlan",
    "case_id",
    "needs_review",
    "plan_for_result",
]
