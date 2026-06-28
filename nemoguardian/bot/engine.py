"""Platform-neutral moderation bot engine."""

from __future__ import annotations

from dataclasses import dataclass

from nemoguardian.bot.audit import AuditLog, AuditRecord, redacted_excerpt, text_hash
from nemoguardian.bot.config import BotConfig, ConfigStore
from nemoguardian.bot.types import ModerationAction, Platform
from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.policy.presets import get_preset
from nemoguardian.schemas import ModerateRequest, ModerateResponse, VerdictLabel


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
    ) -> None:
        self.platform = Platform(platform)
        self.cascade = cascade or Cascade(CascadeConfig.from_env())
        self.config_store = config_store or ConfigStore()
        self.audit_log = audit_log or AuditLog()

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


__all__ = [
    "ModerationContext",
    "ModerationEngine",
    "ModerationEvaluation",
    "ModerationPlan",
    "case_id",
    "plan_for_result",
]
