"""Discord moderation bot adapter.

Run with:
    DISCORD_BOT_TOKEN=xxx python -m nemoguardian.adapters.discord

Optional:
    DISCORD_GUILD_ID=123  # sync slash commands to one guild during testing
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
from typing import Any

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
from nemoguardian.schemas import Mode

WARNING_REACTION = "\N{WARNING SIGN}\N{VARIATION SELECTOR-16}"


def make_handler(
    cascade: Cascade | None = None,
    *,
    config_store: ConfigStore | None = None,
    audit_log: AuditLog | None = None,
):
    """Build an async message handler that runs the Discord moderation flow."""
    engine = ModerationEngine(
        Platform.DISCORD,
        cascade=cascade,
        config_store=config_store,
        audit_log=audit_log,
    )

    async def on_message(message) -> None:
        if getattr(message.author, "bot", False):
            return
        guild = getattr(message, "guild", None)
        if guild is None:
            return

        config = engine.config_for(str(guild.id))
        context = _context_from_message(message)
        evaluation = await asyncio.to_thread(engine.evaluate, context, config)
        if evaluation.skipped:
            return

        status, error = await apply_discord_actions(message, evaluation)
        engine.record(evaluation, execution_status=status, error=error)

    return on_message


async def apply_discord_actions(
    message: Any,
    evaluation: ModerationEvaluation,
) -> tuple[str, str | None]:
    """Apply a moderation plan to a Discord message.

    Returns an execution status string and optional error. Kept separate from
    the real discord.py client so tests can exercise the action flow with fakes.
    """
    config = evaluation.config
    plan = evaluation.plan
    if plan.action == ModerationAction.ALLOW:
        return "allowed", None

    applied: list[str] = []
    errors: list[str] = []

    if config.dry_run:
        await _send_mod_log(message, evaluation, applied=["dry-run"], errors=[])
        return "dry-run", None

    if plan.add_reaction:
        try:
            await message.add_reaction(WARNING_REACTION)
            applied.append("reaction")
        except Exception as exc:
            errors.append(f"reaction:{type(exc).__name__}")

    if plan.delete_message:
        try:
            await message.delete()
            applied.append("delete")
        except Exception as exc:
            errors.append(f"delete:{type(exc).__name__}")

    if plan.timeout_user:
        timeout = getattr(message.author, "timeout", None)
        if timeout is None:
            errors.append("timeout:unsupported-author")
        else:
            try:
                until = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=config.timeout_seconds)
                await timeout(until, reason=_reason(evaluation))
                applied.append("timeout")
            except Exception as exc:
                errors.append(f"timeout:{type(exc).__name__}")

    if plan.public_warning:
        try:
            await message.channel.send(
                f"{WARNING_REACTION} {message.author.mention}, that message was blocked by "
                f"nemoguardian: {_reason(evaluation)}"
            )
            applied.append("public-warning")
        except Exception as exc:
            errors.append(f"public-warning:{type(exc).__name__}")

    if plan.notify_user:
        send = getattr(message.author, "send", None)
        if send is not None:
            try:
                await send(f"Your message in {message.guild.name} was moderated: {_reason(evaluation)}")
                applied.append("dm")
            except Exception as exc:
                errors.append(f"dm:{type(exc).__name__}")

    await _send_mod_log(message, evaluation, applied=applied, errors=errors)

    if errors:
        return "partial" if applied else "failed", ";".join(errors)
    return "+".join(applied) if applied else "planned", None


def run_bot() -> None:
    """Entry point: start the Discord moderation bot."""
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN env var required")

    bot = build_bot()
    bot.run(token)


def build_bot():
    """Create a discord.py commands.Bot with slash commands registered."""
    import discord
    from discord import app_commands
    from discord.ext import commands

    globals()["discord"] = discord
    globals()["app_commands"] = app_commands

    config_store = ConfigStore()
    audit_log = AuditLog()
    handler = make_handler(config_store=config_store, audit_log=audit_log)

    intents = discord.Intents.default()
    intents.guilds = True
    intents.message_content = True

    class NemoguardianDiscordBot(commands.Bot):
        async def setup_hook(self) -> None:
            guild_id = os.environ.get("DISCORD_GUILD_ID")
            if guild_id:
                guild = discord.Object(id=int(guild_id))
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
            else:
                await self.tree.sync()

    bot = NemoguardianDiscordBot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready() -> None:
        print(f"[nemoguardian] discord bot ready as {bot.user}")

    @bot.event
    async def on_message(message) -> None:
        await handler(message)

    group = app_commands.Group(
        name="nemoguardian",
        description="Configure nemoguardian moderation for this server.",
        default_permissions=discord.Permissions(manage_guild=True),
    )

    @group.command(name="setup", description="Enable moderation and set the mod-log channel.")
    @app_commands.default_permissions(manage_guild=True)
    async def setup(interaction, log_channel: discord.TextChannel | None = None) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.get(Platform.DISCORD, str(interaction.guild_id))
        config.enabled = True
        if log_channel is not None:
            config.log_channel_id = str(log_channel.id)
        config_store.save(config)
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="status", description="Show the current moderation configuration.")
    @app_commands.default_permissions(manage_guild=True)
    async def status(interaction) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.get(Platform.DISCORD, str(interaction.guild_id))
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="doctor", description="Check bot permissions, intents, and setup readiness.")
    @app_commands.default_permissions(manage_guild=True)
    async def doctor(interaction) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.get(Platform.DISCORD, str(interaction.guild_id))
        text = _doctor_text(
            config,
            getattr(interaction, "app_permissions", None),
            message_content_enabled=bool(getattr(bot.intents, "message_content", False)),
        )
        await interaction.response.send_message(text, ephemeral=True)

    @group.command(name="mode", description="Set moderation mode: fast, standard, or deep.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="fast", value="fast"),
            app_commands.Choice(name="standard", value="standard"),
            app_commands.Choice(name="deep", value="deep"),
        ]
    )
    async def mode(interaction, mode: app_commands.Choice[str]) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.update(Platform.DISCORD, str(interaction.guild_id), mode=Mode(mode.value))
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="policy", description="Set a plain-English server policy.")
    @app_commands.default_permissions(manage_guild=True)
    async def policy(interaction, text: str) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.update(Platform.DISCORD, str(interaction.guild_id), policy_text=text)
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="log_channel", description="Set where moderation decisions are logged.")
    @app_commands.default_permissions(manage_guild=True)
    async def log_channel(interaction, channel: discord.TextChannel) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.update(
            Platform.DISCORD,
            str(interaction.guild_id),
            log_channel_id=str(channel.id),
        )
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="dry_run", description="Turn dry-run mode on or off.")
    @app_commands.default_permissions(manage_guild=True)
    async def dry_run(interaction, enabled: bool) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.update(Platform.DISCORD, str(interaction.guild_id), dry_run=enabled)
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="enabled", description="Turn passive moderation on or off.")
    @app_commands.default_permissions(manage_guild=True)
    async def set_enabled(interaction, enabled: bool) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.update(Platform.DISCORD, str(interaction.guild_id), enabled=enabled)
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="actions", description="Configure moderation action behavior.")
    @app_commands.default_permissions(manage_guild=True)
    async def actions(
        interaction,
        delete_unsafe: bool | None = None,
        public_warning: bool | None = None,
        react_controversial: bool | None = None,
        dm_users: bool | None = None,
    ) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.get(Platform.DISCORD, str(interaction.guild_id))
        _apply_action_options(
            config,
            delete_unsafe=delete_unsafe,
            public_warning=public_warning,
            react_controversial=react_controversial,
            dm_users=dm_users,
        )
        config_store.save(config)
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="timeout", description="Configure unsafe-message timeout behavior.")
    @app_commands.default_permissions(manage_guild=True)
    async def timeout(interaction, enabled: bool, seconds: int = 600) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.get(Platform.DISCORD, str(interaction.guild_id))
        config.timeout_unsafe = enabled
        config.timeout_seconds = max(60, min(seconds, 2_419_200))
        config_store.save(config)
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="ignore_channel", description="Exclude or include a channel in moderation.")
    @app_commands.default_permissions(manage_guild=True)
    async def ignore_channel(interaction, channel: discord.TextChannel, ignored: bool = True) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.get(Platform.DISCORD, str(interaction.guild_id))
        _toggle_id(config.ignored_channel_ids, str(channel.id), enabled=ignored)
        config_store.save(config)
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="ignore_role", description="Exclude or include a role in moderation.")
    @app_commands.default_permissions(manage_guild=True)
    async def ignore_role(interaction, role: discord.Role, ignored: bool = True) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.get(Platform.DISCORD, str(interaction.guild_id))
        _toggle_id(config.ignored_role_ids, str(role.id), enabled=ignored)
        config_store.save(config)
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="exempt_user", description="Exclude or include a user in moderation.")
    @app_commands.default_permissions(manage_guild=True)
    async def exempt_user(interaction, user: discord.Member, exempt: bool = True) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.get(Platform.DISCORD, str(interaction.guild_id))
        _toggle_id(config.exempt_user_ids, str(user.id), enabled=exempt)
        config_store.save(config)
        await interaction.response.send_message(_status_text(config), ephemeral=True)

    @group.command(name="case", description="Look up a moderation case by case ID.")
    @app_commands.default_permissions(manage_guild=True)
    async def case_lookup(interaction, case_id: str) -> None:
        if not await _require_manage_guild(interaction):
            return
        record = audit_log.find_case(case_id.strip())
        if record is not None and (
            record.get("platform") != Platform.DISCORD.value
            or str(record.get("workspace_id")) != str(interaction.guild_id)
        ):
            record = None
        await interaction.response.send_message(_case_text(record), ephemeral=True)

    @group.command(name="history", description="Show recent moderation cases for this server or user.")
    @app_commands.default_permissions(manage_guild=True)
    async def history(interaction, user: discord.Member | None = None, limit: int = 5) -> None:
        if not await _require_manage_guild(interaction):
            return
        safe_limit = max(1, min(limit, 10))
        records = audit_log.history(
            Platform.DISCORD,
            str(interaction.guild_id),
            user_id=str(user.id) if user is not None else None,
            limit=safe_limit,
        )
        await interaction.response.send_message(_history_text(records), ephemeral=True)

    @group.command(name="stats", description="Summarize recent moderation cases.")
    @app_commands.default_permissions(manage_guild=True)
    async def stats(interaction, user: discord.Member | None = None, limit: int = 100) -> None:
        if not await _require_manage_guild(interaction):
            return
        safe_limit = max(1, min(limit, 500))
        summary = audit_log.summary(
            Platform.DISCORD,
            str(interaction.guild_id),
            user_id=str(user.id) if user is not None else None,
            limit=safe_limit,
        )
        await interaction.response.send_message(_stats_text(summary), ephemeral=True)

    @group.command(name="offenders", description="Show users with the most recent moderation cases.")
    @app_commands.default_permissions(manage_guild=True)
    async def offenders(interaction, limit: int = 5, case_limit: int = 500) -> None:
        if not await _require_manage_guild(interaction):
            return
        safe_limit = max(1, min(limit, 10))
        safe_case_limit = max(1, min(case_limit, 1_000))
        rows = audit_log.top_users(
            Platform.DISCORD,
            str(interaction.guild_id),
            limit=safe_limit,
            case_limit=safe_case_limit,
        )
        await interaction.response.send_message(
            _offenders_text(rows, case_limit=safe_case_limit),
            ephemeral=True,
        )

    @group.command(name="test", description="Test a message against the current policy.")
    @app_commands.default_permissions(manage_guild=True)
    async def test(interaction, text: str) -> None:
        if not await _require_manage_guild(interaction):
            return
        config = config_store.get(Platform.DISCORD, str(interaction.guild_id))
        context = ModerationContext(
            platform=Platform.DISCORD,
            workspace_id=str(interaction.guild_id),
            channel_id=str(interaction.channel_id),
            message_id=f"slash-test-{interaction.id}",
            user_id=str(interaction.user.id),
            username=str(interaction.user),
            text=text,
            user_role_ids=_role_ids(interaction.user),
        )
        engine = ModerationEngine(
            Platform.DISCORD,
            config_store=config_store,
            audit_log=audit_log,
        )
        evaluation = await asyncio.to_thread(engine.evaluate, context, config)
        engine.record(evaluation, execution_status="slash-test")
        await interaction.response.send_message(_test_text(evaluation), ephemeral=True)

    bot.tree.add_command(group)
    return bot


def _context_from_message(message: Any) -> ModerationContext:
    guild_id = str(message.guild.id)
    channel_id = str(message.channel.id)
    return ModerationContext(
        platform=Platform.DISCORD,
        workspace_id=guild_id,
        channel_id=channel_id,
        message_id=str(message.id),
        user_id=str(message.author.id),
        username=str(message.author),
        text=message.content or "",
        user_role_ids=_role_ids(message.author),
        permalink=getattr(message, "jump_url", None),
    )


async def _send_mod_log(
    message: Any,
    evaluation: ModerationEvaluation,
    *,
    applied: list[str],
    errors: list[str],
) -> None:
    channel_id = evaluation.config.log_channel_id
    if not channel_id:
        return
    channel = _find_channel(message, channel_id)
    if channel is None:
        return
    await channel.send(_mod_log_text(message, evaluation, applied=applied, errors=errors))


def _find_channel(message: Any, channel_id: str):
    guild = getattr(message, "guild", None)
    if guild is not None:
        get_channel = getattr(guild, "get_channel", None)
        if get_channel is not None:
            found = get_channel(int(channel_id))
            if found is not None:
                return found
    client = getattr(message, "_state", None)
    client = getattr(client, "_get_client", lambda: None)()
    get_channel = getattr(client, "get_channel", None)
    return get_channel(int(channel_id)) if get_channel is not None else None


def _mod_log_text(
    message: Any,
    evaluation: ModerationEvaluation,
    *,
    applied: list[str],
    errors: list[str],
) -> str:
    result = evaluation.result
    if result is None:
        return "nemoguardian skipped a message."
    category_text = ", ".join(result.categories) or "none"
    applied_text = ", ".join(applied) or "none"
    error_text = ", ".join(errors) or "none"
    return (
        "**nemoguardian moderation**\n"
        f"case: `{evaluation.context.platform.value}-{evaluation.context.workspace_id}-{evaluation.context.message_id}`\n"
        f"user: {message.author.mention} (`{message.author.id}`)\n"
        f"channel: <#{message.channel.id}>\n"
        f"verdict: `{result.verdict.value}` score: `{result.score:.2f}` mode: `{result.mode.value}`\n"
        f"action: `{evaluation.plan.action.value}` applied: `{applied_text}` errors: `{error_text}`\n"
        f"categories: `{category_text}`\n"
        f"rule: `{result.matched_policy_rule or 'none'}` request: `{result.request_id or 'none'}`\n"
        f"message: {redacted_excerpt(evaluation.context.text)}"
    )


async def _require_manage_guild(interaction) -> bool:
    permissions = getattr(interaction.user, "guild_permissions", None)
    if getattr(permissions, "manage_guild", False):
        return True
    await interaction.response.send_message(
        "You need Manage Server permission to configure nemoguardian.",
        ephemeral=True,
    )
    return False


def _role_ids(user: Any) -> set[str]:
    return {str(role.id) for role in getattr(user, "roles", []) if hasattr(role, "id")}


def _reason(evaluation: ModerationEvaluation) -> str:
    return evaluation.plan.reason or "policy violation"


def _status_text(config: BotConfig) -> str:
    return (
        "**nemoguardian status**\n"
        f"enabled: `{config.enabled}`\n"
        f"mode: `{config.mode.value}`\n"
        f"policy preset: `{config.policy_preset}`\n"
        f"log channel: `{config.log_channel_id or 'not set'}`\n"
        f"dry run: `{config.dry_run}`\n"
        f"delete unsafe: `{config.delete_unsafe}` timeout unsafe: `{config.timeout_unsafe}` "
        f"({config.timeout_seconds}s)\n"
        f"public warning: `{config.public_warning}` react controversial: `{config.react_controversial}` "
        f"dm users: `{config.dm_users}`\n"
        f"ignored channels: `{_format_ids(config.ignored_channel_ids)}`\n"
        f"ignored roles: `{_format_ids(config.ignored_role_ids)}`\n"
        f"exempt users: `{_format_ids(config.exempt_user_ids)}`\n"
        f"policy: {config.policy_text}"
    )


def _apply_action_options(
    config: BotConfig,
    *,
    delete_unsafe: bool | None = None,
    public_warning: bool | None = None,
    react_controversial: bool | None = None,
    dm_users: bool | None = None,
) -> BotConfig:
    if delete_unsafe is not None:
        config.delete_unsafe = delete_unsafe
    if public_warning is not None:
        config.public_warning = public_warning
    if react_controversial is not None:
        config.react_controversial = react_controversial
    if dm_users is not None:
        config.dm_users = dm_users
    return config


def _toggle_id(values: set[str], value: str, *, enabled: bool) -> None:
    if enabled:
        values.add(value)
    else:
        values.discard(value)


def _format_ids(values: set[str]) -> str:
    return ", ".join(sorted(values)) if values else "none"


def _doctor_text(config: BotConfig, permissions: Any, *, message_content_enabled: bool) -> str:
    required = {
        "view_channel": "View Channel",
        "read_message_history": "Read Message History",
        "send_messages": "Send Messages",
        "manage_messages": "Manage Messages",
    }
    recommended = {"embed_links": "Embed Links"}
    if config.timeout_unsafe:
        required["moderate_members"] = "Moderate Members"
    else:
        recommended["moderate_members"] = "Moderate Members"

    missing_required = [
        label for name, label in required.items() if not _permission_enabled(permissions, name)
    ]
    missing_recommended = [
        label for name, label in recommended.items() if not _permission_enabled(permissions, name)
    ]

    issues = []
    if not config.enabled:
        issues.append("moderation is disabled")
    if not config.log_channel_id:
        issues.append("mod-log channel is not set")
    if not message_content_enabled:
        issues.append("Message Content intent is not requested")
    if missing_required:
        issues.append(f"missing required permissions: {', '.join(missing_required)}")

    readiness = "ready" if not issues else "needs attention"
    return (
        "**nemoguardian doctor**\n"
        f"readiness: `{readiness}`\n"
        f"enabled: `{config.enabled}` mode: `{config.mode.value}` dry run: `{config.dry_run}`\n"
        f"log channel: `{config.log_channel_id or 'not set'}`\n"
        f"message content intent requested: `{message_content_enabled}`\n"
        f"missing required permissions: `{', '.join(missing_required) or 'none'}`\n"
        f"missing recommended permissions: `{', '.join(missing_recommended) or 'none'}`\n"
        f"issues: `{'; '.join(issues) or 'none'}`"
    )


def _permission_enabled(permissions: Any, name: str) -> bool:
    return bool(getattr(permissions, name, False))


def _test_text(evaluation: ModerationEvaluation) -> str:
    if evaluation.result is None:
        return f"Skipped: `{evaluation.skip_reason}`"
    return (
        "**nemoguardian test result**\n"
        f"verdict: `{evaluation.result.verdict.value}` score: `{evaluation.result.score:.2f}`\n"
        f"planned action: `{evaluation.plan.action.value}` reason: `{evaluation.plan.reason}`\n"
        f"categories: `{', '.join(evaluation.result.categories) or 'none'}`"
    )


def _case_text(record: dict[str, Any] | None) -> str:
    if record is None:
        return "Case not found."

    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    permalink = details.get("permalink") if details else None
    lines = [
        "**nemoguardian case**",
        f"case: `{record.get('case_id', 'unknown')}`",
        (
            f"user: `{record.get('username', 'unknown')}` (`{record.get('user_id', 'unknown')}`) "
            f"channel: <#{record.get('channel_id', 'unknown')}> "
            f"message: `{record.get('message_id', 'unknown')}`"
        ),
        (
            f"verdict: `{record.get('verdict', 'unknown')}` "
            f"score: `{_format_score(record.get('score'))}` "
            f"mode: `{record.get('mode', 'unknown')}`"
        ),
        (
            f"action: `{record.get('action', 'unknown')}` "
            f"status: `{record.get('execution_status', 'unknown')}` "
            f"dry run: `{record.get('dry_run', False)}`"
        ),
        (
            f"categories: `{', '.join(record.get('categories') or []) or 'none'}` "
            f"rule: `{record.get('matched_policy_rule') or 'none'}` "
            f"request: `{record.get('request_id') or 'none'}`"
        ),
        f"created: `{record.get('created_at', 'unknown')}`",
    ]
    if record.get("error"):
        lines.append(f"error: `{record['error']}`")
    if permalink:
        lines.append(f"link: {permalink}")
    if record.get("text_excerpt"):
        lines.append(f"excerpt: {str(record['text_excerpt'])[:300]}")
    return "\n".join(lines)


def _history_text(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No moderation history found."

    lines = ["**nemoguardian history**"]
    for record in records[:10]:
        lines.append(
            f"`{record.get('case_id', 'unknown')}` "
            f"{record.get('action', 'unknown')}/{record.get('verdict', 'unknown')} "
            f"score `{_format_score(record.get('score'))}` "
            f"user `{record.get('user_id', 'unknown')}` "
            f"channel <#{record.get('channel_id', 'unknown')}> "
            f"status `{record.get('execution_status', 'unknown')}`"
        )
    return "\n".join(lines)


def _stats_text(summary: dict[str, Any]) -> str:
    if int(summary.get("total") or 0) <= 0:
        return "**nemoguardian stats**\nNo moderation cases found."

    user_scope = f" user `{summary['user_id']}`" if summary.get("user_id") else ""
    return (
        "**nemoguardian stats**\n"
        f"scope: `{summary.get('platform', 'unknown')}:{summary.get('workspace_id', 'unknown')}`"
        f"{user_scope} last `{summary.get('limit', 0)}` cases\n"
        f"total cases: `{summary.get('total', 0)}` dry run: `{summary.get('dry_run', 0)}` "
        f"errors: `{summary.get('errors', 0)}`\n"
        f"verdicts: `{_format_counts(summary.get('verdicts') or {})}`\n"
        f"actions: `{_format_counts(summary.get('actions') or {})}`\n"
        f"statuses: `{_format_counts(summary.get('statuses') or {})}`\n"
        f"categories: `{_format_counts(summary.get('categories') or {})}`\n"
        f"newest: `{summary.get('newest_case_id') or 'none'}` "
        f"oldest: `{summary.get('oldest_case_id') or 'none'}`"
    )


def _format_counts(counts: dict[str, int], limit: int = 8) -> str:
    if not counts:
        return "none"
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))[:limit]
    return ", ".join(f"{name}:{count}" for name, count in items)


def _offenders_text(rows: list[dict[str, Any]], *, case_limit: int) -> str:
    if not rows:
        return "**nemoguardian offenders**\nNo moderated users found."

    lines = [f"**nemoguardian offenders**\nlast `{case_limit}` cases"]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index}. user `{row.get('username', 'unknown')}` (`{row.get('user_id', 'unknown')}`) "
            f"cases `{row.get('total', 0)}` unsafe `{row.get('unsafe', 0)}` "
            f"controversial `{row.get('controversial', 0)}` "
            f"actions `{_format_counts(row.get('actions') or {}, limit=3)}` "
            f"latest `{row.get('latest_case_id') or 'none'}`"
        )
    return "\n".join(lines)


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "unknown"


if __name__ == "__main__":
    run_bot()


__all__ = [
    "WARNING_REACTION",
    "apply_discord_actions",
    "build_bot",
    "make_handler",
    "run_bot",
]
