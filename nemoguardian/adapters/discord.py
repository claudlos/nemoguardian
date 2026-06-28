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
        f"message: {evaluation.context.text[:500]}"
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
        f"policy: {config.policy_text}"
    )


def _test_text(evaluation: ModerationEvaluation) -> str:
    if evaluation.result is None:
        return f"Skipped: `{evaluation.skip_reason}`"
    return (
        "**nemoguardian test result**\n"
        f"verdict: `{evaluation.result.verdict.value}` score: `{evaluation.result.score:.2f}`\n"
        f"planned action: `{evaluation.plan.action.value}` reason: `{evaluation.plan.reason}`\n"
        f"categories: `{', '.join(evaluation.result.categories) or 'none'}`"
    )


if __name__ == "__main__":
    run_bot()


__all__ = [
    "WARNING_REACTION",
    "apply_discord_actions",
    "build_bot",
    "make_handler",
    "run_bot",
]
