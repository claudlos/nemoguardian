# Moderation Bot Runbook

This is the product path for `nemoguardian`: an installable moderation operator
for communities, starting with Discord.

## Discord MVP

The Discord bot runs the same cascade as the API, but turns verdicts into real
moderation work:

- delete unsafe messages
- optionally timeout users
- react to controversial messages
- write a mod-log case
- append a private audit record
- expose slash commands for setup, policy, mode, dry-run, enable/disable, action
  behavior, timeout, ignore/exempt scopes, diagnostics, case lookup, history,
  stats, repeat-offender review, and test checks

## Discord App Setup

Create a Discord application and bot in the Developer Portal.

Required OAuth2 scopes:

- `bot`
- `applications.commands`

Recommended bot permissions:

- View Channel
- Read Message History
- Send Messages
- Embed Links
- Manage Messages
- Moderate Members

Required Gateway intents:

- Guilds
- Guild Messages
- Message Content

`Message Content` is privileged. Without it, passive message moderation will not
receive full message text. Discord AutoMod can block simple native patterns
before messages post; `nemoguardian` is the contextual second layer with
model-backed case evidence.

Useful references:

- Discord OAuth2 scopes: https://docs.discord.com/developers/topics/oauth2
- Discord permissions: https://docs.discord.com/developers/topics/permissions
- Discord privileged intents: https://discordpy.readthedocs.io/en/stable/intents.html
- discord.py slash commands: https://discordpy.readthedocs.io/en/stable/interactions/api.html

## Run Locally

Install Discord support:

```bash
pip install -e ".[discord]"
```

Set secrets and storage paths through the shell or `.env` tooling:

```bash
export DISCORD_BOT_TOKEN="<secret>"
export DISCORD_GUILD_ID="<test-guild-id>"  # optional, speeds command sync during testing
export NEMOGUARDIAN_BOT_CONFIG_PATH=/tmp/nemoguardian_bot_config.json
export NEMOGUARDIAN_BOT_AUDIT_PATH=/tmp/nemoguardian_bot_audit.jsonl
```

Run:

```bash
nemoguardian discord-bot
# or
python -m nemoguardian.adapters.discord
```

## Slash Commands

Use these in a test server first:

```text
/nemoguardian setup log_channel:#mod-log
/nemoguardian status
/nemoguardian doctor
/nemoguardian mode standard
/nemoguardian policy "block PII, scams, harassment, slurs, and threats"
/nemoguardian dry_run enabled:true
/nemoguardian enabled enabled:true
/nemoguardian actions delete_unsafe:true public_warning:true react_controversial:true dm_users:false
/nemoguardian timeout enabled:true seconds:600
/nemoguardian ignore_channel channel:#off-topic ignored:true
/nemoguardian ignore_role role:@mods ignored:true
/nemoguardian exempt_user user:@trusted-member exempt:true
/nemoguardian test text:"Hey @everyone, drop your SSN for $100"
/nemoguardian history limit:5
/nemoguardian stats limit:100
/nemoguardian offenders limit:5 case_limit:500
/nemoguardian case case_id:discord-<guild-id>-<message-id>
```

`dry_run` is the safest initial deployment mode. It writes mod logs and audit
records without deleting or timing out users.

Use `/nemoguardian enabled enabled:false` to pause passive moderation without
removing config. Use `/nemoguardian actions` to tune enforcement while rolling
out: for example, keep `delete_unsafe:false` and `public_warning:false` during a
silent audit, then turn deletion and warnings on after reviewing the case log.

Run `/nemoguardian doctor` before recording or inviting the bot to a customer
server. It checks the current guild config, mod-log channel, requested Message
Content intent, and the bot's effective channel permissions. `/nemoguardian
history`, `/nemoguardian case`, `/nemoguardian stats`, and `/nemoguardian
offenders` give moderators a quick way to inspect recent decisions, bot
workload, and repeat offenders without reading the JSONL audit file directly.

Use the ignore/exempt commands to keep moderation noise down in trusted or
irrelevant scopes:

- `/nemoguardian ignore_channel ... ignored:true` skips a channel
- `/nemoguardian ignore_role ... ignored:true` skips users with a role
- `/nemoguardian exempt_user ... exempt:true` skips one user

Pass `ignored:false` or `exempt:false` to remove an exclusion.

## Runtime Behavior

Default Discord policy:

- `safe` -> allow
- `controversial` -> add warning reaction and log
- `unsafe` -> delete, optionally timeout, optionally DM, public warning, and log

Configuration is per guild and file-backed by default. The JSON store is meant
for self-hosted deployment and can move to SQLite/Postgres later without
changing adapter behavior.

Audit records are append-only JSONL and include:

- platform, guild, channel, message, and user IDs
- action, verdict, score, categories, mode, policy rule
- request ID, latency, redacted text excerpt
- execution status and any action errors

Mod-log case text and audit excerpts redact common email, SSN, phone, and
payment-card patterns before storage or reposting. The raw message text is still
hashed with SHA-256 for evidence correlation without storing the original
sensitive string in the JSONL case record.

## Offline Audit Inspection

Use the CLI when the bot is running on a host and Discord is unavailable or you
need shell-friendly JSON output:

```bash
nemoguardian bot-audit history --workspace-id "$DISCORD_GUILD_ID" --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit stats --workspace-id "$DISCORD_GUILD_ID" --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit offenders --workspace-id "$DISCORD_GUILD_ID" --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
nemoguardian bot-audit case discord-<guild-id>-<message-id> --path "$NEMOGUARDIAN_BOT_AUDIT_PATH"
```

## Cross-Platform Foundation

The shared bot layer lives under `nemoguardian/bot/`:

- `types.py` - platform and action enums
- `config.py` - per-workspace bot configuration
- `audit.py` - append-only case/audit records
- `engine.py` - cascade call plus platform-neutral action planning

Discord is the flagship adapter. Twitch already uses the same engine and
defaults to `fast` mode. Generic webhooks keep the authenticated API-forwarding
path. Future Slack/Telegram adapters should normalize messages into the same
engine, then execute only the actions their platform supports.

## Product Positioning

The business value is not model hosting. It is reducing moderator labor:

- native AutoMod handles simple keyword and spam blocks
- `nemoguardian` handles context, policy, evidence, and escalation
- cheap GPU/API routing keeps cost predictable
- human moderators get case logs instead of raw chat chaos

The demo should show a community owner installing the bot, setting a mod-log
channel, running `/nemoguardian test`, and watching unsafe messages get handled
with auditable evidence.
