#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${NEMOGUARDIAN_ENV_DIR:-$HOME/.config/nemoguardian}"
ENV_FILE="${NEMOGUARDIAN_DISCORD_ENV_FILE:-$ENV_DIR/discord-live.env}"

fail() {
  printf 'error: %s\n' "$1" >&2
  exit 1
}

validate_token() {
  local label="$1"
  local value="$2"
  local dots

  [[ -n "$value" ]] || fail "$label token is empty"
  [[ "$value" =~ ^[A-Za-z0-9._-]+$ ]] || fail "$label token has invalid characters"
  dots=$(printf '%s' "$value" | tr -cd . | wc -c | tr -d ' ')
  [[ "$dots" == "2" ]] || fail "$label token should contain exactly two dots"
}

validate_snowflake() {
  local label="$1"
  local value="$2"

  [[ "$value" =~ ^[0-9]{17,20}$ ]] || fail "$label must be the raw Discord Copy ID value, 17-20 digits only"
}

printf 'This writes %s\n' "$ENV_FILE"
printf 'Tokens are read silently and are not printed.\n\n'

read -rsp "Moderator token: " DISCORD_BOT_TOKEN
printf '\n'
read -rsp "Sender token: " DISCORD_TEST_SENDER_TOKEN
printf '\n'
read -rp "Server ID: " DISCORD_GUILD_ID
read -rp "Test channel ID: " DISCORD_TEST_CHANNEL_ID
read -rp "Mod log channel ID: " DISCORD_MOD_LOG_CHANNEL_ID

validate_token "Moderator" "$DISCORD_BOT_TOKEN"
validate_token "Sender" "$DISCORD_TEST_SENDER_TOKEN"
validate_snowflake "Server ID" "$DISCORD_GUILD_ID"
validate_snowflake "Test channel ID" "$DISCORD_TEST_CHANNEL_ID"
validate_snowflake "Mod log channel ID" "$DISCORD_MOD_LOG_CHANNEL_ID"

mkdir -p "$ENV_DIR"
chmod 700 "$ENV_DIR"

cat > "$ENV_FILE" <<EOF
export DISCORD_BOT_TOKEN="$DISCORD_BOT_TOKEN"
export DISCORD_TEST_SENDER_TOKEN="$DISCORD_TEST_SENDER_TOKEN"
export DISCORD_GUILD_ID="$DISCORD_GUILD_ID"
export DISCORD_TEST_CHANNEL_ID="$DISCORD_TEST_CHANNEL_ID"
export DISCORD_MOD_LOG_CHANNEL_ID="$DISCORD_MOD_LOG_CHANNEL_ID"
export NEMOGUARDIAN_BOT_CONFIG_PATH="/tmp/nemoguardian_discord_live_config.json"
export NEMOGUARDIAN_BOT_AUDIT_PATH="/tmp/nemoguardian_discord_live_audit.jsonl"
EOF

chmod 600 "$ENV_FILE"

printf '\nWrote %s\n' "$ENV_FILE"
printf 'Next:\n'
printf '  source %s\n' "$ENV_FILE"
printf '  make discord-live-smoke DISCORD_LIVE_SMOKE_FLAGS="--mode fast"\n'
