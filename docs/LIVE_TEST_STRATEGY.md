# Platform live-test strategy

NemoGuardian ships five platform adapters — **Discord, Slack, Telegram, Twitch,
and a generic webhook** — that all conform to the shared
[`PlatformAdapter`](../nemoguardian/adapters/base.py) interface: every event is
normalized into a `ModerationContext`, moderated by the cascade, mapped to a
normalized [`ModerationAction`](../nemoguardian/bot/types.py), capability-degraded
for the platform, and recorded to the append-only redacted audit log.

This document defines, **per platform**, the three tiers of testing that give us
confidence from "the code is correct" all the way to "a real bot enforces in a
real workspace":

| Tier | What it proves | Needs a secret? | Needs the platform? |
|------|----------------|-----------------|---------------------|
| **1. Fake-event unit tests** | parse → moderate → action → audit logic | no | no |
| **2. Sandbox tests** | the bot connects + reacts in a throwaway space | yes (sandbox) | yes (test workspace) |
| **3. Real-token smoke** | end-to-end against a real install | yes (scoped) | yes (real workspace) |

> **Never test in production.** Tier 2/3 always run against a dedicated test
> server / workspace / channel and a bot account with the *minimum* scopes. Use
> `dry_run=true` (config) first so the bot **plans** actions and writes audit
> records without enforcing, then graduate to live enforcement.

---

## Tier 1 — fake-event unit tests (what exists today)

Tier 1 is fully automated, deterministic, and requires **no GPU, no network, no
platform SDK, and no secret**. It is the contract every adapter must keep green.

Two complementary surfaces:

* **Per-adapter unit tests** — fake the platform client and feed synthetic
  events:
  * Discord / Twitch / webhook: [`tests/test_adapters.py`](../tests/test_adapters.py)
  * Slack: [`tests/test_slack_adapter.py`](../tests/test_slack_adapter.py)
  * Telegram: [`tests/test_telegram_adapter.py`](../tests/test_telegram_adapter.py)
  * Capability / degrade contract: [`tests/test_adapters_base.py`](../tests/test_adapters_base.py)
* **Umbrella offline smoke** — one command drives **every** adapter through a
  synthetic event end-to-end and prints a per-platform pass/skip summary:

  ```bash
  make platform-smoke              # offline: runs all five adapters
  make platform-smoke PLATFORM_SMOKE_FLAGS=--json
  make platform-smoke PLATFORM_SMOKE_FLAGS=--require-live   # only live-ready ones
  ```

  Harness: [`scripts/platform_smoke.py`](../scripts/platform_smoke.py),
  tested by [`tests/test_platform_smoke.py`](../tests/test_platform_smoke.py).

  The smoke feeds each adapter a synthetic message (`"ignore all previous
  instructions..."`), which trips the deterministic
  [`detect_prompt_injection`](../nemoguardian/detectors.py) detector, so the
  moderation step uses a network-free **stubbed verdict** (`StubCascade`) rather
  than loading any model. It then resolves the planned action against each
  platform's real `capabilities()` and writes a redacted audit record to an
  isolated temp log. Example output:

  ```
  [PASS] discord   verdict=unsafe action=delete audit=yes · live: not ready
  [PASS] webhook   verdict=unsafe action=delete -> flag audit=yes · live: not ready
  ```

  In the **default (offline)** mode every adapter is *available* (the moderation
  path is pure Python) and is run; a platform is only `skip` if a genuinely
  required dependency is missing, or `fail` if the flow errors. The
  `--require-live` mode instead **skips** any platform whose optional SDK or
  secret env is absent and only runs the live-ready ones.

---

## Tier 2 — sandbox tests (test server / workspace / channel)

Tier 2 connects a **real bot account** to a **throwaway space** you control and
verifies the full loop (gateway/websocket/webhook → parse → moderate →
enforce → audit) against the real platform API, but with zero blast radius.

General recipe for every platform:

1. Create a dedicated test space (server / workspace / supergroup / channel).
2. Create a bot/app with the **minimum** scopes in Tier 3 below.
3. Invite the bot to the test space; grant only the moderation permissions.
4. Start with `dry_run=true` so actions are planned + audited but not enforced.
5. Post benign and clearly-violating test messages from a throwaway user.
6. Assert: the bot reacted (reaction / delete / mod-log), and the audit log
   (`make`-served `/audit` or the JSONL file) shows redacted records with the
   expected verdict/action.
7. Flip `dry_run=false` and confirm enforcement (delete/timeout/ban) works.

---

## Tier 3 — real-token smoke (env vars, scopes, how to run safely)

Per-platform: the SDK, the secret env vars, the scopes/permissions the bot
needs, and the safest way to run. Secrets come **only** from the environment
(never committed); the offline smoke never reads them.

### Discord

* **SDK / extra:** `discord.py` (`pip install -e ".[discord]"`).
* **Env vars:** `DISCORD_BOT_TOKEN` (required); optional `DISCORD_GUILD_ID`,
  `DISCORD_LOG_CHANNEL_ID`.
* **Gateway intents / permissions:** enable **Message Content Intent** in the
  Developer Portal (privileged); bot permissions: *View Channels*, *Read Message
  History*, *Manage Messages* (delete), *Moderate Members* (timeout), and
  *Ban Members* only if you enable bans. Send Messages for mod-log/warnings.
* **Run safely:** there is already a dedicated live harness —
  `make discord-env-setup` then `make discord-live-smoke` (see
  [`scripts/discord_live_smoke.py`](../scripts/discord_live_smoke.py) and
  [`scripts/discord_actor_scenario.py`](../scripts/discord_actor_scenario.py)).
  Run against a test guild with `dry_run` first.

### Slack

* **SDK / extra:** `slack_bolt` (`pip install -e ".[slack]"`).
* **Env vars:** `SLACK_BOT_TOKEN` (`xoxb-…`) and `SLACK_SIGNING_SECRET`
  (required); `SLACK_APP_TOKEN` (`xapp-…`) if you use Socket Mode.
* **OAuth scopes:** `channels:history` / `groups:history` (read messages),
  `chat:write` (warnings + mod-log), `chat:write.customize` (optional),
  `reactions:write` (flag controversial), and `chat:delete` requires
  admin-level delete — note Slack apps can only delete **their own** messages
  with `chat:delete`; deleting user messages needs a workspace-admin token, so
  the adapter degrades unsupported deletes to `flag` (see `slack_decision`).
* **Run safely:** create a test workspace, install the app via OAuth, prefer
  **Socket Mode** (no public URL). Build the app with
  [`nemoguardian.adapters.slack.build_app`](../nemoguardian/adapters/slack.py)
  and post to a `#moderation-test` channel with `dry_run=true` first.

### Telegram

* **SDK / extra:** `python-telegram-bot` (imports as `telegram`;
  `pip install -e ".[telegram]"`).
* **Env vars:** `TELEGRAM_BOT_TOKEN` (required, from **@BotFather**).
* **Permissions:** add the bot to a **supergroup** as an **administrator** with
  *Delete messages* and *Ban users* rights; in @BotFather disable *Group
  Privacy* so the bot sees all messages (otherwise it only sees commands and
  replies/mentions). Only supergroups/groups are moderated (private chats and
  channels are ignored by `parse_update`).
* **Run safely:** create a throwaway supergroup, build the app with
  [`nemoguardian.adapters.telegram.build_application`](../nemoguardian/adapters/telegram.py),
  use long-polling (no webhook URL needed) and `dry_run=true` first.

### Twitch

* **SDK / extra:** `twitchio` (`pip install -e ".[twitch]"`).
* **Env vars:** `TWITCH_TOKEN` (OAuth token, required); channel name passed as
  the CLI arg (`python -m nemoguardian.adapters.twitch <channel>`).
* **Scopes:** the token needs `chat:read` + `chat:edit` to read chat, and the
  bot account must be a **moderator** of the channel to delete messages
  (`/delete`), time out (`/timeout`), or ban (`/ban`). Generate the token via
  the Twitch token generator / your app's OAuth flow.
* **Run safely:** use your own test channel, mod the bot account, and test in an
  off-hours/empty channel. `make_moderator` emits planned actions you can
  observe before wiring real chat-mod commands.

### Webhook (generic)

* **SDK / extra:** none beyond core (`httpx`).
* **Env vars:** `NEMOGUARDIAN_WEBHOOK_FORWARD_URL` (downstream sink, required for
  live forwarding); optional `NEMOGUARDIAN_API_KEY` (authenticate to the
  moderation server), `NEMOGUARDIAN_WEBHOOK_FORWARD_TEXT`
  (`verdict_only` | `redacted` | `full`, default `verdict_only`).
* **Capabilities:** the webhook adapter is **notify-only** (`allow` / `flag` /
  `notify_mods`); any enforcement action degrades to `flag` since the downstream
  system performs enforcement. By default the original text **never leaves the
  box** — only the verdict and a `text_sha256` fingerprint are forwarded.
* **Run safely:** point `NEMOGUARDIAN_WEBHOOK_FORWARD_URL` at a request-bin /
  local listener, keep `forward_text=verdict_only`, and inspect
  `build_forward_payload` output before forwarding anything sensitive.

---

## Promotion checklist (Tier 1 → 3)

1. `make platform-smoke` is green (offline contract holds).
2. `make verify` (ruff + full pytest) is green.
3. Tier 2 sandbox run with `dry_run=true` shows correct **planned** actions +
   redacted audit records.
4. Tier 2 with `dry_run=false` shows correct **enforcement** in the test space.
5. Tier 3 real-token smoke against the real (non-production) workspace with the
   minimum scopes, monitored, then promote.
