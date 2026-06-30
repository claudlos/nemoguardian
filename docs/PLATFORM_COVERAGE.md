# Platform Coverage Matrix

`nemoguardian` is a **self-hostable moderation template**, not a hosted service.
Platform support is added one adapter at a time and lands at different maturity
levels. This page states honestly what each surface can do **in the code today**
so you can decide what to self-host without guessing.

The shared adapter foundation (`nemoguardian/adapters/base.py`) gives every
platform the same normalized action vocabulary
(`nemoguardian/bot/types.py::ModerationAction`) and a `degrade_action()` helper:
an action a platform cannot perform **degrades to `flag`** (surface for review)
with an auditable reason — enforcement is never silently dropped.

## Maturity legend

| Status | Meaning |
|---|---|
| **Full bot** | End-to-end: live ingest, enforced actions, doctor, dry-run, audit. |
| **Evaluate + planned actions** | Live ingest and per-message verdicts; actions are resolved/planned, enforcement is partial. |
| **Adapter skeleton** | Event parsing + action mapping + dry-run + audit are implemented and unit-tested, but there is no live admin surface wired in yet. |
| **Forward** | Verdict is computed and forwarded to a downstream target; this template does not enforce on the source platform. |
| **Planned** | Not implemented yet. |

## Matrix

Columns: **ingest** (can it receive/parse platform events) · **actions enforced**
(does it carry out moderation on the platform) · **doctor** (readiness/permission
check) · **dry-run** (decide-but-don't-act) · **audit** (append-only redacted
record) · **status**.

| Platform | ingest | actions enforced | doctor | dry-run | audit | status |
|---|---|---|---|---|---|---|
| **Discord** | yes (gateway) | yes — delete, timeout, notify-mods/user | yes | yes | yes | **Full bot** |
| **Twitch** | yes (live chat) | partial — delete/timeout/ban resolved via capabilities; enforcement wiring partial | yes | yes | yes | **Evaluate + planned actions** |
| **Slack** | yes (Events API parsing) | mapped — delete, notify-mods/user via injected client; no live admin surface yet | yes | yes | yes | **Adapter skeleton** |
| **Telegram** | yes (webhook / long-poll parsing) | mapped — delete, ban, mute, notify-mods via injected API; no live admin surface yet | yes | yes | yes | **Adapter skeleton** |
| **Webhook** | yes (HTTP POST) | no — forwards verdict (verdict-only by default); enforcement degrades to `flag` | yes | n/a | downstream (forwarded verdict is the record) | **Forward** |
| Matrix | — | — | — | — | — | Planned |
| Reddit | — | — | — | — | — | Planned |
| YouTube Live | — | — | — | — | — | Planned |
| Kick | — | — | — | — | — | Planned |
| Slack slash-commands | — | — | — | — | — | Planned |

## Capabilities per platform

These are the normalized actions each adapter declares via `capabilities()`.
Actions outside this set degrade to `flag` when a policy requests them.

| Platform | Declared capabilities |
|---|---|
| Discord | `allow`, `flag`, `delete`, `timeout`, `notify_mods`, `notify_user` |
| Twitch | `allow`, `flag`, `delete`, `timeout`, `ban` |
| Slack | `allow`, `flag`, `delete`, `notify_mods`, `notify_user` |
| Telegram | `allow`, `flag`, `delete`, `ban`, `mute`, `notify_mods` |
| Webhook | `allow`, `flag`, `notify_mods` |

## What "adapter skeleton" honestly means

For Slack and Telegram, the repo ships:

- **Event parsing** — `parse_slack_event` / Telegram update parsing turn raw
  platform payloads into a normalized message and skip non-moderatable events.
- **Action mapping** — `apply_slack_actions` / `apply_telegram_actions` translate
  a verdict into platform calls against an **injected** client/API object, so the
  flow is unit-testable without live credentials.
- **Dry-run** — the same decide-but-don't-act path the Discord bot uses.
- **Audit** — append-only, redacted JSONL through the shared engine.

What they do **not** yet ship is a turnkey, hosted admin surface (slash-command
setup, live gateway/bot process, permission bootstrapping) the way Discord does.
You can drive them programmatically today; treat the live deployment as
do-it-yourself until that surface lands.

## Honest caveats

- This is a **template you self-host**, not a fully production-ready hosted
  service. You own the GPU, the keys, and the audit data.
- Multi-platform coverage is **partial today** and growing; only Discord is a
  full bot. Do not assume parity across platforms.
- Rented-GPU automation is a convenience, not a guarantee — **you are
  responsible for GPU spend**. Set budgets and tear boxes down after a run.

See `docs/COMMUNITY_TEMPLATE_GUIDE.md` for rollout guidance and
`docs/BOT_RUNBOOK.md` for the Discord bot runbook.
