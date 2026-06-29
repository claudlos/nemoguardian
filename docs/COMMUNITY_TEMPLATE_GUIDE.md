# Community Template Guide

`nemoguardian` is a self-hosted moderation template for communities and
organizations. It is not a hosted business service by default. Clone it, connect
your own Discord/Twitch/webhook surfaces, choose your own GPU/API providers, and
own your audit data.

## Who This Is For

- Discord servers that want cheap first-pass moderation with human review.
- Open-source communities that need transparent policy and audit logs.
- Small organizations that can rent a GPU only when moderation volume warrants it.
- Hackathon/demo teams that need a full working moderation stack, not just a
  model download.

## Recommended Rollout

1. Start in `shadow` enforcement for 24-72 hours.
2. Review `/nemoguardian dry_run_cases`, `/nemoguardian failures`, and
   `/nemoguardian stats`.
3. Move to `review` if moderators want log-only triage without deletes.
4. Move to `balanced` when unsafe messages should be deleted.
5. Move selected high-risk channels to `strict` only after false positives are
   understood.

## Enforcement Presets

| Preset | Deletes unsafe | Deletes controversial | Timeouts | Best use |
|---|---:|---:|---:|---|
| `shadow` | no | no | no | Silent trial and evidence capture |
| `review` | no | no | no | Human moderator queue |
| `balanced` | yes | no | no | Default community moderation |
| `strict` | yes | high-confidence only | high-confidence unsafe | High-risk public channels |

Tune the preset in Discord:

```text
/nemoguardian actions preset:shadow
/nemoguardian actions preset:balanced
/nemoguardian actions unsafe_delete_threshold:0.80 controversial_delete_threshold:0.90
```

The same behavior is testable offline:

```bash
make replay-transcript REPLAY_FLAGS="\
  --scenario-json examples/scenarios/discord_community_smoke.json \
  --base-url http://localhost:8000 \
  --mode standard \
  --enforcement-preset balanced"
```

## Minimum Ownership Checklist

- Create a private Discord test server before inviting the bot to a real server.
- Keep bot tokens, provider API keys, and audit logs outside the public repo.
- Use a mod-log channel visible only to trusted moderators.
- Keep `Message Content` intent enabled for Discord passive moderation.
- Add ignore/exempt scopes for moderator-only and bot-only channels.
- Capture replay evidence before and after changing model, threshold, or policy
  settings.
- Destroy rented GPU boxes as soon as the run is complete.

## GitHub Pages

This repo can serve a static docs page from `docs/index.html` using GitHub Pages:

1. Open the repository on GitHub.
2. Go to Settings -> Pages.
3. Set source to `Deploy from a branch`.
4. Pick branch `main`, folder `/docs`.
5. Save.

That gives the project a free public documentation page without paying for
hosting. For the public Nemoguardian repo, the expected URL is:

```text
https://claudlos.github.io/nemoguardian/
```

The page should stay documentation-focused: what the template does, how to
self-host it, how to verify it, what GPU credits are, and how support/refunds
work. `docs/index.html`, `docs/terms.html`, `docs/privacy.html`, and
`docs/refund.html` are intentionally public so they can satisfy a Stripe
website review without running a separate paid host.

If you fork this for a real organization, replace the business name, support
link, refund language, and contact path with your own public information before
asking Stripe for live processing.

## What Users Bring

- Their own community policy.
- Their own Discord/Twitch/webhook credentials.
- Their own GPU provider account or local GPU.
- Their own NVIDIA/OpenRouter key if they want deep-mode triage.
- Their own decision on when AI moderation should delete versus only flag.

## What The Template Gives Them

- Multi-model cascade with fail-safe behavior.
- Discord bot commands and action planning.
- Append-only redacted audit records.
- Replayable transcript evidence.
- 3090 runbooks and preflight checks.
- Benchmarks and smoke tests for future model changes.
