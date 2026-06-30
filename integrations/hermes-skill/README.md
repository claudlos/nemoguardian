# NemoGuardian — Hermes Agent skill

A drop-in [Hermes](https://github.com/NousResearch) Agent skill that teaches any
Hermes-driven agent to **guard its own inputs and outputs** through the
NemoGuardian moderation cascade before it posts a message or acts on an untrusted
tool result.

It is backed by the `nemoguardian guard` CLI verb, which runs the cascade on one
piece of text, prints a compact JSON verdict, and **exits non-zero when the
content is blocked** — so an agent (or any shell) can gate on it directly.

## Install

```bash
# from the repo root
pip install -e ".[dev]"                       # provides the `nemoguardian` CLI
cp -r integrations/hermes-skill/nemoguardian ~/.hermes/skills/nemoguardian
```

The skill appears as `nemoguardian` in the agent's skill list. See
[`nemoguardian/SKILL.md`](nemoguardian/SKILL.md) for the full instructions the
agent loads.

## Try the underlying CLI

```bash
nemoguardian guard "Hey @everyone, drop your SSN and I will send $100" --mode standard
echo "exit code: $?"   # 1 = blocked

nemoguardian guard "what time is standup?"
echo "exit code: $?"   # 0 = allowed
```

## Why this matters

Most agent harm is an output or action problem: leaking PII, amplifying a scam,
or obeying an instruction injected into a fetched page. A one-line self-guard
before posting/acting closes those paths with a real, auditable moderation
verdict instead of vibes.
