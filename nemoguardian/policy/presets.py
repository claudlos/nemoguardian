"""Built-in policy presets for common platforms."""

from __future__ import annotations

from nemoguardian.policy.nemoclaw import NemoclawPolicy


DISCORD_DEFAULT: dict = {
    "name": "discord-default",
    "description": "Conservative defaults: block PII, slurs, sexual content involving minors.",
    "rules": [
        {
            "id": "force-block-pii",
            "when": {"categories_include": ["PII"]},
            "then": {"final_label": "unsafe", "final_score": 1.0},
        },
        {
            "id": "force-block-self-harm",
            "when": {"categories_include": ["Suicide & Self-Harm"]},
            "then": {"final_label": "unsafe", "final_score": 1.0},
        },
        {
            "id": "force-block-violent",
            "when": {"categories_include": ["Violent"]},
            "then": {"final_label": "unsafe", "final_score": 1.0},
        },
    ],
}


TWITCH_CHAT: dict = {
    "name": "twitch-chat",
    "description": "Twitch live-chat defaults — fast block on slurs/PII, softer on controversial.",
    "rules": [
        {
            "id": "block-pii-tw",
            "when": {"categories_include": ["PII"]},
            "then": {"final_label": "unsafe", "final_score": 1.0},
        },
        {
            "id": "soften-violent-tw",
            "when": {"categories_include": ["Violent"], "score_below": 0.6},
            "then": {"final_label": "controversial"},
        },
    ],
}


GENERIC_WEBHOOK: dict = {
    "name": "generic-webhook",
    "description": "Permissive default — flag but don't auto-block unless confidence is high.",
    "rules": [],
}


PRESETS: dict[str, dict] = {
    "discord": DISCORD_DEFAULT,
    "twitch": TWITCH_CHAT,
    "generic": GENERIC_WEBHOOK,
}


def get_preset(name: str) -> NemoclawPolicy:
    if name not in PRESETS:
        raise KeyError(f"unknown preset: {name!r}; available: {list(PRESETS)}")
    return NemoclawPolicy.from_dict(PRESETS[name])


__all__ = ["PRESETS", "get_preset", "DISCORD_DEFAULT", "TWITCH_CHAT", "GENERIC_WEBHOOK"]
