"""Positioning + platform-coverage docs stay honest and code-grounded.

These tests guard the README/`docs/PLATFORM_COVERAGE.md` framing so the project
keeps presenting itself as a *self-hostable moderation template* (not a hosted
business) and so the coverage matrix never drifts from what the adapters in
``nemoguardian/adapters`` actually declare via ``capabilities()``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nemoguardian.adapters import discord, slack, telegram, twitch, webhook
from nemoguardian.bot.types import Platform

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
COVERAGE = ROOT / "docs" / "PLATFORM_COVERAGE.md"

# Claims the audit told us to avoid making.
FORBIDDEN_CLAIMS = (
    "fully production-ready hosted service",
    "complete multi-platform coverage",
    "auto gpu spend is safe",
    "automatic gpu spend is safe",
)


def _read(path: Path) -> str:
    assert path.exists(), f"missing required doc: {path}"
    return path.read_text(encoding="utf-8")


def test_coverage_doc_exists_and_has_matrix_columns():
    text = _read(COVERAGE)
    for column in ("ingest", "actions enforced", "doctor", "dry-run", "audit", "status"):
        assert column in text, f"coverage matrix missing column: {column}"


def test_readme_positions_as_self_hostable():
    """Guard the night-3 honest positioning: nemoguardian is something you
    *self-host* (run on your own hardware, own the audit data, choose your
    GPU / API providers), not a managed SaaS you subscribe to.

    This is NOT a claim that the cascade runs without any third-party API
    calls — Nemotron 3 Ultra triage is a free API call to NVIDIA or
    OpenRouter and that's fine. The positioning is about *who runs the
    moderation service* (you), not *where the model weights live* (wherever
    is cheapest).

    Asserted signals (at least two must be present, to stay phrasing-agnostic):
    """
    text = _read(README).lower()
    signals = (
        "self-host",          # covers "self-hostable", "self-hosted", "self-hosting"
        "docker",
        "own your audit",
        "choose your own gpu",
        "audit data",         # weaker, present in multiple places
    )
    hits = sum(1 for s in signals if s in text)
    assert hits >= 2, f"README does not signal self-hostable positioning (hits={hits}, signals={signals})"
    # Must link out to the dedicated coverage doc.
    assert "docs/platform_coverage.md" in text


def test_readme_and_coverage_avoid_forbidden_claims():
    for path in (README, COVERAGE):
        text = _read(path).lower()
        for claim in FORBIDDEN_CLAIMS:
            assert claim not in text, f"{path.name} makes an avoided claim: {claim!r}"


def test_readme_has_platform_coverage_section():
    text = _read(README)
    assert "## Platform coverage" in text
    # Every shipped platform appears as a matrix row.
    for platform in ("Discord", "Twitch", "Slack", "Telegram", "Webhook"):
        assert platform in text, f"README coverage matrix missing {platform}"


def test_coverage_doc_lists_planned_platforms():
    text = _read(COVERAGE)
    for planned in ("Matrix", "Reddit", "YouTube", "Kick"):
        assert planned in text, f"coverage doc missing planned platform: {planned}"
    assert "Planned" in text


@pytest.mark.parametrize(
    ("module", "platform"),
    [
        (discord, Platform.DISCORD),
        (twitch, Platform.TWITCH),
        (slack, Platform.SLACK),
        (telegram, Platform.TELEGRAM),
        (webhook, Platform.WEBHOOK),
    ],
)
def test_coverage_doc_matches_declared_capabilities(module, platform):
    """The capabilities table must list exactly each adapter's real actions."""
    text = _read(COVERAGE)
    # Locate the platform's row in the "Capabilities per platform" table.
    label = platform.value.capitalize()
    rows = [
        line
        for line in text.splitlines()
        if line.strip().startswith(f"| {label} |") and "`" in line
    ]
    assert rows, f"no capabilities row found for {label}"
    row = rows[0]
    documented = {token.strip("` ") for token in row.split("`") if token.strip(" `")}
    declared = {action.value for action in module.capabilities()}
    assert declared <= documented, (
        f"{label}: declared capabilities {declared - documented} not documented"
    )
    # No phantom capabilities documented that the code does not actually have.
    documented_actions = documented & {
        "allow",
        "flag",
        "delete",
        "timeout",
        "mute",
        "ban",
        "queue",
        "notify_mods",
        "notify_user",
    }
    assert documented_actions == declared, (
        f"{label}: documented {documented_actions} != declared {declared}"
    )


def test_discord_is_the_full_bot_in_matrix():
    text = _read(COVERAGE)
    # Discord is the platform row tagged a full bot today.
    discord_rows = [
        line
        for line in text.splitlines()
        if line.strip().startswith("| **Discord**") and "**Full bot**" in line
    ]
    assert len(discord_rows) == 1
    # No other platform row claims full-bot status.
    other_full_bot_rows = [
        line
        for line in text.splitlines()
        if "**Full bot**" in line
        and line.strip().startswith("|")
        and not line.strip().startswith("| **Discord**")
        and not line.strip().startswith("| **Full bot**")  # legend row
    ]
    assert other_full_bot_rows == []
