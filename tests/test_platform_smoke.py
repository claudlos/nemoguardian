"""Tests for the offline platform-smoke harness (``scripts/platform_smoke.py``).

No live secret, network, GPU, or platform SDK: the harness drives each adapter
through a synthetic event (parse -> stubbed/deterministic moderate -> planned
action -> audit) and reports a per-platform pass/skip/fail summary.
"""

from __future__ import annotations

import json

import pytest

from nemoguardian.bot.types import ModerationAction, Platform
from nemoguardian.schemas import Mode, ModerateRequest, VerdictLabel
from scripts.platform_smoke import (
    SMOKE_TEXT,
    PlatformSmoke,
    StubCascade,
    build_discord_context,
    build_slack_context,
    build_telegram_context,
    build_twitch_context,
    build_webhook_context,
    default_specs,
    format_report,
    main,
    run_one,
    run_smoke,
)

ALL_PLATFORMS = {"discord", "slack", "telegram", "twitch", "webhook"}


def test_stub_cascade_flags_injection_as_unsafe():
    cascade = StubCascade()
    resp = cascade.moderate(ModerateRequest(text=SMOKE_TEXT, mode=Mode.STANDARD))
    assert resp.verdict == VerdictLabel.UNSAFE
    assert resp.categories == ["Jailbreak"]
    assert cascade.calls == [SMOKE_TEXT]


def test_stub_cascade_passes_benign_text():
    resp = StubCascade().moderate(ModerateRequest(text="hello team, good morning", mode=Mode.FAST))
    assert resp.verdict == VerdictLabel.SAFE
    assert resp.categories == []
    assert resp.mode == Mode.FAST  # echoes the requested mode


@pytest.mark.parametrize(
    ("builder", "platform"),
    [
        (build_discord_context, Platform.DISCORD),
        (build_slack_context, Platform.SLACK),
        (build_telegram_context, Platform.TELEGRAM),
        (build_twitch_context, Platform.TWITCH),
        (build_webhook_context, Platform.WEBHOOK),
    ],
)
def test_every_builder_parses_synthetic_event(builder, platform):
    context = builder()
    assert context.platform == platform
    assert context.text == SMOKE_TEXT
    assert context.workspace_id  # non-empty
    assert context.message_id  # non-empty


def test_default_specs_cover_all_five_platforms():
    platforms = {spec.platform.value for spec in default_specs()}
    assert platforms == ALL_PLATFORMS


def test_run_smoke_offline_all_pass():
    report = run_smoke(env={})  # no secrets -> not live-ready, but still runs offline
    assert report.ok
    assert report.passed == 5
    assert report.skipped == 0
    assert report.failed == 0
    by_platform = {r.platform: r for r in report.results}
    assert set(by_platform) == ALL_PLATFORMS
    for r in report.results:
        assert r.status == "pass"
        assert r.fake_event_ok
        assert r.audit_written
        assert r.verdict == "unsafe"
        # offline runs need no secret -> live-readiness is annotated, not required
        assert r.live_ready is False


def test_webhook_degrades_enforcement_to_flag():
    report = run_smoke(env={})
    webhook = next(r for r in report.results if r.platform == "webhook")
    assert webhook.requested_action == ModerationAction.DELETE.value
    assert webhook.planned_action == ModerationAction.FLAG.value
    assert webhook.degraded is True
    assert any("degraded to flag" in note for note in webhook.notes)


def test_enforcing_platform_does_not_degrade():
    report = run_smoke(env={})
    discord = next(r for r in report.results if r.platform == "discord")
    # discord supports delete natively -> no degrade
    assert discord.requested_action == ModerationAction.DELETE.value
    assert discord.planned_action == ModerationAction.DELETE.value
    assert discord.degraded is False


def test_run_one_skips_on_import_error():
    def _missing_dep():
        raise ImportError("no module named 'pretend_sdk'")

    spec = PlatformSmoke(
        platform=Platform.DISCORD,
        build_context=_missing_dep,
        capabilities=lambda: {ModerationAction.FLAG},
    )
    result = run_one(spec, env={})
    assert result.status == "skip"
    assert any("optional dependency missing" in note for note in result.notes)
    assert result.fake_event_ok is False


def test_run_one_fails_on_unexpected_error():
    def _boom():
        raise ValueError("synthetic parse blew up")

    spec = PlatformSmoke(
        platform=Platform.SLACK,
        build_context=_boom,
        capabilities=lambda: {ModerationAction.FLAG},
    )
    result = run_one(spec, env={})
    assert result.status == "fail"
    assert "ValueError" in (result.error or "")


def test_require_live_skips_when_prerequisites_absent():
    report = run_smoke(require_live=True, env={})
    assert report.skipped == 5
    assert report.passed == 0
    for r in report.results:
        assert r.status == "skip"
        assert "live prerequisites absent" in r.notes


def test_require_live_runs_when_prerequisites_present():
    # A spec with no SDK requirement whose only secret is satisfied via env.
    spec = PlatformSmoke(
        platform=Platform.WEBHOOK,
        build_context=build_webhook_context,
        capabilities=lambda: {ModerationAction.FLAG, ModerationAction.ALLOW},
        sdk_modules=(),
        secret_envs=("FAKE_SECRET",),
    )
    report = run_smoke([spec], require_live=True, env={"FAKE_SECRET": "present"})
    assert report.passed == 1
    assert report.failed == 0
    result = report.results[0]
    assert result.status == "pass"
    assert result.live_ready is True
    assert result.secrets_present is True


def test_live_readiness_reports_missing_sdk_and_secret():
    spec = PlatformSmoke(
        platform=Platform.SLACK,
        build_context=build_slack_context,
        capabilities=lambda: {ModerationAction.FLAG},
        sdk_modules=("definitely_not_installed_sdk_xyz",),
        secret_envs=("MISSING_TOKEN",),
    )
    result = run_one(spec, env={})
    assert result.status == "pass"  # offline still runs
    assert result.sdk_available is False
    assert result.secrets_present is False
    assert result.live_ready is False


def test_report_marks_failure_not_ok():
    bad = PlatformSmoke(
        platform=Platform.TWITCH,
        build_context=lambda: (_ for _ in ()).throw(RuntimeError("nope")),
        capabilities=lambda: {ModerationAction.FLAG},
    )
    report = run_smoke([bad], env={})
    assert report.failed == 1
    assert report.ok is False


def test_format_report_is_readable():
    report = run_smoke(env={})
    text = format_report(report)
    assert "platform-smoke (offline)" in text
    for platform in ALL_PLATFORMS:
        assert platform in text
    assert "[PASS]" in text
    assert "summary: 5 passed, 0 skipped, 0 failed of 5" in text


def test_main_offline_returns_zero(capsys):
    code = main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "summary:" in out


def test_main_json_emits_valid_json(capsys):
    code = main(["--json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["mode"] == "offline"
    assert payload["summary"]["passed"] == 5
    assert {r["platform"] for r in payload["results"]} == ALL_PLATFORMS


def test_to_dict_round_trips_status():
    report = run_smoke(env={})
    data = report.to_dict()
    assert data["mode"] == "offline"
    assert data["summary"]["total"] == 5
    assert all("status" in r for r in data["results"])
