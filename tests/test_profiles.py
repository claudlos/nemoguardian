"""Tests for cascade hardware profiles + VRAM-fit math (audit #41-#43).

All GPU-free: they assert the fit arithmetic and profile-selection logic, never
load weights, and never touch the network.
"""

from __future__ import annotations

import argparse
from dataclasses import fields

import pytest
from typer.testing import CliRunner

from nemoguardian.cascade import CascadeConfig
from nemoguardian.cli import app
from nemoguardian.profiles import (
    DEFAULT_OVERHEAD_GB,
    PROFILES,
    Precision,
    apply_profile_to_config,
    footprint,
    get_profile,
    list_profiles,
    profile_to_config,
)

runner = CliRunner()


# --- footprint math --------------------------------------------------------


def test_footprint_vram_scales_with_precision():
    gen4b = footprint("Qwen/Qwen3Guard-Gen-4B")
    assert gen4b.vram_gb(Precision.FP16) == 8.0
    assert gen4b.vram_gb(Precision.INT8) == 4.0
    assert gen4b.vram_gb(Precision.INT4) == 2.5


def test_remote_triage_footprint_is_zero_local_vram():
    triage = footprint("nvidia/nemotron-3-ultra-550b-a55b")
    assert triage.remote is True
    assert triage.vram_gb(Precision.FP16) == 0.0


def test_footprint_unknown_model_raises():
    with pytest.raises(KeyError):
        footprint("nope/not-a-model")


def test_precision_is_quantized():
    assert Precision.FP16.is_quantized is False
    assert Precision.INT8.is_quantized is True
    assert Precision.INT4.is_quantized is True


# --- #41: 3090 smoke fit ---------------------------------------------------


def test_3090_smoke_two_4b_guards_fit_24gb():
    p = get_profile("3090-smoke")
    # Gen-4B (2.5, int4) + Stream-0.6B (1.5, fp16) + CSR-4B (2.5, int4) = 6.5.
    assert p.weights_vram_gb() == pytest.approx(6.5)
    assert p.local_vram_gb() == pytest.approx(6.5 + DEFAULT_OVERHEAD_GB)
    assert p.fits(24.0) is True
    assert p.headroom_gb(24.0) > 0


def test_3090_smoke_excludes_remote_triage_from_local_budget():
    p = get_profile("3090-smoke")
    local_ids = {fp.hf_id for fp, _ in p.local_models()}
    assert "nvidia/nemotron-3-ultra-550b-a55b" not in local_ids
    assert len(p.local_models()) == 3


def test_3090_smoke_default_full_precision_still_fits_24gb():
    """The shipped 4B model set fits 24 GB even at FP16, per #41."""
    p = get_profile("3090-smoke")
    # FP16: 8 + 1.5 + 8 = 17.5 weights + 6.5 overhead = 24.0 — tight but fits.
    fp16_weights = (
        footprint(p.gen_model).vram_gb(Precision.FP16)
        + footprint(p.stream_model).vram_gb(Precision.FP16)
        + footprint(p.csr_model).vram_gb(Precision.FP16)
    )
    assert fp16_weights == pytest.approx(17.5)
    assert fp16_weights + DEFAULT_OVERHEAD_GB <= 24.0


# --- #42: experiment profiles ---------------------------------------------


def test_8b_profile_needs_quant_for_3090_but_fits_a100_fp16():
    p = get_profile("8b")
    assert p.gen_model == "Qwen/Qwen3Guard-Gen-8B"
    # 4-bit fits the 3090.
    assert p.fits(24.0) is True
    # FP16 (16 + 1.5 + 8 = 25.5 + 6.5 = 32) does NOT fit a 24 GB 3090...
    fp16_total = (
        footprint(p.gen_model).vram_gb(Precision.FP16)
        + footprint(p.stream_model).vram_gb(Precision.FP16)
        + footprint(p.csr_model).vram_gb(Precision.FP16)
        + DEFAULT_OVERHEAD_GB
    )
    assert fp16_total > 24.0
    # ...but fits an A100 40 GB.
    assert fp16_total <= 40.0


def test_14b_profile_is_documented_experiment_not_drop_in():
    p = get_profile("14b")
    assert p.gen_model == "meta-llama/Llama-Guard-4-12B"
    assert p.runnable_with_shipped_wrappers is False
    # Honesty guard: the note must disclose there is no single 14B guard.
    assert "no single" in p.note.lower()
    assert p.fits(24.0) is True  # 6.5 + 1.5 + 2.5 + 6.5 = 17.0


def test_550b_triage_profile_forces_triage_every_request():
    p = get_profile("550b-triage")
    assert p.enable_triage is True
    assert p.triage_on_disagreement_only is False
    # Local footprint matches the smoke profile (only the triage policy differs).
    assert p.weights_vram_gb() == get_profile("3090-smoke").weights_vram_gb()


def test_all_named_experiment_profiles_registered():
    names = {p.name for p in list_profiles()}
    assert {"default", "3090-smoke", "8b", "14b", "550b-triage"} <= names


def test_get_profile_unknown_raises_with_choices():
    with pytest.raises(ValueError) as exc:
        get_profile("does-not-exist")
    assert "3090-smoke" in str(exc.value)


# --- profile -> CascadeConfig (default unchanged) --------------------------


def test_default_profile_roundtrips_to_shipped_config():
    """Selecting the 'default' profile reproduces CascadeConfig() exactly."""
    produced = profile_to_config(get_profile("default"))
    shipped = CascadeConfig()
    for f in fields(CascadeConfig):
        assert getattr(produced, f.name) == getattr(shipped, f.name), f.name


def test_profile_selection_does_not_mutate_default_config():
    before = CascadeConfig()
    profile_to_config(get_profile("8b"))
    after = CascadeConfig()
    # Building a profile config must not change what a fresh default looks like.
    for f in fields(CascadeConfig):
        assert getattr(before, f.name) == getattr(after, f.name), f.name


def test_profile_to_config_maps_precision_to_4bit_flags():
    cfg = profile_to_config(get_profile("3090-smoke"))
    assert cfg.qwen_gen_model == "Qwen/Qwen3Guard-Gen-4B"
    assert cfg.csr_model == "nvidia/Nemotron-Content-Safety-Reasoning-4B"
    assert cfg.qwen_gen_4bit is True
    assert cfg.csr_4bit is True


def test_apply_profile_preserves_unrelated_base_fields():
    base = CascadeConfig(triage_base_url="https://example.test/v1", concurrent_local=False)
    cfg = apply_profile_to_config(base, get_profile("8b"))
    # Profile overrides the models...
    assert cfg.qwen_gen_model == "Qwen/Qwen3Guard-Gen-8B"
    # ...but env-driven fields it does not pin survive.
    assert cfg.triage_base_url == "https://example.test/v1"
    assert cfg.concurrent_local is False


def test_every_profile_summary_is_json_serialisable():
    import json

    for p in list_profiles():
        json.dumps(p.summary())  # must not raise


# --- CLI -------------------------------------------------------------------


def test_cli_profiles_list():
    result = runner.invoke(app, ["profiles", "list"])
    assert result.exit_code == 0, result.output
    assert "3090-smoke" in result.output
    assert "Qwen/Qwen3Guard-Gen-4B" in result.output


def test_cli_profiles_show():
    result = runner.invoke(app, ["profiles", "show", "550b-triage"])
    assert result.exit_code == 0, result.output
    assert "cascade_config" in result.output
    assert "nvidia/nemotron-3-ultra-550b-a55b" in result.output


def test_cli_profiles_fit_pass_and_fail():
    ok = runner.invoke(app, ["profiles", "fit", "3090-smoke", "--gpu-vram-gb", "24"])
    assert ok.exit_code == 0, ok.output

    too_small = runner.invoke(app, ["profiles", "fit", "8b", "--gpu-vram-gb", "8"])
    assert too_small.exit_code == 1


# --- real_model_smoke.py profile wiring (no GPU) ---------------------------


def _smoke_args(**overrides):
    base = dict(
        profile=None,
        min_vram_gb=None,
        qwen_model=None,
        qwen_stream_model=None,
        csr_model=None,
        triage_model=None,
        triage_base_url=None,
        quantize="env",
        qwen_4bit="env",
        csr_4bit="env",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_smoke_resolve_min_vram_uses_profile_floor(monkeypatch):
    from scripts.real_model_smoke import _resolve_min_vram_gb

    monkeypatch.delenv("NEMOGUARDIAN_SMOKE_MIN_VRAM_GB", raising=False)
    # No profile, no flag, no env -> default 20.
    assert _resolve_min_vram_gb(_smoke_args()) == 20.0
    # Profile floor applies when no explicit flag.
    assert _resolve_min_vram_gb(_smoke_args(profile="3090-smoke")) == 20.0
    # Explicit flag always wins.
    assert _resolve_min_vram_gb(_smoke_args(profile="3090-smoke", min_vram_gb=12.0)) == 12.0


def test_smoke_config_from_args_applies_profile(monkeypatch):
    from scripts.real_model_smoke import _config_from_args

    for key in (
        "NEMOGUARDIAN_QWEN_MODEL",
        "NEMOGUARDIAN_CSR_MODEL",
        "NEMOGUARDIAN_TRIAGE_ON_DISAGREEMENT",
    ):
        monkeypatch.delenv(key, raising=False)

    cfg = _config_from_args(_smoke_args(profile="8b"))
    assert cfg.qwen_gen_model == "Qwen/Qwen3Guard-Gen-8B"

    triage_cfg = _config_from_args(_smoke_args(profile="550b-triage"))
    assert triage_cfg.triage_on_disagreement_only is False


def test_profiles_dict_keys_match_profile_names():
    for name, profile in PROFILES.items():
        assert name == profile.name
