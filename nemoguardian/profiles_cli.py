"""``nemoguardian profiles`` sub-app — inspect cascade hardware profiles.

Registered with a single ``app.add_typer(...)`` line in :mod:`nemoguardian.cli`
to keep merge conflicts small. Everything here is GPU-free: it reports the
VRAM-fit math from :mod:`nemoguardian.profiles` so an operator can decide which
profile fits their card before renting one.
"""

from __future__ import annotations

import json
from typing import Any

import typer

from nemoguardian.profiles import get_profile, list_profiles, profile_to_config

profiles_app = typer.Typer(help="Inspect cascade hardware/model profiles (#41-#43).")


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@profiles_app.command("list")
def list_cmd() -> None:
    """List every profile with its VRAM-fit summary as JSON."""
    _echo_json([p.summary() for p in list_profiles()])


@profiles_app.command("show")
def show_cmd(
    name: str = typer.Argument(..., help="Profile name, e.g. 3090-smoke."),
) -> None:
    """Show one profile's fit math and the CascadeConfig it would produce."""
    profile = get_profile(name)
    config = profile_to_config(profile)
    payload = profile.summary()
    payload["cascade_config"] = {
        "qwen_gen_model": config.qwen_gen_model,
        "qwen_stream_model": config.qwen_stream_model,
        "csr_model": config.csr_model,
        "triage_model": config.triage_model,
        "qwen_gen_4bit": config.qwen_gen_4bit,
        "csr_4bit": config.csr_4bit,
        "enable_triage": config.enable_triage,
        "triage_on_disagreement_only": config.triage_on_disagreement_only,
    }
    _echo_json(payload)


@profiles_app.command("fit")
def fit_cmd(
    name: str = typer.Argument(..., help="Profile name, e.g. 3090-smoke."),
    gpu_vram_gb: float = typer.Option(
        None,
        "--gpu-vram-gb",
        help="GPU VRAM budget (GB). Defaults to the profile's target.",
    ),
) -> None:
    """Print whether a profile fits a VRAM budget; exit non-zero if it does not.

    Useful in scripts/CI as a pre-flight gate before a live GPU load.
    """
    profile = get_profile(name)
    budget = profile.target_vram_gb if gpu_vram_gb is None else gpu_vram_gb
    fits = profile.fits(budget)
    _echo_json(
        {
            "profile": profile.name,
            "budget_gb": budget,
            "weights_vram_gb": round(profile.weights_vram_gb(), 1),
            "total_vram_gb": round(profile.local_vram_gb(), 1),
            "headroom_gb": round(profile.headroom_gb(budget), 1),
            "fits": fits,
        }
    )
    if not fits:
        raise typer.Exit(code=1)


__all__ = ["profiles_app"]
