"""Cascade hardware profiles — smoke + experiment selection (audit #41-#43).

A *profile* names which guard models a cascade run loads and at what precision,
plus the VRAM-budget math used to decide whether the run fits a given GPU (for
example a 24 GB RTX 3090). Profiles are pure config + selection logic:

* importing this module loads no weights and needs no GPU, so the VRAM-fit math
  and profile selection are unit-testable in ordinary CI;
* selecting a profile NEVER mutates the shipped :class:`CascadeConfig` default.
  You opt in explicitly via ``--profile <name>`` / :func:`profile_to_config`;
* the per-model VRAM numbers are copied verbatim from ``docs/MODEL_CATALOG.md``
  (FP16 baseline, INT8 ~= half, 4-bit ~= quarter) — they are catalog estimates,
  not measured benchmarks, and are labelled as such.

The flagship use case (#41) is the RTX 3090 operator smoke: prove that
Qwen3Guard-Gen-4B + Nemotron-CSR-4B + the 0.6B stream head fit in 24 GB and
leave headroom, before a (GPU-only) live load is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from nemoguardian.cascade import CascadeConfig

# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------


class Precision(str, Enum):
    """Weight precision a local guard is loaded at."""

    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def is_quantized(self) -> bool:
        return self is not Precision.FP16


# The cascade only exposes a boolean ``load_in_4bit`` flag per local model, so
# anything other than FP16 maps onto 4-bit at runtime. INT8 is representable in
# the VRAM matrix (docs/QUANTIZATION_MATRIX.md) but not yet wired as a distinct
# load path — flagged in ``known_failure_modes`` for the INT8 rows.


# ---------------------------------------------------------------------------
# Per-model VRAM footprints (catalog estimates, not measured)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFootprint:
    """VRAM footprint estimates for one guard model, per precision.

    Numbers are the ``docs/MODEL_CATALOG.md`` estimates (GB). ``remote`` models
    run via an API and consume no local VRAM.
    """

    hf_id: str
    params_b: float
    vram_fp16_gb: float
    vram_int8_gb: float
    vram_int4_gb: float
    remote: bool = False

    def vram_gb(self, precision: Precision) -> float:
        """Estimated resident VRAM (GB) at ``precision``. Remote models are 0."""
        if self.remote:
            return 0.0
        if precision is Precision.FP16:
            return self.vram_fp16_gb
        if precision is Precision.INT8:
            return self.vram_int8_gb
        return self.vram_int4_gb


# Footprints for every model a profile can reference. Sourced from
# docs/MODEL_CATALOG.md sections 1-7 (catalog snapshot: late June 2026).
_FOOTPRINTS: dict[str, ModelFootprint] = {
    f.hf_id: f
    for f in (
        ModelFootprint("Qwen/Qwen3Guard-Gen-0.6B", 0.6, 1.5, 0.8, 0.5),
        ModelFootprint("Qwen/Qwen3Guard-Stream-0.6B", 0.6, 1.5, 0.8, 0.5),
        ModelFootprint("Qwen/Qwen3Guard-Gen-4B", 4.0, 8.0, 4.0, 2.5),
        ModelFootprint("Qwen/Qwen3Guard-Stream-4B", 4.0, 8.0, 4.0, 2.5),
        ModelFootprint("Qwen/Qwen3Guard-Gen-8B", 8.0, 16.0, 8.0, 4.5),
        ModelFootprint("nvidia/Nemotron-Content-Safety-Reasoning-4B", 4.0, 8.0, 4.0, 2.5),
        ModelFootprint("meta-llama/Llama-Guard-4-12B", 12.0, 24.0, 12.0, 6.5),
        # 550B triage runs through the NVIDIA / OpenRouter API — no local VRAM.
        ModelFootprint("nvidia/nemotron-3-ultra-550b-a55b", 550.0, 0.0, 0.0, 0.0, remote=True),
    )
}


def footprint(hf_id: str) -> ModelFootprint:
    """Look up a model's footprint, or raise with the known set."""
    try:
        return _FOOTPRINTS[hf_id]
    except KeyError as exc:
        known = ", ".join(sorted(_FOOTPRINTS))
        raise KeyError(f"unknown model {hf_id!r}; known footprints: {known}") from exc


# Non-weight VRAM: CUDA context + framework + KV cache + activation headroom.
# Cross-checked against providers.base.CASCADE_VRAM_* — the default cascade sums
# to ~17.5 GB of FP16 weights and ~24 GB comfort, i.e. ~6.5 GB of overhead.
DEFAULT_OVERHEAD_GB = 6.5


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """A selectable hardware/model profile for a cascade run.

    A profile is a *recipe*: which gen/stream/CSR guards to load, at what
    precision, whether the remote triage stage is enabled, and which GPU it
    targets. :func:`profile_to_config` turns it into a :class:`CascadeConfig`.
    """

    name: str
    description: str
    gen_model: str
    stream_model: str
    csr_model: str
    triage_model: str
    gen_precision: Precision = Precision.INT4
    csr_precision: Precision = Precision.INT4
    stream_precision: Precision = Precision.FP16
    reasoning: bool = True
    enable_triage: bool = True
    triage_on_disagreement_only: bool = True
    target_gpu: str = "RTX 3090"
    target_vram_gb: float = 24.0
    # Minimum CUDA VRAM the operator smoke should require before loading weights.
    min_smoke_vram_gb: float = 20.0
    # False => the profile references a model the shipped wrappers can't load as
    # a drop-in (needs an extra adapter); it is a documented experiment target,
    # not a runnable cascade. The VRAM math is still valid.
    runnable_with_shipped_wrappers: bool = True
    note: str = ""

    # -- VRAM math ---------------------------------------------------------

    def local_models(self) -> list[tuple[ModelFootprint, Precision]]:
        """The local (VRAM-consuming) guards this profile loads, with precision."""
        pairs = [
            (footprint(self.gen_model), self.gen_precision),
            (footprint(self.stream_model), self.stream_precision),
            (footprint(self.csr_model), self.csr_precision),
        ]
        return [(fp, prec) for fp, prec in pairs if not fp.remote]

    def weights_vram_gb(self) -> float:
        """Sum of resident weight VRAM (GB) across the local guards."""
        return sum(fp.vram_gb(prec) for fp, prec in self.local_models())

    def local_vram_gb(self, *, overhead_gb: float = DEFAULT_OVERHEAD_GB) -> float:
        """Total estimated VRAM (GB): guard weights + framework/KV overhead."""
        return self.weights_vram_gb() + overhead_gb

    def headroom_gb(
        self, budget_gb: float | None = None, *, overhead_gb: float = DEFAULT_OVERHEAD_GB
    ) -> float:
        """Spare VRAM (GB) under ``budget_gb`` (default: the profile target)."""
        budget = self.target_vram_gb if budget_gb is None else budget_gb
        return budget - self.local_vram_gb(overhead_gb=overhead_gb)

    def fits(
        self, budget_gb: float | None = None, *, overhead_gb: float = DEFAULT_OVERHEAD_GB
    ) -> bool:
        """True if the profile's estimated VRAM fits within ``budget_gb``."""
        return self.headroom_gb(budget_gb, overhead_gb=overhead_gb) >= 0.0

    def total_params_b(self) -> float:
        """Total parameters (billions) of the local guards resident on the GPU."""
        return sum(fp.params_b for fp, _ in self.local_models())

    def summary(self) -> dict[str, object]:
        """JSON-friendly description of the profile and its fit math."""
        return {
            "name": self.name,
            "description": self.description,
            "target_gpu": self.target_gpu,
            "target_vram_gb": self.target_vram_gb,
            "local_params_b": round(self.total_params_b(), 1),
            "models": [
                {
                    "hf_id": fp.hf_id,
                    "params_b": fp.params_b,
                    "precision": prec.value,
                    "vram_gb": fp.vram_gb(prec),
                }
                for fp, prec in self.local_models()
            ],
            "remote_triage_model": self.triage_model if self.enable_triage else None,
            "weights_vram_gb": round(self.weights_vram_gb(), 1),
            "overhead_gb": DEFAULT_OVERHEAD_GB,
            "total_vram_gb": round(self.local_vram_gb(), 1),
            "headroom_gb": round(self.headroom_gb(), 1),
            "fits_target": self.fits(),
            "min_smoke_vram_gb": self.min_smoke_vram_gb,
            "runnable_with_shipped_wrappers": self.runnable_with_shipped_wrappers,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

_DEFAULTS = CascadeConfig()

# "default" mirrors the shipped CascadeConfig so a test can prove that adding
# profiles does not change the default cascade. Its precisions reflect the
# shipped 4-bit defaults (qwen_gen_4bit / csr_4bit = True).
_DEFAULT_PROFILE = Profile(
    name="default",
    description="The shipped cascade defaults. Selecting any profile is opt-in; "
    "this one is byte-for-byte the out-of-the-box CascadeConfig().",
    gen_model=_DEFAULTS.qwen_gen_model,
    stream_model=_DEFAULTS.qwen_stream_model,
    csr_model=_DEFAULTS.csr_model,
    triage_model=_DEFAULTS.triage_model,
    gen_precision=Precision.INT4 if _DEFAULTS.qwen_gen_4bit else Precision.FP16,
    csr_precision=Precision.INT4 if _DEFAULTS.csr_4bit else Precision.FP16,
    reasoning=_DEFAULTS.reasoning,
    enable_triage=_DEFAULTS.enable_triage,
    triage_on_disagreement_only=_DEFAULTS.triage_on_disagreement_only,
    target_gpu="RTX 3090",
    target_vram_gb=24.0,
    note="Identical to CascadeConfig() — present so callers can round-trip the "
    "default through the profile machinery without drift.",
)

PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (
        _DEFAULT_PROFILE,
        # --- #41: RTX 3090 operator smoke ---------------------------------
        Profile(
            name="3090-smoke",
            description="RTX 3090 (24 GB) operator smoke: the two 4B guards "
            "(Qwen3Guard-Gen-4B + Nemotron-CSR-4B) at 4-bit plus the 0.6B "
            "stream head. Proves the default model set fits 24 GB with "
            "headroom before a live GPU load.",
            gen_model="Qwen/Qwen3Guard-Gen-4B",
            stream_model="Qwen/Qwen3Guard-Stream-0.6B",
            csr_model="nvidia/Nemotron-Content-Safety-Reasoning-4B",
            triage_model="nvidia/nemotron-3-ultra-550b-a55b",
            gen_precision=Precision.INT4,
            csr_precision=Precision.INT4,
            stream_precision=Precision.FP16,
            target_gpu="RTX 3090",
            target_vram_gb=24.0,
            min_smoke_vram_gb=20.0,
            note="Same models as the shipped default; this profile pins the "
            "4-bit precision and the 20 GB smoke floor so a 24 GB 3090 passes.",
        ),
        # --- #42: experiment profiles (do NOT change the default) ----------
        Profile(
            name="8b",
            description="Experiment: upgrade the generative judge to "
            "Qwen3Guard-Gen-8B (same Apache-2.0 family, drop-in wrapper). "
            "Higher-accuracy tier; needs 4-bit to fit a 24 GB 3090, fits an "
            "A100 at FP16.",
            gen_model="Qwen/Qwen3Guard-Gen-8B",
            stream_model="Qwen/Qwen3Guard-Stream-0.6B",
            csr_model="nvidia/Nemotron-Content-Safety-Reasoning-4B",
            triage_model="nvidia/nemotron-3-ultra-550b-a55b",
            gen_precision=Precision.INT4,
            csr_precision=Precision.INT4,
            stream_precision=Precision.FP16,
            target_gpu="RTX 3090",
            target_vram_gb=24.0,
            min_smoke_vram_gb=20.0,
            note="Runnable with the shipped Qwen3GuardGen wrapper (8B is the "
            "same family as the default 4B). At FP16 it needs an A100 (40 GB).",
        ),
        Profile(
            name="14b",
            description="Experiment / heavy tier: the catalog's largest "
            "single-GPU guard, Llama-Guard-4-12B (12B, multimodal), as the "
            "generative judge alongside the 4B reasoning head — ~16B of "
            "resident guard weights. Multi-image moderation experiment.",
            gen_model="meta-llama/Llama-Guard-4-12B",
            stream_model="Qwen/Qwen3Guard-Stream-0.6B",
            csr_model="nvidia/Nemotron-Content-Safety-Reasoning-4B",
            triage_model="nvidia/nemotron-3-ultra-550b-a55b",
            gen_precision=Precision.INT4,
            csr_precision=Precision.INT4,
            stream_precision=Precision.FP16,
            target_gpu="RTX 3090",
            target_vram_gb=24.0,
            min_smoke_vram_gb=20.0,
            runnable_with_shipped_wrappers=False,
            note="HONEST CAVEAT: there is no single ~14B content-safety model "
            "in docs/MODEL_CATALOG.md (snapshot late June 2026). This profile "
            "uses the real 12B Llama-Guard-4-12B as the heavy-tier stand-in; "
            "it needs a Llama-Guard wrapper (not the Qwen3Guard one), so it is "
            "a documented experiment target, not a drop-in runnable cascade.",
        ),
        Profile(
            name="550b-triage",
            description="Experiment: emphasise the remote 550B triage stage. "
            "Local 4B guards stay at 4-bit (3090-friendly); the Nemotron 3 "
            "Ultra 550B API adjudicates EVERY deep request, not just guard "
            "disagreements. Highest-accuracy, API-cost trade-off.",
            gen_model="Qwen/Qwen3Guard-Gen-4B",
            stream_model="Qwen/Qwen3Guard-Stream-0.6B",
            csr_model="nvidia/Nemotron-Content-Safety-Reasoning-4B",
            triage_model="nvidia/nemotron-3-ultra-550b-a55b",
            gen_precision=Precision.INT4,
            csr_precision=Precision.INT4,
            stream_precision=Precision.FP16,
            enable_triage=True,
            triage_on_disagreement_only=False,
            target_gpu="RTX 3090",
            target_vram_gb=24.0,
            min_smoke_vram_gb=20.0,
            note="Local footprint equals 3090-smoke; the difference is the "
            "triage policy (forced on every deep request) and the API budget "
            "that implies. Requires NVIDIA_API_KEY or OPENROUTER_API_KEY.",
        ),
    )
}

DEFAULT_PROFILE_NAME = "3090-smoke"


def list_profiles() -> list[Profile]:
    """All registered profiles, in registry order."""
    return list(PROFILES.values())


def get_profile(name: str) -> Profile:
    """Look up a profile by name, or raise ``ValueError`` with the known set."""
    try:
        return PROFILES[name]
    except KeyError as exc:
        known = ", ".join(PROFILES)
        raise ValueError(f"unknown profile {name!r}; choose one of: {known}") from exc


# ---------------------------------------------------------------------------
# Profile -> CascadeConfig
# ---------------------------------------------------------------------------


def profile_to_config(profile: Profile, base: CascadeConfig | None = None) -> CascadeConfig:
    """Build a :class:`CascadeConfig` from ``profile``.

    Starts from ``base`` (default: a fresh ``CascadeConfig()``) and overrides
    only the fields the profile pins, so unrelated env-driven settings on
    ``base`` survive. The cascade exposes a boolean 4-bit flag per local model,
    so any non-FP16 precision maps onto ``load_in_4bit=True``.
    """
    cfg = replace(base) if base is not None else CascadeConfig()
    return replace(
        cfg,
        qwen_gen_model=profile.gen_model,
        qwen_stream_model=profile.stream_model,
        csr_model=profile.csr_model,
        triage_model=profile.triage_model,
        qwen_gen_4bit=profile.gen_precision.is_quantized,
        csr_4bit=profile.csr_precision.is_quantized,
        reasoning=profile.reasoning,
        enable_triage=profile.enable_triage,
        triage_on_disagreement_only=profile.triage_on_disagreement_only,
    )


def apply_profile_to_config(config: CascadeConfig, profile: Profile) -> CascadeConfig:
    """Apply ``profile`` on top of an existing ``config`` (e.g. from env).

    Thin wrapper over :func:`profile_to_config` kept as a named seam for the
    real-model smoke script, which builds ``CascadeConfig.from_env()`` first
    and then layers the chosen profile on top.
    """
    return profile_to_config(profile, base=config)


__all__ = [
    "DEFAULT_OVERHEAD_GB",
    "DEFAULT_PROFILE_NAME",
    "PROFILES",
    "ModelFootprint",
    "Precision",
    "Profile",
    "apply_profile_to_config",
    "footprint",
    "get_profile",
    "list_profiles",
    "profile_to_config",
]
