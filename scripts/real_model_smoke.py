"""Real-model smoke check for the hackathon demo host.

This intentionally loads model weights. Run it on the Vast.ai GPU instance, not
in ordinary CI:

    python scripts/real_model_smoke.py --deep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from nemoguardian.cascade import Cascade
    from nemoguardian.policy.presets import get_preset
    from nemoguardian.schemas import Mode, ModerateRequest, VerdictLabel

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        help=(
            "Apply a cascade hardware profile (e.g. 3090-smoke, 8b, 14b, "
            "550b-triage) before any per-flag overrides. See "
            "`nemoguardian profiles list`."
        ),
    )
    parser.add_argument("--deep", action="store_true", help="Also call configured triage API")
    parser.add_argument(
        "--text",
        default="Hey @everyone, drop your SSN and I will send $100 to whoever DMs me first",
        help="Moderation text used for the smoke request",
    )
    parser.add_argument(
        "--policy",
        default="block PII and financial scams",
        help="Policy text used for the smoke request",
    )
    parser.add_argument(
        "--policy-preset",
        default="discord",
        help="Built-in policy preset used for the smoke request",
    )
    parser.add_argument("--qwen-model", help="Override NEMOGUARDIAN_QWEN_MODEL")
    parser.add_argument("--qwen-stream-model", help="Override NEMOGUARDIAN_QWEN_STREAM_MODEL")
    parser.add_argument("--csr-model", help="Override NEMOGUARDIAN_CSR_MODEL")
    parser.add_argument("--triage-model", help="Override NEMOGUARDIAN_TRIAGE_MODEL")
    parser.add_argument("--triage-base-url", help="Override NEMOGUARDIAN_TRIAGE_BASE_URL")
    parser.add_argument(
        "--quantize",
        choices=("env", "on", "off"),
        default="env",
        help="Set both local model 4-bit flags for profile tests",
    )
    parser.add_argument(
        "--qwen-4bit",
        choices=("env", "on", "off"),
        default="env",
        help="Override Qwen3Guard-Gen 4-bit loading",
    )
    parser.add_argument(
        "--csr-4bit",
        choices=("env", "on", "off"),
        default="env",
        help="Override Nemotron-CSR 4-bit loading",
    )
    parser.add_argument(
        "--disable-qwen-gen",
        action="store_true",
        help="Skip Qwen3Guard-Gen in the smoke request",
    )
    parser.add_argument(
        "--disable-qwen-stream",
        action="store_true",
        help="Skip Qwen3Guard-Stream in the smoke request",
    )
    parser.add_argument(
        "--disable-csr",
        action="store_true",
        help="Skip Nemotron-CSR in the smoke request",
    )
    parser.add_argument(
        "--disable-triage",
        action="store_true",
        help="Skip Nemotron triage in the smoke request",
    )
    parser.add_argument(
        "--min-vram-gb",
        type=float,
        default=None,
        help=(
            "Minimum CUDA VRAM required before loading weights. Defaults to the "
            "profile's smoke floor, then NEMOGUARDIAN_SMOKE_MIN_VRAM_GB, then 20."
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip CUDA/API-key checks and attempt model loading anyway",
    )
    args = parser.parse_args()

    min_vram_gb = _resolve_min_vram_gb(args)
    if not args.skip_preflight:
        preflight_code = _preflight(deep=args.deep, min_vram_gb=min_vram_gb)
        if preflight_code:
            return preflight_code

    config = _config_from_args(args)
    cascade = Cascade(config)
    policy = get_preset(args.policy_preset)
    mode = Mode.DEEP if args.deep else Mode.STANDARD
    request = ModerateRequest(
        text=args.text,
        policy=args.policy,
        mode=mode,
        use_qwen_stream=not args.disable_qwen_stream,
        use_qwen_gen=not args.disable_qwen_gen,
        use_nemotron_csr=not args.disable_csr,
        use_nemotron_triage=not args.disable_triage,
    )
    result = cascade.moderate(request, policy_engine=policy)
    print(json.dumps(result.model_dump(), indent=2, default=str))

    errored = {name: verdict.error for name, verdict in result.model_verdicts.items() if verdict.error}
    if errored:
        print(f"model errors: {errored}", file=sys.stderr)
        return 2
    if result.verdict != VerdictLabel.UNSAFE:
        print(f"expected unsafe verdict, got {result.verdict.value}", file=sys.stderr)
        return 3
    return 0


def _resolve_min_vram_gb(args: argparse.Namespace) -> float:
    """Pick the smoke VRAM floor: explicit flag > profile floor > env > 20."""
    if getattr(args, "min_vram_gb", None) is not None:
        return float(args.min_vram_gb)
    profile_name = getattr(args, "profile", None)
    if profile_name:
        from nemoguardian.profiles import get_profile

        return get_profile(profile_name).min_smoke_vram_gb
    return float(os.environ.get("NEMOGUARDIAN_SMOKE_MIN_VRAM_GB", "20"))


def _config_from_args(args: argparse.Namespace):
    from nemoguardian.cascade import CascadeConfig

    config = CascadeConfig.from_env()
    profile_name = getattr(args, "profile", None)
    if profile_name:
        from nemoguardian.profiles import apply_profile_to_config, get_profile

        config = apply_profile_to_config(config, get_profile(profile_name))
    if args.qwen_model:
        config.qwen_gen_model = args.qwen_model
    if args.qwen_stream_model:
        config.qwen_stream_model = args.qwen_stream_model
    if args.csr_model:
        config.csr_model = args.csr_model
    if args.triage_model:
        config.triage_model = args.triage_model
    if args.triage_base_url:
        config.triage_base_url = args.triage_base_url

    quantize = _flag_value(args.quantize)
    if quantize is not None:
        config.qwen_gen_4bit = quantize
        config.csr_4bit = quantize

    qwen_4bit = _flag_value(args.qwen_4bit)
    if qwen_4bit is not None:
        config.qwen_gen_4bit = qwen_4bit

    csr_4bit = _flag_value(args.csr_4bit)
    if csr_4bit is not None:
        config.csr_4bit = csr_4bit

    return config


def _flag_value(raw: str) -> bool | None:
    if raw == "on":
        return True
    if raw == "off":
        return False
    return None


def _preflight(*, deep: bool, min_vram_gb: float) -> int:
    try:
        import torch
    except Exception as exc:
        print(f"torch import failed before smoke test: {exc}", file=sys.stderr)
        return 10

    if not torch.cuda.is_available():
        print("CUDA is not available; run this smoke test on the GPU demo host.", file=sys.stderr)
        return 11

    props = torch.cuda.get_device_properties(0)
    total_vram_gb = props.total_memory / (1024 ** 3)
    print(f"CUDA device: {props.name} ({total_vram_gb:.1f} GB VRAM)", file=sys.stderr)
    if total_vram_gb < min_vram_gb:
        print(
            f"GPU has {total_vram_gb:.1f} GB VRAM; expected at least {min_vram_gb:.1f} GB "
            "for the real-model demo.",
            file=sys.stderr,
        )
        return 12

    if deep and not (os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        print(
            "--deep requires NVIDIA_API_KEY or OPENROUTER_API_KEY for Nemotron triage.",
            file=sys.stderr,
        )
        return 13

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
