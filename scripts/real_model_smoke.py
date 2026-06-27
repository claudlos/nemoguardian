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
    from nemoguardian.cascade import Cascade, CascadeConfig
    from nemoguardian.policy.presets import get_preset
    from nemoguardian.schemas import Mode, ModerateRequest, VerdictLabel

    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true", help="Also call configured triage API")
    parser.add_argument(
        "--min-vram-gb",
        type=float,
        default=float(os.environ.get("NEMOGUARDIAN_SMOKE_MIN_VRAM_GB", "20")),
        help="Minimum CUDA VRAM required before loading weights",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip CUDA/API-key checks and attempt model loading anyway",
    )
    args = parser.parse_args()

    if not args.skip_preflight:
        preflight_code = _preflight(deep=args.deep, min_vram_gb=args.min_vram_gb)
        if preflight_code:
            return preflight_code

    cascade = Cascade(CascadeConfig.from_env())
    policy = get_preset("discord")
    mode = Mode.DEEP if args.deep else Mode.STANDARD
    request = ModerateRequest(
        text="Hey @everyone, drop your SSN and I will send $100 to whoever DMs me first",
        policy="block PII and financial scams",
        mode=mode,
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
