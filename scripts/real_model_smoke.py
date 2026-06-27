"""Real-model smoke check for the hackathon demo host.

This intentionally loads model weights. Run it on the Vast.ai GPU instance, not
in ordinary CI:

    python scripts/real_model_smoke.py --deep
"""

from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()

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


if __name__ == "__main__":
    raise SystemExit(main())
