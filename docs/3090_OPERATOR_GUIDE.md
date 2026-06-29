# RTX 3090 Operator Guide

This guide is for a self-hosted moderation run on a rented RTX 3090. The 3090
loads the local 4B guard models. Nemotron 3 Ultra 550B remains API-backed through
NVIDIA or OpenRouter.

## Guardrails

- Do not stop, recycle, or destroy existing Vast.ai boxes unless their instance
  ID was created for this run.
- Label new boxes with a unique label such as `nemoguardian-3090-YYYYMMDD`.
- Keep Hugging Face, NVIDIA, OpenRouter, Discord, and Vast.ai tokens outside the
  repo.
- Run the API-only triage smoke before loading local 4B models.
- Save evidence JSON under `handoffs/` or another local/private directory if it
  includes host details.

## 1. Read-Only Preflight

Install and authenticate the Vast.ai CLI outside the repo:

```bash
pip install vastai
vastai set api-key <secret>
```

List existing boxes and candidate 3090 offers without renting anything:

```bash
make vast-3090-preflight
```

Useful filters:

```bash
NEMOGUARDIAN_VAST_MAX_PRICE=0.20 make vast-3090-preflight
NEMOGUARDIAN_VAST_OFFER_LIMIT=20 make vast-3090-preflight
```

The helper prints the exact `vastai create instance` command after the offer
list. Pick one offer ID manually.

## 2. Fund GPU Credits Through Stripe

Stripe is the funding rail for rented GPU capacity. Set up local Stripe env
without committing keys:

```bash
make stripe-env-setup
source ~/.config/nemoguardian/stripe.env
```

Then create a GPU-credit Checkout session:

```bash
curl -X POST http://localhost:8000/billing/gpu-credits/checkout \
  -H 'Content-Type: application/json' \
  -d '{"email":"owner@example.com","amount_cents":2500}'
```

After the webhook credits the wallet, provisioning endpoints reserve from that
balance before calling Vast.ai. See `docs/STRIPE_GPU_CREDITS.md`.

## 3. Create One Labeled Box

```bash
vastai create instance <OFFER_ID> \
  --image vastai/pytorch:@vastai-automatic-tag \
  --disk 80 \
  --ssh \
  --direct \
  --label nemoguardian-3090-$(date +%Y%m%d)
```

Poll until `actual_status` is `running`:

```bash
vastai show instance <INSTANCE_ID> --raw
vastai ssh-url <INSTANCE_ID>
```

If status becomes `exited`, `unknown`, or `offline`, destroy only that new
instance and pick a different offer.

## 4. Copy Code Without Pushing Secrets

From the local repo:

```bash
cd /home/carlos/nemoguardian
rsync -az --delete \
  --exclude .git \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude handoffs \
  ./ root@<host>:/workspace/nemoguardian/
```

Or use a git bundle if direct rsync is awkward:

```bash
git bundle create /tmp/nemoguardian.bundle --all
scp /tmp/nemoguardian.bundle root@<host>:/workspace/
```

## 5. Configure Secrets On The Box

```bash
mkdir -p ~/.config/nemoguardian
chmod 700 ~/.config/nemoguardian
$EDITOR ~/.config/nemoguardian/gpu.env
chmod 600 ~/.config/nemoguardian/gpu.env
```

Example:

```bash
export HF_TOKEN="<hf_token>"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export OPENROUTER_API_KEY="<openrouter_key>"
export NEMOGUARDIAN_TRIAGE_BASE_URL="https://openrouter.ai/api/v1"
export NEMOGUARDIAN_TRIAGE_MODEL="nvidia/nemotron-3-ultra-550b-a55b:free"
export NEMOGUARDIAN_QUANTIZE=1
```

If using a local Hugging Face token file that contains more than one token,
extract only one token:

```bash
export HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9]+' /path/to/hf_token.txt | head -1)"
```

## 6. Install And Smoke

```bash
cd /workspace/nemoguardian
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
pip install -e ".[dev,discord]"
source ~/.config/nemoguardian/gpu.env
make triage-api-smoke TRIAGE_API_SMOKE_FLAGS="--expect-verdict unsafe"
make smoke
```

Then run the benchmark:

```bash
make eval EVAL_FLAGS="--mode standard --json"
make replay-transcript REPLAY_FLAGS="\
  --scenario-json examples/scenarios/discord_community_smoke.json \
  --mode standard \
  --enforcement-preset balanced \
  --output handoffs/replay-evidence.json"
```

Use deep mode only after the triage smoke passes:

```bash
make smoke-deep
make eval EVAL_FLAGS="--mode deep --json --limit 20"
```

## 7. Cleanup

Copy evidence back first:

```bash
rsync -az root@<host>:/workspace/nemoguardian/handoffs/ ./handoffs/gpu-run/
```

Destroy only the box created for this run:

```bash
vastai destroy instance <INSTANCE_ID> -y
```
