# Vast.ai Real-Model Demo Runbook

This is the June 30 recording path for `nemoguardian`: real Qwen3Guard +
Nemotron-CSR on a 24GB GPU, with optional Nemotron 3 Ultra triage through NVIDIA
or OpenRouter. The 550B triage model is API-only in this setup; it is not loaded
onto the RTX 3090.

## 1. Provision

Use an RTX 3090/4090 or larger Vast.ai instance with Docker + NVIDIA runtime.
The target is 24GB VRAM minimum.

Clone the published repo directly:

```bash
cd ~
git clone https://github.com/claudlos/nemoguardian nemoguardian
cd nemoguardian
```

If the GPU host cannot reach GitHub, create a git bundle locally and copy that
single file to the GPU host instead:

```bash
cd /home/carlos/nemoguardian
git bundle create /tmp/nemoguardian.bundle --all
scp /tmp/nemoguardian.bundle root@<instance-ip>:/root/
ssh root@<instance-ip>
git clone /root/nemoguardian.bundle nemoguardian
cd nemoguardian
```

## 2. Configure

Create `.env`:

```bash
DEMO_API_KEY="nmg_demo_$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"

cat > .env <<ENV
NEMOGUARDIAN_API_KEY=$DEMO_API_KEY
NEMOGUARDIAN_SELF_HOSTED_EMAIL=self-hosted@nemoguardian.local
NEMOGUARDIAN_TIER=self_hosted
NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT=1

NEMOGUARDIAN_QUANTIZE=1
NEMOGUARDIAN_QWEN_MODEL=Qwen/Qwen3Guard-Gen-4B
NEMOGUARDIAN_QWEN_STREAM_MODEL=Qwen/Qwen3Guard-Stream-0.6B
NEMOGUARDIAN_CSR_MODEL=nvidia/Nemotron-Content-Safety-Reasoning-4B
NEMOGUARDIAN_TRIAGE_MODEL=nvidia/nemotron-3-ultra-550b-a55b

# Set one of these.
NVIDIA_API_KEY=
OPENROUTER_API_KEY=
NEMOGUARDIAN_TRIAGE_BASE_URL=
ENV
```

Do not use `nmg_change_me`, `nmg_default_change_me`,
`nmg_paste_your_key_here`, or `nmg_replace_with_demo_key`; the server rejects
those placeholders.

If using OpenRouter only, set:

```bash
NEMOGUARDIAN_TRIAGE_BASE_URL=https://openrouter.ai/api/v1
NEMOGUARDIAN_TRIAGE_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

If running directly inside a prebuilt Vast PyTorch image instead of the project
Docker image, set `NEMOGUARDIAN_QUANTIZE=0` if 4-bit loading fails in the base
image. The two 4B local models fit on a 24GB RTX 3090 in FP16; Nemotron 3 Ultra
still runs through the API.

## 3. Build And Run

```bash
make docker-build
make docker-run
```

Alternatively with compose:

```bash
docker compose up --build
```

The compose file builds `nemoguardian/self-hosted:latest` from the checked-out
repo, so it does not require a pre-published container registry image.

## 4. Smoke Checks

In a second shell:

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8000/providers/offers?only_fits=true | python3 -m json.tool | head -80
```

Expected:

- `runtime_device` reports CUDA/GPU.
- `model_config` shows Qwen3Guard-Gen-4B and Nemotron-CSR.
- `triage_configured` is `true` when NVIDIA/OpenRouter key is set.
- `triage_provider` reports `nvidia` or `openrouter`; the triage model is an API call, not a local 550B load.

Run real-model smoke:

```bash
docker exec -it <container_id> python scripts/real_model_smoke.py
docker exec -it <container_id> python scripts/real_model_smoke.py --deep
```

The smoke script checks CUDA availability, minimum VRAM, and deep-mode triage
credentials before loading weights. Override the default 20GB preflight only if
you are intentionally testing a smaller host:

```bash
docker exec -e NEMOGUARDIAN_SMOKE_MIN_VRAM_GB=12 -it <container_id> python scripts/real_model_smoke.py
```

If running from a local Python environment on the host instead of Docker:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python scripts/real_model_smoke.py
python scripts/real_model_smoke.py --deep
```

### Optional Larger-Model Profiles

The verified RTX 3090 baseline is Qwen3Guard-Gen-4B + Nemotron-CSR-4B in FP16
on the current Vast PyTorch image. Larger local guards should be tested as
targeted profiles, not as the default recording path.

Qwen3Guard-Gen-8B can be profiled by itself on a compatible 4-bit stack:

```bash
python scripts/real_model_smoke.py \
  --qwen-model Qwen/Qwen3Guard-Gen-8B \
  --quantize on \
  --disable-csr
```

If the base image raises a quantization compatibility error, switch to a newer
PyTorch/bitsandbytes image before rerunning that profile. Do not attempt to load
Nemotron 3 Ultra locally on a 3090; keep it API-backed through NVIDIA or
OpenRouter.

### Framework And Bot Adapter Smoke

After the service is live, run the adapter/framework smoke check. This verifies
the running host plus credential-safe Discord, Twitch, and generic webhook
adapter behavior:

```bash
make framework-smoke DEMO_BASE_URL=http://<instance-ip>:8000 \
  FRAMEWORK_SMOKE_FLAGS="--require-gpu --require-triage --moderate --output framework-evidence.json"
```

To exercise the authenticated webhook path as well, pass the self-hosted API key
only through the shell environment:

```bash
NEMOGUARDIAN_API_KEY=<secret> make framework-smoke DEMO_BASE_URL=http://<instance-ip>:8000 \
  FRAMEWORK_SMOKE_FLAGS="--require-gpu --require-triage --moderate --live-webhook --require-api-key --output framework-evidence.json"
```

Real Discord/Twitch network runs require out-of-repo credentials:

```bash
pip install -e ".[discord,twitch]"
DISCORD_BOT_TOKEN=<secret> python -m nemoguardian.adapters.discord
TWITCH_TOKEN=<secret> python -m nemoguardian.adapters.twitch <channel>
```

After the service is running, capture one evidence file for the submission:

```bash
make demo-check DEMO_CHECK_FLAGS="--wait-seconds 120 --require-gpu --require-triage --moderate --deep --output demo-evidence.json"
```

## 5. Record

Open:

```text
http://<instance-ip>:8000/demo
```

Sequence:

1. Show the header with GPU/model/triage status.
2. Run the PII scam example with `discord` preset.
3. Run the finance example with policy `no financial advice`.
4. Switch to `deep` mode and show triage reasoning.
5. Scroll to the GPU offer table showing the cheap Vast.ai option.

Do not record a final take until both real-model smoke commands pass.
