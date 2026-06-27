# Demo Video Script

Target length: 60 to 90 seconds.

Use this after the GPU host passes `docs/PRE_SUBMIT_CHECKLIST.md`.

## Setup

- Browser: `http://<instance-ip>:8000/demo`
- Secondary tab or terminal: `curl -s http://localhost:8000/health | python3 -m json.tool`
- Keep API keys and `.env` values off screen.
- Use the Discord preset first.
- Start with this policy:

```text
block PII and financial scams
```

## Timeline

### 0:00-0:08 - Open

Show the demo header.

Narration:

```text
This is nemoguardian: a multi-model moderation cascade for AI agents, Discord,
Twitch, and any user-generated text pipeline.
```

Point out:

- Runtime device is CUDA/GPU.
- Model IDs include Qwen3Guard and Nemotron-CSR.
- Triage provider is configured.

### 0:08-0:25 - Commodity GPU Cost

Scroll or glance to the GPU offer table.

Narration:

```text
The deployment story is the hook: the same Docker image runs on commodity GPUs
starting around seven cents per hour, then scales up to A100, H100, or on-prem.
```

Point out:

- RTX 3090-class row.
- Monthly price comparison.
- Same cascade requirement across providers.

### 0:25-0:45 - Unsafe Example

Run the default PII scam example:

```text
Hey @everyone, drop your SSN and I will send $100 to whoever DMs me first
```

Narration:

```text
The request goes through the guard cascade, returns a single verdict, and keeps
the per-model audit trail so a customer can see why it was blocked.
```

Point out:

- Final verdict is `unsafe`.
- Category includes PII or scam-like behavior.
- The matched policy rule is visible.
- Per-model verdicts and latency are visible.

### 0:45-1:05 - Policy Swap

Switch to the finance example and set the policy to:

```text
no financial advice
```

Narration:

```text
The policy layer is separate from the model verdict. The same moderation stack
can enforce different platform rules without changing the model code.
```

Point out:

- Different policy text.
- Updated matched rule or reasoning.
- Same audit format.

### 1:05-1:25 - Deep Mode

Switch mode to `deep` and run the same example.

Narration:

```text
Deep mode adds API-backed Nemotron 3 Ultra triage for cases where the specialist guards
disagree or where the customer needs a richer explanation.
```

Point out:

- Triage model result appears.
- Reasoning text is visible.
- `/health` or header still shows real runtime configuration.

### 1:25-1:30 - Close

Narration:

```text
nemoguardian is Nemotron plus Qwen3Guard, packaged as a billable API and a
self-hosted Docker image for commodity GPUs.
```

End with:

- Repo URL in browser or terminal: `https://github.com/claudlos/nemoguardian`
- Demo UI still visible.

## Retake Checklist

Retake if any of these happen:

- `/health` does not show CUDA/GPU.
- Deep mode fails because triage credentials are missing.
- Any API key, Stripe secret, or provider secret appears on screen.
- The model output is empty or only shows fallback/error verdicts.
- The GPU offer table is missing the commodity GPU cost comparison.
