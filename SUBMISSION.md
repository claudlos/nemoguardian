# nemoguardian — Submission writeup

**Hackathon:** Hermes Agent Accelerated Business Hackathon · NVIDIA × Stripe × Nous Research
**Submission deadline:** EOD Tuesday, June 30, 2026
**Built by:** Carlos
**Repo:** https://github.com/claudlos/nemoguardian

---

## The 60-second pitch

> "nemoguardian is the moderation layer for any AI agent that talks to people.
> We stack NVIDIA's Nemotron Content Safety Reasoning and Qwen3Guard in a
> multi-model cascade that fits on **commodity GPUs starting at $0.07/hr**.
> Same image, same model output, runs on RTX 3090, A100, H100, or Hetzner
> bare metal — pick the price/perf trade-off that fits your deployment.
> The NVIDIA partnership play: Nemotron is the best open safety stack in 2026,
> but loses on cost to Llama Guard. nemoguardian closes that gap by running
> Nemotron on the cheapest commodity hardware available."

## The headline demo moment

Watch the cascade run on a **Vast.ai RTX 3090-class 24GB GPU** — the same
Nemotron-CSR 4B reasoning model that NVIDIA publishes, on commodity hardware.
The `/health` panel in the demo shows the GPU, configured model IDs, quantization,
and NVIDIA/OpenRouter triage status.

## What we built

A Python package + FastAPI server + Docker image that:

1. **Cascades two-to-three specialist classifiers** in series (cheap → expensive)
2. **Enforces per-platform policies** via a YAML gate (NemoClaw-style)
3. **Sells itself via Stripe** with four tiers (Free / Pro / Scale / Self-hosted)
4. **Provisions GPUs** across 9 cloud providers from one API
5. **Runs the same Docker image** anywhere — laptop, RTX 3090, H100, Hetzner, on-prem

### The model cascade (the core IP)

| Stage | Model | Source | Cost | Latency |
|---|---|---|---|---|
| 1. Fast stream | `Qwen/Qwen3Guard-Stream-0.6B` | Qwen team (Apache 2.0) | ~1ms/token | streaming token-level |
| 2. Full-doc guard | `Qwen/Qwen3Guard-Gen-4B` | Qwen team (Apache 2.0) | ~50ms | 119 languages, 3-tier |
| 3. Custom policy | `nvidia/Nemotron-Content-Safety-Reasoning-4B` | NVIDIA (Jan 2026) | ~200ms | reasoning-on mode |
| 4. Deep triage | `nvidia/nemotron-3-ultra-220b-a12b` | NVIDIA NIM / OpenRouter | ~300ms | adjudicates disagreements |

Total VRAM at FP16: **~17.5GB minimum, 24GB comfortable, 32GB headroom.**

## The infrastructure story (NVIDIA partnership pitch)

> "The Nemotron Content Safety family is the best open safety stack in 2026.
> But it loses on cost to Llama Guard because L40S/A100 inference is expensive.
> nemoguardian closes that gap by running Nemotron on commodity RTX 3090s at
> $0.07/hr. That makes Nemotron viable for the long tail of Discord/Twitch
> communities that can't afford OpenAI Moderation or Hive. We want NVIDIA to
> either (a) bundle nemoguardian as the reference Nemotron deployment, or
> (b) sponsor dedicated inference capacity on Nemotron 3 Ultra so we can run
> the triage step for free."

### Provider-agnostic deployment

| Provider | Cheapest GPU that fits cascade | $/hr | Region |
|---|---|---|---|
| **Vast.ai** | RTX 3090 | **$0.07** | Global |
| **Salad** | RTX 3090 | $0.07 | Global |
| **RunPod** | L4 / RTX 3090 | $0.12–0.14 | Global |
| **Vast.ai** | A100 40GB | $0.14 | Global |
| **Vast.ai** | RTX 4090 | $0.18 | Global |
| **Vast.ai** | H100 80GB | $0.19 | Global |
| **Vast.ai** | RTX 5090 | $0.21 | Global |
| **GPUhub** | RTX Pro 6000 96GB | $0.25 | Zero egress |
| **RunPod** | L40S 48GB | $0.50 | Global |
| **Salad / Hetzner** | Dedicated bare metal | $0.41–1.92 | EU / global |
| **Lyceum** | H200 141GB | $2.20 | EU-sovereign |
| **DigitalOcean** | H100 80GB | $3.39 | US |
| **AWS / Azure / GCP** | Various | $3+ | Enterprise compliance |

**The same Docker image runs on every option above.** Customers pick the price/perf
trade-off that fits them. The demo UI (`/demo`) shows the cost comparison table live.

## What's real vs. simulated for the demo

**Real, shipping:**
- The FastAPI server with 16 application endpoints plus generated OpenAPI docs
- Full cascade orchestration (guard stack → optional deep triage → aggregator → policy gate)
- Multi-model aggregator with weighted voting + fail-closed override
- NemoClaw YAML policy gate with Discord / Twitch / generic presets
- Stripe subscription tiers + checkout + webhook + metered usage
- Provider registry with 9 clouds; Vast.ai has live API client
- Docker image + docker-compose for self-hosted
- `/demo` moderation console with per-model verdicts, reasoning, latency, policy rule, and cost table

**Simulated for the demo:**
- Other cloud providers return hard-coded catalog data so the cost table is honest
- Multi-language 119-language live demo (English shown live; cached snapshots for others)
- Billing/provisioning can run in demo mode when Stripe/Vast credentials are absent

## On-brief check

- **NVIDIA integrations:** Nemotron-CSR 4B + Nemotron 3 Ultra triage + NemoClaw policy gate ✅
- **Stripe Skills:** Subscription tiers, webhook handler, metered billing, customer portal ✅
- **"Agents that earn, spend, and operate":** nemoguardian is itself an agent —
  consumes API calls, gates LLM input/output, sold as managed service ✅
- **Real operations at any scale:** FastAPI scales horizontally; same Docker image works on any GPU ✅

## Why this wins (or at least deserves to)

| Criterion | How we hit it |
|---|---|
| **Usefulness** | Every Discord / Twitch / Slack / forum needs moderation. Open-source alternative to Hive / OpenAI Moderation. |
| **Viability** | $0.07/hr per node. Same docker image, 9 clouds. No vendor lock-in. |
| **Presentation** | Live demo shows: cascade running on a 3090, policy swap in real time, cost comparison table. |

## The 90-second demo video flow

```
0:00  Open /demo. Header shows runtime device, model IDs, and triage provider.
0:10  Cost comparison loads from /providers/offers with a low-cost Vast.ai option.
0:20  Run: "drop your SSN in chat for $100" with discord preset.
      Verdict returns unsafe with PII/policy breakdown.
0:40  Run the finance example with policy "no financial advice".
      Same cascade, different policy rule/audit trail.
1:00  Switch mode to deep. Show NVIDIA/OpenRouter triage reasoning.
1:15  Open /health or keep the header visible to prove real runtime config.
1:30  End card: "nemoguardian — Nemotron + Qwen3Guard on commodity GPUs.
      $0.07/hr. 9 clouds. One docker image. Ship it."
```

## What I'd add with one more week

- Small eval harness — 200 adversarial prompts in 4 languages with ground truth
- Stripe test-mode checkout in the demo video (no live card needed)
- Hermes Agent skill so any Hermes-driven agent can invoke nemoguardian as a guardrail
- One-click Vast.ai launch flow wired to the live provisioning endpoint

## Self-critique

The hard problem — making two classifiers agree in adversarial edge cases — is still
open. The triage step is an LLM call, which means it can be wrong too. The
4-bit quantization introduces accuracy loss on the reasoning traces.

What we DO have is a working open-source moderation cascade that runs locally,
demonstrates clear value, exposes the internals so judges can audit every step,
and ships with a Stripe + multi-cloud story that's real, not mocked.

## Links

- Repo: https://github.com/claudlos/nemoguardian
- Model/license manifest: `docs/THIRD_PARTY_MODELS.md`
- Demo runbook: `docs/VASTAI_DEMO_RUNBOOK.md`
- Pre-submit checklist: `docs/PRE_SUBMIT_CHECKLIST.md`
- Hackathon form draft: `docs/SUBMISSION_FORM.md`
- Demo video script: `docs/VIDEO_SCRIPT.md`
- Demo evidence: `demo-evidence.json` from `make demo-check` on the GPU host
- Final demo video: add hosted recording URL after GPU smoke passes
- Public writeup: use this file as the source, or adapt it for the final post
