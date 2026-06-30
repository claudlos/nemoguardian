# Quantization Matrix

> **Audit TODOs #41–#43.** Honest VRAM / load-flag / latency / quality reference
> for the models nemoguardian actually ships, plus the selectable hardware
> *profiles* that pin them. Pairs with `docs/MODEL_CATALOG.md` (the full catalog)
> and `docs/3090_OPERATOR_GUIDE.md` (the live runbook).
>
> **Honesty note.** VRAM and latency figures are **catalog estimates**, not
> measured benchmarks on this repo's hardware. They are copied from
> `docs/MODEL_CATALOG.md` (snapshot: late June 2026). The rule of thumb is FP16
> baseline → INT8 ≈ half → 4-bit ≈ quarter. Treat every number here as a
> *budget*, then confirm with a live `nemoguardian profiles fit` + GPU smoke.

The numbers below are produced by `nemoguardian/profiles.py` and are unit-tested
in `tests/test_profiles.py` **without a GPU** — the fit arithmetic and profile
selection are CI-checkable; only the live load needs a card.

---

## 1. Per-model VRAM (catalog estimates, GB)

| Model | Params | FP16 | INT8 | 4-bit | Load path |
|---|---:|---:|---:|---:|---|
| `Qwen/Qwen3Guard-Gen-0.6B` | 0.6B | 1.5 | 0.8 | 0.5 | `AutoModelForCausalLM`, `torch_dtype="auto"` |
| `Qwen/Qwen3Guard-Stream-0.6B` | 0.6B | 1.5 | 0.8 | 0.5 | `AutoModel`, `trust_remote_code=True` (per-token head) |
| `Qwen/Qwen3Guard-Gen-4B` | 4B | 8.0 | 4.0 | 2.5 | `AutoModelForCausalLM` (+ `BitsAndBytesConfig` for 4-bit) |
| `Qwen/Qwen3Guard-Gen-8B` | 8B | 16.0 | 8.0 | 4.5 | same as 4B; A100 for FP16 |
| `nvidia/Nemotron-Content-Safety-Reasoning-4B` | 4B | 8.0 | 4.0 | 2.5 | `Gemma3ForConditionalGeneration` / `AutoProcessor` |
| `meta-llama/Llama-Guard-4-12B` | 12B | 24.0 | 12.0 | 6.5 | Llama-Guard wrapper (multimodal) — **not** the Qwen wrapper |
| `nvidia/nemotron-3-ultra-550b-a55b` | 550B | — | — | — | **remote API** (NVIDIA NIM / OpenRouter); 0 local VRAM |

"Load path" maps to the model wrappers under `nemoguardian/models/`. The 550B
triage stage never loads locally — it is an HTTP call, so it consumes API budget,
not VRAM.

---

## 2. Profile fit matrix (RTX 3090, 24 GB target)

Each profile = which guards load + at what precision. `total` adds a single
`~6.5 GB` non-weight allowance (CUDA context + framework + KV cache + activation
headroom), cross-checked against `providers.base.CASCADE_VRAM_*` (FP16 cascade ≈
17.5 GB weights, 24 GB comfort). Inspect any profile with
`nemoguardian profiles show <name>`.

| Profile | Generative judge | Precision | Local params | Weights GB | + overhead | Headroom @24 GB | Fits 3090? | Drop-in runnable? |
|---|---|---|---:|---:|---:|---:|:---:|:---:|
| `default` | Qwen3Guard-Gen-4B | 4-bit | 8.6B | 6.5 | 13.0 | 11.0 | ✅ | ✅ |
| `3090-smoke` (#41) | Qwen3Guard-Gen-4B | 4-bit | 8.6B | 6.5 | 13.0 | 11.0 | ✅ | ✅ |
| `8b` (#42) | Qwen3Guard-Gen-8B | 4-bit | 12.6B | 8.5 | 15.0 | 9.0 | ✅ | ✅ |
| `14b` (#42) | Llama-Guard-4-12B | 4-bit | 16.6B | 10.5 | 17.0 | 7.0 | ✅ | ⚠️ experiment |
| `550b-triage` (#42) | Qwen3Guard-Gen-4B + 550B API | 4-bit | 8.6B | 6.5 | 13.0 | 11.0 | ✅ | ✅ |

All profiles also load `Qwen3Guard-Stream-0.6B` (FP16, ~1.5 GB) and
`Nemotron-Content-Safety-Reasoning-4B` (4-bit, ~2.5 GB). The `default` and
`3090-smoke` profiles load the **same model set as the shipped cascade** —
selecting a profile is opt-in and never changes the default
(`tests/test_profiles.py::test_default_profile_roundtrips_to_shipped_config`).

### FP16 note (why the 3090 default is 4-bit)

The shipped 4B model set *does* fit a 24 GB 3090 even at FP16 — but only just:
`8 + 1.5 + 8 = 17.5 GB` weights `+ 6.5 GB` overhead `= 24.0 GB`, i.e. zero
headroom. 4-bit (the shipped default) drops weights to `6.5 GB` and leaves
`~11 GB` of headroom for hot prompts and KV cache. The `8b` profile at FP16
(`25.5 GB` weights `+ 6.5 = 32 GB`) **does not** fit a 24 GB card — use 4-bit on
a 3090 or FP16 on an A100 (40 GB).

---

## 3. Load flags

| Want | Flags |
|---|---|
| Shipped default (4-bit, on-disagreement triage) | `NEMOGUARDIAN_QUANTIZE=1` (default) |
| Force FP16 local guards | `NEMOGUARDIAN_QUANTIZE=0` (needs ≥24 GB tight / prefer A100) |
| 3090 operator smoke | `python scripts/real_model_smoke.py --profile 3090-smoke` |
| 8B judge experiment | `python scripts/real_model_smoke.py --profile 8b` |
| Force 550B triage every deep request | `python scripts/real_model_smoke.py --profile 550b-triage --deep` |
| Pre-flight a profile against a card | `nemoguardian profiles fit <name> --gpu-vram-gb 24` |

The cascade exposes a boolean `load_in_4bit` flag per local guard, so a profile's
precision maps onto FP16 vs 4-bit at runtime. **INT8 appears in the matrix for
budgeting only** — it is not yet a distinct load path (see failure modes). The
profile's `min_smoke_vram_gb` floor (20 GB) gates the live smoke before any
weights load; an explicit `--min-vram-gb` always overrides it.

---

## 4. Expected quality (relative, not absolute)

These are **directional** expectations from the catalog, not measured scores on
this repo's benchmark. Run `make eval` for real numbers on your data.

| Profile | Expected quality trade-off |
|---|---|
| `3090-smoke` / `default` | Baseline. 4B Gen + 4B reasoning + 0.6B stream; 119-lang Qwen coverage, 3-tier severity. 4-bit costs a small accuracy haircut vs FP16 on subtle jailbreaks. |
| `8b` | Higher jailbreak robustness than 4B (bigger judge); still Apache-2.0 Qwen family. 4-bit on a 3090 narrows the gain vs FP16 8B on an A100. |
| `14b` | Heavy/multimodal experiment (text + multi-image). Richest single-GPU guard in the catalog, but Llama-4 Community license + needs its own wrapper. |
| `550b-triage` | Highest ceiling: the 550B API adjudicates **every** deep request, not just guard disagreements — best accuracy on contested cases, at per-call API cost + latency. |

---

## 5. Known failure modes

- **INT8 is budget-only.** The matrix lists INT8 VRAM, but the cascade's runtime
  flag is 4-bit (`load_in_4bit`). A profile that "wants" INT8 will load at 4-bit;
  do not assume an INT8 kernel path exists.
- **`14b` is not drop-in.** `Llama-Guard-4-12B` is a real model and the VRAM math
  is valid, but it needs a Llama-Guard wrapper, **not** the shipped `Qwen3GuardGen`
  wrapper. The profile is flagged `runnable_with_shipped_wrappers = False` and is
  a documented experiment target. There is **no single ~14B content-safety model**
  in `docs/MODEL_CATALOG.md`; the `14b` profile names the heavy *tier* (12B judge +
  4B reasoning ≈ 16B resident guard weights), not a fictional 14B checkpoint.
- **Overhead is an estimate.** The `~6.5 GB` non-weight allowance varies with
  context length, batch size, and KV-cache reuse. Long 128K-context Nemotron calls
  can exceed it — keep headroom, do not run a profile at exactly 0 GB headroom.
- **FP16 on a 3090 is tight, not comfortable.** The default set at FP16 lands at
  ~24.0 GB on a 24 GB card — any extra KV cache will OOM. Use 4-bit (the default)
  unless you are on an A100/H100.
- **`550b-triage` needs an API key.** `NVIDIA_API_KEY` or `OPENROUTER_API_KEY`
  must be set, plus `NEMOGUARDIAN_TRIAGE_BASE_URL` / `NEMOGUARDIAN_TRIAGE_MODEL`.
  Without them the triage stage surfaces `verdict.error` (fail-safe to review),
  it does not fail open.
- **4-bit accuracy haircut.** 4-bit weights trade a small amount of accuracy on
  subtle / adversarial inputs for the VRAM savings. The cascade stays fail-safe
  (degrades to controversial/review on low confidence), so the cost is more
  manual review, not silent fail-open.

---

## 6. How these numbers are kept honest

`nemoguardian/profiles.py` is the single source of truth: per-model footprints,
the overhead allowance, and the fit arithmetic. `tests/test_profiles.py` asserts
the exact GB sums above and that selecting a profile never mutates the shipped
`CascadeConfig`. Regenerate the live view any time with:

```bash
nemoguardian profiles list                       # all profiles + fit math (JSON)
nemoguardian profiles show 3090-smoke            # one profile + its CascadeConfig
nemoguardian profiles fit 8b --gpu-vram-gb 24    # pre-flight gate (exit 1 if no fit)
```
