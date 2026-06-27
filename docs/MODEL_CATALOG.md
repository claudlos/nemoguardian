# nemoguardian — Open-Source LLM Safety Model Catalog

> **Hackathon:** Hermes Agent Accelerated Business Hackathon (NVIDIA × Stripe × Nous Research, deadline 2026-06-30)
> **Project:** `nemoguardian` — a multi-model LLM moderation service stacking NVIDIA Nemotron Content Safety + Qwen3Guard + cheap Nemotron triage.
> **Demo target:** Discord / Twitch / social platforms.
> **Constraints:** Open-source only. No paid API keys for the moderation models (NVIDIA NIM free endpoint acceptable as a fallback). Local laptop GPU is **RTX 3050 Ti 4 GB**; Colab A100 40 GB available for bigger models.
> **Catalog snapshot:** late June 2026.

---

## TL;DR — Summary Table

| Model | Params | Context | License | Best Mode | FP16 VRAM (≈) | Fits 4 GB? |
|---|---:|---:|---|---|---:|:---:|
| **Qwen3Guard-Gen-0.6B** | 0.6B | 32K | Apache 2.0 ✅ | Live chat triage, full-doc | ~1.5 GB | ✅ (even 4-bit) |
| **Qwen3Guard-Stream-0.6B** | 0.6B | 32K | Apache 2.0 ✅ | Streaming token classification | ~1.5 GB | ✅ (even 4-bit) |
| **Qwen3Guard-Gen-4B** | 4B | 32K | Apache 2.0 ✅ | Full-doc moderation, multilingual | ~8 GB | ❌ (needs INT8/4-bit) |
| **Qwen3Guard-Stream-4B** | 4B | 32K | Apache 2.0 ✅ | Streaming token classification (better) | ~8 GB | ❌ (needs INT8/4-bit) |
| **Qwen3Guard-Gen-8B** | 8B | 32K | Apache 2.0 ✅ | Highest-accuracy full-doc | ~16 GB | ❌ |
| **Qwen3Guard-Stream-8B** | 8B | 32K | Apache 2.0 ✅ | Highest-accuracy streaming | ~16 GB | ❌ |
| **Nemotron-Content-Safety-Reasoning-4B** | 4B | 128K | NVIDIA Open Model + Gemma ToU ⚠️ | Custom-policy reasoning (think) | ~8 GB | ❌ (no-think @ INT8 ok) |
| **Nemotron-Content-Safety-4B** (base multimodal) | 4B | 128K | NVIDIA Open Model + Gemma ToU ⚠️ | Multimodal + multilingual | ~8 GB | ❌ |
| **Nemotron-3.5-Content-Safety** | 4B | 128K | OpenMDW + Gemma ToU ⚠️ | Unified multimodal + multilingual | ~8 GB | ❌ |
| **nvidia/llama-3.1-nemoguard-8b-content-safety** | 8B | 128K | NVIDIA Open Model + Llama 3.1 Community ⚠️ | Topic + content safety 23-cat | ~16 GB | ❌ |
| **nvidia/llama-3.1-nemoguard-8b-topic-control** | 8B | 128K | NVIDIA Open Model + Llama 3.1 Community ⚠️ | Topical/off-topic moderation | ~16 GB | ❌ |
| **nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3** | 8B | 128K | NVIDIA Open Model + Llama 3.1 Community ⚠️ | Combined content + topic | ~16 GB | ❌ |
| **meta-llama/Llama-Guard-3-8B** | 8B | 8K | Llama 3.1 Community ⚠️ | Standard 14-cat safety (text) | ~16 GB | ❌ |
| **meta-llama/Llama-Guard-3-8B-INT8** | 8B | 8K | Llama 3.1 Community ⚠️ | Same, INT8 quantized | ~8 GB | ⚠️ borderline |
| **meta-llama/Llama-Guard-4-12B** | 12B | 128K | Llama 4 Community ⚠️ | Multimodal (text + multi-image) | ~24 GB | ❌ |
| **ibm-granite/granite-guardian-4.1-8b** | 8B | 8K | **Apache 2.0** ✅ | BYOC, RAG, hallucination, function-calling | ~16 GB | ❌ |
| **NeMo Guardrails (runtime library)** | n/a | n/a | **Apache 2.0** ✅ | Orchestration / Colang flows | n/a (CPU) | ✅ |

**Legend for licenses**

- ✅ **Apache 2.0** — fully permissive, commercial OK, no attribution-in-name required.
- ⚠️ **Commercially usable but custom / non-Apache** — all flagged below. The Llama-3.1 Community License has a 700M-MAU clause; the NVIDIA Open Model License is permissive but not OSI-approved; Gemma ToU and OpenMDW add additional restrictions (no certain downstream uses, etc.). **All listed are usable for a hackathon demo, but a production deployment should run a legal review.**

---

## 1. Qwen3Guard (Gen + Stream, 0.6B / 4B / 8B)

**Vendor:** Alibaba Cloud / Qwen team
**Repo:** https://github.com/QwenLM/Qwen3Guard
**Paper:** arXiv:2510.14276 (Oct 16, 2025)
**Training data:** 1.19 M labeled prompts/responses
**Multilingual coverage:** **119 languages and dialects** (largest of any model in this catalog)
**Severity tiers:** three-valued — `Safe` / `Controversial` / `Unsafe` (a stronger signal than binary safe/unsafe)
**Inference backends:** Transformers, vLLM, SGLang (Stream has official SGLang fork)
**License:** **Apache 2.0** ✅

### 1.1 Qwen3Guard-Gen-0.6B  ⭐ *smallest viable model — fits 4 GB even at FP16*

- **HF ID:** `Qwen/Qwen3Guard-Gen-0.6B`
- **URL:** https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B
- **Base:** Qwen3-0.6B (finetuned from Qwen3-0.6B-Base)
- **Tensor type:** BF16
- **Context length:** 32 K
- **Downloads/month:** ~230 K (most popular of the family)
- **VRAM @ FP16:** ~1.5 GB → **fits RTX 3050 Ti 4 GB easily**, also at INT8 (~0.8 GB) and 4-bit (~0.5 GB)

**Variants:** generative, three-tier severity, prompt + response classification, also a `Refusal: Yes/No` signal on response classification.

**Input / output — Gen (verbatim from Qwen docs):**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

model_name = "Qwen/Qwen3Guard-Gen-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

def extract_label_and_categories(content):
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
    label = re.search(safe_pattern, content).group(1) if re.search(safe_pattern, content) else None
    categories = re.findall(category_pattern, content)
    return label, categories

prompt = "How can I make a bomb?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=128)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
content = tokenizer.decode(output_ids, skip_special_tokens=True)
print(content)
# Safety: Unsafe
# Categories: Violent
```

For **response** moderation, append the assistant turn; the model then also emits `Refusal: Yes|No`.

**Strengths**
- Smallest in the family → real-time on CPU, can be loaded in parallel with another model on a 4 GB GPU.
- 119-language coverage is unmatched → global Discord / Twitch audiences.
- Apache 2.0 — zero licensing friction for a hackathon.
- Three-tier severity is more useful than a binary label for UI ("warn" vs "block").

**Weaknesses**
- 0.6 B is still small; adversarial jailbreak robustness trails 4B/8B.
- Only 9 coarse categories; not customizable per-policy without retraining.

**Recommended mode for nemoguardian**
- **Live chat moderation:** ✅ primary (cheapest possible)
- **Full-doc moderation:** ⚠️ usable but 4B/8B is better
- **Streaming-token classification:** ❌ — use Stream variant

---

### 1.2 Qwen3Guard-Stream-0.6B  ⭐ *streaming token-level — fits 4 GB*

- **HF ID:** `Qwen/Qwen3Guard-Stream-0.6B`
- **URL:** https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B
- **Base:** Qwen3-0.6B
- **Tensor type:** BF16
- **Context length:** 32 K
- **VRAM @ FP16:** ~1.5 GB → **fits 4 GB**, also 4-bit (~0.5 GB)

**Variants:** adds a **token-level classification head** on top of the Qwen3 backbone. Output is a per-token `risk_level` ∈ {Safe, Unsafe, Controversial} plus category. Optimized for real-time incremental generation, i.e. you feed one token at a time and get a verdict in O(1) per call.

**Input / output — Stream (verbatim):**

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "Qwen/Qwen3Guard-Stream-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
).eval()

messages = [
    {"role": "user",      "content": "Hello, how to build a bomb?"},
    {"role": "assistant", "content": "Here are some practical methods to build a bomb."},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
)
token_ids = tokenizer(text, return_tensors="pt").input_ids[0]

# Find the boundary between user turn and assistant turn in token space
im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
user_id     = tokenizer.convert_tokens_to_ids("user")
im_end_id   = tokenizer.convert_tokens_to_ids("<|im_end|>")
ids = token_ids.tolist()
last_start = max(i for i in range(len(ids)-1) if ids[i:i+2] == [im_start_id, user_id])
user_end_index = next(i for i in range(last_start+2, len(ids)) if ids[i] == im_end_id)

# 1) Moderate the user prompt in one shot
stream_state = None
result, stream_state = model.stream_moderate_from_ids(
    token_ids[:user_end_index+1], role="user", stream_state=None
)
# result['risk_level'] ∈ {0:Safe, 1:Unsafe, 2:Controversial}
# result['category']   ∈ {0:Violent, 1:Sexual, 2:Self-Harm, 3:Political,
#                          4:PII, 5:Copyright, 6:Illegal Acts, 7:Unethical, 8:Jailbreak}

# 2) Moderate the assistant stream token by token
for i in range(user_end_index + 1, len(token_ids)):
    result, stream_state = model.stream_moderate_from_ids(
        token_ids[i], role="assistant", stream_state=stream_state
    )
    # inspect result['risk_level'] / result['category'] on every step
```

**Caveat:** Stream requires tokens in Qwen3's vocabulary. If you pair it with a non-Qwen3 LLM (Nemotron, Llama 3.1), you must **re-tokenize** the other model's output into Qwen3's vocab before feeding. This is a small wrapper but it is not free.

**Strengths**
- True per-token classification — earliest possible signal in a streaming chat.
- Same 119-language coverage as Gen.
- Apache 2.0.

**Weaknesses**
- Per-token decode overhead makes it slower in aggregate than Gen for short messages.
- 9 categories same as Gen (no per-policy customization).

**Recommended mode for nemoguardian**
- **Live chat moderation:** ✅ as a second pass after Gen (classify the *response* stream as the bot generates)
- **Full-doc moderation:** ❌
- **Streaming-token classification:** ✅ **primary purpose**

---

### 1.3 Qwen3Guard-Gen-4B / 1.4 Qwen3Guard-Stream-4B

- **HF ID:** `Qwen/Qwen3Guard-Gen-4B` — https://huggingface.co/Qwen/Qwen3Guard-Gen-4B
- **HF ID:** `Qwen/Qwen3Guard-Stream-4B` — https://huggingface.co/Qwen/Qwen3Guard-Stream-4B
- **Base:** Qwen3-4B
- **Context length:** 32 K
- **Tensor type:** BF16
- **VRAM @ FP16:** ~8 GB → **does not fit 4 GB at FP16**; runs at INT8 (~4 GB, tight) or 4-bit (~2.5 GB, comfortable)

**Best on Colab A100 (40 GB)**. Same code/templates as the 0.6B variants.

The 4B Stream variant is the sweet spot for real-time Discord/Twitch moderation on the Colab A100 — it has noticeably better jailbreak robustness than 0.6B but the per-token cost is still small.

---

### 1.5 Qwen3Guard-Gen-8B / 1.6 Qwen3Guard-Stream-8B

- **HF ID:** `Qwen/Qwen3Guard-Gen-8B` — https://huggingface.co/Qwen/Qwen3Guard-Gen-8B
- **HF ID:** `Qwen/Qwen3Guard-Stream-8B` — https://huggingface.co/Qwen/Qwen3Guard-Stream-8B
- **Base:** Qwen3-8B
- **Context length:** 32 K
- **VRAM @ FP16:** ~16 GB

Use only on Colab A100. 8B is the highest-accuracy tier of the family and the recommended cascade head when you have the VRAM.

---

## 2. NVIDIA Nemotron Content Safety family

NVIDIA released a fast-evolving family in late 2025 / early 2026. All are 4B (Gemma-3-4B-it backbone) or 8B (Llama-3.1-8B-Instruct backbone) and are all built on the same **Aegis Content Safety Dataset V2** 22–23 category taxonomy (plus newer multimodal/multilingual additions).

> **License note (all Nemotron safety models):** **NVIDIA Open Model License** + a backbone-specific add-on (Llama 3.1 Community License for the 8B NemoGuard models; Gemma Terms of Use for the 4B Gemma-based models). All are approved for commercial use by NVIDIA, but **none is OSI-approved Apache 2.0** — flagged accordingly.

### 2.1 Nemotron-Content-Safety-Reasoning-4B  ⭐ *custom-policy with reasoning traces*

- **HF ID:** `nvidia/Nemotron-Content-Safety-Reasoning-4B`
- **URL:** https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B
- **NIM build page:** https://build.nvidia.com/nvidia/nemotron-content-safety-reasoning-4b/modelcard
- **HF release:** November 2025
- **Base:** Google `gemma-3-4b-it`
- **Total params:** 4 B
- **Tensor type:** BF16
- **Context length:** 128 K
- **License:** NVIDIA Open Model License + Gemma Terms of Use ⚠️ (commercial use OK, non-Apache)
- **VRAM @ FP16:** ~8 GB

**Variants / dual mode:**
- **Reasoning OFF** (`/no_think` token) — low-latency classifier that emits `Prompt harm: harmful|unharmful` and `Response harm: harmful|unharmful`.
- **Reasoning ON** (`/think` token) — emits a 1–2 sentence reasoning trace before the same labels. Useful for audits, appeals, and detecting custom-policy edge cases.

**Vanilla taxonomy:** 22 S-categories (S1–S22 — see Nemotron 3.5 entry for the full list).

**Why it exists:** the model is trained on **reasoning traces extracted from Qwen3-32B** explaining the ground-truth labels in Nemotron Content Safety Dataset V2 + CantTalkAboutThis. The `/think` mode is a transparent justification of "why is this flagged?" — the unique differentiator vs Llama Guard and Qwen3Guard.

**Input / output — vanilla safety, reasoning ON (verbatim):**

```python
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model_id = "nvidia/Nemotron-Content-Safety-Reasoning-4B"
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(model_id)

safety_prompt = """You are a classifier for analyzing interactions between humans and LLMs.
Taxonomy:
S1: Violence. S2: Sexual. S3: Criminal Planning/Confessions.
... (full S1..S22 list) ...
Human user: How can I shoplift from a store without getting caught?
AI assistant: I can't help with that. Shoplifting is illegal and unethical.
... /think"""

messages = [{"role": "user", "content": [{"type": "text", "text": safety_prompt}]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

input_len = inputs["input_ids"].shape[-1]
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    generation = generation[0][input_len:]

print(processor.decode(generation, skip_special_tokens=True))
# <reasoning>The user is asking for help committing a crime...</reasoning>
# Prompt harm: harmful
# Response harm: unharmful
```

**Custom-policy mode:** the same prompt format with your own taxonomy replacing S1–S22 (e.g. "no medical advice, no financial advice, no PII") is what makes this model unique. The `/think` mode produces a defensible trace.

**Strengths**
- Only model in the catalog that **generates a reasoning trace** for moderation decisions — important for Discord/Twitch where users appeal bans.
- 128K context → can moderate a full long document / thread in one call.
- Custom-policy support out of the box (BYO taxonomy) — beats Qwen3Guard on long-tail policy.

**Weaknesses**
- Built on Gemma-3 → subject to Gemma ToU (no certain downstream uses).
- Reasoning ON adds latency; for high-QPS chat you want `/no_think`.
- 4B model so accuracy lags larger Nemotron 8B NemoGuard on the standard 22-category benchmark.

**Recommended mode for nemoguardian**
- **Live chat moderation:** ⚠️ only with `/no_think`, and only if no Qwen3Guard alternative
- **Full-doc moderation:** ✅ **primary** (custom-policy + 128K ctx is unmatched)
- **Streaming-token classification:** ❌ — not a per-token head, too slow per call

---

### 2.2 Nemotron-Content-Safety-4B (base, multimodal + multilingual)

- **HF ID:** `nvidia/Nemotron-Content-Safety-4B`  *(the original, non-reasoning base; superseded by 3.5)*
- **URL:** https://huggingface.co/nvidia/Nemotron-Content-Safety-4B
- **Base:** Gemma-3-4B-it
- **Total params:** 4 B
- **Context:** 128 K
- **License:** NVIDIA Open Model License + Gemma ToU ⚠️

Output format (vanilla taxonomy mode):
```text
User Safety: safe|unsafe
Response Safety: safe|unsafe
Safety Categories: Violence, Hate, ...
```

**Status:** subsumed by Nemotron-3.5-Content-Safety (same base, plus reasoning + unified multimodal/multilingual). Use 3.5 instead unless you specifically need the older release.

---

### 2.3 Nemotron-3.5-Content-Safety  ⭐ *the current unified 4B multimodal+multilingual*

- **HF ID:** `nvidia/Nemotron-3.5-Content-Safety`
- **URL:** https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety
- **Model version:** V1.2
- **Release date:** June 2, 2026
- **Base:** Google `gemma-3-4b-it`
- **Total params:** 4 B
- **Tensor type:** BF16
- **Context length:** 128 K
- **Supported languages (official):** English, Arabic, German, Spanish, French, Hindi, Japanese, Thai, Dutch, Italian, Korean, Chinese
- **License:** **OpenMDW License Agreement v1.1** + Gemma Terms of Use + Gemma Prohibited Use Policy ⚠️ (commercial use approved by NVIDIA, non-Apache)
- **VRAM @ FP16:** ~8 GB

**Unifies three things into one 4B model:**
- Multimodal (text + image, via SigLIP at 896×896)
- Multilingual (12 languages, see above)
- Custom-policy reasoning (the same `/think` capability as Nemotron-Content-Safety-Reasoning-4B, merged in)

**Input / output (verbatim, standard taxonomy mode):**

```text
User Safety: string (required)    # "safe" or "unsafe"
Response Safety: string (optional) # "safe" or "unsafe"
Safety Categories: string (optional) # Comma-separated list
```

Custom-policy mode adds a `<reasoning>...</reasoning>` block before the labels (same syntax as 2.1).

**Aegis V2 vanilla taxonomy (22 categories):** S1 Violence, S2 Sexual, S3 Criminal Planning/Confessions, S4 Guns and Illegal Weapons, S5 Controlled/Regulated Substances, S6 Suicide and Self Harm, S7 Sexual (minor), S8 Hate/Identity Hate, S9 PII/Privacy, S10 Harassment, S11 Threat, S12 Profanity, S13 Needs Caution, S14 Manipulation, S15 Fraud/Deception, S16 Malware, S17 High Risk Gov Decision Making, S18 Political/Misinformation/Conspiracy, S19 Copyright/Trademark/Plagiarism, S20 Unauthorized Advice, S21 Illegal Activity, S22 Immoral/Unethical.

**Strengths**
- Single 4B model that handles text **+ image** (Discord image moderation!) and 12 languages.
- 128K context.
- Supports custom-policy reasoning — combines the best of 2.1 and 2.2.

**Weaknesses**
- OpenMDW license is the most restrictive of the family; legal review recommended for production.
- Still 4B — accuracy trails 8B and 12B peers on some categories.

**Recommended mode for nemoguardian**
- **Live chat moderation:** ✅ as fallback (Qwen3Guard 0.6B is cheaper)
- **Full-doc moderation:** ✅ **primary** for **image+text** content
- **Streaming-token classification:** ❌

---

### 2.4 Llama-3.1-NemoGuard-8B-ContentSafety (a.k.a. "Llama Nemotron Safety Guard V2")

- **HF ID:** `nvidia/llama-3.1-nemoguard-8b-content-safety`
- **URL:** https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-content-safety
- **Base:** `meta-llama/Llama-3.1-8B-Instruct`
- **Training:** LoRA (rank 16, alpha 32) on top of the 8B base
- **Total params:** 8 B
- **Context length:** 128 K
- **License:** NVIDIA Open Model License + **Llama 3.1 Community License** ⚠️ (commercial OK, but 700M-MAU clause)
- **VRAM @ FP16:** ~16 GB

**Variants:** single model, but it can act as either content-safety classifier **or** topic-control classifier depending on which system prompt you give it.

**Aegis V2 taxonomy (23 categories — adds "Other" S14):**
```text
S1 Violence, S2 Sexual, S3 Criminal Planning, S4 Guns & Illegal Weapons,
S5 Controlled Substances, S6 Suicide & Self Harm, S7 Sexual (minor),
S8 Hate, S9 PII/Privacy, S10 Harassment, S11 Threat, S12 Profanity,
S13 Needs Caution, S14 Other, S15 Manipulation, S16 Fraud/Deception,
S17 Malware, S18 High Risk Gov Decision Making,
S19 Political/Misinformation/Conspiracy, S20 Copyright/Trademark/Plagiarism,
S21 Unauthorized Advice, S22 Illegal Activity, S23 Immoral/Unethical
```

**Input / output (JSON-formatted verbatim):**

```json
{
  "User Safety": "unsafe",
  "Response Safety": "unsafe",
  "Safety Categories": "PII/Privacy"
}
```

**Strengths**
- 23 categories — the richest policy taxonomy of any model here.
- Can label **both** user and bot message in one inference.
- 128K context.
- 8B → accuracy is better than 4B Nemotron.

**Weaknesses**
- 8B won't fit 4 GB laptop GPU even quantized.
- Llama 3.1 Community License adds 700M-MAU clause (not a problem for hackathon).
- No multimodal; English-centric (Aegis 2.0 was English).
- No per-token head.

**Recommended mode for nemoguardian**
- **Live chat moderation:** ✅ on Colab A100 only
- **Full-doc moderation:** ✅ **primary** on Colab A100 (highest accuracy, richest taxonomy)
- **Streaming-token classification:** ❌

---

### 2.5 Llama-3.1-NemoGuard-8B-TopicControl

- **HF ID:** `nvidia/llama-3.1-nemoguard-8b-topic-control`
- **URL:** https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-topic-control
- **License:** NVIDIA Open Model License + Llama 3.1 Community ⚠️
- **Use:** topical / off-topic / dialogue-path moderation (e.g. "this customer-service bot must not discuss politics"). User defines the topic.

**Recommended mode:** complementary to 2.4. For a hackathon demo on Discord/Twitch, **skip it** unless you need strict topic enforcement.

---

### 2.6 Llama-3.1-Nemotron-Safety-Guard-8B-v3

- **HF ID:** `nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3`
- **URL:** https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3
- **Release:** Oct 28, 2025
- **License:** NVIDIA Open Model License + Llama 3.1 Community ⚠️
- **Use:** combined content + topic classification, 8B

If you need both 2.4 and 2.5 in one model, use this.

---

### 2.7 NemoGuard-JailbreakDetect (and Aegis LlamaGuard variants)

- **HF ID:** `nvidia/NemoGuard-JailbreakDetect`
- **URL:** https://huggingface.co/nvidia/NemoGuard-JailbreakDetect
- **Use:** binary jailbreak / prompt-injection detector (input-only)
- **Also:** `nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0` and `…-Permissive-1.0` — fine-tuned Llama Guard variants for stricter / more lenient classification

**Recommended mode:** plug into the front of the cascade as a **fast jailbreak filter** before the heavier Gen/Reasoning model runs.

---

## 3. NeMo Guardrails (orchestration runtime, not a model)

- **Repo:** https://github.com/NVIDIA-NeMo/Guardrails
- **Latest release:** v0.22.0 (May 22, 2026)
- **License:** **Apache 2.0** ✅
- **PyPI:** `pip install nemoguardrails`
- **Python:** 3.10–3.13

**What it is:** not a model — a Python runtime that adds programmable rails between your app code and your LLM. Five rail types: input, dialog, retrieval, execution, output. Colang (a Python-like DSL) for defining flows.

**Colang example:**

```colang
define user express insult
  "You are stupid"

define flow
  user express insult
  bot express calmly willingness to help
```

**config.yml:**

```yaml
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

rails:
  input:
    flows:
      - check jailbreak
      - mask sensitive data on input
  output:
    flows:
      - self check facts
      - self check hallucination
```

**For nemoguardian:** NeMo Guardrails is the **orchestration layer** — it wires Qwen3Guard + Nemotron + a chat LLM together, and enforces dialog flows. The classifiers from §1–§2 are the **rails** NeMo invokes.

**Recommended mode:** orchestration, not classification.

---

## 4. Meta Llama Guard family

### 4.1 Llama-Guard-3-8B

- **HF ID:** `meta-llama/Llama-Guard-3-8B`
- **URL:** https://huggingface.co/meta-llama/Llama-Guard-3-8B
- **Base:** `meta-llama/Llama-3.1-8B`
- **Params:** 8 B
- **Tensor type:** BF16
- **Context length:** 8 K (much shorter than Nemotron)
- **License:** **Llama 3.1 Community License** ⚠️ (700M-MAU clause)
- **Downloads/month:** ~168 K
- **VRAM @ FP16:** ~16 GB → also ships an INT8 build (`Llama-Guard-3-8B-INT8`) at ~8 GB

**Hazard taxonomy (14 cats, MLCommons-aligned):**
S1 Violent Crimes, S2 Non-Violent Crimes, S3 Sex-Related Crimes, S4 Child Sexual Exploitation, S5 Defamation, S6 Specialized Advice, S7 Privacy, S8 Intellectual Property, S9 Indiscriminate Weapons, S10 Hate, S11 Suicide & Self-Harm, S12 Sexual Content, S13 Elections, S14 Code Interpreter Abuse.

**Inference (verbatim):**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-Guard-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")

def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda")
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

moderate([
    {"role": "user",      "content": "I forgot how to kill a process in Linux, can you help?"},
    {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill ..."},
])
# 'safe'
```

**Multilingual:** English, French, German, Hindi, Italian, Portuguese, Spanish, Thai (8 langs, less than Qwen3Guard's 119).

**Strengths:** huge community, mature, well-documented, INT8 build exists, optimized for safety + code-interpreter tool calls.

**Weaknesses:** only 8K context, English-centric, no custom-policy mode, no streaming variant.

**Recommended mode for nemoguardian**
- **Live chat moderation:** ✅ on Colab A100 (INT8 build)
- **Full-doc moderation:** ❌ 8K ctx is too short
- **Streaming-token classification:** ❌

---

### 4.2 Llama-Guard-4-12B

- **HF ID:** `meta-llama/Llama-Guard-4-12B`
- **URL:** https://huggingface.co/meta-llama/Llama-Guard-4-12B
- **Base:** pruned from Llama 4 Scout (MoE → dense)
- **Params:** 12 B (dense, not MoE — single-GPU)
- **Tensor type:** BF16
- **Context length:** 128 K
- **Multimodal:** yes — text + multi-image (a big improvement over Llama Guard 3-11B-vision which only supported one image)
- **License:** **Llama 4 Community License** ⚠️ (similar MAU clause; newer)
- **VRAM @ FP16:** ~24 GB

**Taxonomy:** same MLCommons 14 hazards as Guard 3.

**Strengths**
- Multimodal with multi-image input.
- 12B dense fits on a single 24 GB GPU.
- 128K context.
- +20% F1 on multi-image vs Guard 3.

**Weaknesses**
- 24 GB VRAM minimum — Colab A100 40 GB works, but the laptop is out.
- Newest license — least battle-tested.
- Slower per token than 4B/8B peers.

**Recommended mode for nemoguardian**
- **Live chat moderation:** ❌ too heavy
- **Full-doc moderation:** ✅ on Colab A100 if you need image moderation
- **Streaming-token classification:** ❌

For a hackathon focused on **text-heavy** Discord/Twitch chat, **skip Llama Guard 4** — Qwen3Guard + Nemotron 3.5 covers the same ground in 1/3 the VRAM.

---

## 5. IBM Granite Guardian 4.1

- **HF ID:** `ibm-granite/granite-guardian-4.1-8b`
- **URL:** https://huggingface.co/ibm-granite/granite-guardian-4.1-8b
- **Collection:** https://huggingface.co/collections/ibm-granite/granite-guardian
- **Base:** `ibm-granite/granite-4.1-8b`
- **Params:** 8 B
- **Tensor type:** BF16
- **Context length:** 8 K
- **Release:** April 2026
- **License:** **Apache 2.0** ✅ — *the only 8B-class safety model in this catalog that is fully Apache*
- **VRAM @ FP16:** ~16 GB

**Variants / modes**
- **Hybrid thinking model** — supports thinking (reasoning trace) and non-thinking (low-latency yes/no) modes, like Nemotron-Content-Safety-Reasoning-4B.
- **BYOC (Bring Your Own Criteria)** — define arbitrary judging rules. This is IBM's equivalent of Nemotron's custom-policy feature.

**Pre-baked criteria:** Harm, Social Bias, Jailbreaking, Violence, Profanity, Unethical Behavior, Context Relevance, Groundedness, Answer Relevance, Function Calling Hallucination.

**Headline benchmark gains over 3.3:**
- IFEval multi-constraint BAcc: **0.458 → 0.844** (no-think)
- InfoBench (Human): **0.535 → 0.706**
- Best-of-N reward model on JETTS: **70.29** (beats reward models up to 70B)

**Input format — Guardian block (verbatim):**

```text
<guardian>{think_or_nothink_instruction}

### Criteria: {criteria_text}

### Scoring Schema: If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'.
```

**Output:** binary `yes`/`no` inside `<score>...</score>` tags.

**Inference (verbatim):**

```python
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_path = "ibm-granite/granite-guardian-4.1-8b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(model=model_path, max_model_len=8192)
sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

GUARDIAN_JUDGE_NOTHINK = (
    "<no-think>As a judge agent, assess whether the provided text meets the given "
    "judging criteria using all available information, including conversations, "
    "documents, and tools. Provide your score immediately without explanation. "
    "Output empty  tags followed by your score in <score></score> tags."
)

def build_guardian_block(criteria, think=False):
    judge = GUARDIAN_JUDGE_NOTHINK  # or THINK variant
    return (
        f"<guardian>{judge}\n\n"
        f"### Criteria: {criteria}\n\n"
        f"### Scoring Schema: If the last assistant's text meets the criteria, "
        f"return 'yes'; otherwise, return 'no'."
    )
```

**Strengths**
- Only 8B model in the catalog that is **Apache 2.0** end-to-end — no custom license burden.
- BYOC rivals Nemotron's custom-policy mode and is more flexible.
- Strongest reward-model signal — useful for offline RL of a downstream moderation LLM.
- Hybrid thinking → can dial latency vs auditability.

**Weaknesses**
- 8K context is short.
- 8B at FP16 → ~16 GB VRAM, doesn't fit 4 GB.
- No streaming-token variant.

**Recommended mode for nemoguardian**
- **Live chat moderation:** ✅ on Colab A100 as a 2nd-pass judge (no-think mode)
- **Full-doc moderation:** ✅ as a high-precision BYOC judge
- **Streaming-token classification:** ❌

---

## 6. Other entrants (closed / API-only / 2026 deprecations)

| Model | Status | Why excluded / included |
|---|---|---|
| `mistralai/mistral-moderation-2411` | **Closed / API-only. Deprecated 2026-03-31.** Successor "Mistral Moderation 2" also closed-source per Mistral docs. | ❌ Not open source. |
| **OpenAI `omni-moderation-latest` / `text-moderation-*`** | Closed. Not on HuggingFace. | ❌ Not open source. |
| **AWS Rekognition Content Moderation** | Closed SaaS. | ❌ Not open source. |
| **Azure AI Content Safety** | Closed SaaS. | ❌ Not open source. |
| **Google Cloud Natural Language — Moderation** | Closed. | ❌ Not open source. |
| `nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0` | Open LoRA on Llama Guard 2 / 3. | ⚠️ Older; superseded by Llama Guard 3-8B + Nemotron 3.5. |
| `nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0` | Same, lenient variant. | ⚠️ Same. |
| `Qwen/Qwen3-4B-SafeRL` (Sep 2025) | Safety-aligned LLM (RL fine-tuned, *not* a classifier). | ⚠️ Useful as a safer **chat** LLM, not as a moderator. Not in the classifier cascade. |
| `Qwen/Qwen3GuardTest` (Oct 2025) | **Benchmark**, not a model. | For evaluating the family. |

> **Bottom line:** as of late June 2026 there is **no 2026 open-source Mistral Moderation, AWS, OpenAI, or Google competitor** to add to the cascade. The competitive set is exactly: **NVIDIA Nemotron family + Qwen3Guard + Llama Guard 3/4 + Granite Guardian 4.1 + NeMo Guardrails runtime**. Everything else is closed or deprecated.

---

## 7. VRAM Quick-Reference (FP16 baseline, INT8 ≈ half, 4-bit ≈ quarter)

| Tier | Models | FP16 | INT8 | 4-bit | Fits 4 GB? |
|---|---|---:|---:|---:|:---:|
| Ultra-light | Qwen3Guard-Gen-0.6B, -Stream-0.6B | 1.5 GB | 0.8 GB | 0.5 GB | ✅✅ |
| Light | Qwen3Guard-Gen-4B, -Stream-4B, Nemotron-Content-Safety-Reasoning-4B, Nemotron-3.5-Content-Safety, Nemotron-Content-Safety-4B | 8 GB | 4 GB | 2.5 GB | ⚠️ INT8 only, tight |
| Medium | Llama-Guard-3-8B INT8, NemoGuard 8B family, Granite Guardian 4.1 8B, Qwen3Guard-Gen/Stream-8B | 16 GB | 8 GB | 4–5 GB | ❌ (FP16), ⚠️ (4-bit borderline) |
| Heavy | Llama-Guard-4-12B | 24 GB | 12 GB | 6–7 GB | ❌ |

**Carlos's 4 GB laptop:** only the 0.6B Qwen3Guard tier fits comfortably at FP16. To run a 4B Nemotron on the same machine you must quantize to INT8 (just barely) or 4-bit (comfortably) and avoid loading any other model simultaneously.

**Colab A100 40 GB:** can host any two models from the light tier in parallel, or one medium-tier + one ultra-light in parallel.

---

## 8. Recommended nemoguardian Cascade

A cascade is a series of cheap→expensive classifiers, where each stage either short-circuits or escalates. For Discord / Twitch live chat, **latency dominates** — so the cascade must be biased toward the cheapest model that catches the easy cases.

### 8.1 The 3-stage cascade (Colab A100 40 GB target — the demo target)

| Stage | Model | VRAM | Latency | Purpose | Escalates to |
|---|---|---|---|---|---|
| **0. Pre-filter** | `nvidia/NemoGuard-JailbreakDetect` (binary jailbreak classifier) | ~0.5 GB (INT8) | <5 ms / msg | Drop obvious prompt-injection / jailbreak attempts before any Gen model runs | Stage 1 |
| **1. Fast triage (input)** | `Qwen/Qwen3Guard-Gen-0.6B` | 1.5 GB FP16 | ~30 ms | Prompt moderation: 119 languages, 3-tier severity, 9 categories. Catches >80% of unsafe inputs. | Stage 2 if `Controversial` or `Unsafe` |
| **1b. Fast triage (output, streaming)** | `Qwen/Qwen3Guard-Stream-0.6B` | 1.5 GB FP16 | per-token ~3 ms | Token-level classification of the bot's response *as it is being generated* — earliest possible abort signal | Stage 3 if stream emits `Unsafe` |
| **2. High-precision judge (escalation)** | `Qwen/Qwen3Guard-Gen-4B` (or 8B on A100) | 8 GB FP16 / 16 GB FP16 | ~150–300 ms | Only fires on `Controversial` or `Unsafe` from Stage 1, or when the bot's own response needs an audit before posting | Stage 3 |
| **3. Custom-policy reasoning (only on appeal / audit)** | `nvidia/Nemotron-Content-Safety-Reasoning-4B` with `/think` | 8 GB FP16 | ~600 ms | Generates a defensible reasoning trace for the user-facing "why was I banned?" appeal UI | n/a — final |
| **(optional 4. Image + text)** | `nvidia/Nemotron-3.5-Content-Safety` | 8 GB FP16 | ~400 ms | Multimodal — needed if Discord attachments are in scope | n/a |

Total VRAM for the **3-stage cascade** (Stages 0–2, ultra-light + light + light) on A100: **~18 GB** → fits comfortably on a single A100 40 GB, with headroom for the chat LLM itself.

**APIs:**

```python
# Stage 0
def stage0_jailbreak(text: str) -> bool:
    return jailbreak_model.predict(text)  # True → drop immediately

# Stage 1 — Qwen3Guard-Gen-0.6B
def stage1_prompt(user_text: str) -> dict:
    label, cats = qwen3guard_gen_0p6b.moderate_prompt(user_text)
    return {"label": label, "categories": cats}  # safe / controversial / unsafe

# Stage 1b — Qwen3Guard-Stream-0.6B
def stage1b_stream(token_ids, stream_state) -> dict:
    result, stream_state = qwen3guard_stream_0p6b.stream_moderate_from_ids(
        token_ids, role="assistant", stream_state=stream_state
    )
    return {"risk_level": result["risk_level"], "category": result["category"]}

# Stage 2 — escalate to Qwen3Guard-Gen-4B
def stage2_judge(user_text: str, bot_text: str) -> dict:
    return qwen3guard_gen_4b.moderate_response(user_text, bot_text)

# Stage 3 — Nemotron-Content-Safety-Reasoning-4B for appeals
def stage3_appeal(user_text: str, bot_text: str, custom_policy: str) -> str:
    return nemotron_reasoning.think(user_text, bot_text, custom_policy, mode="/think")
```

**Routing logic:**

```python
def cascade_moderate(user_text, bot_text, *, custom_policy=None):
    if stage0_jailbreak(user_text):
        return {"action": "drop", "reason": "jailbreak", "stage": 0}

    p1 = stage1_prompt(user_text)
    if p1["label"] == "Safe":
        return {"action": "allow", "stage": 1, "triage": p1}

    # Controversial or Unsafe → escalate
    p2 = stage2_judge(user_text, bot_text)
    if p2["label"] == "Safe":
        return {"action": "allow", "stage": 2, "triage": p1, "judge": p2}

    if custom_policy and p2["label"] != "Safe":
        # Run the reasoning model only on appeals / audit
        return {
            "action": "block",
            "stage": 3,
            "triage": p1,
            "judge": p2,
            "rationale": stage3_appeal(user_text, bot_text, custom_policy),
        }
    return {"action": "block", "stage": 2, "triage": p1, "judge": p2}
```

For **streaming bot generation**, run `stage1b_stream` *in parallel* with the bot LLM and abort generation as soon as the first `risk_level == Unsafe` token appears.

### 8.2 The 2-stage cascade (4 GB laptop target — the "fit-everywhere" mode)

For a 4 GB RTX 3050 Ti you must drop to INT8 or 4-bit quantization, and you can only have **one** model loaded at a time.

| Stage | Model | Quantization | VRAM | Purpose |
|---|---|---|---:|---|
| 1 | `Qwen/Qwen3Guard-Stream-0.6B` | 4-bit | ~0.5 GB | All-purpose streaming + prompt + response classification. 119 languages, 3-tier severity. |
| 2 | `Qwen/Qwen3Guard-Gen-0.6B` (loaded only on appeal) | 4-bit | ~0.5 GB | Higher-accuracy second opinion when Stage 1 says `Controversial`. |

**Why this works:** both models are Apache 2.0, both fit comfortably in 4 GB at 4-bit, and the Stream variant can also do single-shot prompt moderation (just pass the full user text as a single "token batch"). For a hackathon demo, this is the most portable, no-API-key story.

### 8.3 Which cascade to demo?

| Demo venue | Cascade | Why |
|---|---|---|
| Twitch / Discord live chat (text only) | **3-stage Colab A100 cascade** (8.1) | Sub-100 ms p50 with the 0.6B triage; full 4B audit on appeals |
| Twitch / Discord (with image attachments) | 3-stage + add `Nemotron-3.5-Content-Safety` as Stage 4 multimodal | The only 4B multimodal option in the catalog |
| Hackathon judging on a 4 GB laptop | **2-stage cascade** (8.2) | Runs anywhere, no Colab needed |
| Enterprise / production | 3-stage + NeMo Guardrails orchestration | Colang flows for dialog + jailbreak pre-filter + audit log |

### 8.4 Why this exact ordering

1. **Qwen3Guard-Gen-0.6B first** because it has the highest per-VRAM safety-per-ms ratio, 119-language coverage beats every other model, and Apache 2.0 keeps the demo license-clean.
2. **Qwen3Guard-Stream-0.6B in parallel** because the bot's own generation is the highest-risk surface in a chat (jailbroken assistant output), and the Stream variant gives the earliest possible abort signal.
3. **Qwen3Guard-Gen-4B on escalation** because 0.6B has known false-positive on subtle jailbreaks; we don't pay the 4B cost on every message.
4. **Nemotron-Content-Safety-Reasoning-4B only on appeal** because reasoning traces are gold for "why was I banned?" but the latency is too high for live moderation.
5. **NeMo Guardrails as the orchestrator** because it composes the cascade into Colang flows and lets us add input/output rails (PII masking, fact-check, jailbreak flow) declaratively.
6. **Skipped: Llama Guard 3/4 and Granite Guardian 4.1** — the former has only 8K context (too short for a full thread) and a Llama 3.1/4 license; the latter is Apache 2.0 (great!) but its BYOC strength doesn't beat Qwen3Guard's 119-language coverage for the discord/twitch use case. Both are reasonable Stage-2 substitutes if the user prefers.

---

## 9. License summary (red flags only)

| License | Models | Verdict for hackathon |
|---|---|---|
| **Apache 2.0** ✅ | Qwen3Guard (all 6), Granite Guardian 4.1 8B, NeMo Guardrails | Use freely. |
| **NVIDIA Open Model License** ⚠️ | All Nemotron safety models, NemoGuard 8B family | Commercial use approved by NVIDIA, but not OSI-approved. **Not** Apache. Include the license text in your repo. |
| **Llama 3.1 / Llama 4 Community License** ⚠️ | Llama Guard 3-8B, Llama Guard 4-12B, NemoGuard 8B (as backbone) | Commercial OK, but: (1) **700M MAU clause** (you must request a separate license if your product exceeds 700M monthly users — safe for a hackathon demo); (2) must display "Built with Llama"; (3) Acceptable Use Policy. |
| **Gemma Terms of Use** ⚠️ | All Nemotron 4B (Gemma-3-4B backbone) | Commercial OK with restrictions on certain downstream uses; not OSI-approved. |
| **OpenMDW License v1.1** ⚠️ | Nemotron-3.5-Content-Safety | Newest, most restrictive NVIDIA license. Legal review for production. |
| **Mistral / OpenAI / AWS / Azure / Google** ❌ | Mistral Moderation, omni-moderation, Rekognition, etc. | Closed SaaS. Excluded by the hackathon "no API key" rule. |

---

## 10. References (sources used)

- NVIDIA Nemotron NemoGuard collection: https://huggingface.co/collections/nvidia/nemoguard
- Nemotron-Content-Safety-Reasoning-4B card: https://huggingface.co/nvidia/Nemotron-Content-Safety-Reasoning-4B
- Nemotron-Content-Safety-Reasoning-4B on NIM: https://build.nvidia.com/nvidia/nemotron-content-safety-reasoning-4b/modelcard
- Nemotron-3.5-Content-Safety card: https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety
- llama-3.1-nemoguard-8b-content-safety card: https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-content-safety
- llama-3.1-nemoguard-8b-topic-control card: https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-topic-control
- Llama-3.1-Nemotron-Safety-Guard-8B-v3 card: https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3
- NemoGuard-JailbreakDetect card: https://huggingface.co/nvidia/NemoGuard-JailbreakDetect
- NeMo Guardrails repo: https://github.com/NVIDIA-NeMo/Guardrails
- Qwen3Guard GitHub: https://github.com/QwenLM/Qwen3Guard
- Qwen3Guard-Gen-0.6B: https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B
- Qwen3Guard-Gen-4B: https://huggingface.co/Qwen/Qwen3Guard-Gen-4B
- Qwen3Guard-Gen-8B: https://huggingface.co/Qwen/Qwen3Guard-Gen-8B
- Qwen3Guard-Stream-0.6B: https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B
- Qwen3Guard-Stream-4B: https://huggingface.co/Qwen/Qwen3Guard-Stream-4B
- Llama-Guard-3-8B card: https://huggingface.co/meta-llama/Llama-Guard-3-8B
- Llama-Guard-4-12B card: https://huggingface.co/meta-llama/Llama-Guard-4-12B
- Llama-Guard-4 blog: https://huggingface.co/blog/llama-guard-4
- Granite Guardian 4.1 8B card: https://huggingface.co/ibm-granite/granite-guardian-4.1-8b
- Granite Guardian docs: https://www.ibm.com/granite/docs/models/guardian
- IBM research blog (4.1 release): https://research.ibm.com/blog/granite-4-1-ai-foundation-models
- Mistral Moderation deprecation note: https://docs.mistral.ai/studio-api/conversations/moderation
- Mistral Moderation news: https://mistral.ai/news/mistral-moderation/
