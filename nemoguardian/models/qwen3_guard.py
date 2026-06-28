"""Qwen3Guard wrappers — Gen (full doc) and Stream (token-level).

Qwen3Guard is the Qwen team's safety classification family (Sep 2025).
- Gen variants: instruction-following classification of full prompts/responses.
- Stream variants: classification head attached to the transformer for real-time
  per-token scoring during generation.
- Three sizes: 0.6B, 4B, 8B (base Qwen3).
- Three-tier label: Safe / Controversial / Unsafe.
- 119 languages.
- 9 categories: Violent, Non-violent Illegal Acts, Sexual Content, PII,
  Suicide & Self-Harm, Unethical Acts, Politically Sensitive, Copyright, Jailbreak.
"""

from __future__ import annotations

import re
from typing import Any

from nemoguardian.models.base import ModerationModel
from nemoguardian.models.torch_runtime import runtime_torch_dtype
from nemoguardian.schemas import VerdictLabel

# Label and category regexes (verbatim from Qwen3Guard model card)
_LABEL_RE = re.compile(r"Safety: (Safe|Unsafe|Controversial)")
_CATEGORY_RE = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)"
)
_REFUSAL_RE = re.compile(r"Refusal: (Yes|No)")


def _parse_gen_output(text: str) -> dict[str, Any]:
    label_match = _LABEL_RE.search(text)
    categories = _CATEGORY_RE.findall(text)
    refusal_match = _REFUSAL_RE.search(text)
    label = label_match.group(1) if label_match else "Safe"
    return {
        "label": label,
        "categories": [c for c in categories if c != "None"],
        "refusal": refusal_match.group(1) if refusal_match else None,
    }


class Qwen3GuardGen(ModerationModel):
    """Qwen3Guard-Gen: generative classifier of full prompts and responses.

    Default size: 4B (FP16 ~8GB, INT4 ~2.5GB). For 4GB GPUs use the 0.6B INT4 variant.
    """

    DEFAULT_MODEL = "Qwen/Qwen3Guard-Gen-4B"
    # For 4GB GPUs (RTX 3050 Ti) use 0.6B at INT4 — fits in ~0.5 GB.

    def __init__(self, model_name: str | None = None, *, load_in_4bit: bool = False) -> None:
        super().__init__()
        self.model_name = model_name or self.DEFAULT_MODEL
        self.load_in_4bit = load_in_4bit
        self.model_id = self.model_name  # reported back in verdicts

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {"torch_dtype": runtime_torch_dtype(torch), "device_map": "auto"}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.eval()

    def _moderate_impl(
        self,
        text: str,
        *,
        policy: str | None = None,
        response: str | None = None,
    ) -> dict[str, Any]:
        import torch

        if response is None:
            # Prompt-only moderation
            messages = [{"role": "user", "content": text}]
        else:
            # Prompt + response moderation (Qwen3Gen format)
            messages = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": response},
            ]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        out_ids = gen[0][len(inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        parsed = _parse_gen_output(content)

        label = parsed["label"]
        verdict = (
            VerdictLabel.UNSAFE
            if label == "Unsafe"
            else VerdictLabel.CONTROVERSIAL
            if label == "Controversial"
            else VerdictLabel.SAFE
        )
        # Map to 0..1 score. Controversial sits in the middle.
        score = 0.0 if verdict == VerdictLabel.SAFE else 0.5 if verdict == VerdictLabel.CONTROVERSIAL else 1.0
        return {
            "verdict": verdict,
            "score": score,
            "categories": parsed["categories"],
            "reasoning": content.strip()[:512],
        }


class Qwen3GuardStream:
    """Qwen3Guard-Stream: real-time token-level classifier.

    Stream variants attach a lightweight classification head to the final layer of
    the Qwen3 transformer and emit a per-token Safe/Controversial/Unsafe label.
    Use this when you need to flag content WHILE an LLM is generating (live chat).
    """

    DEFAULT_MODEL = "Qwen/Qwen3Guard-Stream-0.6B"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self._loaded = False

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self) -> None:
        # The Stream variants use a different model class — handled at runtime
        # to keep this module importable without torch installed.
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            torch_dtype=runtime_torch_dtype(torch),
            device_map="auto",
        )
        self.model.eval()

    def stream_classify(self, text: str):
        """Yield per-token (label, score) tuples. Cheap — ~1ms/token on CPU/GPU."""
        import torch

        self.ensure_loaded()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, T, 3)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        # Per Qwen3Guard mapping: 0=Safe, 1=Controversial, 2=Unsafe
        label_names = ["safe", "controversial", "unsafe"]
        for i, row in enumerate(probs):
            label_idx = int(row.argmax())
            yield label_names[label_idx], float(row[label_idx]), i


__all__ = ["Qwen3GuardGen", "Qwen3GuardStream"]
