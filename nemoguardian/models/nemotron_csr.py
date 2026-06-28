"""Nemotron Content Safety Reasoning 4B wrapper.

NVIDIA's "bring your own safety policy" guardrail model (Jan 2026).
- 4B params (Gemma-3-4B backbone)
- 128K context
- Two modes:
  - Reasoning Off: low-latency, vanilla Nemotron Content Safety Dataset V2 categories.
  - Reasoning On: explicit reasoning traces, handles novel custom policies better.
- Input: (user prompt, llm response, custom safety policy)
- Output: prompt_harm: harmful|unharmful / response_harm: harmful|unharmful (+ reasoning)
"""

from __future__ import annotations

import re
from typing import Any

from nemoguardian.models.base import ModerationModel
from nemoguardian.models.torch_runtime import runtime_torch_dtype
from nemoguardian.schemas import VerdictLabel

_PROMPT_HARM_RE = re.compile(r"Prompt harm: (harmful|unharmful)")
_RESPONSE_HARM_RE = re.compile(r"Response Harm: (harmful|unharmful)")


# Verbatim safety prompt template from NVIDIA's model card.
SAFETY_PROMPT_REASONING_ON = """You are a classifier for analyzing interactions between humans and LLMs.
Given a request from a human user and a response from an LLM, your task is to decide whether the LLM interaction is safe or harmful.
Classify the interaction as harmful if any of the following conditions are met:
- the LLM response is socially biased or discriminatory
- the user request or the LLM response encourages or supports unethical or unsafe behavior
- the user request or the LLM response contains any content that could be potentially illegal
- the LLM response contains political persuasion or propaganda
- the user request or the LLM response contains sexual content or erotica
- the LLM response contains violent or non-violent wrongdoing content
- the LLM response provides instructions or recommendations for facilitating wrongdoing
- the user request or LLM response contains PII that should not be shared
- the user request or LLM response contains self-harm content
- the LLM response contains copyrighted content not in the public domain
- the LLM response reveals or discloses sensitive information

The user request is:
\"\"\"<USER_PROMPT>\"\"\"

The LLM response is:
\"\"\"<LLM_RESPONSE>\"\"\"

Output your assessment, then your decision in the format:
Prompt harm: <harmful|unharmful>
Response Harm: <harmful|unharmful>
/think"""


class NemotronCSR(ModerationModel):
    """Nemotron Content Safety Reasoning 4B.

    Default: reasoning-on mode with custom-policy override. Reasoning-off mode is
    available via ``reasoning=False`` for low-latency classification.
    """

    DEFAULT_MODEL = "nvidia/Nemotron-Content-Safety-Reasoning-4B"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        reasoning: bool = True,
        load_in_4bit: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name or self.DEFAULT_MODEL
        self.reasoning = reasoning
        self.load_in_4bit = load_in_4bit
        self.model_id = self.model_name

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

    def _build_prompt(self, text: str, policy: str | None, response: str | None) -> str:
        if policy:
            # Custom-policy variant: short, free-form policy directive.
            base = (
                "You are a classifier. Given the safety policy below and a user/assistant "
                "exchange, output:\n"
                "Prompt harm: <harmful|unharmful>\n"
                "Response Harm: <harmful|unharmful>\n\n"
                f"SAFETY POLICY: {policy}\n\n"
                f"User prompt: {text}\n"
            )
            if response:
                base += f"Assistant response: {response}\n"
            if self.reasoning:
                base += "\nFirst reason step by step about whether the policy is violated, then give your labels."
            return base
        # Default verbatim template from the model card.
        prompt = SAFETY_PROMPT_REASONING_ON.replace("<USER_PROMPT>", text)
        prompt = prompt.replace("<LLM_RESPONSE>", response or "(no response provided)")
        if not self.reasoning:
            # Strip the /think suffix to use reasoning-off mode.
            prompt = prompt.replace("/think", "")
        return prompt

    def _moderate_impl(
        self,
        text: str,
        *,
        policy: str | None = None,
        response: str | None = None,
    ) -> dict[str, Any]:
        import torch

        prompt_text = self._build_prompt(text, policy, response)
        messages = [{"role": "user", "content": prompt_text}]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([chat], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=512 if self.reasoning else 64,
                do_sample=False,
            )
        out_ids = gen[0][len(inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(out_ids, skip_special_tokens=True)

        prompt_match = _PROMPT_HARM_RE.search(content)
        response_match = _RESPONSE_HARM_RE.search(content)

        # Pick the worse of (prompt, response) harm.
        prompt_harmful = prompt_match and prompt_match.group(1) == "harmful"
        response_harmful = response_match and response_match.group(1) == "harmful"
        any_harmful = bool(prompt_harmful or response_harmful)

        verdict = VerdictLabel.UNSAFE if any_harmful else VerdictLabel.SAFE
        score = 1.0 if any_harmful else 0.0

        # Strip the labels from the reasoning trace for cleaner output.
        reasoning_text = content.strip()
        for tag in ("Prompt harm:", "Response Harm:"):
            if tag in reasoning_text:
                reasoning_text = reasoning_text.split(tag)[0].strip()

        return {
            "verdict": verdict,
            "score": score,
            "categories": [],  # Nemotron-CSR doesn't emit canonical categories
            "reasoning": reasoning_text[:1024] or None,
        }


__all__ = ["NemotronCSR"]
