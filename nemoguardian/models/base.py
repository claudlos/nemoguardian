"""Base protocol for moderation models.

Each model wraps one HuggingFace checkpoint. Implementations live in qwen3_guard.py
and nemotron_csr.py. All models must:

- report their HF id
- load lazily (only on first call)
- expose `moderate(...)` returning a ModelVerdict
- never raise on inference errors — surface them via `error` field
"""

from __future__ import annotations

import abc
import threading
import time
from typing import Any

from nemoguardian.schemas import ModelVerdict, VerdictLabel

# Process-wide lock serializing first-time model loads. The cascade loads the
# local guards concurrently, and a transformers _LazyModule does NOT tolerate
# two threads doing their first `from transformers import ...` at once (raises
# "cannot import name 'AutoModelForCausalLM'"). Serializing only the load keeps
# inference fully concurrent — after warmup the fast path skips the lock.
_LOAD_LOCK = threading.Lock()


class ModerationModel(abc.ABC):
    """Abstract moderation model."""

    model_id: str  # subclasses set this

    def __init__(self) -> None:
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        if not self._loaded:
            with _LOAD_LOCK:
                if not self._loaded:  # double-checked: only the first thread loads
                    self._load()
                    self._loaded = True

    @abc.abstractmethod
    def _load(self) -> None:
        """Download/initialize the model. Subclasses implement."""

    @abc.abstractmethod
    def _moderate_impl(
        self,
        text: str,
        *,
        policy: str | None = None,
        response: str | None = None,
    ) -> dict[str, Any]:
        """Run inference. Return dict matching ModelVerdict fields minus id+latency."""

    def moderate(
        self,
        text: str,
        *,
        policy: str | None = None,
        response: str | None = None,
        **kwargs: Any,
    ) -> ModelVerdict:
        """Public entry point. Times the call and surfaces errors.

        Extra keyword args are forwarded to ``_moderate_impl`` so a specific
        model can accept per-call options (e.g. Nemotron-CSR's ``reasoning``).
        """
        try:
            self.ensure_loaded()
            start = time.perf_counter()
            payload = self._moderate_impl(text, policy=policy, response=response, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000.0
            verdict = payload.get("verdict", VerdictLabel.SAFE)
            if isinstance(verdict, str):
                verdict = VerdictLabel(verdict.lower())
            # An impl may flag its own output as unusable (e.g. a reasoning model
            # that ran out of tokens before emitting a label). Surface it via
            # ``error`` so the aggregator DROPS this vote instead of counting a
            # silent "safe" — see aggregator fail-safe handling.
            return ModelVerdict(
                model_id=self.model_id,
                verdict=verdict,
                score=float(payload.get("score", 0.0)),
                categories=list(payload.get("categories", [])),
                reasoning=payload.get("reasoning"),
                latency_ms=latency_ms,
                error=payload.get("error"),
            )
        except Exception as exc:
            return ModelVerdict(
                model_id=self.model_id,
                verdict=VerdictLabel.SAFE,  # fail-open: better wrong-safe than wrong-block
                score=0.0,
                categories=[],
                reasoning=None,
                latency_ms=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
