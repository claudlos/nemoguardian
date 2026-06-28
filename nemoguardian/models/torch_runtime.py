"""Torch runtime helpers for local model loading."""

from __future__ import annotations

import os
from typing import Any


def runtime_torch_dtype(torch_module: Any) -> Any:
    """Pick a dtype that works on the active runtime.

    Hugging Face ``auto`` may select bfloat16 for checkpoints that advertise it,
    but CPU inference can fail on operations that do not support bfloat16. The
    default here is conservative: float32 on CPU and float16 on CUDA.
    """

    override = os.environ.get("NEMOGUARDIAN_TORCH_DTYPE")
    if override:
        normalized = override.strip().lower()
        if normalized == "auto":
            return "auto"
        names = {
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp16": "float16",
            "float16": "float16",
            "fp32": "float32",
            "float32": "float32",
        }
        dtype_name = names.get(normalized)
        if dtype_name is None or not hasattr(torch_module, dtype_name):
            raise ValueError(f"unsupported NEMOGUARDIAN_TORCH_DTYPE: {override}")
        return getattr(torch_module, dtype_name)

    cuda = getattr(torch_module, "cuda", None)
    try:
        cuda_available = bool(cuda is not None and cuda.is_available())
    except Exception:
        cuda_available = False

    if cuda_available and hasattr(torch_module, "float16"):
        return torch_module.float16
    return getattr(torch_module, "float32", "auto")


__all__ = ["runtime_torch_dtype"]
