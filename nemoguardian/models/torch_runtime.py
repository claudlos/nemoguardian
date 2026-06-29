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


def dtype_kwarg_name() -> str:
    """Name of the ``from_pretrained`` dtype kwarg for the installed transformers.

    transformers >= 4.56 renamed ``torch_dtype`` -> ``dtype`` and warns on the
    old name; older releases (the repo floor is 4.51) only accept ``torch_dtype``.
    Falls back to the legacy name if the version can't be determined (e.g. tests
    that inject a stub transformers module).
    """
    try:
        import transformers

        major, minor, *_ = (int(p) for p in transformers.__version__.split(".")[:2])
        if (major, minor) >= (4, 56):
            return "dtype"
    except Exception:
        pass
    return "torch_dtype"


def dtype_kwargs(torch_module: Any) -> dict[str, Any]:
    """``{<dtype-kwarg>: <runtime dtype>}`` for the installed transformers."""
    return {dtype_kwarg_name(): runtime_torch_dtype(torch_module)}


def attn_impl_kwargs() -> dict[str, str]:
    """Optional ``attn_implementation`` for ``from_pretrained``.

    Opt-in via ``NEMOGUARDIAN_ATTN_IMPL`` (e.g. ``flash_attention_2`` or ``sdpa``)
    to speed up the local 4B guards on capable GPUs. Empty by default so the
    transformers default is used and CPU/test runs are unaffected.
    """
    impl = os.environ.get("NEMOGUARDIAN_ATTN_IMPL", "").strip()
    return {"attn_implementation": impl} if impl else {}


__all__ = [
    "attn_impl_kwargs",
    "dtype_kwarg_name",
    "dtype_kwargs",
    "runtime_torch_dtype",
]
