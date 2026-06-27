FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        git curl ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/nemoguardian

# Python deps first (cache layer)
COPY pyproject.toml ./
COPY nemoguardian/__init__.py nemoguardian/__init__.py
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "pydantic>=2.6" \
        "httpx>=0.27" \
        "pyyaml>=6.0" \
        "transformers>=4.51.0" \
        "torch>=2.2" \
        "accelerate>=0.27" \
        "bitsandbytes>=0.43" \
        "tenacity>=8.2" \
        "structlog>=24.1" \
        "typer>=0.12"

ENV PATH="/opt/venv/bin:${PATH}"

# Application code
COPY nemoguardian ./nemoguardian

# Pre-download default models at build time so first boot is fast.
# Override with NEMOGUARDIAN_SKIP_PREDOWNLOAD=1 for slim builds.
ARG NEMOGUARDIAN_SKIP_PREDOWNLOAD=0
RUN if [ "$NEMOGUARDIAN_SKIP_PREDOWNLOAD" != "1" ]; then \
        python -c "from transformers import AutoTokenizer; \
            from nemoguardian.models import Qwen3GuardGen; \
            Qwen3GuardGen('Qwen/Qwen3Guard-Gen-0.6B').ensure_loaded(); \
            print('Qwen3Guard-Gen-0.6B cached')" || true; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Boot with uvicorn. Customers override NEMOGUARDIAN_API_KEY via env or runtime secret.
ENV NEMOGUARDIAN_API_KEY=${NEMOGUARDIAN_API_KEY:-nmg_default_change_me} \
    NEMOGUARDIAN_TIER=${NEMOGUARDIAN_TIER:-self_hosted} \
    NEMOGUARDIAN_CASCADE_MODE=${NEMOGUARDIAN_CASCADE_MODE:-standard} \
    NEMOGUARDIAN_QUANTIZE=${NEMOGUARDIAN_QUANTIZE:-1}

CMD ["uvicorn", "nemoguardian.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
