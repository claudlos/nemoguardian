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

# Python deps first (cache layer). pyproject.toml is the single source of truth
# for dependencies — the stub __init__ lets `pip install .` resolve them before
# the full source is copied, so this layer only rebuilds when pyproject changes.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY nemoguardian/__init__.py nemoguardian/__init__.py
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install .

ENV PATH="/opt/venv/bin:${PATH}"

# Application code
COPY nemoguardian ./nemoguardian
COPY demo ./demo
COPY docs ./docs
COPY scripts ./scripts
COPY SUBMISSION.md ./
# Re-install the (now complete) package over the dep-resolution stub.
RUN /opt/venv/bin/pip install --no-deps --force-reinstall .

# Create an unprivileged runtime user and give it ownership of the app tree and
# the HuggingFace cache so model weights download on first boot. Build steps run
# as root; only the runtime (uvicorn) drops privileges.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --no-create-home --home-dir /opt/nemoguardian app \
    && mkdir -p /home/app/.cache/huggingface /var/lib/nemoguardian \
    && chown -R app:app /opt/nemoguardian /opt/venv /home/app /var/lib/nemoguardian

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

# Boot with uvicorn. Supply NEMOGUARDIAN_API_KEY at runtime via env or secret.
ENV NEMOGUARDIAN_SELF_HOSTED_EMAIL=self-hosted@nemoguardian.local \
    NEMOGUARDIAN_TIER=self_hosted \
    NEMOGUARDIAN_QUANTIZE=1 \
    NEMOGUARDIAN_QWEN_MODEL=Qwen/Qwen3Guard-Gen-4B \
    NEMOGUARDIAN_QWEN_STREAM_MODEL=Qwen/Qwen3Guard-Stream-0.6B \
    NEMOGUARDIAN_CSR_MODEL=nvidia/Nemotron-Content-Safety-Reasoning-4B \
    NEMOGUARDIAN_TRIAGE_MODEL=nvidia/nemotron-3-ultra-550b-a55b \
    NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT=0 \
    HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    NEMOGUARDIAN_DB_PATH=/var/lib/nemoguardian/nemoguardian.db

USER app

CMD ["uvicorn", "nemoguardian.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
