# syntax=docker/dockerfile:1
# Build from the repository root:
#   docker build -t <registry>/amd-swecrew:<tag> .
#
# Grader hard gates this file respects:
#   * FINAL stage is FROM the mandated ROCm base (checked by layer identity -> never --squash)
#   * image <= 60 GiB uncompressed
#   * code at /app/app.py, deps at /app/requirements.txt, optional weights at /models
#   * no secrets baked in (nothing from .env is copied; see .dockerignore)
#
# This image needs `git` (to diff/checkout target repositories) and a coding-capable model:
# code-gen quality matters a lot more here than for the other challenges, so the default model
# is a dedicated code model rather than a general chat model.
ARG BASE_IMAGE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
FROM ${BASE_IMAGE}

ARG INSTALL_VLLM=1
ARG VLLM_SPEC="vllm"
ARG BAKE_MODEL=1
ARG MODEL_ID=Qwen/Qwen2.5-Coder-7B-Instruct

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HARNESS_OUTPUT_DIR=/app/output \
    LOG_FORMAT=json \
    LLM_BACKEND=vllm \
    LLM_SERVE_MODEL=/models/llm \
    LLM_SERVED_NAME=swecrew-coder \
    LLM_BASE_URL=http://127.0.0.1:8000/v1 \
    LLM_VRAM_BUDGET_GIB=36 \
    LLM_MAX_MODEL_LEN=32768 \
    SWECREW_TIME_BUDGET_S=240

WORKDIR /app

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# 1) third-party deps first (best layer caching)
COPY requirements.txt /app/requirements.txt
COPY docker/requirements-gpu.txt /tmp/requirements-gpu.txt
RUN python3 -m pip install -r /app/requirements.txt -r /tmp/requirements-gpu.txt \
 && if [ "$INSTALL_VLLM" = "1" ]; then python3 -m pip install "$VLLM_SPEC" || echo "WARN: vLLM install failed; hf backend will be used"; fi

# 2) model weights baked in so startup never depends on a multi-GB download
RUN if [ "$BAKE_MODEL" = "1" ]; then \
      python3 -c "from huggingface_hub import snapshot_download as s; s('${MODEL_ID}', local_dir='/models/llm', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.tiktoken','merges.txt','vocab.*'])"; \
    else mkdir -p /models; fi

# 3) our code
COPY academy-core /opt/src/academy-core
COPY pyproject.toml app.py /opt/src/swecrew/
COPY src /opt/src/swecrew/src
RUN python3 -m pip install --no-deps /opt/src/academy-core /opt/src/swecrew \
 && cp /opt/src/swecrew/app.py /app/app.py \
 && mkdir -p /app/input /app/output \
 && git config --global user.email "swecrew@local" && git config --global user.name "swecrew" \
 && git config --global --add safe.directory '*'

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s CMD test -f /tmp/academy_ready || exit 1

# Starts the model server during the 10-minute startup budget, then stays alive; the harness runs
# `python3 /app/app.py ...` per item. For the REST API set APP_SERVE_CMD (see docker-compose.yml).
ENTRYPOINT ["python3", "-m", "academy_core.server.entrypoint"]
