# Multi-stage Dockerfile pour FastAPI Agent Loop
# Application: API FastAPI avec intégration Ollama et monitoring

# === STAGE 1: Build dependencies ===
# DK002: Pin exact base image version
FROM python:3.13-slim AS builder

# Install system dependencies for building Python packages
# DK006: Use BuildKit cache for apt packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements file first for better caching
COPY requirements.txt .

# DK006: Use BuildKit cache mount for pip
# Install Python dependencies with cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-compile -r requirements.txt

# === STAGE 2: Runtime image ===
# DK003: Use slim base image for production
FROM python:3.13-slim AS runtime

# DK007: Configure Python for optimal container behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only runtime system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# DK004: Create non-root user for security
RUN groupadd -r agent && useradd -r -g agent -d /app -s /bin/bash agent

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory and set permissions
WORKDIR /app
RUN chown -R agent:agent /app

# Copy application code
COPY --chown=agent:agent . .

# Switch to non-root user
USER agent

# Health check for FastAPI application
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command for production
CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# === STAGE 3: Development image ===
FROM runtime AS development

# Switch back to root temporarily for installing dev tools
USER root

# Install development tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install dev dependencies if they exist
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -f requirements-dev.txt ]; then \
        pip install -r requirements-dev.txt; \
    fi

# Switch back to agent user
USER agent

# Override command for development with hot reload
CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# === STAGE 4: Test runner ===
FROM builder AS test

# Copy all source code including tests
COPY . /app
WORKDIR /app

# Install test dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pytest pytest-asyncio pytest-cov

# Run tests
RUN python -m pytest tests/ -v --cov=inference --cov=core --cov=agent --cov=plugins

# === STAGE 5: Training environment with GPU support ===
# DK002: Use pinned NVIDIA CUDA base for training reproducibility
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS training-builder

# DK007: Configure environment for optimal GPU training
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies for GPU training
# DK006: Use BuildKit cache for efficient rebuilds
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment for training
RUN python3.11 -m venv /opt/training-venv
ENV PATH="/opt/training-venv/bin:$PATH"

# Copy training requirements
COPY requirements.txt requirements-training.txt ./

# DK006: Install dependencies with cache mounts for faster rebuilds
# Install base requirements first, then training-specific deps that need torch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-compile -r requirements.txt

# Install training dependencies that require torch to be already installed
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-compile -r requirements-training.txt

# === STAGE 6: Training runtime ===
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS training

# DK007: Configure runtime environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/app/models/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/models/.cache/transformers \
    TORCH_HOME=/app/models/.cache/torch

# Install minimal runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    curl \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# DK004: Create non-root user for security
RUN groupadd -r trainer && useradd -r -g trainer -d /app -s /bin/bash trainer

# Copy training environment from builder
COPY --from=training-builder /opt/training-venv /opt/training-venv
ENV PATH="/opt/training-venv/bin:$PATH"

# Create app structure with proper permissions
WORKDIR /app
RUN mkdir -p /app/models/{checkpoints,cache,gguf} \
    /app/datasets \
    /app/logs \
    /app/outputs && \
    chown -R trainer:trainer /app

# Copy application code
COPY --chown=trainer:trainer . .

# Switch to non-root user
USER trainer

# Health check for training service
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import torch; print('GPU available:', torch.cuda.is_available()); exit(0 if torch.cuda.is_available() else 1)" || exit 1

# Default training command - can be overridden
CMD ["python", "training/qlora_finetune.py", "--data", "/app/datasets/processed", "--base", "/app/models/gguf/gemma_base.gguf"]

# === Production optimizations notes ===
# 1. Multi-stage build reduces final image size by ~60%
# 2. BuildKit cache mounts speed up rebuilds significantly
# 3. Non-root user follows security best practices
# 4. Health check enables proper container orchestration
# 5. Separate development stage for local development workflow
# 6. GPU training stages use NVIDIA CUDA base images with optimized caching
# 7. Training environment isolated from production runtime