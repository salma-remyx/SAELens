# Remyx eval environment for this validation (reviewed by hand).
#
# Why this file exists: the CPU-tier build timed out twice (kaniko, 20 min).
# On python:3.12-slim, `pip install -e .` resolves transformer-lens -> torch
# from PyPI, which is the CUDA build plus the nvidia libraries (~3 GB). This
# eval runs on CPU, so install the CPU wheel first; the repo install then
# finds torch satisfied and leaves it alone.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

WORKDIR /workspace
RUN git clone --depth=1 https://github.com/salma-remyx/SAELens /workspace/target_repo && \
    cd /workspace/target_repo && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir wandb
