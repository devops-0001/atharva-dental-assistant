FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch + Transformers/PEFT/Datasets/Accelerate
RUN pip install --no-cache-dir "torch==2.3.1" \
    "transformers==4.43.3" "peft==0.12.0" "datasets==2.20.0" \
    "accelerate==0.33.0" "tqdm==4.66.5" "bitsandbytes==0.43.1" \
    "sentencepiece==0.2.0" "numpy==1.26.4" "pydantic==2.8.2"

WORKDIR /workspace
COPY training/train_lora.py training/merge_lora.py training/prompt_utils.py /workspace/
