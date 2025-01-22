FROM ghcr.io/alexjackson1/mathlib4:deepseek AS lean_base
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Install dependencies including Python 3.10
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  curl \
  git \
  software-properties-common \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y \
  python3.10 \
  python3.10-distutils \
  python3.10-venv \
  psmisc \
  && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
  && rm -rf /var/lib/apt/lists/*

# Create symlinks to make python3.10 the default python3 and python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install base dependencies
RUN pip install --no-cache-dir \
  wheel \
  setuptools \
  numpy==1.26.4 \
  torch==2.2.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# Install flash-attn (there were issues with install without --no-build-isolation)
RUN pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation

# Install remaining dependencies
RUN pip install --no-cache-dir \
  pytz==2022.1 \
  easydict==1.13 \
  transformers==4.40.1 \
  vllm==0.4.1 \
  pandas==1.4.3 \
  tabulate==0.9.0 \
  termcolor==2.4.0 \
  accelerate==0.33.0 \
  hf_transfer

# NOTE: It seems that you cannot elevate privileges in hugging face spaces
USER ubuntu

WORKDIR /workspace

# Install Lean from the pre-built binary
COPY --from=lean_base --chown=ubuntu:ubuntu /root/.elan /home/ubuntu/.elan

# Copy DeepSeek scaffold
COPY --chown=ubuntu=ubuntu . /workspace

# Copy Mathlib4 into workspace
COPY --from=lean_base --chown=ubuntu:ubuntu /mathlib4 /workspace/mathlib4

CMD ["python", "quick_start.py"]