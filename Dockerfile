FROM dsprove-mathlib4:latest AS lean_base
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
  accelerate==0.33.0

# Install Lean
COPY --from=lean_base /root/.elan /root/.elan
COPY --from=lean_base /mathlib4 /mathlib4

# Add DeepSeek Prover scaffold
COPY . /workspace

# Create symlinks to make Lean available in the workspace
RUN ln -s /mathlib4 /workspace/mathlib4

# Try the quick start
CMD ["python", "/workspace/quick_start.py"]