# GPU training image for the SeMoCo tokenizer.
#
# Build (repo root):
#   docker build -t semoco-tokenizer .
#
# The image installs the [soma] extras so live SOMA-X FK works out of the box.
# The SOMA-X submodule must be present in third_party/SOMA-X at build time
# (clone with --recursive, or `git submodule update --init` before building).
#
# Train (single node, 8 GPUs, data root mounted at /data):
#   docker run --rm --gpus all --ipc=host --shm-size=8g \
#     -v /path/to/omega-MotionVerse:/data:ro -v /path/to/runs:/workspace/runs \
#     -e MOTIONVERSE_DATA_ROOT=/data \
#     semoco-tokenizer \
#     torchrun --nproc_per_node=8 -m experiments.train_native \
#       --config experiments/configs/split_fp_w64_s4.yaml \
#       --output-dir runs/semoco_split_fp

ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (single-binary Python package manager).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Full tree (see .dockerignore). third_party/SOMA-X ships inside the image so
# live FK runs without extra mounts.
COPY . /workspace

# Install into the system Python, resolving the [soma] extras
# (smplx / trimesh / warp-lang / rtree) at the same time.
RUN uv pip install --system -e "/workspace[soma]"

# Default to a shell; pass the training command after the image name, e.g.:
#   docker run --rm --gpus all semoco-tokenizer \
#     torchrun --nproc_per_node=8 -m experiments.train_native --config ... --output-dir ...
CMD ["bash"]
