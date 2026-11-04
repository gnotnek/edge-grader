# edge-grader on the lab's DGX Spark (GB10, aarch64, CUDA 13.0).
# One image for everything: the Python env (uv.lock, finetune + dev groups)
# and llama-server built at the same llama.cpp commit as the Kaggle notebooks.
# Usage: docs/guides/spark-docker.md

ARG CUDA_VERSION=13.0.1
ARG UBUNTU=ubuntu24.04

# --- llama-server, pinned like notebooks/kaggle_baseline.ipynb -------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS llama
ARG LLAMA_CPP_COMMIT=a3a1c4747fdc0dcad40b3946108b89375d9a7d0e
# GB10 is compute capability 12.1. Kaggle builds for 75 (T4).
ARG CUDA_ARCH=121
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/llama.cpp
RUN git init -q && git remote add origin https://github.com/ggml-org/llama.cpp \
    && git fetch -q --depth 1 origin "$LLAMA_CPP_COMMIT" && git checkout -q FETCH_HEAD
# Same cmake options as the notebook; only the CUDA arch differs. There is no
# driver at build time, so link against the toolkit's libcuda stub; libggml-cuda
# asks for libcuda.so.1, so give the stub that name for the final link.
RUN ln -s libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DLLAMA_CURL=OFF \
        -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCUDA_cuda_driver_LIBRARY=/usr/local/cuda/lib64/stubs/libcuda.so \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build build -j"$(nproc)" --target llama-server \
    && git rev-parse --short HEAD > build/bin/COMMIT

# --- runtime ------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
ARG UID=1001
ARG GID=1001
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev git curl ca-certificates libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /usr/local/bin/
COPY --from=llama /opt/llama.cpp/build/bin /opt/llama.cpp/bin
# The LoRA → GGUF converter at the same commit (scripts/spark_finetune.sh gives it its own venv).
COPY --from=llama /opt/llama.cpp/convert_lora_to_gguf.py /opt/llama.cpp/convert_hf_to_gguf.py /opt/llama.cpp/
COPY --from=llama /opt/llama.cpp/conversion /opt/llama.cpp/conversion
COPY --from=llama /opt/llama.cpp/gguf-py /opt/llama.cpp/gguf-py
COPY --from=llama /opt/llama.cpp/requirements /opt/llama.cpp/requirements
ENV PATH=/opt/llama.cpp/bin:$PATH \
    LD_LIBRARY_PATH=/opt/llama.cpp/bin:$LD_LIBRARY_PATH

# Run as the host user so files written to the mounted repo stay theirs.
# Ubuntu 24.04 images ship a "ubuntu" user at 1000; leave it alone.
RUN groupadd -g "$GID" grader 2>/dev/null || true \
    && useradd -m -u "$UID" -g "$GID" -s /bin/bash grader \
    && mkdir -p /opt/venv /workspace && chown "$UID:$GID" /opt/venv /workspace
USER grader

# The venv lives outside /workspace so the repo bind mount doesn't hide it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/workspace/data/hf
WORKDIR /workspace
COPY --chown=grader pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --group finetune --group dev
CMD ["bash"]
