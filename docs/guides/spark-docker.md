# Running on the lab's DGX Spark (Docker)

The lab asks for work on the DGX Spark to run in Docker, so everything here runs in one image built from
`docker/Dockerfile`. The image contains:

- **Python:** the `uv.lock` environment (default dependencies plus the `finetune` and `dev` groups), in `/opt/venv`.
- **Grading server:** `llama-server` built at llama.cpp `a3a1c47`, the same commit as the Kaggle notebooks, for the GB10 (`CMAKE_CUDA_ARCHITECTURES=121`).

The Spark is shared with other people's services, so:

- The server listens on `127.0.0.1:8099` only.
- Use only `docker compose` commands for this project's services. Never stop or remove other containers.

## Machine facts

| | |
|---|---|
| GPU | NVIDIA GB10, compute capability 12.1, driver 580.142, CUDA 13.0 |
| Memory | 121 GB unified (CPU and GPU share it, together with the other tenants) |
| CPU | aarch64 |
| GPU in Docker | through CDI (`nvidia.com/gpu=all`); there is no `nvidia` runtime, so `--gpus` is not used |

## First time

```bash
cd ~/research/edge-grader
mkdir -p ~/research/models ~/.cache/huggingface   # create these first, so Docker doesn't make them root-owned
docker compose build                              # takes a while: llama.cpp CUDA build and torch
```

The image runs as uid/gid 1001 (`ari`). For another user, build with
`docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)`.

## Everyday use

The repo is mounted at `/workspace`, so edits on the host apply in the container without a rebuild.
Rebuild only when `uv.lock` or `docker/Dockerfile` changes.

```bash
docker compose run --rm dev uv run pytest -q                 # tests
docker compose run --rm dev bash                             # shell
docker compose run --rm dev uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Grading server

The server reads models from `~/research/models`, mounted at `/models`. E2B is the default model; pick
another with `MODEL`:

```bash
docker compose up -d llama-server                                          # E2B
MODEL=gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf docker compose up -d llama-server # E4B
curl -s 127.0.0.1:8099/health
docker compose logs -f llama-server
docker compose down                                                        # stop it
```

To grade against the server from the host, run:

```bash
uv run edge-grader-baseline --base-url http://127.0.0.1:8099/v1 ...
```

To grade from inside the container, use the server's compose service name: `http://llama-server:8080/v1`.

## Settings

| Variable | Default | What it sets |
|---|---|---|
| `MODEL` | `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` | GGUF file in `MODELS_DIR` |
| `MODELS_DIR` | `~/research/models` | host folder mounted at `/models` |
| `HF_CACHE` | `~/.cache/huggingface` | Hugging Face cache and token, shared with the host |
| `LLAMA_PORT` | `8099` | host port, bound to 127.0.0.1 |

Gated models (the training base) need a Hugging Face login. Logging in once on the host is enough, because
the cache folder, which holds the token, is mounted into the container.
