# Running on the lab's DGX Spark (Docker)

The lab asks for work on the DGX Spark to run in Docker, so everything here runs in one image built from
`docker/Dockerfile`. The image contains:

- **Python:** the `uv.lock` environment (default dependencies plus the `finetune` and `dev` groups), in `/opt/venv`.
- **Grading server:** `llama-server` built at llama.cpp `a3a1c47`, the same commit as the Kaggle notebooks, for the GB10 (`CMAKE_CUDA_ARCHITECTURES=121`).
- **LoRA converter:** llama.cpp's `convert_lora_to_gguf.py` at the same commit.

**Everything stays in the project folder** (all under `data/`, which git ignores, except the summaries):

| Folder | What |
|---|---|
| `data/models/` | the GGUF models (mounted at `/models` for the server) |
| `data/hf/` | Hugging Face cache, and the token if you log in (`HF_HOME`) |
| `data/adapters/` | trained adapters and their GGUFs |
| `data/logs/` | one log per fine-tuning run, plus its server log |
| `results/finetune/`, `results/spark/` | the run summaries, committed like every other result |

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
mkdir -p data/models data/hf data/adapters data/logs   # first, so Docker doesn't make them root-owned
docker compose build                                    # takes a while: llama.cpp CUDA build and torch
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

The server reads models from `data/models`, mounted at `/models`. E2B is the default model; pick
another with `MODEL`:

```bash
docker compose up -d llama-server                                          # E2B
MODEL=gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf docker compose up -d llama-server # E4B
curl -s 127.0.0.1:8099/health
docker compose logs -f llama-server
docker compose down                                                        # stop it
```

The server's container is named `edge-grader-llama-server`.

To grade against the server from the host, run:

```bash
uv run edge-grader-baseline --base-url http://127.0.0.1:8099/v1 ...
```

To grade from inside the container, use the server's compose service name: `http://llama-server:8080/v1`.

## Settings

| Variable | Default | What it sets |
|---|---|---|
| `MODEL` | `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` | GGUF file in `MODELS_DIR` |
| `LLAMA_PORT` | `8099` | host port, bound to 127.0.0.1 |

The training bases (`google/gemma-4-E*-it-qat-q4_0-unquantized`) downloaded on Kaggle without a login. If one
ever needs it, log in inside the container so the token lands in `data/hf`, and keep it private there:
`docker compose run --rm dev uv run hf auth login`, then `chmod 600 data/hf/token`.

## Fine-tuning (stage 3b, E4B)

Each run has its own service and a container named after it, so anyone in the lab can see what is running:

| Service | Container | Runs |
|---|---|---|
| `finetune-e4b-lf` | `edge-grader-finetune-e4b-lf` | `scripts/spark_finetune.sh e4b label-first` |
| `finetune-e4b-rf` | `edge-grader-finetune-e4b-rf` | `scripts/spark_finetune.sh e4b reason-first` |

The script does what `notebooks/kaggle_finetune.ipynb` does with `SMOKE = False`: train (3 epochs, validation
picks one), convert the adapter to GGUF, then grade 100 SciEntsBank test_ua answers through `llama-server --lora`
as a sanity check. The recipe is unchanged from Kaggle (QLoRA, nf4, fp16 compute); only the hardware differs.

```bash
docker compose up -d finetune-e4b-lf               # start (one run at a time: the GPU is shared)
docker ps --filter name=edge-grader                # what is running
docker logs -f edge-grader-finetune-e4b-lf         # follow it (also in data/logs/finetune-e4b-lf-*.log)
docker compose up -d --force-recreate finetune-e4b-lf   # run it again after it finished
```

When it finishes, the container stays (exited) with its log until it is recreated or removed with
`docker compose rm finetune-e4b-lf`. Outputs: `data/adapters/e4b-ft-lf.gguf`, the training summary in
`results/finetune/`, and the 100-answer run in `results/spark/`.
