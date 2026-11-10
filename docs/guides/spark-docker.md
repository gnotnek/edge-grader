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
docker compose run --rm -e SMOKE=1 dev scripts/spark_finetune.sh e4b label-first   # first: a few minutes, output in data/smoke
docker compose up -d finetune-e4b-lf               # start (one run at a time: the GPU is shared)
docker ps --filter name=edge-grader                # what is running
docker logs -f edge-grader-finetune-e4b-lf         # follow it (also in data/logs/finetune-e4b-lf-*.log)
docker compose up -d --force-recreate finetune-e4b-lf   # run it again after it finished
```

When it finishes, the container stays (exited) with its log until it is recreated or removed with
`docker compose rm finetune-e4b-lf`. Outputs: `data/adapters/e4b-ft-lf.gguf`, the training summary in
`results/finetune/`, and the 100-answer run in `results/spark/`.

## Trained on Beetle and SAF, tested on SciEntsBank

The adapters never see SciEntsBank, so all three of its test splits are held out. Same recipe as the Oct 28 adapters; only the training data changes. Label-first and reason-first train on reasons from Qwen3-14B, which `reasons-beetle-saf` writes first, and the owner spot-checks them before those trainings start. Run everything one at a time:

| Service | Container | Runs |
|---|---|---|
| `finetune-e2b-beetle-saf` | `edge-grader-finetune-e2b-beetle-saf` | trains `data/adapters/e2b-ft-beetle-saf.gguf` |
| `finetune-e4b-beetle-saf` | `edge-grader-finetune-e4b-beetle-saf` | trains `data/adapters/e4b-ft-beetle-saf.gguf` |
| `reasons-beetle-saf` | `edge-grader-reasons-beetle-saf` | writes `results/finetune/raw/reasons-qwen3-14b-q4_k_m-beetle-saf.jsonl` |
| `finetune-{e2b,e4b}-{lf,rf}-beetle-saf` | `edge-grader-finetune-e2b-lf-beetle-saf` and so on | train label-first and reason-first on those reasons |
| `bench-scientsbank` | `edge-grader-bench-scientsbank` | grades test_ua, test_uq and test_ud with every adapter that has a file, and skips what's already graded |
| `bench-scientsbank-baseline` | `edge-grader-bench-scientsbank-baseline` | the same with E2B and E4B untouched, queued last |

`scripts/spark_queue.sh` runs services one after the other on the host and stops at the first that fails:

```bash
AFTER=edge-grader-bench-asap2-think nohup scripts/spark_queue.sh reasons-beetle-saf finetune-e2b-beetle-saf \
  finetune-e4b-beetle-saf bench-scientsbank > data/logs/queue.log 2>&1 &
```

Grading the untouched models and Oct 28 adapters here too keeps the comparison on one machine, and those runs can be checked against Kaggle's. Summaries land in `results/spark/`.

## Adapter soup (E4B, SciEntsBank)

Five E4B label-only adapters, the same recipe with seeds 0–4, then one adapter that is their exact average, then all of them graded on the three test splits beside untouched E4B ([spec](../superpowers/specs/2026-11-10-adapter-soup-design.md)):

| Service | Container | Runs |
|---|---|---|
| `finetune-e4b-s0` … `-s4` | `edge-grader-finetune-e4b-s0` and so on | `SEED=<n> scripts/spark_finetune.sh e4b label`: `data/adapters/e4b-ft-s<n>.gguf` |
| `soup-e4b` | `edge-grader-soup-e4b` | `SOUP="0 1 2 3 4" scripts/spark_finetune.sh e4b label`: stacks the five adapters into `data/adapters/e4b-ft-soup.gguf` |
| `bench-scientsbank-soup` | `edge-grader-bench-scientsbank-soup` | grades E4B untouched, the five seeds and the soup; skips what's already graded |

```bash
docker compose run --rm -e SMOKE=1 -e SEED=0 dev scripts/spark_finetune.sh e4b label   # smoke: two seeds,
docker compose run --rm -e SMOKE=1 -e SEED=1 dev scripts/spark_finetune.sh e4b label   # then their soup,
docker compose run --rm -e SMOKE=1 -e SOUP="0 1" dev scripts/spark_finetune.sh e4b label  # all in data/smoke
AFTER=<the container running now> nohup scripts/spark_queue.sh finetune-e4b-s0 finetune-e4b-s1 finetune-e4b-s2 \
  finetune-e4b-s3 finetune-e4b-s4 soup-e4b bench-scientsbank-soup > data/logs/queue-soup.log 2>&1 &
```

Then, on the Mac with the results pulled: `uv run python -m edge_grader.compare_soup`.

## ASAP 2.0 (stage 5)

One container, `edge-grader-bench-asap2`, runs the fixed 600-essay sample on each version in turn: E2B and E4B untouched, then `e2b-ft`, `e2b-ft-lf`, `e4b-ft` and `e4b-ft-lf`. Adapters are read from `data/adapters/`; a missing one is logged and skipped.

```bash
docker compose run --rm -e N=5 -e OUT=data/smoke dev scripts/spark_bench_asap2.sh   # first: 5 essays each
docker compose up -d bench-asap2
docker logs -f edge-grader-bench-asap2      # also in data/logs/bench-asap2-*.log
docker compose up -d bench-asap2-think      # thinking on: E2B and E4B untouched, container edge-grader-bench-asap2-think
```

Summaries land in `results/spark/`, named `asap2-baseline-<label>-sample-*.json`. The adapters were trained on SciEntsBank's three labels, not 1–6 scores, so read `unparsed` before QWK.
