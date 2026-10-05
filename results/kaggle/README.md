# Kaggle runs: grading on a Kaggle GPU

- **`baseline-*`:** the stage-1 baseline, 2026-09-29. Described on this page.
- **`rag-*-en-k3-*`:** stage 2, RAG-en (top 3 English passages), 2026-10-05, llama.cpp build `a3a1c47`. Results and analysis: [`docs/results/rag.md`](../../docs/results/rag.md#grading-results-rag-en-kaggle-t4-2026-10-05). The runs at `071301` and `071359` are the 100-answer test runs.

# Stage-1 grading baseline

*Run 2026-09-29 on Kaggle (Tesla T4, one GPU) with [`notebooks/kaggle_baseline.ipynb`](../../notebooks/kaggle_baseline.ipynb).*

## Setup

| | |
|---|---|
| Task | SciEntsBank (SemEval-2013 Task 7), 3-way labels: correct / contradictory / incorrect |
| Models | Gemma 4 **E2B** and **E4B**, `unsloth/gemma-4-E*B-it-qat-GGUF`, file `*-qat-UD-Q4_K_XL.gguf`, the same files as `agentic-edge` |
| Server | llama.cpp `llama-server` build `86ea01d`, `-c 8192 -rea off --reasoning-budget -1 --cache-ram 0 -ngl 99` (the agentic-edge baseline flags, thinking off) |
| Prompt | Zero-shot: question, reference answer, student answer, label definitions; "Reply with the label only." Temperature 0 |
| Grader state | **Untouched**: no RAG, no fine-tuning, no examples |

## Results (full test splits)

| Model | Split | n | Accuracy | Macro-F1 | Weighted-F1 | Median latency |
|---|---|---|---|---|---|---|
| E2B | test_ua (unseen answers) | 540 | 55.2% | 0.518 | 0.567 | 0.106 s |
| E2B | test_uq (unseen questions) | 733 | 59.8% | 0.553 | 0.621 | 0.108 s |
| E2B | test_ud (unseen domains) | 4,562 | 60.3% | 0.570 | 0.618 | 0.105 s |
| **E4B** | test_ua | 540 | **67.0%** | **0.664** | 0.658 | 0.137 s |
| **E4B** | test_uq | 733 | **67.5%** | **0.655** | 0.655 | 0.150 s |
| **E4B** | test_ud | 4,562 | **70.4%** | **0.705** | 0.700 | 0.139 s |

- **Every reply was a valid label** (0 unparsed) and **no request failed** (0 errors), in every run.
- **Latency is on a T4 GPU,** so it says nothing about the Pi or Jetson. Board numbers come from `scripts/board_grade.sh`.
- **Cost per answer:** about 155 prompt tokens and 2 completion tokens, so almost all the work is reading the prompt, not writing the answer.

**Pilot runs** (first 100 answers of test_ua, seed 0): E2B 55.0%, E4B 68.0%. These agree with the full runs. The files are kept, with timestamps `0956*` / `0957*`.

## What the models get wrong

Recall per class, i.e. of the answers the human graders gave each label, the share the model also gave that label:

| | correct | contradictory | incorrect |
|---|---|---|---|
| E2B, test_ud | 60% | 79% | 57% |
| E4B, test_ud | 57% | 69% | 82% |
| E4B, test_ua | 45% | 64% | 88% |
| E4B, test_uq | 41% | 67% | 90% |

1. **E4B is a strict grader.** It marks many correct answers as incorrect: 126 of 233 on test_ua, 803 of 1,917 on test_ud. It rarely marks a wrong answer correct: 6% of non-correct answers on test_ua, 13% on test_ud. This is the opposite of the "too generous" failure reported for LLM judges *without* a reference answer. Here the reference answer is given.
2. **E2B overuses "contradictory".** It gives that label 1,105 times on test_ud, where the human graders gave it 417 times. Only 30% of its "contradictory" labels are right; most of the rest are answers the human graders called plain incorrect.
3. **E4B is about 10–15 points better than E2B** on every split. That's a similar gap to the lab's MMLU-Pro baseline (65.7% vs 51.7% on the Pi).
4. **"Unseen domains" isn't harder here, and is even slightly easier.** That's expected: an untouched model was never trained on any SciEntsBank split, so the answers, questions and domains are *all* new to it, and the splits only differ in their mix of questions and labels. The split distinction becomes meaningful in **stage 3**, after fine-tuning on the train split. That's where "does it transfer to unseen domains" gets tested.

## What this means for the next stages

- **Most room to improve is in recognising correct answers**, especially for E4B, and in E2B's "contradictory" confusion. Check whether RAG (stage 2) and fine-tuning (stage 3) move those numbers.
- **Compare every later stage against these rows,** with the same splits, prompt and server flags.

## Files

- `baseline-<model>-kaggle-t4-<split>-3way-<timestamp>.json`: run summary (settings, server model path and build, metrics, token totals)
- `raw/<same name>.jsonl`: one line per answer (id, human label, model label, raw reply, latency, tokens)
