# Design: translating Beetle's "contradictory" into SciEntsBank's definition (proposal)

*2026-11-09. Draft for the supervisor and the lab, not approved yet. Written after the Nov 8 fine-tuning results were known, so this is a new experiment, not a re-reading of those results. Calls marked **(Claude's call)** were not discussed and can be changed. Once approved, the decisions go into [`research-outline.md`](../../research-outline.md) and [`plan.md`](../../plan.md).*

## 1. Why

The adapters trained on Beetle + SAF and tested on SciEntsBank help E2B but not E4B on the 3-way labels ([results](../../results/finetuning.md#result-transfer-helps-e2b-not-e4b-2026-11-08)). Scored 2-way (correct vs not correct), all nine E4B cells go up; on 3-way, test_ud drops 3–9 points ([boundary check](../../results/finetuning.md#why-e4b-loses-the-contradictoryincorrect-boundary-2026-11-09-post-hoc)). The loss sits on one label: "contradictory".

The two datasets use the label **"contradictory" for two different things.** Both were labeled correctly under their own rules; the rules differ (Dzikovska et al., 2012, §2.1–2.2):

| | Beetle | SciEntsBank |
|---|---|---|
| How the labels were made | Trained annotators labeled each answer by hand (κ = 0.69), then the grades were mapped to the shared labels | Computed by rules from annotations on each part ("facet") of the reference answer (facet κ = 0.728) |
| What becomes "contradictory" | The original grades "incorrect" (completely wrong), "partly correct with some error" and "mix of correct, incorrect and missing" | Only answers in which **at least one facet of the reference answer is contradicted** |
| A wrong answer that conflicts with nothing in the reference | contradictory | incorrect (via "irrelevant" or "partially correct incomplete") |
| Share of train answers labeled contradictory | 1,049 of 3,941 (26.6%) | 499 of 4,969 (10.0%) |

So an adapter trained on Beetle learns a wider "contradictory" than the one it is tested on. That fits what we saw: the adapters that write reasons call 20% of SciEntsBank answers contradictory against 9–11% gold, and all three of the owner's spot-check doubts were Beetle answers graded contradictory that the owner would call incorrect.

**This is a mismatch between two definitions, not label noise.** We don't claim the experts made mistakes. We translate Beetle's labels into the definition used by the set we test on. In dataset-shift terms the label rule differs between training and test data (concept shift; Moreno-Torres et al., 2012).

## 2. Research question

Part of **RQ3**: *Is E4B's 3-way loss after Beetle + SAF training caused by Beetle's wider "contradictory"? If Beetle's contradictory answers are relabeled under SciEntsBank's rule, does the 3-way loss go away?*

**What "it worked" means:** the relabeled E4B adapter scores higher on SciEntsBank 3-way accuracy than the Nov 8 E4B label-only Beetle + SAF adapter, significant after Holm (§5.5), on at least one split. **(Claude's call)** Its "contradictory" rate should also move toward the gold rate; this is reported but not part of the test.

**Why this test is clean:** relabeling only moves answers between "contradictory" and "incorrect". Both count as "not correct" in 2-way, so the 2-way training targets are identical to Nov 8. Any change in 3-way comes from the contradictory/incorrect boundary and nothing else.

## 3. Goals, non-goals, anti-goals

**Goals:**
- Beetle's 1,049 training answers labeled contradictory are each relabeled **contradictory or incorrect** under SciEntsBank's rule.
- An LLM proposes the labels; the LLM is checked against two references (§5.2, §5.3) before any label is used.
- **E4B label-only** retrained on Beetle (relabeled) + SAF with the Nov 8 recipe, graded on SciEntsBank test_ua, test_uq and test_ud on Kaggle.

**Non-goals:**
- **Relabeling anything else.** Beetle's "correct" and "incorrect" answers stay as they are. Beetle's incorrect is "partially correct incomplete" (by definition it doesn't contradict the reference), "irrelevant" (doesn't answer the question) or "non domain" (no domain content), so none of them is defined by conflicting with the reference. SAF is 2-way and has no contradictory. **(Claude's call)**
- **Touching any test label.** SciEntsBank, Beetle and SAF test splits keep their published labels.
- **Reason adapters.** Label-only scored highest in all 15 cells on Nov 8, and it costs the device 2 tokens per answer. **(Claude's call)**
- **E2B.** Its Beetle + SAF adapter already passes the 5-point bar on all three splits. An E2B run can be added if the supervisor wants to see whether relabeling costs E2B anything. **(Claude's call)**

**Anti-goals (a failure even if it runs):**
- Looking at any SciEntsBank, Beetle or SAF test split while writing the relabeling prompt or setting thresholds.
- Humans labeling the Beetle sample after seeing the LLM's labels. Wrong LLM labels pull people toward them (Wang et al., 2024).
- Loosening a threshold in §5 after seeing the numbers.
- A comparison with the Nov 8 adapter that differs in more than the labels: same base model, recipe, seed, validation split, grading prompt, GGUF, llama.cpp build and server flags.
- Replacing the original Beetle labels. They are kept, and the relabels are a separate file.

## 4. Constraints

- Fixed setup as on Nov 8: Gemma 4 E4B, the lab's QAT `UD-Q4_K_XL` GGUF, llama.cpp `a3a1c47` on Kaggle, `-c 8192 -rea off --reasoning-budget -1 --cache-ram 0 -ngl 99`, temperature 0, thinking off, adapter served with `--lora`.
- Models run on Kaggle; the Mac runs no models.
- Every run writes a summary JSON with its settings to `results/`, including the relabeling run and the two checks.
- Metrics: accuracy and macro-F1, 3-way (main) and 2-way; per-label precision and recall for "contradictory".

## 5. Design

### 5.1 The guideline (the definition)

One short text, used by the LLM and by the human annotators alike, written from SciEntsBank's rule (Dzikovska et al., 2012, §2.2):

> **Contradictory:** the answer states something that conflicts with at least one part of the reference answer.
> **Incorrect:** the answer is wrong or incomplete, but nothing it states conflicts with the reference answer (for example, it misses the point, talks about something else, or leaves parts out).

It goes in `src/edge_grader/prompts/` and is fixed before any check is run. It has no examples drawn from SciEntsBank, so the adapter's training still holds no SciEntsBank answer. **(Claude's call)**

### 5.2 Check 1: can the LLM apply SciEntsBank's rule? (SciEntsBank train)

- **Relabeler:** Qwen3-14B, Q4_K_M, in `llama-server` on Kaggle, thinking off, temperature 0, as already set up for the reason writing. It sees the question, reference answer and student answer, plus the guideline, and replies with one of the two labels. **(Claude's call)**
- **Data:** SciEntsBank **train** answers whose gold label is contradictory or incorrect: all 499 contradictory and 499 incorrect drawn with seed 0. These already carry labels under the target rule, so they are ground truth for this check. Caveat: they came from the automatic conversion, which its authors call "not perfectly accurate", and only the test split was checked by hand (Dzikovska et al., 2013, §3.3).
- **Gate:** Cohen's κ between the LLM and the gold labels **≥ 0.60** ("substantial" on Landis & Koch's 1977 scale). **(Claude's call)**

### 5.3 Check 2: does the LLM agree with people on Beetle? (human sample)

- **Sample:** 100 of Beetle's 1,049 contradictory training answers, drawn with seed 0. **(Claude's call)**
- **Annotators:** the owner and one lab member, each labeling all 100 alone, with only the guideline, without seeing the LLM's labels or each other's. Disagreements are then settled by the two together, and the settled labels are final for these 100.
- **Gates:**
  1. Human vs human κ **≥ 0.60.** If two people can't apply the definition consistently, the definition isn't usable and work stops here.
  2. The LLM's κ with each annotator is **no more than 0.10 below** the human vs human κ. **(Claude's call)**

### 5.4 Relabeling and training

- If both checks pass, the LLM relabels all 1,049 answers. The 100 human-settled labels replace the LLM's for those answers. If the LLM gives different answers on two runs, the answer goes to the owner. **(Claude's call)**
- **Output:** a JSONL of answer id → new label, with the original label beside it, in `results/finetune/raw/`, plus a summary JSON with the model file, prompt sha256, both checks' κ, and how many labels changed.
- **Training:** the Nov 8 Beetle + SAF label-only run, unchanged except that Beetle's labels come from the relabel file. Validation is the same held-out Beetle + SAF questions, with their labels relabeled the same way, so the epoch is picked under the definition being trained.
- **If either check fails:** no LLM labels are used. The owner chooses between relabeling all 1,049 by hand (two annotators on a share, as in §5.3) or stopping, and reporting 2-way scores, which need no relabeling.

### 5.5 Testing

- The relabeled E4B adapter on SciEntsBank test_ua, test_uq and test_ud, whole splits, Kaggle, build `a3a1c47`.
- Two comparisons per split, on the same answers: against the **Nov 8 E4B label-only Beetle + SAF adapter** (the "it worked" test) and against **untouched E4B** (the 5-point bar). McNemar on the 3-way labels, Holm over these 6 tests, set before grading.
- Also reported, not tested: macro-F1, 2-way accuracy, the share of answers predicted contradictory against gold, and per-label recall.

### 5.6 Optional arm: merge, no LLM

Beetle's contradictory all set to incorrect, the same training run. This answers "was the LLM needed?": if merging everything does as well as relabeling, the simpler fixed rule wins. Its cost is that the training set then has no contradictory answers at all, so the adapter has nothing to learn the label from. One extra training run and three gradings on Kaggle. **(Supervisor's choice)**

## 6. What the supervisor is asked to decide

1. Go ahead with relabeling, or report 2-way only.
2. Who the second annotator is for the 100-answer sample (§5.3).
3. Whether to add the E2B run (§3) and the merge arm (§5.6).
4. Whether the thresholds in §5.2 and §5.3 are acceptable.

## 7. References

- Dzikovska, M. O., Nielsen, R. D., & Brew, C. (2012). Towards effective tutorial feedback for explanation questions: A dataset and baselines. *NAACL-HLT*, 200–210. https://aclanthology.org/N12-1021/
- Dzikovska, M. O., et al. (2013). SemEval-2013 Task 7: The joint student response analysis and 8th recognizing textual entailment challenge. *SemEval*. https://aclanthology.org/S13-2045/
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174.
- Moreno-Torres, J. G., et al. (2012). A unifying view on dataset shift in classification. *Pattern Recognition*, 45(1), 521–530.
- Pangakis, N., Wolken, S., & Fasching, N. (2023). Automated annotation with generative AI requires validation. arXiv:2306.00176.
- Wang, X., Kim, H., Miao, Z., Mitra, K., & Rahman, S. (2024). Human-LLM collaborative annotation through effective verification of LLM labels. *CHI*. doi:10.1145/3613904.3641960
- Background on why "noise" is the wrong frame: Frénay & Verleysen (2014), *IEEE TNNLS* 25(5); Northcutt, Athalye & Mueller (2021), NeurIPS Datasets & Benchmarks; Plank (2022), EMNLP; Pavlick & Kwiatkowski (2019), *TACL* 7.
