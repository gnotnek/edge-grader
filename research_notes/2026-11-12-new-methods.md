# New methods to raise E4B's accuracy (after graded examples)

*2026-11-12. Follows the six methods of 2026-11-09 and the graded-examples result ([examples.md](../docs/results/examples.md)). Every paper below was checked in IEEE Xplore (lab access), arXiv, Crossref or the ACL Anthology.*

## First: where the E4B adapter still goes wrong

Errors of the E4B label-only SciEntsBank adapter (Kaggle, 2026-10-28), split by the answer's original **5-way** label (computed from `results/kaggle/raw/`):

| Split | Errors | Correct answer called "incorrect" | Partly-correct answer called "correct" | Together | Contradictory answer called "incorrect" |
|---|---|---|---|---|---|
| test_ua | 113 | 51 | 24 | **66%** | 20 |
| test_uq | 147 | 64 | 38 | **69%** | 17 |
| test_ud | 1,075 | 435 | 348 | **73%** | 95 |

**Two-thirds or more of the remaining errors are about completeness:** is the answer fully correct, or only partly? "Contradictory" errors are 9–18%. The 3-way label "incorrect" hides the reason. In SciEntsBank train it's a mix of three original labels: partly correct but incomplete (1,324), irrelevant (1,115) and non-domain (23). So the model is asked to learn one label that means three different things.

That shifts the priorities. Methods that help the model judge completeness should come first.

## Methods, best first

### 1. Train on the 5-way labels, report 3-way
**How it works:** train the adapter to output the original 5 labels (correct, partially correct incomplete, contradictory, irrelevant, non domain). At test time, map them to 3-way the usual way, so the scores stay comparable. The model learns "partly correct" as its own idea instead of mixing it with "irrelevant".
**Evidence:**
- Chen, Z., Ding, R., Chin, T.-W., & Marculescu, D. (2018). Understanding the impact of label granularity on CNN-based image classification. *IEEE ICDM Workshops*, 895–904. doi:10.1109/ICDMW.2018.00131. Training on finer labels raised accuracy on the coarse classes on every dataset they tried, and with 40% of the data it beat coarse training on all of it. (Images, not text.)
- Zhang, Y., & Yang, Q. (2022). A survey on multi-task learning. *IEEE Transactions on Knowledge and Data Engineering*, 34(12), 5586–5609. doi:10.1109/TKDE.2021.3070203. Learning related tasks together (here, 5-way and 3-way) usually helps each one.
- The 5-way labels are part of SciEntsBank (Dzikovska et al., 2013) and our loader already reads them (`data.load(..., scheme="5way")`).
**Cost:** one more training run. On the device, the same speed (one label, a few tokens). Code change: `finetune.py` fixes the scheme to 3-way in two places.
**Risk:** none found on SciEntsBank itself, so this is new. "Non domain" has only 23 training answers.

### 2. Confidence gating: auto-grade only the answers the model is sure about
**How it works:** read how confident E4B is in its label, using the token probabilities that `llama-server` already returns. Answers above a threshold are graded automatically. The rest go to the teacher, or to a costlier step such as thinking mode in the cascade. Accuracy on the auto-graded answers goes up, and you report the share that was auto-graded (the coverage) next to it.
**Evidence:**
- Gwozdz, J., & Both, A. (2026). Auditing LLM grading for short-answer responses: Confidence gating and cross-model agreement. *IEEE ICALT 2026*, 332–336. doi:10.1109/ICALT68022.2026.00102. LLM graders, including a locally served open model, gated by self-reported confidence and agreement between models.
- Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*. arXiv:1705.08500. A model that may refuse some inputs can guarantee a chosen error rate on the rest.
**Cost:** almost none on the device. The probabilities come with the same reply. Our `label_prior.py` already reads them. Needs one grading run that saves them.
**Note:** this doesn't raise accuracy on *all* answers, so it answers a different question: "how many answers can the device grade alone at teacher-level accuracy?" That fits an edge-deployment thesis well.

### 3. More training answers written by an LLM, aimed at the hard boundary
**How it works:** a larger model writes extra training answers from SciEntsBank's train questions. In particular it writes "near misses": take a correct answer and drop one key part, so it becomes partly correct. Then train E4B on the real plus the new answers. This targets the completeness errors directly.
**Evidence:**
- Lai, P., Zhang, K., Wang, Y., & Cui, B. (2026). Improving low-resource short answer scoring through large language model-based data augmentation. *IEEE Transactions on Knowledge and Data Engineering*, 38(5), 2914–2926. doi:10.1109/TKDE.2026.3673734. LLM-written answers, made by swapping key phrases and rewriting style, improved short-answer scoring over BERT, RoBERTa and ERNIE.
- Zhu, B., Chen, Z., & Liu, S. (2025). Feedback-integrated short answer grading with data augmentation and large language models. *IEEE BigData 2025*, 1895–1902. doi:10.1109/BigData66926.2025.11402058.
**Cost:** a run on the Spark to generate the answers, plus one training run. Nothing extra on the device.
**Risk:** the new answers need labels we can trust. A sample should be checked by hand, as in the relabel plan.

### 4. Train the adapter *with* the graded examples in its prompt
**How it works:** our graded-examples run put examples in front of an adapter that had never seen examples during training, and that hurt on test_ud. Here the adapter is trained with the retrieved examples already in its prompt, so it learns how to use them.
**Evidence:**
- Chen, Y., Zhong, R., Zha, S., Karypis, G., & He, H. (2022). Meta-learning via language model in-context tuning. *ACL*. arXiv:2110.07814.
- Min, S., Lewis, M., Zettlemoyer, L., & Hajishirzi, H. (2022). MetaICL: Learning to learn in context. *NAACL*. arXiv:2110.15943.
- Why plain examples lost to fine-tuning: Scherp, A., et al. (2026). Are we really making much progress in text classification? A comparative review. *IEEE Transactions on Knowledge and Data Engineering*, 38(12), 7959–7978. doi:10.1109/TKDE.2026.3735248. With enough training data, fine-tuned models beat few-shot prompting.
**Cost:** prompts about 3.5× longer on the device, as measured in our examples run.
**Expectation:** most likely to help test_ua only, as before, since test_uq and test_ud have no same-question examples.

### 5. Learn from a larger teacher model (distillation)
**How it works:** a bigger model on the Spark grades the training answers, giving its label or its probability for each label. E4B is trained to copy it alongside the gold labels.
**Evidence:**
- Li, J., Gui, L., Zhou, Y., West, D., Aloisi, C., & He, Y. (2023). Distilling ChatGPT for explainable automated student answer assessment. *Findings of EMNLP*. arXiv:2305.12962. Specific to short-answer grading.
- Zhang, D., Listiyani, D., Singh, P., & Mohanty, M. (2025). Distilling wisdom: A review on optimizing learning from massive language models. *IEEE Access*, 13, 56296–56325. doi:10.1109/ACCESS.2025.3554586.
- Liang, Y., et al. (2026). A comprehensive survey on large language model compression for AI applications in edge systems. *IEEE Internet of Things Journal*, 13(10), 20583–20599. doi:10.1109/JIOT.2026.3675866. Distillation as a route to edge deployment.
**Caution, which is why it ranks last:** a teacher must grade better than the student. Our E4B adapter (79–80%) already beats every published LLM result on SciEntsBank, including Claude 3.5 Sonnet + RAG. Check the teacher on a train sample first, and only go ahead if it's clearly better.

## Not recommended now
- **Two-stage grading** (correct vs not first, then contradictory vs incorrect). Most errors are in the *first* stage, so a second specialist stage would fix little.
- **Loss weighting for rare labels** (e.g., focal loss). The adapter's "contradictory" F1 is already 0.71–0.80, so the rare label isn't the problem.
- **NLI-first for the SciEntsBank adapter.** The adapter is already past Kazi & Kahanda's result; keep it for the Beetle + SAF adapter ([reference check](2026-11-12-method-references-check.md)).

## Earlier method worth moving up
**Part-by-part grading** (method 6 of 2026-11-09) also targets completeness: it asks which parts of the reference answer the student covered. With completeness now the main error, it fits better than it did. It's still untested on SciEntsBank, though, and it costs more tokens.

## Suggested order
1. **5-way training:** the smallest change, aimed at the main error, no cost on the device.
2. **Confidence gating:** one grading run, no cost on the device, and it speaks directly to edge deployment.
3. **LLM-written near-miss answers:** more work (generation plus a hand check).
4. Then 4 or 5 if the supervisor wants more.
