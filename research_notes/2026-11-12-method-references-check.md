# Reference check for the six accuracy methods, plus IEEE Xplore search

*2026-11-12. Covers the references behind the six methods proposed on 2026-11-09 (majority vote, NLI first, graded examples, adapter averaging, self-consistency, part-by-part grading). Each one was checked against arXiv, Crossref, the ACL Anthology or the paper's full text. IEEE Xplore was searched with the lab's institutional access (NTUST).*

## 1. The original references

| Reference | Status | What to fix or add |
|---|---|---|
| Camus & Filighera (2020), AIED, LNCS 12164, pp. 43–48, doi:10.1007/978-3-030-52240-7_8 | Correct | Numbers (76.7/64.1/66.8 → 78.8/66.4/71.8) were read from the paper earlier |
| Dietterich (2000), Ensemble Methods in Machine Learning, MCS, LNCS 1857, pp. 1–15, doi:10.1007/3-540-45014-9_1 | Correct | Was cited from memory; now confirmed in Crossref |
| Dzikovska, Nielsen & Brew (2012), NAACL-HLT, pp. 200–210 | Correct | |
| Liu et al. (2022), What Makes Good In-Context Examples for GPT-3?, DeeLIO, pp. 100–114, doi:10.18653/v1/2022.deelio-1.10 | Correct | Abstract: retrieved similar examples "consistently" beat random ones. **Our SciEntsBank run did not reproduce this** (§3) |
| Ormerod & Kwako (2024), arXiv:2407.01873 | Correct, wording needs care | The ensemble they compare against is from an earlier paper (Ormerod, "Short-answer scoring with ensembles of pretrained language models"), not run in this one. Their fine-tuned models are 3.8–8.5B (Phi-3, Mistral, Llama-3, Gemma 1.1), not "7–8B". Their conclusion: ensembles of smaller LMs "remain more efficient and performant" than these models |
| Pruksachatkun et al. (2020), ACL | **Title was wrong** | Correct title: *Intermediate-Task Transfer Learning with Pretrained Models for Natural Language Understanding: When and Why Does It Work?* doi:10.18653/v1/2020.acl-main.467. Claim holds with nuance: NLI (MNLI) is "generally useful", but Cosmos QA and HellaSwag worked best |
| SteLLA | **Year and venue were wrong** | Qiu et al. (2024), *2024 IEEE International Conference on Big Data*, pp. 8154–8163, doi:10.1109/BigData62323.2024.10825385. Cite the IEEE version, not arXiv 2025 |
| Wang & Ormerod (2024), arXiv:2408.03811 | Correct | Numbers (78.71/76.94/73.35) were read from the paper earlier |
| Wang et al. (2023), Self-Consistency, ICLR, arXiv:2203.11171 | Correct | +3.9 (ARC-challenge) to +17.9 (GSM8K) confirmed in the abstract |
| Wortsman et al. (2022), Model soups, ICML, arXiv:2203.05482 | Correct | Their soups average models trained with different hyperparameters, not only different seeds. Our soup uses seeds only |

## 2. New from IEEE Xplore, by method

### Method 1, majority vote
- **Sahu, A., & Bhowmick, P. K. (2020).** Feature engineering and ensemble-based approach for improving automatic short-answer grading performance. *IEEE Transactions on Learning Technologies*, 13(1), 77–90. doi:10.1109/TLT.2019.2897997. **Journal.** Ensembles (stacked models) beat single models on **SciEntsBank and Beetle**. Pre-LLM features, but the same datasets.
- **Gwozdz, J., & Both, A. (2026).** Auditing LLM grading for short-answer responses: Confidence gating and cross-model agreement. *IEEE ICALT 2026*, 332–336. doi:10.1109/ICALT68022.2026.00102. Uses agreement between models to decide which answers to auto-grade. This supports our cascade idea: send only the answers where models disagree to a costlier step.

### Method 2, NLI first, then SciEntsBank (strongest new find)
- **Kazi, N., & Kahanda, I. (2023).** Enhancing transfer learning of LLMs through fine-tuning on task-related corpora for automated short-answer grading. *2023 ICMLA*, 1687–1691. doi:10.1109/ICMLA58977.2023.00255. This is **exactly method 2 on our test set**: RoBERTa-large trained on MNLI, then on SciEntsBank 3-way, reaching weighted-F1 0.77/0.72/0.72 (UA/UQ/UD). The MNLI step mainly improved the **contradictory** label and the harder UQ and UD splits, which are our weak spots. Nazmul Kazi also published the `nkazi/SciEntsBank` and `nkazi/Beetle` copies we load.
- **Full text read (2026-11-12, IEEE Xplore HTML).** RoBERTa-large (355M), trained for 20 epochs at learning rate 2e-5 and batch size 5, once on SciEntsBank alone and once on MNLI then SciEntsBank. One run each, no significance test, a 5-page conference paper. Tables IV and V:

  | 3-way, SciEntsBank | UA acc / macro-F1 | UQ acc / macro-F1 | UD acc / macro-F1 | "contradictory" F1 (UA / UQ / UD) |
  |---|---|---|---|---|
  | RoBERTa-large | 76 / 0.71 | 65 / 0.54 | 67 / 0.60 | 0.58 / 0.24 / 0.40 |
  | RoBERTa-large, MNLI first | 77 / 0.74 | 72 / 0.67 | 72 / 0.70 | 0.65 / 0.55 / 0.64 |
  | Camus & Filighera (2020), MNLI first, as quoted in their table | 79 / 0.78 | 66 / 0.66 | 72 / 0.71 | not reported |
  | **Our E4B label-only adapter** (Kaggle, 2026-10-28) | **79.1 / 0.769** | **80.0 / 0.786** | **76.4 / 0.774** | **0.71 / 0.75 / 0.80** |

  MNLI added 1, 7 and 5 accuracy points, and +0.07, +0.31 and +0.24 F1 on "contradictory". **Our SciEntsBank adapter already beats their MNLI model on every split, including on "contradictory".** So for the SciEntsBank adapter, "contradictory" is not the weak label, and the gain left for NLI-first is probably smaller than theirs. Where "contradictory" *is* weak is untouched E4B and the Beetle + SAF adapter (see the [boundary check](../docs/results/finetuning.md)). That is where NLI-first is most likely to help. Their future-work section also suggests adding Beetle, SNLI and SciTail. (Our per-label F1 was computed from the raw predictions in `results/kaggle/raw/`.)
- **Zhu, X., Wu, H., & Zhang, L. (2022).** Automatic short-answer grading via BERT-based deep neural networks. *IEEE Transactions on Learning Technologies*, 15(3), 364–375. doi:10.1109/TLT.2022.3175537. **Journal.** BERT baseline on SemEval-2013; useful as a comparison row.

### Method 3, graded examples in the prompt
- **Fateen, M., Wang, B., & Mine, T. (2024).** Beyond scores: A modular RAG-based system for automatic short answer scoring with feedback. *IEEE Access*, 12, 185371–185385. doi:10.1109/ACCESS.2024.3508747. **Journal.** Uses retrieval to pick few-shot examples and reports +9% accuracy on unseen questions **over fine-tuning**. Our result went the other way: the adapter beat examples on every split. Which dataset they used is not in the abstract; I haven't read the full text.
- **Al Zubaer, A., et al. (2025).** Comparative study of language models and prompt paradigms in short answer grading. *IEEE ICTAI 2025*, 321–329. doi:10.1109/ICTAI66417.2025.00049. Few-shot examples help on difficult answers, but **without a sample solution they do significantly worse**. This fits our finding that examples shift the model rather than teach it the boundary.

### Method 4, adapter averaging
- **Mahmoud, S., Nabil, E., & Torki, M. (2024).** Automatic scoring of Arabic essays: A parameter-efficient approach for grammatical assessment. *IEEE Access*, 12, 142555–142568. doi:10.1109/ACCESS.2024.3470728. **Journal.** Applies a model soup to parameter-efficient (adapter) models for scoring, with results similar to or better than full fine-tuning. Essays and grammar, not short answers.
- Nothing on IEEE averages LoRA adapters for short-answer grading. Wortsman et al. (2022) remains the main source.

### Method 5, self-consistency
- No IEEE paper applies it to grading. Wang et al. (2023) remains the only source.

### Method 6, part-by-part grading
- **SteLLA** (above) is an IEEE paper.
- **Chu, Y., Li, H., Yang, K., Copur-Gencturk, Y., & Tang, J. (2025).** LLM-based automated grading with human-in-the-loop. *IEEE TALE 2025*, 1–8. doi:10.1109/TALE66047.2025.11346771. Rubric-based grading where the LLM asks experts questions to refine the rubric. Related, not the same.
- Gwozdz & Both (2026) found that rubric-style guidance makes LLM graders **stricter** (more false fails). That's a risk to watch if we try method 6.

## 3. What this changes

1. **Method 2 now has a direct precedent on our exact task:** MNLI first, SciEntsBank 3-way, with the gain on "contradictory". But the full text shows our SciEntsBank adapter is already past their result, so the best place to try NLI-first is the **Beetle + SAF adapter**, whose 3-way loss sits on "contradictory". For the SciEntsBank adapter, expect a small gain at most. Its cost on the device is zero either way, because it's still one adapter. Their model is 355M parameters against E4B's ~4B, and an instruction-tuned LLM may already have seen NLI data, so their gain may not carry over. That's my reasoning, not tested.
2. **Fix two citations** in anything sent out: the Pruksachatkun title, and SteLLA (IEEE BigData 2024).
3. **Our graded-examples result disagrees with two papers:** Liu et al. (similar beats random) and Fateen et al. (examples beat fine-tuning on unseen questions). Worth one sentence in the write-up. Possible reasons: a small model, 4 examples, and an untuned embedder.
4. **Journal (not conference) backing found:** IEEE TLT for methods 1 and 2 (comparison), and IEEE Access for methods 3 and 4. Nothing in IEEE journals for method 5.

## Not done
- Full text read only for Kazi & Kahanda (2023). The other IEEE papers were checked by abstract and metadata only. The PDFs are available through the lab's access.
- The search was IEEE Xplore only, with about a dozen queries.
