# How others fine-tune graders on SciEntsBank, and what they test on

*2026-11-13. Prompted by the supervisor's worry that fine-tuning on SciEntsBank risks leaking data. Searched IEEE Xplore (search and abstracts; full text wasn't reachable from this machine), arXiv, Frontiers and the web. "Read" = methods section read; "abstract" = abstract only.*

## Short answer

**Training on the SciEntsBank train split and testing on its test splits is the standard, published protocol.** It comes from the dataset's own release (SemEval-2013 Task 7, Dzikovska et al. 2013), and the LLM papers below still follow it. That is what this project does.

**The data-leak worry is real for one split only.** test_ua ("unseen answers") contains new students' answers to the *same questions* used in training. Papers report it, but call it the easy case. test_uq (new questions) and test_ud (new domains) share nothing with training, and they are the fair test of a fine-tuned grader.

**Nobody checks whether the base model saw SciEntsBank during pretraining.** None of the papers found here tests for it.

## LLM papers that fine-tune on SciEntsBank, or on data built the same way

| Paper | Venue | Model and method | Train on | Test on | Ev. |
|---|---|---|---|---|---|
| Raikote et al., *CHiL(L)Grader* ([arXiv 2603.11957](https://arxiv.org/html/2603.11957)) | Preprint, 2026 | Qwen2.5-7B-Instruct, **LoRA** (r 16, α 32), plus Llama-3.1-8B and 3.2-3B baselines | SciEntsBank train (4,969, of which 540 used for calibration) | **test_uq only (733)**: "Following the standard setup, we evaluate in the fully UQ setting" | Read |
| Saur & Toth, *Grading at scale* ([Frontiers in CS 2026](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1829089/full)) | Journal, 2026 | Qwen3 0.6B, Llama 3.2 1B, Flan-T5, GPT-2, **LoRA**, 5 seeds; runs on a laptop CPU | Their own synthetic set, or SciEntsBank | **The other dataset** (cross-dataset). Within one set, split by question "to avoid leakage" | Read |
| Kazi & Kahanda, *Enhancing transfer learning of LLMs…* ([IEEE ICMLA 2023](https://ieeexplore.ieee.org/document/10459755/), doi:10.1109/ICMLA58977.2023.00255) | **IEEE**, 2023 | RoBERTa-large (355M encoder; the paper calls it an LLM), fine-tuned on MNLI first, then on SciEntsBank 3-way | SciEntsBank train | test_ua / test_uq / test_ud: weighted F1 0.77 / 0.72 / 0.72 | Abstract |
| Aggarwal et al., *EngSAF* ([arXiv 2407.12818](https://arxiv.org/html/2407.12818)) | Preprint (later version cited as AIED 2025) | Llama-2, Llama-3.1, **Mistral-7B** fine-tuned | EngSAF train, split "following Dzikovska et al." | Unseen answers 75.4%, unseen questions 58.7% (a 23-point drop on new questions) | Read |
| Walsh et al. 2025 ([arXiv 2508.04063](https://arxiv.org/pdf/2508.04063)) | Workshop | Llama 3.1 8B, **QLoRA** | OpenTutor | OpenTutor | Earlier note |
| Ormerod & Kwako 2024 ([arXiv 2407.01873](https://arxiv.org/abs/2407.01873)) | Preprint | Mistral 7B, Gemma, Llama-3-8B, Phi-3, **QLoRA** | ASAP-SAS train | ASAP-SAS test | Earlier note |

**IEEE, encoder models (BERT era), SciEntsBank:**
- Zhu, Wu & Zhang, *ASAG via BERT-based deep neural networks*, IEEE TLT 15(3), 2022 ([doi:10.1109/TLT.2022.3175537](https://ieeexplore.ieee.org/document/9779091)). Fine-tuned BERT on SemEval-2013 and Mohler (abstract).
- Sahu & Bhowmick, IEEE TLT 13(1), 2020 ([doi:10.1109/TLT.2019.2897997](https://ieeexplore.ieee.org/document/8636967)). Feature ensembles, evaluated on SciEntsBank and Beetle (abstract).

**Not found:** an IEEE paper that fine-tunes a decoder LLM (Llama, Qwen, Gemma, Mistral) with LoRA or QLoRA on SciEntsBank. IEEE Xplore has 6 records that mention SciEntsBank at all.

## What it means for this project

1. **Keep SciEntsBank.** Our fine-tuning setup (train split; test on test_ua, test_uq, test_ud; 3-way accuracy and F1) is the same one the literature uses, including the closest LLM paper (CHiL(L)Grader, Qwen2.5-7B + LoRA).
2. **Report test_uq and test_ud as the headline for any fine-tuned E2B or E4B.** Mark test_ua as "questions seen in training". CHiL(L)Grader drops test_ua altogether. EngSAF shows the gap: 23 points between seen and unseen questions.
3. **Answering "leakage" with evidence we already have:** the Beetle + SAF adapters never saw SciEntsBank. Saur & Toth use the same cross-dataset idea as their generalisation test. E2B's Beetle + SAF adapter scores 69.3 / 76.4 / 68.4%, against untouched E4B's 66.7 / 67.5 / 70.4%.
4. **Pretraining contamination** isn't tested by anyone found here. A cheap check exists: give the model a SciEntsBank question plus the start of a reference or student answer, and see whether it completes the text word for word (Golchin & Surdeanu, *Time travel in LLMs: tracing data contamination*, ICLR 2024). This is optional, and it hasn't been checked against our setup yet.

## Gaps

- IEEE full texts weren't readable here, only abstracts. The lab's IEEE access would confirm Kazi & Kahanda's exact splits and numbers.
- Aggarwal et al.'s final venue isn't confirmed. The Frontiers paper cites it as 2025 in "Artificial Intelligence in Education".
