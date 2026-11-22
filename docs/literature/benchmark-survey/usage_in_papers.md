# Benchmarks used in recent LLM short-answer grading papers (2024-2026)

Searched 2026-11-05. **How to read this file:**

- **This is a search-led sample, not a census.** Several queries named SciEntsBank, Beetle, Mohler, SAF and ASAP-SAS, so those sets are over-represented. The counts show that a dataset is still in use. They do not give its true share of the field.
- **Evidence codes in the table:** **F** = paper text read (HTML or PDF). **A** = abstract or listing page only. **S** = search snippet only, so treat as unverified.
- **Publisher pages:** ScienceDirect, ACM DL, MDPI and Springer refused automated reading (403 or login redirect). Journal papers are therefore under-sampled against arXiv and ACL Anthology.

## Which datasets, models and metrics did recent papers use?

### Takeaway
In 35 short-answer papers, SciEntsBank is the most used set (12 rows, 9 of them from 2025-2026), then Mohler and ASAP-SAS (8 each), then Beetle (6). SAF is rare (2). Metrics follow the dataset: accuracy and macro-F1 on SemEval sets, QWK on ASAP-SAS, RMSE and Pearson on Mohler and SAF.

### Cited Findings

**Usage table.** "Status" is PR (peer-reviewed), WS (workshop or work-in-progress track; review depth not checked) or Preprint. Models under 10B are in **bold**.

| # | Paper | Year | Status | Ev. | Datasets | Models | Metrics |
|---|---|---|---|---|---|---|---|
| 1 | [Cong et al., IRT for LLM graders](https://aclanthology.org/2026.bea-1.19/) | 2026 | PR (BEA) | F | SciEntsBank, Beetle | 17 open models, 0.8B-14B, 14 under 10B (**Gemma 3 1B/4B, Qwen3.5 0.8-9B, Llama 3.x 1-8B, Phi**) | Accuracy, macro-F1, IRT ability |
| 2 | [Cong et al., Confidence estimation](https://link.springer.com/chapter/10.1007/978-3-032-29755-6_8) ([arXiv](https://arxiv.org/abs/2605.00200)) | 2026 | PR (AIED) | F | SciEntsBank | gpt-oss-20b | ROC / accuracy-rejection, Brier, ECE, MCE |
| 3 | [Vasquez Ferrer et al., When can we trust LLM graders?](https://arxiv.org/html/2603.29559v1) | 2026 | Preprint | F | RiceChem, SciEntsBank, Beetle | 7 models incl. **Llama 3.1 8B, Qwen3-4B-Thinking**; GPT-OSS 20B/120B | ECE, Brier, AUC, coverage, accuracy |
| 4 | [Seuti & Ray Choudhury, S-GRADES](https://arxiv.org/abs/2603.10233) | 2026 | Preprint | F | 14 sets. Short: ASAP-SAS, Mohler, Beetle, SciEntsBank, RiceChem, OS_Dataset, ReGrading. Essay: ASAP-AES, ASAP++, ASAP 2.0, PERSUADE 2, IELTS x2, CSEE | GPT-4o-mini, Gemini 2.5 Flash, Llama 4 Scout | QWK, Pearson, MAE, RMSE; accuracy, F1 for SemEval sets |
| 5 | [Raikote et al., CHiL(L)Grader](https://arxiv.org/html/2603.11957) | 2026 | Preprint | F | DAMI (course data, release not stated), SciEntsBank, EngSAF | **Qwen2.5-7B LoRA**; baselines **Llama 3.1 8B, Llama 3.2 3B, Gemma 3 4B, Mistral 7B**, Gemma 3 12B | QWK, ECE, exact match, off-by-1, MAE, coverage |
| 6 | [Saur & Toth, Grading at scale](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1829089/full) | 2026 | PR (Frontiers in CS) | F | GRAS (new, synthetic), SciEntsBank | **Qwen3 0.6B, Llama 3.2 1B, Flan-T5 L, GPT-2 L, LoRA**; GPT-4o | Macro-F1, QWK |
| 7 | [McGrane & Torres Irribarra, How much training data is enough?](https://aclanthology.org/2026.aimecon-wip.39.pdf) | 2026 | WS (AIME-Con WIP) | F | ASAP-SAS | Fine-tuned GPT-4o-mini, gpt-oss-20b, gpt-oss-120b | QWK, Williamson criteria, PRMSE |
| 8 | [Wang et al., Learnable assessment skills](https://arxiv.org/html/2605.29274) | 2026 | Preprint | F | ASAP-SAS | **Qwen3.5-9B** scorer; GPT-5.4 Thinking diagnoser | QWK |
| 9 | [Gurin Schleifer et al., Quality-conditioned agreement](https://arxiv.org/abs/2605.07647) | 2026 | Preprint | F | Own Hebrew biology items (2 items) | GPT-5.2, GPT-4o, Claude Opus 4.5, fine-tuned DictaBERT | Own agreement analysis; states it does not use standard metrics such as QWK |
| 10 | [Chu et al., CARO](https://arxiv.org/html/2603.00451) | 2026 | Preprint | F | Own: physics, elementary science, teacher education | GPT-4o-mini | Accuracy, Cohen's kappa |
| 11 | [BeSTraP paper, Data 11(3)](https://doi.org/10.3390/data11030057) (authors not captured) | 2026 | PR (MDPI Data) | A | BeSTraP (new, 651 answers), Mohler | Fine-tuned GPT-2, LLMs (names not seen) | Pearson |
| 12 | [Kuichua Kamdem et al., Per-question adaptation](https://publications.polymtl.ca/78521/) | 2026 | PR (UMAP) | S | Mohler | Gemini, Claude, GPT (versions not seen) | RMSE, QWK, Pearson |
| 13 | [Deng et al., Rubric-conditioned LLM grading](https://arxiv.org/html/2601.08843) | 2025-26 | Preprint | F | SciEntsBank | Qwen2.5-72B | Accuracy, Cohen's kappa, Spearman |
| 14 | [Zhao et al., Language models are few-shot graders](https://arxiv.org/html/2502.13337) | 2025 | Preprint | F | Mohler, SAF, SciEntsBank, Proof | GPT-4, GPT-4o, o1-preview | RMSE, Pearson |
| 15 | [Wang et al., AutoSCORE](https://arxiv.org/html/2509.21910) | 2025 | Preprint | F | ASAP-SAS (subsets), ASAP-AES set 1 | GPT-4o, **Llama 3.1 8B**, Llama 3.1 70B | Accuracy, QWK, Pearson, Spearman, MAE, RMSE |
| 16 | [Lai et al., SAS-Bench](https://arxiv.org/abs/2505.07247) | 2025 | Preprint | F | SAS-Bench (new, Chinese, Gaokao-based) | Many, incl. **Qwen3-8B, Llama3-8B, DeepSeek-R1-Distill-7B** up to DeepSeek-V3/R1 | CCS (new), QWK |
| 17 | [Walsh et al., Fine-tuning for better few-shot prompting](https://arxiv.org/html/2508.04063) | 2025 | WS (EvalLAC) | F | OpenTutor (148 labelled; release not stated) | GPT-4o-mini, **Llama 3.1 8B QLoRA** | Accuracy, precision, recall, F1 |
| 18 | [Todorov et al., Better short answer scoring](https://www.scitepress.org/Papers/2025/132917/132917.pdf) | 2025 | PR (CSEDU) | F | ASAP-SAS | BERT, full and LoRA fine-tuning (title says LLM; model is BERT) | QWK |
| 19 | [Desmarais et al., Sentence similarity and few grades](https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.short-papers.124/) | 2025 | PR (EDM short) | F | Mohler | Embeddings: mxbai-embed-large, text-embedding-3-small | RMSE |
| 20 | [Ferreira Mello et al., GPT-4 vs traditional models](https://dl.acm.org/doi/10.1145/3706468.3706481) ([open copy](https://researchmgt.monash.edu/ws/files/722528341/675322817-oa.pdf)) | 2025 | PR (LAK) | F | Mohler, PT_ASAG (Portuguese) | GPT-4 family; BERT, SVM, RF, BiLSTM | MAE, RMSE |
| 21 | [Chu et al., GradeRAG](https://arxiv.org/html/2504.05276) | 2025 | Preprint | F | Own NGSS middle-school science | GPT-4o-mini | Accuracy, weighted F1, Cohen's kappa |
| 22 | [Dadu et al., Grade Guard](https://arxiv.org/abs/2504.01253) | 2025 | Preprint | F | Mohler | Solar Pro, Solar Mini, Gemini 1.5 Flash, GPT-4o-mini | RMSE |
| 23 | [NodeGrade evaluation](https://dl.acm.org/doi/10.1145/3723010.3723021) | 2025 | PR (ACM; venue not seen) | S | SciEntsBank | LLM-based tool (not seen) | F1 |
| 24 | [Funayama et al., Cross-prompt pre-finetuning](https://doi.org/10.1007/s40593-025-00474-w) ([2024 preprint](https://arxiv.org/html/2408.13966)) | 2025 | PR (IJAIED) | A | RIKEN (Japanese, public); checked on the preprint only | BERT; journal version adds open LLMs incl. **8B** | QWK |
| 25 | [Frohn et al., Impacts of model, item, rubric](https://tylerburleigh.com/research/articles/frohn-et-al-2025) | 2025 | PR (AIED) | A | Not named; [later paper](https://arxiv.org/abs/2604.26954) describes high-school maths and ELA items | Several LLMs | Not seen |
| 26 | [Jiang & Bosch, Short answer scoring with GPT-4](https://pnigel.com/papers/jiang-2024-2TEQ5JEF.pdf) | 2024 | PR (L@S) | F | ASAP-SAS | GPT-4 | Accuracy, QWK |
| 27 | [Chamieh et al., LLMs in short answer scoring](https://aclanthology.org/2024.bea-1.25) | 2024 | PR (BEA) | F | ASAP-SAS, Powergrading, MindReading | GPT-3.5, GPT-4, **LLaMA-2 7B**, 13B, 70B; BERT, SVM | QWK |
| 28 | [Wang & Ormerod, RAG for short answer scoring](https://arxiv.org/html/2408.03811v1) | 2024 | Preprint | F | Beetle, SciEntsBank | Claude 3 Haiku, 3.5 Sonnet | Accuracy, macro-F1, weighted F1 |
| 29 | [Ormerod & Kwako, GPU-poor](https://arxiv.org/abs/2407.01873) | 2024 | Preprint | F | ASAP-SAS, ASAP-AES | **Mistral 7B, Gemma 1.1, Llama-3-8B, Phi-3, QLoRA** | QWK |
| 30 | [Meyer et al., ASAG2024](https://arxiv.org/abs/2409.18596) ([thesis](https://arxiv.org/pdf/2506.12066)) | 2024 | PR (SIGCSE Virtual) | F | SciEntsBank, Beetle, Mohler, SAF, Stita, CU-NLP, DigiKlausur | **Llama-3-8B, PrometheusII-7B**, GPT-3.5, GPT-4o, BART-SAF | Per-dataset error on a 0-1 scale (exact metric not re-checked) |
| 31 | [Aggarwal et al., EngSAF](https://arxiv.org/abs/2407.12818) | 2024-25 | Preprint (venue not confirmed) | A | EngSAF (new, about 5.8k) | **Mistral-7B** fine-tuned; Llama-2 | Accuracy |
| 32 | [Chu et al., GradeOpt](https://arxiv.org/abs/2410.02165) | 2024 | Preprint | F | Own: teacher maths knowledge, high-school physics | GPT-4o and optimised prompts | Accuracy, Cohen's kappa, QWK |
| 33 | [Lee et al., LLMs and chain-of-thought](https://doaj.org/article/928c0f01d762405a987c831e3968deb5) | 2024 | PR (C&E: AI) | S | Own: 6 science tasks, 1,650 answers | GPT-3.5, GPT-4 | Accuracy |
| 34 | [Henkel et al., Can LLMs grade short-answer reading comprehension?](https://arxiv.org/abs/2310.18373) | 2023-24 | Preprint | A | New Ghana reading set | GPT-4 and others | QWK, F1 |
| 35 | [Kortemeyer, GPT-4 on ASAG](https://arxiv.org/abs/2309.09338) ([Springer](https://link.springer.com/article/10.1007/s44163-024-00147-y)) | 2023-24 | PR (Springer; page not readable) | A | SciEntsBank, Beetle | GPT-4 | Not re-checked |

**Dataset counts (rows above; a row can use several sets).** Counts were checked by script against the table.

| Dataset | All 35 rows | 2025-26 rows (1-25) | Same, dropping S rows | Status in 2025-26 |
|---|---|---|---|---|
| SciEntsBank | 12 | 9 | 8 | Most used here; still current |
| Mohler (Texas) | 8 | 7 | 6 | Still in use, mostly for RMSE or Pearson |
| ASAP-SAS | 8 | 5 | 5 | Still in use, QWK by default |
| Beetle | 6 | 3 | 3 | In use, usually paired with SciEntsBank |
| RiceChem | 2 | 2 | 2 | Newer, long-answer |
| SAF | 2 | 1 | 1 | Rare; one direct evaluation in 2025-26 |
| EngSAF | 2 | 1 | 1 | Newer; rare |

- Everything else appeared once as a named short-answer set (PT_ASAG, RIKEN, Powergrading, MindReading, SAS-Bench, GRAS, BeSTraP, Proof, OS_Dataset, ReGrading, Stita, CU-NLP, DigiKlausur).
- **Metric conventions:**
  - The IRT paper says ASAG with LLMs is commonly judged with macro-F1 and Cohen's kappa — [Cong et al.](https://arxiv.org/html/2605.00238v1)
  - S-GRADES uses QWK as primary, with accuracy, F1, precision and recall for Beetle and SciEntsBank because they are classification sets — [Seuti & Ray Choudhury](https://arxiv.org/abs/2603.10233)
  - Zhao et al. normalise Mohler, SAF and SciEntsBank to percentages and report RMSE and Pearson — [Zhao et al.](https://arxiv.org/html/2502.13337)
- **Comparable figures on the project's dataset family:**
  - Five-way SciEntsBank, zero training: gemma-3-4b-it accuracy 0.467, macro-F1 0.377; gemma-3-12b-it 0.546, 0.474. This uses all responses, with no train/test split described, so it does not match test_uq or test_ud — [Cong et al.](https://arxiv.org/html/2605.00238v1)
  - Qwen2.5-72B: 76% (2-way) falling to 57% (5-way) — [Deng et al.](https://arxiv.org/html/2601.08843)
- **ASAP-SAS headline:** GPT-4 reached QWK .677 across the 10 items — [Jiang & Bosch](https://pnigel.com/papers/jiang-2024-2TEQ5JEF.pdf). Open-weight fine-tuned models met operational criteria on 7 of 10 items, with a median of 50 responses per score point among passing items — [McGrane & Torres Irribarra](https://aclanthology.org/2026.aimecon-wip.39.pdf).
- **Correction to a search lead:** the "Cohen's kappa 0.70" attached to the LAK 2025 paper is that paper's summary of Henkel et al., not its own result. Its own comparison is MAE and RMSE, with BERT at 0.59 / 1.02 on Mohler — [Ferreira Mello et al.](https://researchmgt.monash.edu/ws/files/722528341/675322817-oa.pdf)

**Essay-only papers, kept out of the counts (mostly snippet-level, unverified):**
- [Sunkavalli 2026](https://arxiv.org/abs/2608.29517): ASAP and ENEM, 12 LLM judges, preprint (F).
- [ASAP 2.0 corpus paper](https://www.sciencedirect.com/science/article/pii/S1075293525000418), Assessing Writing (S).
- [LLMs Do Not Grade Essays Like Humans](https://www.alphaxiv.org/abs/2603.23714): snippet reports LLM QWK typically below 0.30 against about 0.72 human agreement on ASAP (S).
- [EssayJudge](https://arxiv.org/pdf/2502.11916) (S).
- [Long-context AES](https://arxiv.org/pdf/2509.10417) (S).
- [LLM essay scoring under holistic and analytic rubrics](https://arxiv.org/html/2604.00259) (S).
- [Evaluating LLM-based AES](https://aclanthology.org/2025.aimecon-wip.9.pdf) (S).

### Inferences
- All five named sets are still alive in 2024-2026, but not equally. SciEntsBank, Mohler and ASAP-SAS are the working standards; Beetle mostly rides along with SciEntsBank; SAF is marginal.
- There is no single standard. Each community keeps its own set and metric, which is why S-GRADES and ASAG2024 exist. Results across sets are rarely comparable.
- Group effects inflate some counts. Two SciEntsBank rows are one group (Cong et al.), and three private-data rows are one group (Chu et al.).
- ASAP 2.0, the project's grading benchmark since today, is an essay set. Among the 35 short-answer rows it appears only in S-GRADES. So "ASAP 2.0 with QWK" fits essay-scoring practice, and reporting it as short-answer grading would need care.

### Gaps
- Row 30's metric, row 35's metrics and row 31's venue were not confirmed. Row 12's and 23's datasets rest on snippets.
- [Cong et al., LAK 2026, "From Memorization to Reasoning"](https://www.solaresearch.org/events/lak/lak26/accepted-papers/) is a real listing but its text was not reachable, so it is not in the table. The AIED paper cites it for gpt-oss-20b being state of the art for ASAG.
- No Expert Systems with Applications, IEEE TLT or Education and Information Technologies 2025-2026 paper was confirmed. This may be a search and access limit, not an absence.
- IJAIED leads not read: [Andersen et al. on fairness in ASAS; Impey et al.; Pecuchova & Benko](https://link.springer.com/journal/40593/articles).

## Are there 2024-2026 surveys, and what do they say about benchmarks?

### Takeaway
Yes, but few and thin on benchmark choice. The one read in full reviews 30 studies and criticises mixed datasets and metrics. No survey found discusses contamination. The clearest benchmark critiques come from the benchmark papers themselves.

### Cited Findings
- **BEA 2025 PRISMA survey** (peer-reviewed): search from 1 Jan 2022 to 14 Jan 2025; 104 records, 48 full-text checks, 30 included (19 peer-reviewed, 11 preprints) — [Tania-Amanda Nkoyo et al.](https://aclanthology.org/2025.bea-1.35.pdf)
  - Different datasets, metrics and protocols limit comparison, for example ASAP against proprietary essay data.
  - Over 60% of studies are in higher education. Real classroom use is scarce and most studies are controlled experiments.
  - It calls ASAP the standard essay benchmark.
  - It recommends domain-specific fine-tuning, standard prompting practice, multimodal assessment and more classroom validation.
  - The word "contamination" does not appear in the text.
- **S-GRADES:** AES and ASAG progressed in isolation with "fragmented datasets, inconsistent metrics"; all 14 of its datasets are English and text-only — [Seuti & Ray Choudhury](https://arxiv.org/abs/2603.10233)
- **ASAG2024:** no comprehensive benchmark existed across subjects, scales and score distributions; LLM methods set new highs but remain "far from reaching human performance" — [Meyer et al.](https://arxiv.org/abs/2409.18596)
- **On QWK:** reporting it alone can overstate operational readiness. The paper adds the Williamson criteria and PRMSE — [McGrane & Torres Irribarra](https://aclanthology.org/2026.aimecon-wip.39.pdf)
- **On aggregate metrics:** macro-F1 and kappa say little about how grading varies with response difficulty — [Cong et al.](https://arxiv.org/html/2605.00238v1)
- **On annotation depth:** most existing datasets hold raw responses and ignore step-wise answers — [SAS-Bench](https://arxiv.org/abs/2505.07247)
- **English dominance** (snippet level): applications generally use English datasets — [Kara et al. 2024](https://asianjde.com/ojs/index.php/AsianJDE/article/view/768). [Cüvitoğlu 2025 (OSF preprint)](https://www.researchgate.net/publication/397771822_Automated_Short_Answer_Grading_from_an_Educational_Assessment_Perspective_A_Systematic_Review) covers ASAG 2014-2024, but only its existence and scope were confirmed.
- **Non-English sets exist but are scattered:** PT_ASAG in [Ferreira Mello et al.](https://researchmgt.monash.edu/ws/files/722528341/675322817-oa.pdf), RIKEN in [Funayama et al.](https://arxiv.org/html/2408.13966), Chinese in [SAS-Bench](https://arxiv.org/abs/2505.07247), Hebrew in [Gurin Schleifer et al.](https://arxiv.org/abs/2605.07647)

### Inferences
- The field's stated complaints are fragmentation, English dominance, metric choice and weak classroom validity. Dataset age is seldom argued as a problem on its own, and contamination is mostly missing from reviews.
- The reviews do not tell authors which benchmark to pick. The closest thing to a recommendation is "use several sets and report more than one metric".

### Gaps
- No dedicated 2026 survey of LLM grading benchmarks was found.
- [Other review leads](https://www.researchgate.net/publication/404520474_Automatic_Short_Answer_Grading_ASAG_In_Education_A_Review_Of_AI_And_Deep_Learning_Techniques) were not readable. An unattributed snippet said most 2014-2024 ASAG studies use a few English sets; that source was not pinned down.

## Do recent papers use private classroom data instead of public benchmarks?

### Takeaway
Often, but public sets are still the majority here. In 35 rows, 7 rely mainly on own or unreleased data, 1 mixes it with public sets, and 1 does not name its data. That is about 1 in 4, probably an underestimate because searches targeted public sets.

### Cited Findings
- **Mainly own data (rows 9, 10, 17, 21, 32, 33, 34):**
  - Two Hebrew biology items, 152 students in the second cycle — [Gurin Schleifer et al.](https://arxiv.org/abs/2605.07647)
  - Three Michigan State group sets — [CARO](https://arxiv.org/html/2603.00451), [GradeOpt](https://arxiv.org/abs/2410.02165), [GradeRAG](https://arxiv.org/html/2504.05276)
  - 148 labelled OpenTutor answers — [Walsh et al.](https://arxiv.org/html/2508.04063)
  - Six science tasks, 1,650 answers — [Lee et al.](https://doaj.org/article/928c0f01d762405a987c831e3968deb5)
  - Over 150 students in Ghana — [Henkel et al.](https://arxiv.org/abs/2310.18373)
- **Mixed:** DAMI, 4,031 course answers, next to SciEntsBank and EngSAF — [CHiL(L)Grader](https://arxiv.org/html/2603.11957)
- **Newly built and public or described as such:** [GRAS](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1829089/full) (synthetic; GPT-4o wrote answers and also served as grader, which the authors say may favour it), [SAS-Bench](https://arxiv.org/abs/2505.07247), [EngSAF](https://arxiv.org/abs/2407.12818).
- **Review view:** some essay studies use ASAP and others proprietary data, which blocks comparison — [BEA 2025 survey](https://aclanthology.org/2025.bea-1.35.pdf)
- **Finnish course study** (second-hand): GPT-4 reached QWK of at least 0.6 in only some settings. Jiang & Bosch report "44 out of 100 courses", while a search summary said 44% of one-shot settings over ten courses, so the exact unit is unclear — [Jiang & Bosch](https://pnigel.com/papers/jiang-2024-2TEQ5JEF.pdf)

### Inferences
- Private data is the norm for education-science groups studying a real course or curriculum. Public sets are the norm for method papers from NLP groups.
- Private sets avoid contamination but cannot be replicated. The project's public-benchmark route gives comparability at the cost of that risk.

### Gaps
- Release status was often unstated ("does not say"), so "private" here means "not shown to be public". No precise field-wide rate was found.

## Do papers use small or open models, fine-tuning or on-device settings?

### Takeaway
Small open models and LoRA or QLoRA are now common, on SciEntsBank, ASAP-SAS and own data. On-device work is almost absent: one paper timed CPU inference on a laptop. No paper found measured power or energy, or ran on an edge board.

### Cited Findings
- **Under-10B models tested in 13 rows** (1, 3, 5, 6, 8, 15, 16, 17, 24, 27, 29, 30, 31).
- **Size is not a simple predictor:** on five-way SciEntsBank, Qwen3.5-4B (0.538) beat Qwen3.5-9B (0.508), and Llama-3.1-8B got 0.379 — [Cong et al.](https://arxiv.org/html/2605.00238v1)
- **On-device:** Qwen3 0.6B graded 521 answers in 273.65 s on a ThinkPad i7-8665U, about 0.52 s each, with loading excluded. Training was on two RTX 6000 GPUs — [Saur & Toth](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1829089/full)
  - Small models beat GPT-4o in-domain on synthetic GRAS (QWK 0.96 against 0.93).
  - Cross-domain they lost: GRAS to SciEntsBank QWK 0.43 against 0.67.
- **Fine-tuning results:**
  - QLoRA at 7-8B on one 24 GB GPU gave adequate but not state-of-the-art ASAP-SAS scores; ensembles of smaller LMs stayed better — [Ormerod & Kwako](https://arxiv.org/abs/2407.01873)
  - Mistral-7B was best on EngSAF: 75.4% unseen answers, 58.7% unseen questions — [Aggarwal et al.](https://arxiv.org/abs/2407.12818)
  - Qwen2.5-7B LoRA with calibration and human corrections (simulated from gold labels) reached QWK 0.882 on DAMI's held-out split, against 0.669 for plain fine-tuning — [CHiL(L)Grader](https://arxiv.org/html/2603.11957)
  - Llama 3.1 8B QLoRA on 148 examples moved little; adding synthetic data lifted F1 to 0.653 — [Walsh et al.](https://arxiv.org/html/2508.04063)
  - Full fine-tuning beat LoRA across all ASAP-SAS sets, though the model was BERT — [Todorov et al.](https://www.scitepress.org/Papers/2025/132917/132917.pdf)
  - An 8B model reached human-level performance with two-step pre-finetuning (search summary of the journal version) — [Funayama et al.](https://doi.org/10.1007/s40593-025-00474-w)
- **Adjacent, not short-answer benchmarks:** fine-tuned Llama 3.2 3B beat zero-shot Llama 3.3 70B on learning-strategy rubrics, weighted F1 0.69 against 0.43 on self-explanation — [Dutulescu et al.](https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.short-papers.109/index.html)
- **Argument for local models:** API use raises transparency and security concerns — [Ormerod & Kwako](https://arxiv.org/abs/2407.01873); privacy and GPU cost — [Saur & Toth](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1829089/full)

### Inferences
- The project's niche looks open: accuracy against measured energy and latency on an edge board. Nothing in this sample reports J per answer, and no Gemma 4 grading paper appeared.
- The best-matched published comparisons are rows 1, 5 and 6 on SciEntsBank, and rows 7, 8 and 29 on ASAP-SAS. For ASAP 2.0 the sample holds no small-model, edge or fine-tuning comparison among the short-answer rows.
- Cross-domain results suggest reporting unseen-question and unseen-domain splits, as the project already does for SciEntsBank.

### Gaps
- No edge-board study was found, but only one direct search targeted on-device grading. This is "not found", not "does not exist".

## Is there evidence of benchmark contamination?

### Takeaway
No direct evidence that old short-answer sets were memorised, and no direct proof they were not. Only one short-answer paper in this sample discusses it, and it finds no clear sign while saying it cannot be ruled out. One essay paper's verbatim test found nothing, with stated limits.

### Cited Findings
- **Short-answer, the only explicit discussion found:** ASAP-SAS has been public since 2012, so items, rubrics and responses may have been seen in pretraining. Zero-shot baselines were poor on every item, and gpt-oss models were weakest despite later release. That gives no clear evidence of recall but cannot rule out contamination; the authors call for replication on an unreleased item bank — [McGrane & Torres Irribarra](https://aclanthology.org/2026.aimecon-wip.39.pdf)
- **Essay, a direct test:** continuation probes on 150-word prefixes of ASAP and ENEM essays. 0 of 22 judge-by-corpus cells were flagged. The author reads this as no detectable verbatim memorisation, "not proof of no contamination", and calls such probes weak detectors. Preprint, under review — [Sunkavalli 2026](https://arxiv.org/html/2608.29517)
- **Silence elsewhere:**
  - A text search for "contaminat" over 13 extracted papers hit only McGrane & Torres Irribarra. It covered rows 2, 4, 7, 9, 16, 18, 19, 20, 26, 27, 29, 32 and the BEA survey; a word split across lines could be missed.
  - Reads of rows 1, 3, 5, 6, 13 and 17 found no statement on pretraining overlap.
- **Exposure of the public sets:** ASAP 2.0 is described as having publicly available test labels — [S-GRADES](https://arxiv.org/abs/2603.10233). SciEntsBank's five-way version was said to be no longer publicly accessible in 2024 — [Wang & Ormerod](https://arxiv.org/html/2408.03811v1). The project's outline lists an HF mirror, so it has been re-hosted.
- **General LLM background** (not grading-specific, snippet level): small amounts of contamination can inflate scores, and many open models do not disclose training data — [EMNLP 2025 survey](https://aclanthology.org/2025.emnlp-main.511.pdf).

### Inferences
- The risk is plausible for all five sets, since they have been public for years. Evidence is indirect. Weak zero-shot scores on 5-way SciEntsBank and ASAP-SAS argue against strong label memorisation but do not settle it.
- Memorised labels and real skill cannot be told apart from accuracy alone. A cheap check for this project would be a continuation probe on Gemma 4, or a small unreleased set, in line with the McGrane & Torres Irribarra call.
- Gemma 4's January 2025 cutoff (project outline) comes before ASAP 2.0's 2025 paper, but the source corpus PERSUADE was published earlier and the Kaggle release date was not checked here.

### Gaps
- No study was found that tests SciEntsBank, Beetle, Mohler, SAF or ASAP-SAS for memorisation directly.
- The LAK 2026 "From Memorization to Reasoning" paper might bear on this, but its listing describes fine-tuning against prompting, and the full text could not be read.
