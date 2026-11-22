# English short answer grading datasets and benchmarks, 2023-2026

Checked 2026-11-05. "Own check" means I read the dataset's API, repository tree or files myself, and the linked URL is where to repeat it. MDPI pages returned 403 to my fetcher, so facts on JorGPT, BeSTraP and the Covenant set come from Zenodo, Crossref or search snippets of those pages, and are marked.

## Which new English short-answer grading datasets or benchmarks exist?

### Takeaway
Few are genuinely new. Real-student English sets released since 2023 are EngSAF, JorGPT, the OS dataset, MIME, the Ghana ROARS set, RiceChem, BeSTraP, the Covenant set, DAMI and Regrading 2JC. ASAG2024 and S-GRADES are repackagings of older data, and SAS-Bench is neither English nor real-student, despite calling its data "student responses".

### Cited Findings

**New real-student data**
- EngSAF (2024 preprint, AIED 2025): about 5.8k answers, 119 questions, engineering exams — [arXiv](https://arxiv.org/html/2407.12818v2); [repo README](https://github.com/dishankaggarwal/EngSAF)
- JorGPT (Data, March 2026): about 3,041 answers, 50 CS questions, 79 students — [Zenodo](https://zenodo.org/records/18981627)
- OS dataset, "Grade Like a Human" (2024): undergraduate operating systems course at City University of Hong Kong, 6 questions, three TAs per answer — [arXiv](https://arxiv.org/pdf/2405.19694); [repo](https://github.com/wenjing1170/llm_grader)
- MIME / MAP Kaggle competition (2025): 52,463 explanations on 15 maths items, of which 38,095 are authentic and 14,368 synthetic — [AIME-Con 2026 paper](https://aclanthology.org/2026.aimecon-main.33.pdf)
- Ghana ROARS dataset (2023): about 1,000 short answers from reading assessments in Ghana — [arXiv](https://arxiv.org/pdf/2310.18373)
- RiceChem (2024): 1,264 college chemistry answers, 27 rubric items, 8,392 data points, framed as long-answer grading — [arXiv](https://www.arxiv.org/pdf/2404.14316); its S-GRADES card gives an average of 120 words per answer — [HF card](https://huggingface.co/datasets/nlpatunt/D_Rice_Chem)
- BeSTraP (Data, 2026): 651 answers from a Database Transaction exam at Beni-Suef University — [DOI](https://doi.org/10.3390/data11030057)
- Covenant University set (Data, June 2025): handwritten answers from two courses, transcribed — [Crossref](https://api.crossref.org/works/10.3390%2FDATA10060087)
- DAMI (2026): 4,031 anonymised answers from a second-cycle Data Mining course — [CHiL(L)Grader](https://arxiv.org/html/2603.11957)
- Regrading 2JC (Gao et al. 2024): rubric-graded conceptual engineering answers, shared by its authors — [HF card](https://huggingface.co/datasets/nlpatunt/D_Regrading_Dataset_J2C)
- LLM-as-a-Grader (Nov 2025): about 50 undergraduates in Computational Linguistics, 258 graded quiz cases compared — [arXiv](https://arxiv.org/html/2511.10819v1)

**Combined benchmarks of older data**
- ASAG2024 (2024): seven existing English sets, 18,988 rows — [paper](https://ar5iv.labs.arxiv.org/html/2409.18596)
- S-GRADES (LREC 2026): web benchmark of 14 datasets with a public leaderboard, covering essays and short answers — [arXiv](https://arxiv.org/abs/2603.10233); [repo](https://github.com/nlpatunt/sgrades). ASAG side: ASAP-SAS, Regrading 2JC, CSEE, Mohler, Beetle, SciEntsBank, RiceChem, OS dataset (Table 1 of the paper). It lists no new data of its own.

**Named in the brief, but not a fit**
- SAS-Bench is LLM-generated, not student-written: "we leverage LLMs to simulate student responses" using GPT-4o-mini and DeepSeek-V3, from Gaokao questions — [arXiv v2](https://arxiv.org/html/2505.07247v2). The README and abstract still say "student responses" — [repo](https://github.com/PKU-DAIR/SAS-Bench)
- SAS-Bench is tagged `zh`; English is 49 of 1,030 questions and 245 of 4,109 responses, all gap-filling — [HF card](https://huggingface.co/datasets/aleversn/SAS-Bench); [arXiv v2](https://arxiv.org/html/2505.07247v2)
- NAEP reading challenge (2021): 20 item-specific and 2 generic items, confidential data — [challenge repo](https://github.com/NAEP-AS-Challenge/reading-prediction)
- NAEP maths challenge (2023): ten items for grades 4 and 8 — [UF notice](https://education.ufl.edu/educational-research/2023/04/04/spring-2023-naep-math-assessment-item-score-prediction-challenge/)
- "Pistis": no dataset of that name found. The top search hit, the bachelor's thesis behind ASAG2024, does not contain the word (own check) — [thesis](https://arxiv.org/pdf/2506.12066)
- "CLASSE": no dataset of that name found in search.
- ASSISTments open-response maths data is used in many papers, but one reuse was "with permission from the original authors and in compliance with IRB" — [paper](https://par.nsf.gov/servlets/purl/10285771)
- DrawEduMath (NAACL 2025) is 2,030 images of handwritten maths work, for vision models rather than text grading — [arXiv](https://arxiv.org/html/2501.14877v1)
- MathDial is tutoring dialogue, not grading — [arXiv](https://arxiv.org/abs/2305.14536v2)

**Synthetic, so out of scope for "real students"**
- GRAS (2026): student answers generated with GPT-4o, in neuroscience, psychology and AI; gated on request, Apache-2.0 tag — [listing](https://app.alphaneural.io/datasets/saurluca/GRAS); [HF](https://huggingface.co/datasets/saurluca/GRAS)
- Šindelář et al. (DTF at LREC 2026): LLM-generated surrogates of a PISA set that is itself under NDA, with full, partial and no-credit rubrics — [arXiv](https://arxiv.org/abs/2603.28537); [repo](https://github.com/sindelarp/Context-Dependent-Automated-Short-Answer-Grading-materials)
- Kaggle "Automatic Short Answer Grading Dataset": tagged synthetic, 4000+ primary science Q&As — [Kaggle](https://www.kaggle.com/datasets/mubeenfurqanahmed/automatic-short-answer-grading-dataset)

**Contrast only**
- ASAP-SAS (2012): 10 prompts, QWK; a later paper's table sums to 22,267 public responses (my sum) — [paper](https://ar5iv.labs.arxiv.org/html/2202.11558); press gave "over 27,000" scored — [EdSurge](https://www.edsurge.com/news/hewlett-foundation-announces-winners-of-short-answer-scoring-competition)
- PERSUADE 2.0 (essays): over 25,000 argumentative essays, grades 6-12, 15 prompts — [repo](https://github.com/scrosseye/PERSUADE_corpus)
- BEA 2026 shared task on rubric-based short answer scoring is German, with seen and unseen questions and 2-way and 3-way labels — [report](https://aclanthology.org/2026.bea-1.85/)

### Inferences
- A public, openly licensed, English, real-student set with a built-in unseen-question test did not turn up. EngSAF has that design but is request-only.
- ASAG2024 adds little for this project: SciEntsBank, Beetle, SAF and Mohler are 17,838 of its 18,988 rows (94%), which leaves 1,150 rows from CU-NLP, DigiKlausur and Stita.
- The field's "benchmark" gap is partly a data-rights gap. Many studies built custom datasets ([review of 49 studies](https://mdpi.com/2076-3417/15/10/5683), n = 23), and large item banks (NAEP, PISA) stay confidential.

### Gaps
- Learning Agency Lab: only MIME is short-text; PERSUADE and ASAP 2.0 are essays. I did not check whether any other of its competitions holds short answers.
- "ScienceQA-like grading sets": none found. Stanford, Khan Academy and Carnegie Learning: no public short-answer grading set found, and no link to the NAEP challenges confirmed.
- Medical education: studies exist but no release found. [Grévisse 2024](https://orbilu.uni.lu/handle/10993/62124) graded 2,288 answers in 3 languages; its data statement could not be read (the Springer redirect blocked my fetcher). [Jade and Yartsev 2025](https://arxiv.org/abs/2505.04645) used 557 postgraduate answers. A Harvard study ([listing](https://ctisus.com/learning/journalclub/349828)) and Bolgova et al. (804 answers) came from a search summary only, so treat those as unverified.
- NGSS science: [arXiv 2604.19754](https://arxiv.org/html/2604.19754) uses 1,466 high-school responses. Release status not checked. "PASTA" returned nothing as a dataset.
- EDM, AIED and LAK 2026 proceedings were not read directly; search returned no dataset paper from them, so a recent release could be missing.
- Whether [LAK 2026 "From Memorization to Reasoning"](https://dx.doi.org/10.1145/3785022.3785031) releases data is unconfirmed.

## What are each dataset's details?

### Takeaway
EngSAF has the fullest design: real answers, 3-way labels, reference answers, and unseen-answer and unseen-question tests. JorGPT is the most open recent set, though its English is translated. Most others have fewer than 1,000 answers or no standard split.

### Cited Findings

Fields are given as far as verified; "?" means not verified.

| Dataset | Year, venue | Answers / questions | Subject, level | Real? | Labels | Reference / rubric | Split | Metric used |
|---|---|---|---|---|---|---|---|---|
| EngSAF | 2024; AIED 2025 | 5,814 / 119 | 25 engineering courses, UG and PG | Real answers and grades; feedback by Gemini-1.5-flash | 3-way (0/1/2) + feedback | Reference yes | Train 4,069, UA 980, UQ 765 | Accuracy, macro-F1 |
| JorGPT | 2026, Data | ~3,041 / 50 | CS, university, 79 students | Real; English is a translation | 0-10 + teacher feedback + LLM evaluations | Ideal answer yes | ? | ? |
| OS dataset | 2024, arXiv | ? / 6 | Operating systems, UG | Real | Numeric to question maximum, 3 TAs | Sample answer + criteria | None upstream; S-GRADES uses 5 questions | S-GRADES: QWK and others |
| MIME | 2025-26 | 52,463 / 15 | Maths, grades 4-8, Eedi | 38,095 real, 14,368 paraphrased | 6 categories + misconception type | Correct MCQ option, no reference | ~70% train; all 15 items in train | MAP@3 |
| Ghana ROARS | 2023, arXiv | ~1,000 short / ? | Reading, ages 9-18 | Real | Expert scores | ? | ? | QWK, F1 |
| RiceChem | 2024, arXiv; AIED per search | 1,264 / 4 | Chemistry, college | Real | Binary per rubric item (27 items) | Rubric yes | Script supports unseen-question split | ? |
| BeSTraP | 2026, Data | 651 / ? | Databases, university | Real | 0-5, two instructors | ? | ? | ? |
| Covenant set | 2025, Data | ? / ? | MIS221, MIS415, university | Real, transcribed | Human scores | Marking guide | ? | Spearman, MSE |
| DAMI | 2026, arXiv | 4,031 / 53+ | Data mining, graduate | Real | Max 5, 8 or 10 | Not described | Train 3,770; cal 260; test 261 | QWK, ECE |
| Regrading 2JC | 2024 | 1,979 rows / ? | Engineering concepts | Real (per card) | 0-8 | Rubric-based | 1,583 / 198 / 198 | S-GRADES: QWK |
| ASAG2024 | 2024, SIGCSE Virtual | 18,988 / ? | 7 sets, mixed | Real | 0-1 normalised | Reference yes | 15,190 / 1,899 / 1,899, random | wRMSE |

Sources by row:
- EngSAF: [arXiv](https://arxiv.org/html/2407.12818v2); venue from [README](https://github.com/dishankaggarwal/EngSAF). Best fine-tuned model Mistral-7B reached 75.4 accuracy on unseen answers and 58.7 on unseen questions, while zero-shot GPT-4o got 48.9 on unseen answers.
- EngSAF count differs by source: 5,814 in the paper's split sizes, 5,798 in [CHiL(L)Grader](https://arxiv.org/html/2603.11957).
- JorGPT: [Zenodo](https://zenodo.org/records/18981627); the 0-10 scale, three LLM evaluators and translation come from search snippets of the [MDPI page](https://www.mdpi.com/2306-5729/11/3/59), not full text.
- OS dataset: fields from [repo README](https://github.com/wenjing1170/llm_grader); score ranges and very small test splits (3 to 8 per question) from [S-GRADES Table 1](https://arxiv.org/abs/2603.10233).
- MIME: [paper](https://aclanthology.org/2026.aimecon-main.33.pdf). Top MAP@3 scores were slightly above 0.948.
- Ghana ROARS: [arXiv](https://arxiv.org/pdf/2310.18373). GPT-4 reached QWK 0.92, F1 0.89. Student count is 130 in the PDF and "over 150" in the [abstract listing](https://arxiv.org/abs/2310.18373).
- RiceChem: [arXiv](https://www.arxiv.org/pdf/2404.14316); split script in [repo](https://github.com/luffycodes/Automated-Long-Answer-Grading); venue from a search snippet.
- BeSTraP: search snippets of the [paper](https://doi.org/10.3390/data11030057).
- Covenant: [Crossref](https://api.crossref.org/works/10.3390%2FDATA10060087). Longformer reached Spearman 0.77, MSE 0.04.
- DAMI: [paper](https://arxiv.org/html/2603.11957). Its split sizes sum to 4,291, above the stated 4,031, and the paper does not explain this.
- Regrading 2JC: [card](https://huggingface.co/datasets/nlpatunt/D_Regrading_Dataset_J2C); row counts from the [viewer API](https://datasets-server.huggingface.co/size?dataset=nlpatunt/D_Regrading_Dataset_J2C); range from [S-GRADES](https://arxiv.org/abs/2603.10233).
- ASAG2024: [paper](https://ar5iv.labs.arxiv.org/html/2409.18596); split sizes are an own check of the [HF files](https://huggingface.co/datasets/Meyerger/ASAG2024). All 276 distinct test questions also appear in train, so there is no unseen-question test (own check). The paper reports wRMSE 0.27 for GPT-4o, 0.39 for Llama-3-8B, and 0.40 for a mean predictor.
- ASAG2024's HF viewer API reports 56,646 rows, against 18,988 in the combined file — [size API](https://datasets-server.huggingface.co/size?dataset=Meyerger/ASAG2024).
- SAS-Bench, for contrast: 1,030 questions, 4,109 responses, 9 subjects, test split only, step-wise scores and 65 error causes; metrics CCS, ECS, QWK — [arXiv v2](https://arxiv.org/html/2505.07247v2); [HF](https://huggingface.co/datasets/aleversn/SAS-Bench).
- S-GRADES metrics: QWK, Pearson, MAE, RMSE and accuracy for scores; F1, precision, recall and accuracy for classes — [repo](https://github.com/nlpatunt/sgrades).

### Inferences
- For a project reporting QWK, ordinal sets (JorGPT, BeSTraP, DAMI, Regrading, OS) suit QWK. EngSAF's three ordered labels allow QWK, as CHiL(L)Grader does, though its authors use accuracy and macro-F1.
- The EngSAF gap between 75.4 and 58.7 suggests unseen questions remain the hard case, in line with the BEA 2026 report's finding on German.
- MIME is the only large recent set, but with 15 items it tests diagnosis on known questions, and 27% of it is paraphrased text. It measures something different from grading against a reference.
- RiceChem and the OS dataset sit at the long end of "short answers", and CSEE in S-GRADES is letters and short essays, so check lengths before calling them short-answer results.

### Gaps
- Number of OS dataset answers, and JorGPT's split: not verified. The OS count could be read from the repository's JSON files.
- BeSTraP and Covenant: language, question counts, and splits unverified because MDPI blocked full text. Covenant's answer count was not found.
- DAMI's language and reference answers are not stated.
- Ghana ROARS label scale is in its data dictionary, which I did not open.

## Is the data actually public today?

### Takeaway
Truly open today, with real students and labels: JorGPT, MIME (with a licence conflict), ASAG2024, the OS dataset, and Ghana ROARS. EngSAF and RiceChem are officially request-only, though unofficial or derived copies exist. NAEP is closed.

### Cited Findings

| Dataset | Status on 2026-11-05 | Licence | Location |
|---|---|---|---|
| JorGPT | Open; `dataset_en.csv` 14.6 MB | CC BY 4.0 | [Zenodo v3](https://zenodo.org/records/18981627); [Kaggle mirror](https://www.kaggle.com/datasets/javiersanchezsoriano/jorgpt-student-answers-and-multi-llm-grading) |
| MIME | Open except 700 held-out samples | Conflict, see below | [GitHub](https://github.com/scrosseye/mime_dataset); [Kaggle](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) |
| ASAG2024 | Open, not gated | "Data Source Licenses apply" | [HF](https://huggingface.co/datasets/Meyerger/ASAG2024) |
| OS dataset | Open; no licence file | None stated | [GitHub](https://github.com/wenjing1170/llm_grader) |
| Ghana ROARS | Open; `ROARS.csv` 4.6 MB, dictionary, test PDF; no licence file | None stated | [GitHub](https://github.com/owenhenkel/ROARS-dataset) |
| EngSAF | **Request-only**; repo holds no data | Terms bar redistribution | [Request form via README](https://github.com/dishankaggarwal/EngSAF) |
| RiceChem | **Request-only** upstream | ? | [Google form via README](https://github.com/luffycodes/Automated-Long-Answer-Grading) |
| S-GRADES copies | Open on HF; test labels removed | No licence tag | [HF collection](https://huggingface.co/nlpatunt); [leaderboard](https://sgrades.eng.unt.edu) |
| LLM-as-a-Grader | Sample only (`student_answer.json`, 2.5 KB) | None stated | [GitHub](https://github.com/emorynlp/LLM-Grading) |
| DigiKlausur (in ASAG2024) | Open | MPL-2.0 | [GitHub](https://github.com/DigiKlausur/ASAG-Dataset) |
| SAS-Bench | Open, but Chinese and synthetic | Apache-2.0 | [HF](https://huggingface.co/datasets/aleversn/SAS-Bench) |
| DAMI | No release found | ? | [paper](https://arxiv.org/html/2603.11957) |
| BeSTraP | Not verified | ? | [DOI](https://doi.org/10.3390/data11030057) |
| Covenant set | No repository found | Listing says "None" | [listing](https://www.bohrium.com/paper-details/a-structured-dataset-for-automated-grading-from-raw-data-to-processed-dataset/1137786341291982849-41127) |
| NAEP 2021 | **Closed**: confidential, destroy within 30 days of the challenge | n/a | [challenge repo](https://github.com/NAEP-AS-Challenge/reading-prediction) |
| NAEP maths | **Closed**: no public-use file, NCES Remote Access | n/a | [NSF catalogue](https://sap.nsf.gov/dataset/ed_nces-2024072/metadata/access) |
| ASAP-SAS | Kaggle, after accepting rules | "Subject to Competition Rules" | [Kaggle](https://www.kaggle.com/competitions/asap-sas) |

- **EngSAF mirror.** [IsmaelMousa/engsaf](https://huggingface.co/datasets/IsmaelMousa/engsaf) (created 2026-04-06) is open and tagged Apache-2.0, while the official README says "Redistribution of the dataset is prohibited" — [README](https://github.com/dishankaggarwal/EngSAF).
- The mirror differs from the paper: 5,579 rows, 118 questions, splits 4,735 / 338 / 506, and 31 of its 42 test questions appear in train (own check). The UA and UQ tests are not recoverable from it, and its `mark_scheme` column holds only generic label text (own check).
- **MIME licence conflict.** The README reads "Creative Commons Attribution 4.0 International License (CC BY-NC 4.0)" — [repo](https://github.com/scrosseye/mime_dataset). Kaggle lists the competition licence as MIT — [Kaggle](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings).
- **RiceChem status conflict.** S-GRADES Table 1 marks it public — [arXiv](https://arxiv.org/abs/2603.10233). The authors' repo still asks for a form — [repo](https://github.com/luffycodes/Automated-Long-Answer-Grading). S-GRADES says non-hosted sets were shared with creator permission.
- **S-GRADES labels.** Paper says train and validation carry labels and test does not — [arXiv](https://arxiv.org/abs/2603.10233). The [D_ASAP-SAS card](https://huggingface.co/datasets/nlpatunt/D_ASAP-SAS) says only "Ground truth labels have been removed", without naming a split. Test scoring goes through its [leaderboard](https://sgrades.eng.unt.edu).
- This project recorded a Kaggle 403 on ASAP-SAS because the competition rules were not accepted on the account — `docs/research-outline.md`.
- The Ghana paper calls the set publicly available, yet a later line says it "is not publicly available" — [arXiv](https://arxiv.org/pdf/2310.18373). The repo is live, so that line probably refers to the time of model training (my reading).
- The Ghana repo URL given in search, `owenhenkel/Ghanaian_Student_Reading_Comprehension_Short_Answer_Dataset`, resolves through the GitHub API; the owner's list shows `ROARS-dataset` — [owner](https://github.com/owenhenkel).

### Inferences
- Adoption order for a small-LLM project, on current evidence:
  1. **EngSAF, via the request form.** Best fit to the project's unseen-question design. Costs a wait and a no-redistribution term. Avoid the mirror for headline numbers.
  2. **JorGPT.** Usable today with no approval; run on `dataset_en.csv`, and report that English is translated, from one institution. It needs a project-made split, by question.
  3. **OS dataset or Ghana ROARS** as small out-of-domain checks. No licence file means asking the authors before redistributing anything.
  4. **MIME** only if the project wants misconception diagnosis rather than grading.
- S-GRADES could give the supervisor's "benchmark rather than dataset" point a public leaderboard across ASAP-SAS, Mohler, Beetle and SciEntsBank. Its test labels stay hidden, so results depend on the site staying up.
- Neither SAS-Bench nor ASAG2024 should be cited as a new real-student English test.

### Gaps
- Ghana and OS sets: terms of use are unknown, as no licence file exists.
- Whether MIME marks which rows are synthetic: I saw the README and paper only, not the CSV columns.
- The [S-GRADES site](https://sgrades.eng.unt.edu) was not opened, so whether it is up today is unverified. Its GitHub repo was last pushed 2026-03-24.
- EngSAF request turnaround and approval criteria are unknown.
