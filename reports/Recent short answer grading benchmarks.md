# Old sets still grade new LLMs

Recent papers mostly still test LLM graders on old data: in a sample of 35 short-answer papers from 2024-2026, **SciEntsBank appears 12 times, Mohler and ASAP-SAS 8 each and Beetle 6, while no newer short-answer set appears more than twice**. That count is a sample, not a census, because many searches named the old sets. Newer real-student sets do exist, led by EngSAF, JorGPT and the German ALICE, but **no public, openly licensed English set with real answers and a built-in unseen-question test turned up**. ASAP 2.0 is an essay set and appears among the 35 papers only inside the S-GRADES benchmark, so it can stay the supervisor's main benchmark but cannot alone support a claim about short answers. The best additions are **JorGPT now** (open, CC BY 4.0, answers written after Gemma 4's training cutoff), **EngSAF through its official request form**, and **Haidir & Purwarianti 2020 for the optional Indonesian condition**. The EngSAF Hugging Face mirror should not be used: its Apache-2.0 tag conflicts with EngSAF's ban on redistribution. No paper in the sample reports energy per answer on an edge board, so the project's cost measurements remain its clearest contribution, although Gemma 4 E2B and E4B have already been run as grading baselines elsewhere.

## Three warnings and the terms to know first

> **(a) Usage counts are a sample, not a census.** Several searches named SciEntsBank, Beetle, Mohler, SAF and ASAP-SAS, so those sets are over-represented. The counts show a set is still in use. They do not show its share of the field. ScienceDirect, ACM, MDPI and Springer pages also refused automated reading, so journal papers are under-sampled against arXiv and the ACL Anthology.
>
> **(b) `[deleted copy]` marks numbers from unapproved downloads.** Row and overlap counts for ASAG2024 and the EngSAF Hugging Face mirror came from copies downloaded without the owner's prior approval and then deleted. They cannot be re-checked without a new download. Details are in the section on open checks below.
>
> **(c) `[unverified]` marks anything the notes record as snippet-only or unconfirmed.** Treat it as a lead, not a fact, until someone reads the source.

| Term | Plain meaning |
|---|---|
| **QWK** (quadratic weighted kappa) | Agreement between two graders on an ordered score, after removing agreement expected by chance. 1 is perfect and 0 is chance level. A miss by two points costs four times a miss by one, so it suits ordered scores and not unordered categories. |
| **Accuracy, macro-F1** | Accuracy is the share of exact matches. Macro-F1 averages performance across labels, so rare labels count as much as common ones. |
| **RMSE, Pearson** | RMSE is the typical size of the score error. Pearson shows whether predicted and true scores rise together. |
| **Unseen-answer test** | New students' answers to questions the model trained on. |
| **Unseen-question test** | Answers to questions absent from training. This is the harder test and is closer to a teacher writing a new exam. |
| **Reference answer, rubric** | A model answer, and the written marking rules. A grading prompt needs at least one. |
| **Zero-shot, LoRA/QLoRA** | Zero-shot means no training or examples. LoRA and QLoRA train a small add-on (an adapter) rather than the whole model. |
| **Contamination** | The test data was in the model's training text, which can inflate scores. |

## SciEntsBank leads a sample of 35 papers, and each community keeps its own metric

The old sets are alive, but not equally. SciEntsBank, Mohler and ASAP-SAS are the working standards, Beetle mostly travels with SciEntsBank, and SAF is marginal, with one direct evaluation in 2025-2026. The field has no single standard: metrics follow the dataset rather than a shared rule.

| Dataset | All 35 papers | 2025-26 papers | 2025-26, dropping snippet-only rows | Usual metric | Example papers |
|---|---|---|---|---|---|
| SciEntsBank | 12 | 9 | 8 | Accuracy, macro-F1 | [Cong et al., BEA 2026](https://aclanthology.org/2026.bea-1.19/); [Saur & Toth 2026](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1829089/full); [CHiL(L)Grader](https://arxiv.org/html/2603.11957) |
| Mohler | 8 | 7 | 6 | RMSE, Pearson | [Ferreira Mello et al., LAK 2025](https://researchmgt.monash.edu/ws/files/722528341/675322817-oa.pdf); [Zhao et al. 2025](https://arxiv.org/html/2502.13337); [Desmarais et al., EDM 2025](https://educationaldatamining.org/EDM2025/proceedings/2025.EDM.short-papers.124/) |
| ASAP-SAS | 8 | 5 | 5 | QWK | [McGrane & Torres Irribarra 2026](https://aclanthology.org/2026.aimecon-wip.39.pdf); [Jiang & Bosch 2024](https://pnigel.com/papers/jiang-2024-2TEQ5JEF.pdf); [Chamieh et al., BEA 2024](https://aclanthology.org/2024.bea-1.25); [Ormerod & Kwako 2024](https://arxiv.org/abs/2407.01873) |
| Beetle | 6 | 3 | 3 | Accuracy, macro-F1 | [Vasquez Ferrer et al. 2026](https://arxiv.org/html/2603.29559v1); [Wang & Ormerod 2024](https://arxiv.org/html/2408.03811v1) |
| RiceChem | 2 | 2 | 2 | Varies | [Vasquez Ferrer et al. 2026](https://arxiv.org/html/2603.29559v1); [S-GRADES](https://arxiv.org/abs/2603.10233) |
| SAF | 2 | 1 | 1 | RMSE, Pearson | [Zhao et al. 2025](https://arxiv.org/html/2502.13337); [ASAG2024](https://arxiv.org/abs/2409.18596) |
| EngSAF | 2 | 1 | 1 | Accuracy; QWK in one paper | [Aggarwal et al.](https://arxiv.org/abs/2407.12818); [CHiL(L)Grader](https://arxiv.org/html/2603.11957) |

*Sample, not a census (warning a). A paper can use several sets. Two SciEntsBank rows come from one research group.*

**ASAP 2.0 appears in one of the 35 papers, inside [S-GRADES](https://arxiv.org/abs/2603.10233)**, which groups it with the essay sets. The [BEA 2025 survey](https://aclanthology.org/2025.bea-1.35.pdf) calls ASAP the standard essay benchmark. "ASAP 2.0 with QWK" therefore matches essay-scoring practice, and the supervisor's choice is defensible as an essay benchmark. The project's own note says the same thing: these are [~24.7k source-based argumentative essays scored 1-6](../docs/research-outline.md). Reporting it as short-answer grading would invite an easy objection.

About 1 paper in 4 (9 of 35) relies on private data, mixes it in, or does not name its data, and that figure is probably low because searches targeted public sets. Private data is typical of education groups studying a real course, while public sets are typical of NLP method papers. This split explains much of the gap in new benchmarks: [one review of 49 studies](https://mdpi.com/2076-3417/15/10/5683) found 23 built custom datasets `[unverified]`, and large item banks such as [NAEP](https://github.com/NAEP-AS-Challenge/reading-prediction) stay confidential.

Age is not the field's main complaint. The BEA 2025 survey, which covered 30 studies, criticises mixed datasets, metrics and protocols, and it never uses the word "contamination" ([survey](https://aclanthology.org/2025.bea-1.35.pdf)). S-GRADES calls the field fragmented ([paper](https://arxiv.org/abs/2603.10233)). Direct evidence on memorised test sets is thin in both directions: the one short-answer discussion found, on ASAP-SAS, saw no clear sign of recall but could not rule it out and called for a test on unreleased items ([McGrane & Torres Irribarra](https://aclanthology.org/2026.aimecon-wip.39.pdf)). The same paper warns that QWK alone can overstate readiness for real use, which supports the project's habit of reporting exact and within-one agreement beside it.

Small models are now common: 13 of the 35 papers tested a model under 10B, and several used LoRA or QLoRA. Edge work is nearly absent. The closest case timed Qwen3 0.6B on a laptop CPU, at about 0.52 s per answer ([Saur & Toth](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1829089/full)), and no paper in the sample measured power or energy or ran on an edge board. Only one direct search targeted this, so read that as "not found", not "does not exist".

## Few new sets hold real answers, and fewer are open

Three things thin out the list. Several "new" benchmarks repackage old data, some "student answers" were written by an LLM, and the best-designed set needs a request form. The tables below cover sets with real student answers first.

### English sets

| Name | Year | Language | Answers / questions | Subject, level | Labels or scale | Metric used | Licence | Access today | Used by |
|---|---|---|---|---|---|---|---|---|---|
| [EngSAF](https://arxiv.org/html/2407.12818v2) | 2024; AIED 2025 per [README](https://github.com/dishankaggarwal/EngSAF), venue `[unverified]` by the paper check | English | 5,814 / 119 (5,798 in CHiL(L)Grader) | 25 engineering courses, university | 3 ordered labels (0/1/2), plus Gemini-written feedback | Accuracy, macro-F1; QWK in CHiL(L)Grader | Terms bar redistribution | **Request form only** | Aggarwal et al.; [CHiL(L)Grader](https://arxiv.org/html/2603.11957) |
| [JorGPT](https://zenodo.org/records/18981627) | 2026 | Spanish, with English translation | [3,041 / 50](https://api.crossref.org/works/10.3390/data11030059) | Computer science, university | 0-10 `[unverified]`, teacher feedback, LLM gradings | Not read | CC BY 4.0 | **Open** on Zenodo and [Kaggle](https://www.kaggle.com/datasets/javiersanchezsoriano/jorgpt-student-answers-and-multi-llm-grading) | Its data paper only |
| [OS dataset](https://github.com/wenjing1170/llm_grader) | 2024 | English | Count not verified / 6 | Operating systems, undergraduate | Numeric, 3 TAs per answer | QWK and others in S-GRADES | None stated | Open | [Grade Like a Human](https://arxiv.org/pdf/2405.19694); S-GRADES |
| [Ghana ROARS](https://github.com/owenhenkel/ROARS-dataset) | 2023 | English | About 1,000 short answers | Reading, ages 9-18 | Expert scores; scale not opened | QWK, F1 | None stated | Open | [Henkel et al.](https://arxiv.org/abs/2310.18373) |
| [RiceChem](https://www.arxiv.org/pdf/2404.14316) | 2024 | English | 1,264 / 4 | Chemistry, college | Binary on 27 rubric items | Not verified | Not verified | [Form](https://github.com/luffycodes/Automated-Long-Answer-Grading) upstream; S-GRADES says public | Vasquez Ferrer et al.; S-GRADES |
| [BeSTraP](https://doi.org/10.3390/data11030057) | 2026 | `[unverified]` | 651 / `[unverified]` | Databases, university | 0-5, two instructors `[unverified]` | Pearson | `[unverified]` | `[unverified]` | Its data paper, beside Mohler |
| [Covenant set](https://api.crossref.org/works/10.3390%2FDATA10060087) | 2025 | `[unverified]` | Not found | Two university courses, transcribed from handwriting | Human scores | Spearman, MSE | Listing says "None" | No repository found | Its data paper |
| [DAMI](https://arxiv.org/html/2603.11957) | 2026 | Not stated | 4,031 / 53+ | Data mining, graduate | Maximum 5, 8 or 10 | QWK, calibration | Unknown | **No release found** | CHiL(L)Grader |
| [Regrading 2JC](https://huggingface.co/datasets/nlpatunt/D_Regrading_Dataset_J2C) | 2024 | English | 1,979 rows | Engineering concepts | 0-8 | QWK in S-GRADES | No licence tag | S-GRADES copy, test labels removed | Gao et al.; S-GRADES |
| [MIME](https://aclanthology.org/2026.aimecon-main.33.pdf) | 2025-26 | English | 52,463 / 15 (14,368 synthetic paraphrases) | Maths, grades 4-8 | 6 categories plus misconception | MAP@3 | [Conflict](https://github.com/scrosseye/mime_dataset): CC BY-NC text vs MIT on Kaggle | Open except 700 held out | [MAP Kaggle competition](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) |

For contrast, [ASAP-SAS](https://www.kaggle.com/competitions/asap-sas) (2012) has 10 prompts and scores of 0-3 with rubrics ([outline](../docs/research-outline.md)). A later paper's table sums to [22,267 public responses](https://ar5iv.labs.arxiv.org/html/2202.11558). It is available on Kaggle after accepting the competition rules.

Two "benchmarks" add little new data. [**ASAG2024**](https://ar5iv.labs.arxiv.org/html/2409.18596) combines seven existing sets into 18,988 rows on a 0-1 scale, scored with weighted RMSE. SciEntsBank, Beetle, SAF and Mohler make up 94% of it, and it has no unseen-question test `[deleted copy]`. It offers one useful result: Llama-3-8B scored 0.39 against 0.40 for simply predicting the mean, so that 8B model barely beat a constant guess. [**S-GRADES**](https://arxiv.org/abs/2603.10233) (LREC 2026 in one note, preprint in another) combines 14 English datasets, covering both essays and short answers, with a [public leaderboard](https://sgrades.eng.unt.edu). It uses QWK as primary, adds accuracy and F1 for class labels, and holds no new data. It is the closest thing to the "benchmark rather than dataset" the supervisor asked for, but test scoring depends on its site, which nobody opened `[unverified]`.

### Non-English and multilingual sets

| Name | Year | Language | Answers / questions | Subject, level | Labels or scale | Metric used | Licence | Access today | Used by |
|---|---|---|---|---|---|---|---|---|---|
| [ALICE](https://github.com/Szhifan/emnlp2026alice) | 2026 | German | 16,572 / 112 | STEM, school | Three label sets: LP 0-2, KE 0-3, SK 0-2 | Macro-F1 | CC BY-NC-SA 4.0 | **Open** (1.4 MB zip) | [Sun et al., EMNLP 2026](https://aclanthology.org/2026.emnlp-main.667/) |
| [ALICE-LP-1.0](https://edutec.science/bea-2026-shared-task/) | 2026 | German | 13,075 (summed) | STEM, secondary | 3-way and 2-way | **QWK** | Not stated | `[unverified]` | [BEA 2026 shared task](https://aclanthology.org/2026.bea-1.85/) |
| [PT_ASAG_2018](https://www.kaggle.com/datasets/lucasbgalhardi/pt-asag-2018) | 2018 | Portuguese | Sources conflict: 7,473 / 15 is one figure | Biology, grades 8-9 `[unverified]` | 0-2 or 0-3 `[unverified]` | MAE, RMSE | CC BY-NC-SA 4.0 | Open | [Ferreira Mello et al.](https://researchmgt.monash.edu/ws/files/722528341/675322817-oa.pdf) |
| [AR-ASAG](https://data.mendeley.com/datasets/dj95jh332j/1) | 2020 | Arabic | 2,133 / 48 | Three exams | Two marks and a mean | Pearson, RMSE | CC BY 4.0 | Open | [Ouahrani & Bennouar](https://aclanthology.org/2020.lrec-1.321) |
| [ScAA](https://github.com/PrathamOrg/ScAA-Dataset) | 2020 | Hindi, Marathi | 10,988 + 1,955 / 32 | Science, ages 8-14 | Binary | Not read | CC BY-NC-SA 4.0 | Open | [Agarwal et al.](https://aclanthology.org/2020.icon-main.58) |
| [Khmer ASAG](https://huggingface.co/datasets/phorknorak/khmer-asag-benchmark) | 2026 | Khmer | 1,184 / 41 | High school | Marks, maxima 5-20 | Not read | CC BY-NC 4.0 | Open; no answer key, best student answer stands in | Its dataset card |
| [RIKEN-SAS](https://aclanthology.org/2026.emnlp-main.667.pdf) | 2019 | Japanese | 31K / 34 | Reading, high school | Analytic scores | QWK | `[unverified]` | Listed public, terms `[unverified]` | [Funayama et al.](https://doi.org/10.1007/s40593-025-00474-w) |
| [ELOQUENT 2026](https://ceur-ws.org/Vol-4283/paper98.pdf) | 2026 | 13, no Indonesian | About 35,000 PISA + 47,000 other test items per track | Mixed | 0-2 on the rubric track | QWK | `[unverified]` | No download found; PISA items under NDA | Its organisers; [WSE-Research](https://ceur-ws.org/Vol-4283/paper103.pdf) |

### Names to avoid as real-student benchmarks

[**SAS-Bench**](https://arxiv.org/abs/2505.07247) (2025; 4,109 responses, 1,030 questions, Chinese with a small English share, Apache-2.0) is not real student writing: its [paper](https://arxiv.org/html/2505.07247v2) says LLMs simulate the responses, though its README still says "student responses". [**GRAS**](https://huggingface.co/datasets/saurluca/GRAS) answers came from GPT-4o, the [Kaggle Indonesian ASAG set](https://www.kaggle.com/datasets/yuvenr/indonesian-automated-short-answer-grading-asag) was LLM-generated according to its owner, and the [Kaggle "Automatic Short Answer Grading Dataset"](https://www.kaggle.com/datasets/mubeenfurqanahmed/automatic-short-answer-grading-dataset) is tagged synthetic. [**NAEP**](https://sap.nsf.gov/dataset/ed_nces-2024072/metadata/access) is closed. [**MIME**](https://aclanthology.org/2026.aimecon-main.33.pdf) is large but has only 15 items and tests misconception diagnosis rather than grading against a reference. [**RiceChem**](https://huggingface.co/datasets/nlpatunt/D_Rice_Chem) averages 120 words per answer and frames itself as long-answer grading. No dataset called "Pistis" or "CLASSE" was found.

## ASAP 2.0 needs one short-answer partner to back any short-answer claim

The project needs a set that meets six tests: real student answers, short text, ordered labels so QWK works next to ASAP 2.0, a reference answer or rubric, an unseen-question test, and usable access. The notes support the ratings below, and the final column is this report's judgement.

| Candidate | Real, short answers | Works with QWK | Reference or rubric | Unseen-question test | Usable today | Role |
|---|---|---|---|---|---|---|
| **JorGPT** | Yes; English is translated; length unchecked | Yes, if the 0-10 scale is confirmed | Ideal answer | Project must make one | **Yes** | **Add first** |
| **EngSAF (official)** | Yes | Yes, three ordered labels | Reference | **Built in** (765 answers) | After request | **Request now**, best design |
| **ASAP-SAS** | Yes, from 2012 | Native metric | Rubrics | Not checked | After accepting Kaggle rules | Comparison anchor, not new |
| **ALICE** | Yes, in German | Yes; its shared task used QWK | Reference and rubric | **Built in** (3,096 answers) | **Yes** | Strongest recent set, if German is acceptable |
| **Haidir & Purwarianti** | Yes, in Indonesian | Yes, 0-5 | Reference | **Built in** (6 questions) | Yes, after label cleaning | Indonesian condition |
| **S-GRADES** | Repackaged mix | Primary metric | Varies | Varies | Site `[unverified]` | Optional public scoreboard |

**JorGPT comes first because it needs nobody's permission and it answers the contamination worry.** The answers come from [academic year 2025-2026](https://api.crossref.org/works/10.3390/data11030059), after Gemma 4's [January 2025 cutoff](../docs/research-outline.md), so the model cannot have seen them in training (this report's inference; the questions themselves could be older). No project-made split is needed for the untouched models or the existing SciEntsBank adapters, because all 3,041 answers are already held out from them. A split by question is needed only if the project trains on JorGPT. Three limits should be reported: one institution, translated English, and included LLM gradings that must stay out of the prompt. The Spanish original also offers a free language comparison, though a Spanish-English gap could reflect translation as well as language. The two Zenodo versions name the translated file differently ([`dataset_en.csv`](https://zenodo.org/records/18981627) and [`dataset_translated.csv`](https://zenodo.org/records/18434167)), so pin one version.

**EngSAF has the design the project wants, so the request should go in now.** Its authors report Mistral-7B at [75.4 on unseen answers and 58.7 on unseen questions](https://arxiv.org/html/2407.12818v2), while zero-shot GPT-4o reached 48.9 on unseen answers. That gives a small-model reference point of the right size. Turnaround and approval criteria are unknown, and the terms forbid redistribution, so nothing derived from the data should be re-hosted.

**ASAP-SAS is older but gives the most published comparison points.** It shares ASAP 2.0's metric, and small-model results already exist ([Ormerod & Kwako](https://arxiv.org/abs/2407.01873); [Wang et al.](https://arxiv.org/html/2605.29274)). GPT-4 reached [QWK .677](https://pnigel.com/papers/jiang-2024-2TEQ5JEF.pdf) across its 10 items. The project skipped it because [Kaggle returned 403](../docs/research-outline.md) when the rules were not accepted. Accepting those rules is the owner's own step. The [S-GRADES copy](https://huggingface.co/datasets/nlpatunt/D_ASAP-SAS) should not be a way around them: it has no licence tag, and the EngSAF mirror shows why that matters.

**ALICE is the strongest recent set overall, but its language changes the question.** It is real, recent, public, rubric-based and has both unseen tests. Its sister task gives a QWK scoreboard: the top team reached [0.796 on unseen answers and 0.681 on unseen questions](https://aclanthology.org/2026.bea-1.85/) on 3-way labels. Its [split differs from ALICE-LP-1.0](https://aclanthology.org/2026.emnlp-main.667.pdf), so those scores cannot be compared directly with a run on the public ALICE release. Llama-3.2-3B with rubric retrieval reached [macro-F1 76.7 and 69.5](https://aclanthology.org/2026.emnlp-main.667.pdf) on the two tests. A German result would mix grading skill with German skill, so it suits a secondary role.

### A short-answer set would test a surprising ASAP 2.0 result

On the [600-essay sample](../docs/results/asap2.md), untouched E2B scored QWK 0.568 and untouched E4B 0.466, the reverse of SciEntsBank. The one published Gemma 4 grading result found goes the other way: on the multilingual ELOQUENT rubric track, [E4B scored QWK 0.506 and E2B 0.286](https://ceur-ws.org/Vol-4283/paper98.pdf). The setups differ, since those baselines reasoned before answering and used AWQ quantisation through vLLM, while the project runs `llama-server` with thinking off. Even so, the contrast raises a simple question for the supervisor: is the reversal an essay effect or a real property of these models? This report's hypothesis is an essay or prompt effect, because the project's own results show E4B piling two thirds of its scores on 3. A QWK run on one short-answer set would test that cheaply.

The ELOQUENT result also corrects a novelty claim. The paper sample found no Gemma 4 grading paper, but ELOQUENT did run both models, and E4B fell from [dev QWK 0.764 to 0.475 on PISA items](https://ceur-ws.org/Vol-4283/paper98.pdf). The project should not say it is the first to test Gemma 4 as a grader. It can say that no paper found reports accuracy against measured energy on an edge board.

Two caveats apply to ASAP 2.0 itself. S-GRADES describes it as having [publicly available test labels](https://arxiv.org/abs/2603.10233), and the source corpus PERSUADE predates Gemma 4's cutoff, although the Kaggle release date was not checked `[unverified]`. Essays are also much longer than short answers, and [Jetson cost for ASAP 2.0 is still to come](../docs/results/asap2.md), so the project's short-answer energy figures should not be quoted as if they carry over.

## Haidir & Purwarianti is the only strong Indonesian option

Only two public Indonesian sets contain real graded answers. Recent Indonesian work mostly uses small private school sets, and the one recent public "Indonesian ASAG" set is LLM-generated.

| Name | Year | Answers / questions | Subject, level | Labels | Metric used | Licence | Access today |
|---|---|---|---|---|---|---|---|
| [Haidir & Purwarianti](https://github.com/AgeMagi/tugas-akhir) | 2020 | 9,163 rows / 36 (paper says 9,165) | Biology, Geography; level unconfirmed | 0-5, with reference answers | RMSE, QWK | [CC BY-NC 4.0](https://github.com/AgeMagi/tugas-akhir/blob/master/LICENSE) in the repo, ["Unknown"](https://huggingface.co/datasets/SEACrowd/id_short_answer_grading) on SEACrowd | Open on GitHub; the Hugging Face entry is only a loader |
| [UKARA 1.0](https://github.com/ilhamfp/ukara-1.0-challenge) | 2019 | 2,861, only 573 labelled / 2 | Two tasks from an assessment challenge | Binary, with coding guide | [F1](https://arxiv.org/abs/2002.12540) | None stated | Partly; dev and test labels absent |
| [Rahutomo & Roshinta](https://data.mendeley.com/datasets/6gp8m72s9p/1) | 2018 | `[unverified]` | Online essay test | `[unverified]` | Pearson | CC BY 4.0 | Open; contents unchecked |

Haidir & Purwarianti fits best. It has the same shape as the project's grading input (question, reference answer, student answer), and its [official test files hold 6 questions that never appear in training](https://github.com/AgeMagi/tugas-akhir/tree/master/data). Its 0-5 scale also allows QWK, so the Indonesian condition can share the headline metric. **No published small-LLM result on it was found, so a Gemma run would be a real gap to fill.** The only recent baselines come from a [personal repository](https://github.com/vacelni/indo-short-answer-scoring), not peer-reviewed: QWK 0.573 for lexical overlap with gradient boosting, and 0.509 for fine-tuned IndoBERT. The labels need a written cleaning rule before any run, because the [score files](https://github.com/AgeMagi/tugas-akhir/tree/master/data) hold blanks and stray values such as "5?", and row counts of 9,165, 9,163 and 8,827 already circulate. The non-commercial licence suits research, but any adapter trained on it stays non-commercial.

UKARA is a fallback, scored with accuracy and F1 because its labels are binary. It has two questions, no reference answers, and training labels that [one team called highly inconsistent](https://arxiv.org/abs/2002.12540). The closest small-model paper is [Faza et al. 2025](https://ijecbe.ui.ac.id/go/article/download/148/84), which fine-tuned Gemma-3-1b-it on a private 220-pair set. Reasoning-guided training reached QWK 0.7791 against 0.7279 for standard tuning, at about 35 times the inference time. Its test has only 44 items and one grader, so the gap is uncertain. The cost pattern still matters for edge use.

## Open checks before quoting these numbers

### Unapproved downloads and the EngSAF mirror

**These numbers came from copies downloaded without the owner's prior approval and later deleted.** The notes label them "own check", and they should be treated as provisional until the owner approves a fresh check or an official source confirms them.

| Source | Numbers affected |
|---|---|
| [ASAG2024](https://huggingface.co/datasets/Meyerger/ASAG2024) | Split sizes 15,190 / 1,899 / 1,899, and all 276 distinct test questions also appearing in train, which is the basis for "no unseen-question test" |
| [IsmaelMousa/engsaf](https://huggingface.co/datasets/IsmaelMousa/engsaf) | 5,579 rows, 118 questions, splits 4,735 / 338 / 506, 31 of 42 test questions appearing in train, and a `mark_scheme` column holding only generic label text |
| [Haidir & Purwarianti](https://github.com/AgeMagi/tugas-akhir), [UKARA 1.0](https://github.com/ilhamfp/ukara-1.0-challenge) | Haidir's 9,163 rows and 36 questions, and UKARA's 573 labelled answers of 2,861. These small CSVs were also downloaded without prior approval, from the authors' own public repositories, so no redistribution conflict applies. |

**Do not use the EngSAF mirror until this is resolved.** It is tagged Apache-2.0, while the [official README](https://github.com/dishankaggarwal/EngSAF) says "Redistribution of the dataset is prohibited". The mirror's tag does not override the authors' terms. The official request route avoids the problem, and it also gives the real unseen-answer and unseen-question tests, which the mirror's counts suggest it lacks `[deleted copy]`. ASAG2024's stated licence is ["Data Source Licenses apply"](https://huggingface.co/datasets/Meyerger/ASAG2024), so each source set's terms would also need checking. Its [viewer API reports 56,646 rows](https://datasets-server.huggingface.co/size?dataset=Meyerger/ASAG2024) against the 18,988 in the combined file, and that difference is unexplained.

### Unverified or conflicting items

| Item | What is not settled |
|---|---|
| JorGPT | The 0-10 scale, 79 students and three LLM evaluators come from search snippets, because MDPI blocked full text. Its splits are not verified. |
| BeSTraP, Covenant | Mostly snippets or metadata. Language, question counts, access and BeSTraP's licence are not verified. The Covenant answer count was not found. |
| EngSAF | The venue comes from its README only. Answer counts differ (5,814 and 5,798). Request turnaround is unknown. |
| RiceChem | S-GRADES marks it public, but the authors' repo still asks for a form. Its AIED venue is snippet-only. |
| DAMI | Its split sizes sum to 4,291, above the stated 4,031, unexplained. Language and reference answers are not stated. |
| S-GRADES | The leaderboard site was not opened. The notes disagree on whether it is LREC 2026 or a preprint. The D_ASAP-SAS card does not name which split lost its labels. |
| ALICE-LP-1.0, ELOQUENT | No licence or download location was verified. ELOQUENT mixes auto-translated and LLM-written material. |
| PT_ASAG_2018, RIKEN-SAS | Portuguese sizes conflict across sources. RIKEN's access terms and prompt counts are unverified. |
| Haidir, UKARA, Rahutomo | Haidir's school level is unconfirmed. UKARA's official licence and hidden labels are unconfirmed. Rahutomo's size and scores are unknown. |
| OS dataset, Ghana ROARS | There is no licence file, so ask the authors before sharing anything. The OS answer count was not read. |
| Paper table rows | Datasets for [Kuichua Kamdem et al.](https://publications.polymtl.ca/78521/) and [NodeGrade](https://dl.acm.org/doi/10.1145/3723010.3723021), and all of [Lee et al.](https://doaj.org/article/928c0f01d762405a987c831e3968deb5), are snippet-only. [Funayama et al.](https://doi.org/10.1007/s40593-025-00474-w)'s 8B result is from a search summary. |
| Coverage | No Expert Systems with Applications, IEEE TLT or Education and Information Technologies paper from 2025-2026 was confirmed. EDM, AIED and LAK 2026 proceedings were not read directly. Garuda and SINTA were not searched directly, and one possibly relevant quantised-LLM study on `journal.laksita.id` failed to resolve. |

## Conclusion

The useful answer for the supervisor is that newness alone will not fix the benchmark. Old sets persist because they let papers compare results, while most new real-student data is small, private, request-only or from one course. The project gains more by pairing sets with different jobs: ASAP 2.0 for the agreed headline, one short-answer set so the short-answer claim stands, and a post-cutoff set so a memorisation objection has an answer. JorGPT covers the last two jobs at once, with no approval needed.

The next decisions are small. Confirm JorGPT's columns and run the untouched models on it, send the EngSAF request, decide whether to accept the ASAP-SAS Kaggle rules, and write the Haidir cleaning rule if the Indonesian condition goes ahead. Whichever set is chosen, describing ASAP 2.0 as essay scoring and reporting cost on the same data as accuracy will keep the claims defensible.
