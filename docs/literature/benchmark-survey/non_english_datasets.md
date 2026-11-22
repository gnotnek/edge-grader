# Non-English and multilingual short answer grading benchmarks, with a focus on Indonesian (2023-2026)

Notes checked on 2026-11-05. Evidence tags used below:
- **[files]** I read the data files, repo tree or dataset API myself.
- **[paper]** I read the paper text or the dataset page.
- **[snippet]** I saw only a search summary, so treat it as UNVERIFIED.

## Which Indonesian short-answer or essay-answer grading datasets exist?

### Takeaway
Only two public Indonesian sets contain real graded student answers and can be downloaded today: Haidir & Purwarianti 2020 (9,163 rows, 36 questions, 0-5 scores, reference answers) and UKARA 1.0 (2019, 2 questions, binary). Of UKARA's 2,861 answers, only 573 have labels in the public copy. I found no newer public Indonesian set of real answers. The 2024-2026 Indonesian work mostly uses small private school sets, and the one recent public "Indonesian ASAG" set is LLM-generated.

### Cited Findings

**A. Haidir & Purwarianti 2020 ("id_short_answer_grading"): the strongest Indonesian option**
- Paper: Haidir and Purwarianti, "Short Answer Grading Using Contextual Word Embedding and Linear Regression", Jurnal Linguistik Komputasional 3(2), 2020, pp. 54-61, DOI 10.26418/jlk.v3i2.38; the abstract gives 36 questions, 9,165 answers, 534 respondents, 7 expert graders and best RMSE 1.893 on the Indonesian set — [SEACrowd loader citation block](https://huggingface.co/datasets/SEACrowd/id_short_answer_grading/raw/main/id_short_answer_grading.py); [journal page, not opened](https://inacl.id/journal/index.php/jlk/article/view/38)
- Subject: Biology ("saintek") and Geography ("soshum"). Level: "respondents" only; I could not confirm the school level — [SEACrowd card](https://huggingface.co/datasets/SEACrowd/id_short_answer_grading)
- [files] The repo has 8 CSVs, with questions, reference answers and scores. Row counts: score-saintek 3,583, score-saintek-test 780, score-soshum 4,020, score-soshum-test 780, total 9,163 (the paper says 9,165). Question files: 15 + 3 + 15 + 3 = 36 — [AgeMagi/tugas-akhir data folder](https://github.com/AgeMagi/tugas-akhir/tree/master/data)
- [files] The labels need cleaning. score-soshum has 266 blank scores, 2 "-" and 4 "5?". score-saintek has 3 blank and 1 "`0". Score 0 is very common in the train files (1,148 of 3,583 and 1,351 of 4,020) — [score files](https://github.com/AgeMagi/tugas-akhir/tree/master/data)
- Split: the official test files hold 6 questions that never appear in training. A third-party project keeps 8,827 of 9,163 rows, with a 1,499-answer test set — [vacelni/indo-short-answer-scoring](https://github.com/vacelni/indo-short-answer-scoring)
- Licence conflict: the repo LICENSE now says CC BY-NC 4.0 (copyright 2020-2026) — [LICENSE](https://github.com/AgeMagi/tugas-akhir/blob/master/LICENSE); SEACrowd lists it as "Unknown" — [SEACrowd card](https://huggingface.co/datasets/SEACrowd/id_short_answer_grading)
- The HF entry is a loader only. It holds no data and pulls from the GitHub raw URLs, so GitHub is the true location — [HF file list via API](https://huggingface.co/api/datasets/SEACrowd/id_short_answer_grading)
- Recent baseline on the unseen-question test: lexical overlap + gradient boosting QWK 0.573; fine-tuned IndoBERT cross-encoder QWK 0.509, exact match 40.9% — [vacelni README](https://github.com/vacelni/indo-short-answer-scoring). This is a personal repo, not peer reviewed.

**B. UKARA 1.0 (2019)**
- Challenge run by the UGM NLP group with PUSPENDIK, the assessment unit of the Ministry of Education and Culture; students usually write 2-3 sentences — [ilhamfp README](https://github.com/ilhamfp/ukara-1.0-challenge)
- Two questions in Bahasa Indonesia. Task: right or wrong, scored by F1 — [arXiv 2002.12540](https://arxiv.org/abs/2002.12540)
- [files] Answers per split: A train 268 / dev 215 / test 855; B train 305 / dev 244 / test 974, total 2,861. Train labels: A 191 positive, 77 negative; B 168 positive, 137 negative. These match Table 1 of the paper — [repo data folder](https://github.com/ilhamfp/ukara-1.0-challenge/tree/master/data); [arXiv 2002.12540](https://arxiv.org/abs/2002.12540)
- [files] Only the train CSVs have a LABEL column; the dev and test files have only RES_ID and RESPONSE. So 573 answers are usable as labelled data — [repo data folder](https://github.com/ilhamfp/ukara-1.0-challenge/tree/master/data)
- [files] The repo also has a stimulus text, the question and a coding guideline for each task. These work as a rubric. There is no per-question reference answer — [stimulus A file](https://github.com/ilhamfp/ukara-1.0-challenge/tree/master/data/data_A)
- Label quality: one team called the training labels highly inconsistent and hand-corrected them — [arXiv 2002.12540](https://arxiv.org/abs/2002.12540)
- Licence: the GitHub API reports no licence on the participant repo that holds the data — [repo](https://github.com/ilhamfp/ukara-1.0-challenge). The [official page](https://nlp.mipa.ugm.ac.id/ukara-1-0-challenge/) returned no readable text to my fetcher. The [EvalAI page](https://evalai.cloudcv.org/web/challenges/challenge-page/410/overview) is linked from the repo but I did not open it.
- Results: final-leaderboard F1 0.81 (Bi-LSTM, 2nd place) — [ilhamfp README](https://github.com/ilhamfp/ukara-1.0-challenge); F1 0.812 — [arXiv 2002.12540](https://arxiv.org/abs/2002.12540); single classical model within 2% of deep learning at 1/18 of the training time — [Septiandri et al., SustaiNLP 2020](https://aclanthology.org/2020.sustainlp-1.1.pdf); F1 0.821 [snippet] — [Kinetik paper](https://kinetik.umm.ac.id/index.php/kinetik/article/download/1196/124124250/124125899)
- Still in use in 2026: UKARA A accuracy 93.49% (BiLSTM-FastText + GMP, Fadilah et al. 2026a), as reported in — [Fadilah 2026 review, Table 3](https://journal.ininnawaparaedu.com/jmsme/article/download/825/539/3980)

**C. Rahutomo & Roshinta 2018 ("Rahutomo dataset")**
- "Indonesian Query Answering Dataset for Online Essay Test System", Mendeley Data, 30 Aug 2018, DOI 10.17632/6gp8m72s9p.1, CC BY 4.0, Excel files. The page gives no counts and does not mention scores — [Mendeley record](https://data.mendeley.com/datasets/6gp8m72s9p/1)
- The 2026 review says it is used for cross-prompt evaluation and reports Pearson 0.9504 (Wicaksono et al. 2024, Direct Scoring Transformer) — [Fadilah 2026](https://journal.ininnawaparaedu.com/jmsme/article/download/825/539/3980)

**D. Named in the literature, no public download found**
- 2026 survey: Indonesian ASAS still rests on UKARA, PISA Indonesia, Rahutomo and domain sets; it names limited data, semantic noise, overfitting and informal language as the main problems — [Fadilah, GAUSS 2(1), 2026](https://journal.ininnawaparaedu.com/jmsme/article/view/825)
- PISA Indonesia: dichotomous and polytomous scoring, cited to Hidayatulloh 2018 — [same review, Table 1](https://journal.ininnawaparaedu.com/jmsme/article/download/825/539/3980). I did not locate the data.
- SMKN 1 Rao Selatan: vocational Computer and Network Engineering (TKJ) answers, cited to Ayu & Nurjanah 2025 — [same review, Table 1](https://journal.ininnawaparaedu.com/jmsme/article/download/825/539/3980). I did not locate the data.
- Ratna et al. corpus, expanded to 220 lecturer/student answer pairs on a 0-100 scale, one expert grader, computer science and engineering topics, 176/44 split — [Faza et al., IJECBE 2025](https://ijecbe.ui.ac.id/go/article/download/148/84). I found no data link in the PDF text.
- Salim et al. 2022, BEEI 11(3): two unnamed ASAG sets, Pearson 0.9508 and RMSE 0.4138. The Zenodo record holds only the PDF — [Zenodo](https://zenodo.org/records/6781703)
- Situmeang et al. 2025, IJICT 14(3): IndoBERTLite-Base-P2 had the best QWK among BERT variants. The dataset is not named on the record, which holds only the PDF — [Zenodo](https://zenodo.org/records/18464887)
- Amalia et al. 2025, IAENG IJCS 52(3): translated SQuAD 2.0 plus 1,000 lecture-based QA pairs, so this is not a student-answer grading set — [PDF](https://www.iaeng.org/IJCS/issues_v52/issue_3/IJCS_52_3_01.pdf)

**E. Synthetic: do not treat as student data**
- Kaggle "Indonesian Automated Short Answer Grading (ASAG)": about 4,000 rows, SMK computer networking, MIT, May 2026. The owner says the student answers and scores were generated purely by LLMs — [Kaggle](https://www.kaggle.com/datasets/yuvenr/indonesian-automated-short-answer-grading-asag)

**F. Essay-type Indonesian sets (kept separate; all [snippet], none confirmed public)**
- 630 high-school essay answers, 504/126 split, Sentence-BERT — [ResearchGate](https://www.researchgate.net/publication/405805256_Penilaian_Otomatis_Jawaban_Esai_SMA_Menggunakan_Sentence-BERT_dan_Hybrid_Levenshtein-Jaccard_dengan_Akurasi_Hybrid)
- Translated ASAP essays with IndoBERT, reported QWK 0.9, and 300 Research Methodology quiz essays — [ResearchGate](https://www.researchgate.net/publication/388813936_Automated_Indonesian_Essay_Scoring_and_Holistic_Feedback_Using_Bidirectional_Encoder_Representations_for_Transformers) and [ITS](https://scholar.its.ac.id/en/publications/enhancing-automated-essay-scoring-in-bahasa-indonesia-with-indobe/). The search summary did not make clear which result belongs to which source.
- Many GitHub student projects exist (for example biology and Indonesian-history essay scorers), but I did not check any for data — [example](https://github.com/ZkyWan/penilaian-esai-otomatis-sejarah-indonesia)

### Inferences
- For the optional Indonesian condition, Haidir & Purwarianti fits best. It has the same input shape as SciEntsBank (question, reference answer, student answer), 36 questions and an unseen-question test. UKARA fits worse: 2 questions, 573 labelled answers and no reference answers.
- The metric follows the label type. Haidir's 0-5 scale is ordinal, so QWK is reasonable there, alongside accuracy and F1. UKARA is binary, so accuracy and F1 are right. This matches the project rule for SciEntsBank.
- The 0-5 labels need a written cleaning rule before any run. Otherwise row counts will differ between papers: 9,165, 9,163 and 8,827 already appear.
- CC BY-NC should be fine for non-commercial research, but anything built on it stays non-commercial. UKARA's redistribution status is unclear, so cite the challenge and do not re-host the files.

### Gaps
- Could not confirm UKARA's official licence or whether its dev and test labels can still be obtained.
- Could not confirm the Rahutomo set's size or whether it has scores, because I did not open the Excel files.
- Garuda and SINTA could not be searched directly; I used web search and Garuda result links only, so small Indonesian journal sets may be missing.
- `journal.laksita.id` ("Evaluasi Kinerja Model LLM Terkuantisasi...", a quantised-LLM study that could be very relevant) did not resolve (DNS failure), so its task and content are unknown.

## Which other non-English sets are in current use?

### Takeaway
German is the strongest area: ALICE (EMNLP 2026) and the BEA 2026 shared task are both recent, rubric-based, from real students and public. Portuguese, Arabic, Hindi/Marathi, Khmer, Japanese and Spanish each have a usable set. Two names in the brief need correcting: SAS-Bench's "student responses" are LLM-simulated, and ALICE is German, not Finnish. The largest multilingual effort (CLEF 2026 ELOQUENT, 13 languages) has no Indonesian, and its genuine PISA test data are under NDA.

### Cited Findings

**German**
- ALICE: 16,572 answers, 112 questions, German K-12 biology, chemistry, mathematics and physics. LP 0-2, KE 0-3, SK 0-2. Split: train 10,781 / test_ua 2,695 / test_uq 3,096. Data licence CC BY-NC-SA 4.0, distributed as `alice_data_public.zip` (1.4 MB) in the repo — [GitHub README](https://github.com/Szhifan/emnlp2026alice)
- ALICE paper: Sun et al., EMNLP 2026, pp. 15366-15387. Answers come from routine classroom assessment. Macro-F1 is reported in the paper's tables — [ACL Anthology](https://aclanthology.org/2026.emnlp-main.667/)
- ALICE-LP-1.0, BEA 2026 Shared Task 2: German mid- and high-school STEM answers from Schleswig-Holstein; 3-way labels with a 2-way version; train 7,899, unseen answers 2,008, unseen questions 3,168; QWK is the primary metric. Top QWK (IWM-DKM): 0.796 for 3-way unseen answers, 0.681 for 3-way unseen questions — [EduTec page](https://edutec.science/bea-2026-shared-task/); [report, pp. 1179-1192](https://aclanthology.org/2026.bea-1.85/)
- The winning system used LoRA fine-tuning of Qwen models; generalising to unseen questions remains the main difficulty — [report](https://aclanthology.org/2026.bea-1.85/)
- ALICE results are not directly comparable with earlier ALICE-LP results, because the split differs — [ALICE paper](https://aclanthology.org/2026.emnlp-main.667.pdf)
- SAF (Filighera et al., ACL 2022): reference answers plus elaborated feedback. HF splits [files]: micro-job German 1,226 / 308 / 271 / 602, legal German 1,596 / 400 / 221 / 275, each with unseen-answer and unseen-question tests; CC BY 4.0 — [micro job](https://huggingface.co/datasets/Short-Answer-Feedback/saf_micro_job_german); [legal](https://huggingface.co/datasets/Short-Answer-Feedback/saf_legal_domain_german); [English set](https://huggingface.co/datasets/Short-Answer-Feedback/saf_communication_networks_english)
- ASAP-DE (Horbach et al., BEA 2018): German answers to three ASAP prompts; machine translation caused a considerable drop in scoring quality — [ACL Anthology](https://aclanthology.org/W18-0550/). No download found on that page.
- No dataset called "ASAG-DE" turned up. The closest are ASAP-DE and the German SAF sets.

**Chinese**
- SAS-Bench (Lai et al., 2025): 1,030 questions, 4,109 responses, 9 subjects, one test split [files]; reference answers, analysis, step scores and error types; Apache-2.0 in metadata, while the README adds "research purposes only" — [HF](https://huggingface.co/datasets/aleversn/SAS-Bench); [GitHub](https://github.com/PKU-DAIR/SAS-Bench)
- Its answers are not real: questions come from GAOKAO-Bench, and the authors use LLMs to simulate student responses, then have them graded by experts. Their limitations section names this as a key limitation — [arXiv 2505.07247](https://arxiv.org/abs/2505.07247)
- CSEE is a set of English essays by Chinese students: 13,372 essays, 2 prompts, described as private [snippet] — [arXiv 2401.06431](https://arxiv.org/pdf/2401.06431). It is essay-type, not Chinese-language and not public.

**Portuguese**
- PT_ASAG_2018 (Galhardi et al., SBIE 2018): real Brazilian students and teachers; questions, concepts, graded answers and reference answers; CC BY-NC-SA 4.0; v1.0 and v2.0 files, 544 KB, public — [Kaggle](https://www.kaggle.com/datasets/lucasbgalhardi/pt-asag-2018); [paper page](https://sol.sbc.org.br/index.php/sbie/article/view/41544)
- Sizes conflict: 7,473 answers from 659 students with 9,558 grades from 14 evaluators — [Galhardi 2019 thesis](https://repositorio.uel.br/bitstreams/54f197e7-7b9d-4eac-81d9-ad6e7cf260a0/download); 3,675 answers from 245 students on 15 questions [snippet] — [2025 conference paper](https://anais-siiepe.ufpel.edu.br/2025/CE_02054.pdf); 13K answers and 8 questions — [ALICE Table 1](https://aclanthology.org/2026.emnlp-main.667.pdf)
- Used with GPT-4 in 2025: 7,473 answers, 659 students, 15 questions [snippet] — [ACM LAK 2025](https://dl.acm.org/doi/10.1145/3706468.3706481)

**Arabic**
- AR-ASAG (Ouahrani & Bennouar, LREC 2020): 2,133 pairs, 48 questions, 3 exams, two human marks plus a gold average, Pearson 0.8384 between graders; CC BY 4.0 — [Mendeley](https://data.mendeley.com/datasets/dj95jh332j/1); [paper](https://aclanthology.org/2020.lrec-1.321)
- Nabil et al. 2025 (IJIET 15(12)): custom Environmental Science Corpus; DeepSeek-V3 best at QWK 0.8273, Llama 3-8B lowest. Availability not stated on the page — [IJIET](https://www.ijiet.org/show-234-3131-1.html)
- Minor: Kaggle `arabic_answers`, primary-school Islamic questions, 13.7 KB, GPL 2 — [Kaggle](https://www.kaggle.com/datasets/lotfielaachak/asag2022)

**Spanish**
- JorGPT (Cisneros-Gonzalez et al., Data 11(3):59, Mar 2026): 3,041 real student answers, 50 computer science questions, academic year 2025-2026; parallel English translations; ideal answers; blind human grades with feedback; plus DeepSeek, Qwen and Gemini gradings — [Crossref abstract](https://api.crossref.org/works/10.3390/data11030059); [MDPI](https://www.mdpi.com/2306-5729/11/3/59) (blocked for me). 79 students and a 0-10 scale come from search snippets [snippet].
- JorGPT files: `dataset.csv` 12.3 MB, `dataset_translated.csv` 14.6 MB, CC BY 4.0, open; a newer version than v2 exists — [Zenodo](https://zenodo.org/records/18434167)
- Capdehourat et al. 2025: 51 human-written questions for 8-10-year-olds with GPT-4-generated answers in three tiers, 24 per question (1,224 by my multiplication), so synthetic — [arXiv 2503.18072](https://arxiv.org/abs/2503.18072)
- Mardini et al. 2023: reading-comprehension ASAG, 199 undergraduates [snippet]; availability unverified — [Springer](https://link.springer.com/article/10.1007/s10639-023-11890-7)

**South and Southeast Asia**
- ScAA (Agarwal et al., ICON 2020): Hindi 10,988 and Marathi 1,955 answers, 32 parallel questions on 8 science topics, children aged 8-14, binary labels, model answers — [ACL Anthology](https://aclanthology.org/2020.icon-main.58). Excel files, CC BY-NC-SA 4.0 — [GitHub](https://github.com/PrathamOrg/ScAA-Dataset)
- Khmer ASAG Benchmark (Phork, v1.0.1, 17 Oct 2026): 1,184 real answers transcribed from handwritten exams (909 curated), 41 questions, 203 students, 2 high schools, one teacher grading, maximum marks 5-20. There is no answer key, so the best student answer stands in as the reference. CC BY-NC 4.0 — [HF](https://huggingface.co/datasets/phorknorak/khmer-asag-benchmark); [Kaggle mirror](https://www.kaggle.com/datasets/phorknorak/khmer-asag-benchmark)
- Bangla (Farzana et al., June 2026): 490,391 synthetic instances from Gemini, 0-10 scores, 259-record teacher-graded subset, no dataset URL in the paper — [arXiv 2606.11931](https://arxiv.org/html/2606.11931)
- Vietnamese: no real ASAG set found. Thai: only 1,073 questions inside the private Classtime data (see below).

**Japanese, Finnish, others**
- RIKEN-SAS: 31K answers, 34 questions, listed public, reading passage + analytic rubrics — [ALICE Table 1](https://aclanthology.org/2026.emnlp-main.667.pdf). Launched July 2020 through the NII repository [snippet] — [Tohoku NLP news](https://www.nlp.ecei.tohoku.ac.jp/?p=4035). Prompt counts differ between papers: 17 and 109 analytic criteria [snippet] — [arXiv 2408.13966](https://arxiv.org/abs/2408.13966)
- Finnish, Chang & Ginter (AAAI 2024): ChatGPT on ten undergraduate courses, QWK 0.6+ in 44% of one-shot settings for GPT-4 [snippet] — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/30363). Marked not public, and called proprietary, in — [ALICE Table 1](https://aclanthology.org/2026.emnlp-main.667.pdf). Size conflict: 2,000 answers [snippet] vs 76K in that table.
- Finnish matriculation exam: 1,016 geography answers, 0-15 scale, research permit, no stated public release. Translating to English raised within-2-points agreement from 75% to 85% with GPT-4o — [Jauhiainen & Garagorry Guerra 2026](https://journal.hep.com.cn/fde/EN/10.1007/s44366-026-0091-1)
- Latvian 502 and Lithuanian 690 QA pairs: answer variants made by alteration rules, so this is answer matching, not student grading. No data link on the arXiv page — [arXiv 2501.09164](https://arxiv.org/abs/2501.09164)
- Turkish physics open-ended answers, 42 KB, "Data files © Original Authors" — [Kaggle](https://www.kaggle.com/datasets/elifince/ml-for-openended-physics-questions-in-turkish). Not inspected.

**Multilingual and cross-language**
- CLEF 2026 ELOQUENT Sensemaking: 13 languages (English, Czech, Portuguese, German, Hungarian, Finnish, Danish, Swedish, Serbian, Greek, Irish, Romanian, Ukrainian); no Indonesian. Dev 4,146 items, train 13,856 multilingual + 18,377 English, test about 35,000 PISA items under OECD NDA + about 47,000 non-PISA per track; rubric track 0-2 with rubrics always in English; QWK — [WSE-Research note](https://ceur-ws.org/Vol-4283/paper103.pdf)
- Data origin there is mixed: contexts mostly auto-translated, rubrics largely LLM-generated with human correction, answers human and LLM. The dev PISA responses were written by the organisers; PISA test items include genuine student answers — [WSE-Research note](https://ceur-ws.org/Vol-4283/paper103.pdf); [overview, Sindelar et al.](https://ceur-ws.org/Vol-4283/paper98.pdf)
- Horbach et al. 2023 (IJAIED 34): five languages (Chinese, English, French, German, Spanish) over three ASAP prompts; machine translation plus multilingual models beat either alone [snippet] — [Springer](https://link.springer.com/article/10.1007/s40593-023-00370-1)
- Classtime (Schneider et al.): about 10 million answers and 990,000 questions, correct/incorrect teacher marks, many languages led by Ukrainian. The language table shows no Indonesian. I found no release statement in the paper text — [arXiv 2201.03425](https://arxiv.org/abs/2201.03425)
- Omethi scored 3.8 million PISA responses in 11 languages at kappa 0.804 (Zehner et al.), as cited in — [WSE-Research note](https://ceur-ws.org/Vol-4283/paper103.pdf)
- S-GRADES combines 14 grading datasets; language coverage is not stated on the summary page — [papers.cool](https://papers.cool/arxiv/2603.10233)

### Inferences
- Best cross-language test with parallel content: JorGPT (the same answers in Spanish and English, CC BY 4.0). Its English side is translated, so a Spanish/English gap could reflect translation as well as language.
- Best real-answer comparison sets with reference answers: ALICE or SAF (German), PT_ASAG_2018 (Portuguese), AR-ASAG (Arabic), ScAA (Hindi/Marathi). They differ in subject and scale, so cross-language comparisons would be confounded unless reported per dataset.
- ScAA and the Khmer set are the nearest regional relatives to Indonesian in low-resource terms, though Khmer's reference answers are weak.
- An Indonesian result cannot be compared directly with ELOQUENT, because Indonesian is not one of its languages.

### Gaps
- ASAP-DE and the Spanish, French and Chinese ASAP variants: no download location verified. Springer pages were blocked for me.
- ELOQUENT 2026 train and dev: no public download location found. The task page I tried returned 404 and the Eloquent HF organisation lists no matching dataset.
- RIKEN access terms unverified: the GitLab page redirected to a login and my guessed NII URL returned 404.
- PT_ASAG_2018's real row counts unverified. v2 answer files are about three times the size of v1 (855,910 vs 289,250 bytes) — [Kaggle](https://www.kaggle.com/datasets/lucasbgalhardi/pt-asag-2018) — which may mean 3,675 and 7,473 are different versions, but that is a guess.
- The search did not cover Korean, Malay, Filipino or Persian in depth. One combined search found nothing for Korean, Malay or Swahili.

## For each dataset: fields and whether it is really public today

### Takeaway
Real answers, with a working download, reference answers or rubrics, and a stated licence: ALICE, SAF, PT_ASAG_2018, AR-ASAG, JorGPT, ScAA, Haidir (licence disputed between sources) and Khmer (reference answers are proxies).

### Cited Findings
Sources for each row are in the sections above. "Link" is the download location. "Ref" means reference answer, rubric or both.

| Set | Year | Lang | Answers / questions | Subject, level | Real? | Labels | Ref | Split | Metric used | Licence | Link | Public today? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Haidir & Purwarianti | 2020 | id | 9,163 rows / 36 | Biology, Geography; level unconfirmed | Real | 0-5 | Yes | Train / test, unseen questions | RMSE, QWK | CC BY-NC 4.0 (repo) vs Unknown (SEACrowd) | [GitHub](https://github.com/AgeMagi/tugas-akhir) | Yes |
| UKARA 1.0 | 2019 | id | 2,861 (573 labelled) / 2 | Disaster, social topics | Real | Binary | Coding guide | Train / dev / test | F1 | None stated | [GitHub copy](https://github.com/ilhamfp/ukara-1.0-challenge) | Partly |
| Rahutomo | 2018 | id | Not stated | Online essay test | Unverified | Unverified | Unverified | Unverified | Pearson (2024 use) | CC BY 4.0 | [Mendeley](https://data.mendeley.com/datasets/6gp8m72s9p/1) | Yes, content unchecked |
| Kaggle id ASAG | 2026 | id | ~4,000 rows | SMK networking | Synthetic | Not checked | Yes | Not checked | n/a | MIT | [Kaggle](https://www.kaggle.com/datasets/yuvenr/indonesian-automated-short-answer-grading-asag) | Yes, but LLM-made |
| ALICE | 2026 | de | 16,572 / 112 | STEM, K-12 | Real | LP 0-2, KE 0-3, SK 0-2 | Both | Train / UA / UQ | Macro-F1 | CC BY-NC-SA 4.0 | [GitHub](https://github.com/Szhifan/emnlp2026alice) | Yes |
| ALICE-LP-1.0 | 2026 | de | 13,075 (summed) | STEM, secondary | Real | 3-way, 2-way | Rubric | Train / UA / UQ | QWK | Not stated | Not verified | Unverified |
| SAF | 2022 | de, en | 2,407 + 2,492 German | Micro-job, legal | Real, not confirmed by me | Score + feedback | Yes | Train / val / UA / UQ | n/a | CC BY 4.0 | [HF](https://huggingface.co/datasets/Short-Answer-Feedback/saf_micro_job_german) | Yes |
| SAS-Bench | 2025 | zh | 4,109 / 1,030 | 9 Gaokao subjects | LLM-simulated, expert-graded | Holistic + step | Both | Test only | QWK, CCS, ECS | Apache-2.0, research use | [HF](https://huggingface.co/datasets/aleversn/SAS-Bench) | Yes |
| PT_ASAG_2018 | 2018 | pt | Conflict: 7,473 / 15 or others | Biology, 8th-9th grade [snippet] | Real | 0-2 or 0-3 [snippet] | Yes | Not checked | Accuracy, others | CC BY-NC-SA 4.0 | [Kaggle](https://www.kaggle.com/datasets/lucasbgalhardi/pt-asag-2018) | Yes |
| AR-ASAG | 2020 | ar | 2,133 / 48 | Three exams; level not stated | Real | Two marks + mean | Yes | Not stated | Pearson, RMSE | CC BY 4.0 | [Mendeley](https://data.mendeley.com/datasets/dj95jh332j/1) | Yes |
| JorGPT | 2026 | es + en | 3,041 / 50 | CS, university | Real; English translated | 0-10 [snippet] + feedback | Yes | Not checked | Not read | CC BY 4.0 | [Zenodo](https://zenodo.org/records/18434167) | Yes |
| ScAA | 2020 | hi, mr | 10,988 + 1,955 / 32 | Science, ages 8-14 | Real | Binary | Yes | Not checked | Not read | CC BY-NC-SA 4.0 | [GitHub](https://github.com/PrathamOrg/ScAA-Dataset) | Yes |
| Khmer ASAG | 2026 | km | 1,184 / 41 | High school | Real | Marks, maxima 5-20 | Student proxy | Single file | Not read | CC BY-NC 4.0 | [HF](https://huggingface.co/datasets/phorknorak/khmer-asag-benchmark) | Yes |
| RIKEN-SAS | 2019 | ja | 31K / 34 | Reading, high school | Real | Analytic | Both | Not checked | Not read | Unverified | NII repository | Listed public, terms unverified |
| ELOQUENT 2026 | 2026 | 13 | See above | Mixed | Mixed | 0-2; 5-way | Rubric | Train / dev / test | QWK | Unverified | Not found | Test under NDA |
| Chang & Ginter | 2024 | fi | Conflict / 10 courses | University | Real | Not checked | Question only | n/a | QWK | n/a | None | No |
| CSEE (essay) | 2024 | en by zh students | 13,372 / 2 | School | Real | Essay score | n/a | 8:2 | n/a | Private | None | No |

### Inferences
- "Public" on a card is not the same as usable: UKARA lacks dev and test labels, the SEACrowd entry contains no data, and SAS-Bench and the Kaggle Indonesian set are not real student writing.
- Most real sets carry a non-commercial clause. That needs checking before any fine-tuned adapter is released beyond research use.

### Gaps
- Cells marked "Not checked" or "Not read" were not confirmed and were left blank rather than guessed.

## Do any recent papers test small models on these sets?

### Takeaway
Yes, and one result directly matches this project: the ELOQUENT organisers ran `gemma-4-E4B-it` and `gemma-4-E2B-it` as multilingual baselines. For Indonesian, the only small-LLM paper I found fine-tunes Gemma-3-1b-it on a private 220-pair set. I found no paper testing small LLMs on UKARA or the Haidir set.

### Cited Findings
- **Indonesian, Gemma-3-1b-it** (Faza et al., IJECBE 3(3), Sept 2025): 44-item test. Reasoning-guided model QWK 0.7791, Spearman 0.8276; standard fine-tune QWK 0.7279 with severe overfitting; SVR QWK 0.6952. Rationales came from DeepSeek R1-0528. The reasoning model took 54.63 s vs 1.55 s on the test set and 10,464 MB vs 2,989 MB of VRAM, on an RTX 4090 — [PDF](https://ijecbe.ui.ac.id/go/article/download/148/84)
- **13 languages, Gemma 4** (ELOQUENT 2026 rubric track): Qwen3.5-27B overall QWK 0.648; Gemma4-E4B 0.506 (PISA test 0.475, Additional test 0.536); Gemma4-E2B 0.286; EuroBERT-210m 0.331. Gemma baselines used vLLM with AWQ quantisation, temperature 0, an untuned prompt and reasoning before the answer. E4B fell from dev QWK 0.764 to 0.475 on PISA items — [overview](https://ceur-ws.org/Vol-4283/paper98.pdf)
- **German, Llama 1-8B** (ALICE LP macro-F1): rubric-retrieval Llama-3.2-3B-Instruct with question and solution input reached 76.7 on unseen answers and 69.5 on unseen questions. Zero-shot on unseen questions with rubric: Llama-3.1-8B 45.9, Mistral-7B 48.4, Llama-3.3-70B 60.6, GPT-5-mini 64.0 — [ALICE PDF](https://aclanthology.org/2026.emnlp-main.667.pdf)
- **German, Qwen LoRA**: BEA 2026 winner, see above — [report](https://aclanthology.org/2026.bea-1.85/). Also [SDPA, efficient LLM fine-tuning](https://aclanthology.org/2026.bea-1.92/) and [WSE Research](https://aclanthology.org/2026.bea-1.88/), neither fully read.
- **Chinese, SAS-Bench** average CCS: Qwen2.5-14B 64.44, Qwen3-8B 58.43, GLM4-9B 46.85, Llama3-8B 36.08 — [HF card](https://huggingface.co/datasets/aleversn/SAS-Bench)
- **Spanish** (synthetic answers): Gemma2-2B, Llama3.2-3B, Qwen2.5-7B and Gemma2-9B run through Ollama. The best models exceeded 95% on three-level grading; Llama3.2-3B tended to over-rate — [arXiv 2503.18072](https://arxiv.org/abs/2503.18072)
- **Bangla** (synthetic, not released): QLoRA Qwen3-8B human correlation 0.936 vs base 0.904 on a 259-record subset — [arXiv 2606.11931](https://arxiv.org/html/2606.11931)
- **Arabic**: Llama 3-8B lowest of four models — [IJIET](https://www.ijiet.org/show-234-3131-1.html)
- **Latvian/Lithuanian**: 7-12B models varied more than 70B models; Mistral Nemo 12B struggled with subtle Lithuanian changes — [arXiv 2501.09164](https://arxiv.org/abs/2501.09164)

### Inferences
- Across these papers, fine-tuned or rubric-aware small models do well on seen questions and drop on unseen questions or real items. The E4B fall on PISA items and the ALICE unseen-question gap are the clearest cases.
- The Faza result rests on 44 test items and one grader, so the QWK gap between methods is uncertain. Its 35x inference cost also matters directly for edge use.
- Testing Gemma on the Haidir set looks like an open gap, because I found no published small-LLM result on it.
- The ELOQUENT Gemma 4 numbers are a useful outside reference for the project's E4B work, but its baselines let the model reason first and used AWQ through vLLM. A `llama-server` run with thinking off would not be the same setup.

### Gaps
- Whether the Khmer report tests small models was not checked.
- No Indonesian result on quantised models or on edge hardware was found. See the `journal.laksita.id` gap above.
- An early search summary linked a Frontiers 2026 paper to Indonesian fine-tuning. That was wrong: the paper uses English data and only cites Faza et al. Its CPU results are in English — [Frontiers](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1829089/full).
