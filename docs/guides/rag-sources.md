# RAG source material (English and Indonesian)

*Gathered 2026-09-30. How the material is used: [`rag.md`](rag.md). The machine-readable list, with checksums and Wikipedia revision ids, is [`src/edge_grader/rag/sources.json`](../../src/edge_grader/rag/sources.json).*

There is material for both baselines. Each book in `sources.json` names the benchmark it serves (`"benchmark"`):
- **[SciEntsBank](#part-a-scientsbank)**: school science, grades 3–9
- **[MMLU-Pro](#part-b-mmlu-pro)**: college subjects

## Get the files

```bash
uv run --group rag python -m edge_grader.rag.sources                     # everything
uv run --group rag python -m edge_grader.rag.sources --only scientsbank  # ~550 MB
uv run --group rag python -m edge_grader.rag.sources --only mmlupro      # ~3.5 GB
```

This downloads into `data/rag/raw/<lang>/` (git-ignored) and checks each book against its pinned sha256. Files already there aren't downloaded again. `--pin` records the checksums and Wikipedia revisions after adding a source.

# Part A: SciEntsBank

## What SciEntsBank needs

SciEntsBank's answers come from 15 units of the FOSS science curriculum (grades 3–6). The unit code is the start of each answer's `id`:

| Code | FOSS unit | Split |
|---|---|---|
| EM | Earth Materials | train, test_ua, test_uq |
| FN | Food and Nutrition | 〃 |
| II | Ideas and Inventions | 〃 |
| LF | Landforms | 〃 |
| LP | Levers and Pulleys | 〃 |
| ME | Magnetism and Electricity | 〃 |
| MS | Measurement | 〃 |
| MX | Mixtures and Solutions | 〃 |
| PS | Physics of Sound | 〃 |
| SE | Solar Energy | 〃 |
| ST | Structures of Life | 〃 |
| VB | Variables | 〃 |
| **EV** | **Environments** | **test_ud only** |
| **HB** | **Human Body** | **test_ud only** |
| **WA** | **Water** | **test_ud only** |

FOSS's own texts are copyrighted, so the corpus uses open material on the same topics.

## Sources

### 1. English textbooks: Utah OER SEEd, grades 3–8

- **Publisher:** Utah State Board of Education (Utah OER Textbook Project), 2025–2026 editions
- **Licence:** CC BY-NC
- **Where:** [UEN eMedia](https://emedia.uen.org/curated-collections/108); the files are on Google Drive
- **Size:** 6 books, 937 pages, ~125k words
- **Fit:** the right grade level, but written for Utah's SEEd standards (NGSS-style), not for FOSS

### 2. Indonesian textbooks: Kurikulum Merdeka IPAS / IPA, kelas III–IX

- **Publisher:** Pusat Perbukuan, Kemendikdasmen; the latest *edisi revisi* student books (*buku siswa*)
  - IPAS SD kelas III–VI
  - IPA SMP kelas VII–IX
- **Where:** [SIBI](https://buku.kemendikdasmen.go.id). The catalogue API (`api.buku.cloudapp.web.id/api/catalogue/getPenggerakTextBooks`) gives the direct PDF links.
- **Licence:** "Hak Cipta … Dilindungi Undang-Undang" (all rights reserved), but free to download and read. **Use them for research, and don't commit or redistribute the text.**
- **Size:** 7 books, 1,650 pages, ~300k words
- **Why the SMP books too:** they hold most of the human body, electricity, levers and mixtures material.
- **IPAS is science and social studies combined** (*Ilmu Pengetahuan Alam dan Sosial*), and runs from kelas III; kelas I–II have no IPAS. Roughly half of each IPAS book is social studies (history, economics, culture). The books are used whole for now, with no chapter filter (decided 2026-09-30).

### 3. Wikipedia, to fill the gaps (both languages)

- **Licence:** CC BY-SA 4.0
- **English:** Simple English Wikipedia (written for learners), or English Wikipedia where Simple has no article
- **Indonesian:** Indonesian Wikipedia, through the English article's interlanguage link, so both languages cover the same topics
- **Topics:** 4–8 per unit, chosen from the FOSS unit titles. The list is in `sources.json` under `wikipedia.topics`.
- **Reproducibility:** each article's revision id is pinned. Wikipedia changes, so a re-fetch reports any article that has changed since.
- **Size:** 85 topics.
  - **English:** 84 articles, ~48k words. 78 are from Simple English Wikipedia and 6 from English Wikipedia.
  - **Indonesian:** 79 articles, ~90k words.
- **Not found:** *Tolerance (ecology)* in either language. Five topics have no Indonesian article: Reflection symmetry, Loudness, Absorption (electromagnetic radiation), Scientific control, Dependent and independent variables.

## Coverage of the books

Counted as book pages that mention at least two of a unit's keywords. This is a rough screen, not a relevance judgement.

| Unit | English books | Indonesian books | Gap filled by Wikipedia |
|---|---|---|---|
| EM Earth Materials | 10 | 8 | yes |
| FN Food and Nutrition | 4 | 43 | English |
| II Ideas and Inventions | 0 | 0 | **both, and still thin** |
| LF Landforms | 15 | 1 | Indonesian |
| LP Levers and Pulleys | **0** | 14 | English |
| ME Magnetism and Electricity | 9 | 44 | |
| MS Measurement | 24 | 43 | |
| MX Mixtures and Solutions | 6 | 22 | English |
| PS Physics of Sound | 25 | 32 | |
| SE Solar Energy | 123 | 176 | |
| ST Structures of Life | 24 | 19 | |
| VB Variables | 5 | 31 | English |
| **EV Environments** | 64 | 62 | |
| **HB Human Body** | 8 | 29 | English |
| **WA Water** | 9 | 8 | both |

**Ideas and Inventions** is a mixed unit (chromatography, symmetry, fingerprints, inventions). Expect RAG to help it least.

## Things to know before using the corpus

- **Most of the Indonesian books break some ligatures on extraction.** "Penafian" comes out as "Penaian", or with stray characters such as `昀椀` or `Ƽ` in place of "fi"/"ti". Clean these up in the extraction step (step 1 in [`rag.md`](rag.md)).
- **The questions are in English.** Retrieving Indonesian passages for them needs a multilingual embedding model, e.g. `intfloat/multilingual-e5-small`; `bge-small-en-v1.5` is English-only. Keep the English and Indonesian indexes separate, so each can be tested on its own (see below).
- **Test-set caution:** the unit list comes from the answer ids, and the Wikipedia topics from the unit titles. A few test_ud question texts were looked at while identifying the units. Don't tune the topic list further by reading test questions or answers. Use the train split for any tuning.

## Possible experiments with the two languages

| Condition | Question | Corpus | What it tests |
|---|---|---|---|
| RAG-en | English | English | **the main RAG condition** |
| RAG-id | English | Indonesian | cross-lingual retrieval: does material in another language still help? **Optional** (2026-10-03): run only if time allows |
| ~~RAG-both~~ | English | both | **Dropped** (2026-10-03): it doesn't answer a research question |

A fully Indonesian condition (questions and answers translated) would be a separate, larger piece of work.

# Part B: MMLU-Pro

## What MMLU-Pro needs

The subject-knowledge baseline is the lab's MMLU-Pro subset: 3 × 100 questions (s1/s2/s3), stratified over MMLU-Pro's 14 subjects ([`../results/baselines.md`](../results/baselines.md)). Questions per subject, pooled over the 300:

| Subject | Questions | Subject | Questions |
|---|---|---|---|
| math | 33 | economics | 21 |
| physics | 33 | health | 21 |
| chemistry | 27 | psychology | 21 |
| law | 27 | biology | 18 |
| engineering | 24 | philosophy | 12 |
| other | 24 | computer science | 9 |
| business | 21 | history | 9 |

The questions are college level, so this material is too.

*The subset files (`results/reference/agentic-edge/lm-eval/mmlupro_subset100_*_ids.json`) hold MMLU-Pro's global `question_id`s, even though their `method` text says "positions". lm-eval's positions are in the separate `*_samples.json` files.*

## Sources

### 4. English: OpenStax college textbooks

- **Publisher:** OpenStax, Rice University; the current web editions
- **Licence:** CC BY-NC-SA 4.0 (a few titles CC BY 4.0; each book's licence is in `sources.json`)
- **Where:** [openstax.org](https://openstax.org). The CMS API (`openstax.org/apps/cms/api/v2/pages/?type=books.Book`) gives the PDF links.
- **Size:** 25 books, 20,250 pages, ~8.0M words (~3.0 GB)

| Subject | Books |
|---|---|
| math | Calculus Vol. 1 and 2, Introductory Statistics 2e |
| physics, engineering | University Physics Vol. 1–3 |
| chemistry | Chemistry 2e, Organic Chemistry |
| law | Business Law I Essentials 2e, Introduction to Intellectual Property, American Government 4e |
| business | Introduction to Business 2e, Principles of Financial Accounting, Principles of Finance 2e (plus Business Law, Management, Marketing) |
| other | Principles of Management, Principles of Marketing |
| economics | Principles of Economics 3e |
| health | Anatomy and Physiology 2e |
| psychology | Psychology 2e |
| biology | Biology 2e |
| philosophy | Introduction to Philosophy |
| computer science | Introduction to Computer Science |
| history | World History Vol. 1 and 2, U.S. History |

### 5. Indonesian: Kurikulum Merdeka SMA subject books, kelas XI–XII

- **Publisher:** Pusat Perbukuan, Kemendikdasmen (SIBI, same API and licence as source 2: all rights reserved, research use only)
- **Books:** Biologi, Fisika, Kimia, Matematika Tingkat Lanjut, Ekonomi, Sejarah and Informatika, kelas XI and XII each
- **Size:** 14 books, 3,680 pages, ~640k words (~430 MB)
- **Level:** upper secondary, one step below MMLU-Pro. There are no Indonesian books for law, philosophy, psychology or engineering, and no health book beyond Biologi.
- **Informatika kelas XII has no text layer:** 0 words extracted from 248 pages. It needs OCR before it can be used; `sources.json` flags it with a `note`.

## Coverage and gaps

- **Well covered:** math, physics, chemistry, biology, health, psychology, economics, business, philosophy, computer science and history each have a full college textbook.
- **Engineering (24 questions):** OpenStax has no engineering textbook. University Physics covers circuits, thermodynamics and mechanics, which is only part of it.
- **Law (27 questions):** MMLU-Pro's law questions are mostly US bar-exam style. The OpenStax books cover business law, intellectual property and government, not criminal law, torts or evidence, so expect little help here.
- **Other (24 questions):** a mixed category; management and marketing cover only part of it.
- **History:** MMLU-Pro history is mostly US and European. The Indonesian Sejarah books are about Indonesian history, so they won't help it.

**Leakage check (2026-09-30):** none of the 300 subset questions shares a 12-word verbatim run with the OpenStax books. Retrieval can't simply hand the model a copied question and its answer.

## Using RAG on MMLU-Pro

- **The baseline protocol has to change.** The lab's baseline uses lm-eval's 5-shot CoT prompt. Adding retrieved passages needs a custom lm-eval task, or this project's own runner, so the RAG run is a new protocol. Report it next to the baseline, not as a drop-in replacement.
- **Context length:** the longest baseline prompt is 2,362 tokens and the server runs with `-c 8192`, so 3 passages of ~300 tokens fit.
- **Query:** search with the question text, not the answer options. The options would pull in passages for wrong answers.
