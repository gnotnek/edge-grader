# RAG source material (English and Indonesian)

*Gathered 2026-09-30. How the material is used: [`rag.md`](rag.md). The machine-readable list, with checksums and Wikipedia revision ids, is [`src/edge_grader/rag/sources.json`](../../src/edge_grader/rag/sources.json).*

## Get the files

```bash
uv run --group rag python -m edge_grader.rag.sources
```

This downloads everything into `data/rag/raw/<lang>/` (about 550 MB, git-ignored; ~10 minutes, mostly Wikipedia's rate limit) and checks each book against its pinned sha256. Files already there aren't downloaded again. `--pin` records the checksums and Wikipedia revisions after adding a source.

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
- **Why the SMP books too:** they hold most of the human body, electricity, levers and mixtures material. IPAS SD mixes science with social studies.

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
| RAG-en | English | English | the main RAG condition |
| RAG-id | English | Indonesian | cross-lingual retrieval: does material in another language still help? |
| RAG-both | English | both | does more material help or add noise? |

A fully Indonesian condition (questions and answers translated) would be a separate, larger piece of work.
