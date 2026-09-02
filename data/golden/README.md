# Docling hierarchy gold standard

`docling_hierarchy_golden_100.jsonl` is the dataset to use when testing whether
Docling assigned the correct PDF section hierarchy. Unlike the older page-text set
below, every record is an **actual chunk from `data/processed/docling/*.chunks.jsonl`**.

Each of the 100 records contains:

- `chunk`: the normalized Docling chunk text;
- `expected_section_heading`: the independently verified heading that must be present;
- `expected_hierarchy`: the PDF root-to-parent heading path;
- `observed_docling_heading` and `observed_docling_hierarchy`: Docling's output;
- `hierarchy_verdict`: whether the required heading/path is present;
- source PDF, page, original chunk index, text hash, and annotation method.

Expected labels are sourced from native PDF outlines where they are semantic and
from rendered-page adjudication where no useful outline exists. Docling headings are
never used to create expected labels. Generic bookmarks such as `Page 43`, `Cover`,
and `Slide Number 5` are excluded from the gold labels.

Current 100-chunk result:

- 8 exact hierarchy paths;
- 28 required headings present but with an incomplete hierarchy path;
- 64 required PDF headings missing from Docling's hierarchy;
- required-heading recall: **36%**;
- visually reviewed immediate-heading accuracy: **93.33%** (14/15).

This indicates that Docling usually recognizes the local printed heading, but often omits
the broader PDF section ancestor. The set covers all 22 PDFs: 85 labels use semantic PDF
bookmark intervals and 15 use rendered-page review with text anchors.

Regenerate with the bundled PDF runtime:

```powershell
$env:PYTHONPATH='scripts'
python scripts\evaluation\build_docling_hierarchy_gold.py
```

The manifest is `docling_hierarchy_golden_100.manifest.json`.

## Legacy PDF-extracted section-aware dataset

`section_aware_chunking_golden.jsonl` contains 100 reproducibly sampled chunks from the
PDFs in `data/raw`. It is intended to test whether a chunker preserves the section
context in which each chunk occurs.

This older file re-chunks text extracted directly from PDF pages. It does **not** label
the actual Docling chunks and therefore should not be used for the Docling hierarchy
evaluation requested here.

Each JSONL record contains:

- `id`: stable record identifier within this generated dataset.
- `source_document` and `source_pages`: source provenance (1-based PDF pages).
- `chunk` and `chunk_sha256`: extracted chunk text and its integrity hash.
- `expected.section_contexts`: one or more expected contexts. Each context has the
  immediate `parent_section`, root-to-parent `ancestry`, and `annotation_method`.
- `corner_cases`: overlapping labels such as `deep_ancestry`,
  `table_or_numeric_dense`, `multiple_sections`, and `short_chunk`.

For a chunk spanning a section boundary, `section_contexts` contains both expected
paths in reading order. Native PDF outlines are the preferred source of truth.
`visual_reviewed` marks labels adjudicated against rendered source pages for PDFs without
usable outlines. `layout_inference_review_required` is reserved for unreviewed inferred
labels and should not appear in a released strict-gold dataset.

Regenerate from the repository root with:

```powershell
python scripts/evaluation/generate_section_chunking_golden.py
```

The fixed default seed is `20260831`. The manifest records corpus coverage, extraction
methods, and corner-case counts.

## Retrieval gold standard (100 queries)

`retrieval_golden_100.jsonl` evaluates retrieval against the **current hierarchical
Docling chunks**, rather than evaluating PDF section extraction. It contains 100 curated
questions spanning all 22 PDFs and all four URLs (26 sources total), with an 80/20
development/test split.

Each record contains:

- `query`, `query_type`, `difficulty`, and `split`;
- `gold`: a relevance-3 direct evidence chunk plus up to three relevance-1
  same-section supporting chunks;
- stable, content-derived `chunk_id` values, source locator, pages, headings, evidence
  excerpt, and full-text SHA-256;
- three same-document hard-negative chunk IDs where available.

The relevance scale is `3 = direct answer`, `1 = supporting section context`, and
`0 = not relevant`. Use relevance 3 for Recall/Hit/MRR and both grades for nDCG.

Regenerate and evaluate from the repository root:

```powershell
.\.uvenv\Scripts\python.exe scripts\evaluation\build_retrieval_gold.py
.\.uvenv\Scripts\python.exe scripts\audit_docling_chunks.py
.\.uvenv\Scripts\python.exe scripts\evaluate_retrieval_gold.py
```

The measured lexical baselines are stored in `retrieval_baseline_metrics.json`:

| Baseline | Hit@1 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25, body only | 0.260 | 0.490 | 0.610 | 0.3717 | 0.3923 |
| BM25, headings + body | 0.260 | 0.530 | 0.630 | 0.3904 | 0.4193 |

These are corpus-specific baselines, not production acceptance targets. A candidate
embedding or hybrid retriever should be compared on the frozen test split and should
beat the heading-enriched BM25 baseline without tuning on those 20 test questions.

## Retrieval gold standard (200 questions)

`retrieval_golden_200.jsonl` is the expanded retrieval/grounding set. It contains 200
unique questions: 100 independently anchored source questions, 80 semantic robustness
variants, and 20 manually adjudicated reconciliation, false-premise, ambiguous,
prompt-injection, privacy, future-period, and no-answer cases. A flat
`retrieval_golden_200.csv` companion is provided for review.

Positive labels live in `expected_chunks` and include a content-derived `chunk_id`, the
current JSONL `chunk_index`, pages/headings, an evidence excerpt, and a full-text SHA-256.
An empty `expected_chunks` array is the gold label for abstention. The manifest records
the corpus fingerprint and known source-date caveats.

Regenerate and run the lexical smoke baseline with:

```powershell
.\.uvenv\Scripts\python.exe scripts\evaluation\build_retrieval_gold_200.py
.\.uvenv\Scripts\python.exe scripts\evaluate_retrieval_gold_200.py
```

The BM25 report is stored in `retrieval_golden_200_baseline.json`. Empty-gold cases are
excluded from ranking metrics because a rank-only baseline has no abstention threshold.

## Vector-ingestion decision

The current corpus is **conditionally ready after cleanup**, not ready to ingest as-is.
The complete machine-readable audit is in `chunk_quality_audit.json`. The primary gates
that currently fail are missing native chunk IDs, 4 chunks without hierarchy, 857 chunks
under 300 characters (13.84%), 4 known web navigation/footer chunks, and 60 chunks with
control characters. All PDF chunks retain page provenance and every chunk respects the
1,500-character ceiling.

Before indexing, derive/persist a stable `chunk_id`, strip control characters and web
chrome, merge or deliberately retain short table fragments, and prepend the hierarchy
to the embedding text while storing the unmodified body separately for citation.
