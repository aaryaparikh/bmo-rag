# Section-aware chunking golden dataset

`section_aware_chunking_golden.jsonl` contains 100 reproducibly sampled chunks from the
PDFs in `data/raw`. It is intended to test whether a chunker preserves the section
context in which each chunk occurs.

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
python scripts/generate_section_chunking_golden.py
```

The fixed default seed is `20260831`. The manifest records corpus coverage, extraction
methods, and corner-case counts.
