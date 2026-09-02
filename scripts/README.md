# Script guide

Run scripts from the project root. They are grouped by the pipeline stage they support.

## `ingestion/`

- `ingest.py` converts PDFs and configured web sources into cleaned Docling chunks under
  `data/processed/docling/`. Run it as
  `.uvenv\Scripts\python.exe scripts\ingestion\ingest.py ingest`.

## `indexing/`

- `build_hybrid_index.py` embeds processed chunks and stores dense vectors plus Qdrant-native
  BM25 data in the hybrid Qdrant collection. Run it as
  `.uvenv\Scripts\python.exe scripts\indexing\build_hybrid_index.py`.

## `querying/`

- `test_retrieval.py` is the normal interactive or one-question retrieval command.
- `chat.py` runs the GPT-5 grounded answer pipeline; omit the question for a memory-enabled session.
- `corpus_search.py` searches the processed JSONL chunks directly with a regular expression;
  it is mainly useful when curating or debugging evaluation data.

## `evaluation/`

- `benchmark_embeddings.py` compares one running embedding model against the retrieval gold set.
- `run_local_embedding_benchmark.py` starts each local model in turn and orchestrates the embedding
  comparison.
- `benchmark_hybrid_retrieval.py` compares dense retrieval, hybrid RRF, and hybrid plus reranking.
- `evaluate_retrieval_gold*.py` runs lightweight lexical baselines.
- `build_retrieval_gold*.py`, `build_docling_hierarchy_gold.py`, and
  `generate_section_chunking_golden.py` build reproducible evaluation datasets.
- `audit_docling_chunks.py` checks processed chunks against data-quality gates.

Generated reports belong under `outputs/`; curated test fixtures belong under `data/golden/`.

## `monitoring/`

- `observability.py` lists recent latency/token summaries or emits a complete JSON trace selected
  with `--trace-id`.
