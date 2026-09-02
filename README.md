# BMO RAG

Baseline Python project layout for a retrieval-augmented generation system.

## Structure

```text
.
├── config/                 # Runtime configuration templates
├── data/                   # Local documents, transformed chunks, and vector indexes
├── docs/                   # Architecture notes and project documentation
├── notebooks/              # Exploration notebooks
├── scripts/                # Utilities grouped by pipeline stage (see scripts/README.md)
├── src/bmo_rag/            # Application package
├── tests/                  # Unit and integration tests
└── outputs/                # Generated benchmarks, audits, and exports (see outputs/README.md)
```

## First Steps

1. Create a virtual environment.
2. Install dependencies with `pip install -e ".[dev]"`.
3. Copy `.env.example` to `.env` and fill in provider keys.
4. Add source documents under `data/raw/`.
5. Run ingestion, indexing, and query scripts as they are implemented.

## Core Pipeline

1. Load documents from configured sources.
2. Split and normalize content into chunks.
3. Embed chunks and store them in a vector index.
4. Retrieve relevant context for a user question.
5. Generate a grounded GPT-5 answer with source/page citations and bounded conversational memory.

## Embedding benchmark and Qdrant indexing

The benchmark indexes the current Docling chunks into a separate Qdrant collection for
each model and evaluates `retrieval_golden_200.jsonl` at k = 5, 10, 20, and 30.
It compares Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, Qwen3-Embedding-8B,
BGE-M3, and Nomic Embed v1.5.
Each model uses its native dense dimension, cosine distance, identical chunk content
(headings plus body), and exact Qdrant search. The report includes macro Precision@k,
Recall@k, MRR@k, and HitRate@k for all/development/test records.

Start Docker Desktop, then run the sequential local orchestrator:

```powershell
.uvenv\Scripts\python.exe scripts\evaluation\run_local_embedding_benchmark.py
```

The script starts Qdrant, then hosts exactly one model at a time with vLLM's pooling
runner on port 8000. It waits for the model, indexes/evaluates it, stops that server,
and moves to the next model. vLLM performs continuous batching behind its
OpenAI-compatible embeddings endpoint. Model and compile caches persist in named
Docker volumes, so later runs do not download everything again.

The RTX 4060 has 8 GB VRAM. The two Qwen checkpoints are therefore loaded with vLLM's
in-flight 4-bit BitsAndBytes quantization and a 2,048-token serving limit. This limit is
well above the current chunk sizes. BGE-M3 and Nomic run in FP16. Native dimensions are
4,096 (Qwen 8B), 2,560 (Qwen 4B), 1,024 (BGE-M3), and 768 (Nomic v1.5).

To run or resume only selected models:

```powershell
.uvenv\Scripts\python.exe scripts\evaluation\run_local_embedding_benchmark.py `
  --models bge-m3 nomic-embed-v1.5
```

Interrupted indexes resume from the chunk IDs already stored in Qdrant. Use `--reindex`
only to intentionally delete and rebuild the selected collections.

Qdrant binds only to localhost and stores data in the persistent Docker volume
`bmo-rag-qdrant-storage`. Confirm it at `http://localhost:6333/` or open its dashboard
at `http://localhost:6333/dashboard`.

The default report is `outputs/benchmarks/embedding_model_comparison/summary.json`. To benchmark one
already-running vLLM server directly:

```powershell
.uvenv\Scripts\python.exe scripts\evaluation\benchmark_embeddings.py `
  --models bge-m3 `
  --base-url http://127.0.0.1:8000/v1
```

The 11 empty-gold/abstention questions are counted in the report but excluded from the
three requested metrics because fixed top-k retrieval always returns candidates.

## Quick retrieval testing

Build the BGE-M3 hybrid index once. It stores named dense vectors and Qdrant-native
BM25 sparse vectors in a separate collection, preserving the dense benchmark index:

```powershell
.uvenv\Scripts\python.exe scripts\indexing\build_hybrid_index.py
```

Then inspect retrieval results directly. By default the command uses Qdrant's native
dense + BM25 Reciprocal Rank Fusion and reranks 30 candidates with
`BAAI/bge-reranker-v2-m3`. It starts or reuses local Qdrant and both vLLM services:

```powershell
.uvenv\Scripts\python.exe scripts\querying\test_retrieval.py "What was BMO's CET1 ratio?"
```

Omit the question for an interactive prompt, or change the number of returned chunks:

```powershell
.uvenv\Scripts\python.exe scripts\querying\test_retrieval.py -k 10
```

Use `--no-start-local` when Qdrant and a compatible vLLM endpoint are already running.
If the project has been installed as a CLI, the equivalent command is
`bmo-rag retrieve "your question"`.

Useful comparison modes:

```powershell
# Native Qdrant hybrid fusion without the cross-encoder
.uvenv\Scripts\python.exe scripts\querying\test_retrieval.py "your question" --no-rerank

# Original dense-only BGE retrieval
.uvenv\Scripts\python.exe scripts\querying\test_retrieval.py "your question" --dense
```

Run the 200-query comparison of dense BGE, native Qdrant hybrid RRF, and hybrid plus
cross-encoder reranking with:

```powershell
.uvenv\Scripts\python.exe scripts\evaluation\benchmark_hybrid_retrieval.py
```

The summary is written to `outputs/benchmarks/hybrid_retrieval_comparison/summary.json`,
with question-level retrieved and reranked chunks beside it in `query_details.jsonl`.

For a complete explanation of indexing, hybrid search, reranking, citations, and output scores, see
[`docs/querying-workflow.md`](docs/querying-workflow.md).
