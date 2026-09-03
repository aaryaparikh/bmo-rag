# Querying workflow

The project supports both retrieval inspection and a conversational answer pipeline. The `retrieve`
command prints ranked chunks; the `ask` command uses GPT-5 to answer from reranked, selectively
expanded evidence with citations and bounded in-process memory.

## Before a query

1. Put source PDFs and `urls.txt` under `data/raw/`.
2. Run `.uvenv\Scripts\python.exe scripts\ingestion\ingest.py ingest`. Docling extracts document structure, text, tables, page
   provenance, and headings. The ingestion pipeline cleans, splits, merges, and deduplicates that
   material into JSONL chunks under `data/processed/docling/`.
3. Run `.uvenv\Scripts\python.exe scripts\indexing\build_hybrid_index.py`. Each chunk is represented as its heading path plus
   body text. BGE-M3 creates a dense vector, while Qdrant also builds a native BM25 sparse
   representation. Qdrant stores both representations with the original text, headings, source,
   page numbers, and stable chunk ID as payload metadata.

The index is persistent, so these preparation steps do not need to be repeated for every question.
Re-run ingestion and indexing when the source corpus changes.

## What happens for one query

```text
Question
  -> BGE-M3 query embedding
  -> Qdrant dense search ----+
  -> Qdrant BM25 search -----+-> Reciprocal Rank Fusion (30 candidates)
                               -> BGE cross-encoder reranker
                               -> exact/contained-passage diversification
                               -> top-k chunks with citations
```

1. `scripts/querying/test_retrieval.py` forwards the command to the `retrieve` command in
   `src/bmo_rag/cli.py`.
2. Unless `--no-start-local` is used, the command starts or reuses local Docker services: Qdrant,
   a vLLM embedding endpoint, and—when enabled—a vLLM reranker endpoint.
3. The selected embedding model converts the question to a query vector. Model-specific prefixes
   are applied where required.
4. In the default hybrid mode, Qdrant runs two searches against the same corpus:
   dense semantic similarity and BM25 keyword matching. Reciprocal Rank Fusion (RRF) combines their
   ranks into a candidate list. The default candidate pool is 30 chunks.
5. The BGE reranker scores each candidate as a question–passage pair. Unlike independent embedding
   similarity, this cross-encoder reads the question and chunk together, giving a more precise final
   ordering.
6. Exact copies and chunks that fully contain an already higher-ranked passage are collapsed so
   repeated disclosures do not consume the evidence budget. Copies from explicitly requested
   sources remain available to preserve source-qualified provenance.
7. The command returns the requested `top_k` chunks and prints the rerank score, fusion score,
   stable chunk ID, source/page citation, section path, and text excerpt.

## Run a query

```powershell
.uvenv\Scripts\python.exe scripts\querying\test_retrieval.py "What was BMO's CET1 ratio?"
```

Omit the question for an interactive prompt. Use `-k 10` to return ten chunks.

The useful comparison modes are:

- Default: `--hybrid --rerank` uses dense + BM25 RRF, followed by cross-encoder reranking.
- `--no-rerank` returns the hybrid RRF order directly.
- `--dense` searches the older dense-only per-model collection.
- `--no-start-local` uses services that you have already started yourself.

## Where query results go

The quick query command prints results to the terminal and does not create files. Reproducible bulk
results are produced by the evaluation scripts and saved under `outputs/benchmarks/`. See
`outputs/README.md` for the meaning of each report.

## How to read the scores

- Dense score measures semantic vector similarity.
- BM25 rewards exact term overlap, which helps with names, acronyms, and financial labels.
- Fusion score reflects the combined dense and BM25 ranks; it is not a probability.
- Rerank score is the cross-encoder relevance score used for the final order; it is also not a
  calibrated probability.

The returned chunks are evidence candidates. The `ask` pipeline passes selected evidence to the
answer model and preserves source/page citations.

## Ask with GPT-5

Set the API key locally (do not commit it):

```powershell
$env:OPENAI_API_KEY="your-key"
```

Start a chat with an initial question:

```powershell
.uvenv\Scripts\python.exe scripts\querying\chat.py "What was BMO's CET1 ratio?"
```

The launcher stays open after answering so you can ask follow-up questions in the same
memory-enabled session. Type `exit` or `quit` to stop. GPT-5 receives a 5,000-token answer budget by
default. For unusually broad comparisons that need many cited figures, increase it explicitly:

```powershell
.uvenv\Scripts\python.exe scripts\querying\chat.py --max-answer-tokens 8000 "Compare ..."
```

Answer text streams to the terminal by default while it is generated. Token usage, the completed
answer, conversation memory, and observability traces are still recorded from the final Responses
API event. Use `--no-stream` when buffered output is preferable.

When the package is installed, use `bmo-rag ask --loop` to get the same behavior. You can also omit
the question to start the interactive session with an empty history:

```powershell
.uvenv\Scripts\python.exe scripts\querying\chat.py
```

For each turn, the pipeline retrieves 30 hybrid candidates, reranks and diversifies them to eight
seeds, and selectively adds same-source, same-section neighbors for broad or boundary-limited
evidence. All seed chunks are preserved before expansion. Context is capped at 32,000 characters, and every block
includes its document, pages, and heading path. GPT-5 receives the evidence through the Responses API
with remote response storage disabled.

When the question explicitly names a report, the pipeline resolves that human-readable name against
the indexed source catalog. It adds a filtered retrieval lane for each named report and guarantees up
to two reranked seeds from each lane before filling the remaining seed budget from global retrieval.
This prevents popular quarterly documents from crowding a named source out of comparison questions.
The CLI prints the resolved source constraints so name-to-file matching is auditable.

Conversation memory is intentionally session-local. It retains six recent question/answer pairs and
a compact structured state used only to rewrite follow-ups into standalone retrieval queries. It does
not persist financial facts or API keys.

Full local observability is enabled by default. See [`observability.md`](observability.md) for the
SQLite schema, captured fields, monitor commands, and data-retention warning.
