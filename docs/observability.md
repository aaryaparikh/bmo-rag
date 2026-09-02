# RAG observability

Chat tracing is enabled by default and stored locally in
`data/observability/rag_observability.sqlite3`. The database uses SQLite WAL mode and is excluded
from Git because it contains complete user queries, document evidence, prompts, and responses.
It never stores `OPENAI_API_KEY`.

## Inspect recent traces

```powershell
.uvenv\Scripts\python.exe scripts\monitoring\observability.py
```

The summary reports request count, failures, average/p50/p95/max end-to-end latency, and aggregate
input/output/total token use. Each recent row includes its trace ID, status, end-to-end latency,
input/output tokens, evidence tokens, and query.

Use the trace ID printed after every chat answer to inspect the complete record:

```powershell
.uvenv\Scripts\python.exe scripts\monitoring\observability.py --trace-id TRACE_ID
```

Use `chat.py --no-observe` to disable collection, or `--observability-db PATH` on both commands to
select another database.

## Database contents

- `traces`: macro request latency in nanoseconds, microseconds, milliseconds, and seconds; query;
  rewritten retrieval query; main system prompt; complete model input; evidence; evidence token
  count; aggregate model tokens; source constraints; response; and failure information.
- `stage_timings`: ordered micro-timings for normalization, conversational rewrite, catalog loading,
  source resolution, embedding, global and source-filtered retrieval, reranking, source quota
  selection, expansion, packing, token counting, generation, and memory update. Durations are stored
  in nanoseconds, microseconds, and milliseconds.
- `llm_calls`: one row per OpenAI call, including operation, response ID, model, system prompt,
  complete input/output, and server-reported input, output, total, cached-input, and reasoning tokens.
- `retrieved_chunks`: snapshots of global candidates, source-filtered candidates, complete reranked
  candidates, final seeds, expanded chunks, and final evidence chunks. Each row preserves rank, lane,
  text, source metadata, context role, and every score available at that stage (fusion and reranker).

Evidence token counts use `tiktoken` locally. OpenAI input/output totals come from the API response's
`usage` object and are therefore kept separate from locally counted evidence tokens.

## Retention and privacy

This configuration deliberately provides full forensic logging and therefore stores potentially
sensitive query and document content. Restrict filesystem access to the database and establish a
retention/deletion policy before deploying beyond a development machine.
