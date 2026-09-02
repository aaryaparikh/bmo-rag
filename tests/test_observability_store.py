from __future__ import annotations

from bmo_rag.generation.openai_responses import ResponseTelemetry
from bmo_rag.observability.store import SQLiteObservabilityStore


def test_sqlite_store_persists_macro_micro_llm_and_chunk_data(tmp_path) -> None:
    store = SQLiteObservabilityStore(tmp_path / "observability.sqlite3")
    trace = store.new_trace(
        session_id="session-1",
        original_query="What is CET1?",
        model="gpt-5",
        embedding_model="bge-m3",
        candidate_k=30,
        seed_k=8,
    )
    with trace.stage("query_embedding", metadata={"dimension": 1024}):
        pass
    trace.retrieval_query = "BMO CET1 ratio"
    trace.system_prompt = "Use evidence."
    trace.llm_input = "Evidence and question"
    trace.evidence_text = "The CET1 ratio was 13.0%."
    trace.evidence_tokens = 8
    trace.evidence_token_method = "test"
    trace.record_llm_call(
        ResponseTelemetry(
            operation="answer_generation",
            response_id="resp-1",
            model="gpt-5",
            instructions="Use evidence.",
            input_text="Evidence and question",
            output_text="13.0% [S1]",
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            cached_input_tokens=10,
            reasoning_tokens=5,
        )
    )
    trace.record_chunks(
        "reranked",
        [
            {
                "score": 0.91,
                "fusion_score": 0.03,
                "rerank_score": 0.91,
                "payload": {
                    "chunk_id": "chunk-1",
                    "source_id": "report",
                    "chunk_index": 2,
                    "text": "The CET1 ratio was 13.0%.",
                    "headings": ["Capital"],
                    "pages": [3],
                },
            }
        ],
        lane="merged",
    )
    trace.complete("13.0% [S1]")
    store.save(trace)

    detail = store.trace_detail(trace.trace_id)
    assert detail is not None
    assert detail["trace"]["duration_ns"] >= 0
    assert detail["trace"]["duration_us"] >= 0
    assert detail["trace"]["duration_ms"] >= 0
    assert detail["trace"]["duration_seconds"] >= 0
    assert detail["trace"]["input_tokens"] == 100
    assert detail["trace"]["output_tokens"] == 20
    assert detail["llm_calls"][0]["system_prompt"] == "Use evidence."
    assert '"rerank_score": 0.91' in detail["retrieved_chunks"][0]["scores_json"]
    assert store.recent(limit=1)[0]["trace_id"] == trace.trace_id
    summary = store.summary(limit=10)
    assert summary["trace_count"] == 1
    assert summary["failure_count"] == 0
    assert summary["total_tokens"] == 120


def test_failed_trace_is_persisted(tmp_path) -> None:
    store = SQLiteObservabilityStore(tmp_path / "observability.sqlite3")
    trace = store.new_trace(
        session_id="session-1",
        original_query="bad query",
        model="gpt-5",
        embedding_model="bge-m3",
        candidate_k=30,
        seed_k=8,
    )
    trace.fail(RuntimeError("retrieval unavailable"))
    store.save(trace)

    detail = store.trace_detail(trace.trace_id)
    assert detail is not None
    assert detail["trace"]["status"] == "failed"
    assert detail["trace"]["error_type"] == "RuntimeError"
    assert detail["trace"]["error_message"] == "retrieval unavailable"


def test_unsaved_running_trace_is_rejected(tmp_path) -> None:
    store = SQLiteObservabilityStore(tmp_path / "observability.sqlite3")
    trace = store.new_trace(
        session_id="session-1",
        original_query="query",
        model="gpt-5",
        embedding_model="bge-m3",
        candidate_k=30,
        seed_k=8,
    )
    try:
        store.save(trace)
    except ValueError as exc:
        assert "completed or failed" in str(exc)
    else:
        raise AssertionError("expected ValueError")
