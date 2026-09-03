from __future__ import annotations

from bmo_rag.generation.openai_responses import ResponseTelemetry
from bmo_rag.observability.store import SQLiteObservabilityStore
from bmo_rag.pipeline.chat import RAGChatbot


class FakeLLM:
    model = "gpt-5"

    def __init__(self) -> None:
        self.input_text = ""
        self.max_output_tokens = 0

    def text(self, *, input_text: str, **kwargs: object) -> str:
        self.input_text = input_text
        self.max_output_tokens = int(kwargs.get("max_output_tokens") or 0)
        output = "The ratio was 13.6% [S1]."
        on_text_delta = kwargs.get("on_text_delta")
        if callable(on_text_delta):
            on_text_delta(output)
        callback = kwargs.get("telemetry")
        if callable(callback):
            callback(
                ResponseTelemetry(
                    operation=str(kwargs.get("operation") or "generation"),
                    response_id="resp-test",
                    model=self.model,
                    instructions=str(kwargs.get("instructions") or ""),
                    input_text=input_text,
                    output_text=output,
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    cached_input_tokens=2,
                    reasoning_tokens=1,
                )
            )
        return output


class FakeStore:
    def __init__(self, url: str) -> None:
        self.url = url

    def related_points(self, *args: object, **kwargs: object) -> list[dict]:
        return []

    def source_catalog(self, name: str) -> dict[str, str | None]:
        return {"report": "report.pdf"}


def test_chatbot_retrieves_packs_answers_remembers_and_traces(monkeypatch, tmp_path) -> None:
    seed = {
        "score": 0.9,
        "payload": {
            "chunk_id": "chunk-1",
            "source_id": "report",
            "chunk_index": 4,
            "origin_filename": "report.pdf",
            "pages": [7],
            "headings": ["Capital"],
            "text": "The CET1 ratio was 13.6%.",
        },
    }
    captured: dict = {}

    def fake_retrieve(question: str, **kwargs: object) -> list[dict]:
        captured.update(question=question, **kwargs)
        return [seed]

    monkeypatch.setattr(
        "bmo_rag.pipeline.chat.retrieve_source_aware_hybrid_chunks", fake_retrieve
    )
    monkeypatch.setattr("bmo_rag.pipeline.chat.QdrantStore", FakeStore)
    llm = FakeLLM()
    observability = SQLiteObservabilityStore(tmp_path / "traces.sqlite3")
    chatbot = RAGChatbot(llm=llm, seed_k=1, observability_store=observability)

    result = chatbot.answer("What was the CET1 ratio?")

    assert captured["question"] == "What was the CET1 ratio?"
    assert captured["candidate_k"] == 30
    assert "rerank" not in captured
    assert "[S1] Source: report.pdf, p. 7" in llm.input_text
    assert llm.max_output_tokens == 5000
    assert result.answer == "The ratio was 13.6% [S1]."
    assert result.sources[0].chunk_id == "chunk-1"
    assert chatbot.memory.turns[0].question == "What was the CET1 ratio?"
    assert result.trace_id is not None
    detail = observability.trace_detail(result.trace_id)
    assert detail is not None
    assert detail["trace"]["input_tokens"] == 10
    assert detail["trace"]["output_tokens"] == 5
    assert detail["trace"]["system_prompt"]
    assert detail["trace"]["evidence_tokens"] > 0
    assert {row["name"] for row in detail["stages"]} >= {
        "query_normalization",
        "context_expansion",
        "answer_generation",
    }
    answer_stage = next(
        row for row in detail["stages"] if row["name"] == "answer_generation"
    )
    assert '"max_output_tokens": 5000' in answer_stage["metadata_json"]
    assert {row["stage"] for row in detail["retrieved_chunks"]} == {
        "expanded",
        "context",
    }
