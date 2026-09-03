from __future__ import annotations

from typer.testing import CliRunner

from bmo_rag.cli import app
from bmo_rag.retrieval.semantic import citation, deduplicate_retrieved_points


def test_citation_uses_filename_and_pages() -> None:
    assert citation({"origin_filename": "report.pdf", "pages": [4]}) == "report.pdf, p. 4"
    assert citation({"source_url": "https://example.test", "pages": [2, 3]}) == (
        "https://example.test, pp. 2, 3"
    )


def test_retrieval_deduplicates_content_but_can_preserve_requested_sources() -> None:
    passage = "A complete evidence passage with enough detail to be safely recognized. " * 3
    points = [
        {"id": 1, "payload": {"chunk_id": "a", "source_id": "one", "text": passage}},
        {"id": 2, "payload": {"chunk_id": "b", "source_id": "two", "text": passage.upper()}},
        {
            "id": 4,
            "payload": {
                "chunk_id": "d",
                "source_id": "four",
                "text": f"Introduction. {passage} Appendix.",
            },
        },
        {"id": 3, "payload": {"chunk_id": "c", "source_id": "three", "text": "Different."}},
    ]

    deduplicated = deduplicate_retrieved_points(points)
    source_preserved = deduplicate_retrieved_points(points, preserve_source_ids={"two"})

    assert [point["payload"]["chunk_id"] for point in deduplicated] == ["a", "c"]
    assert deduplicated[0]["duplicate_chunk_ids"] == ["b", "d"]
    assert [point["payload"]["chunk_id"] for point in source_preserved] == ["a", "b", "c"]


def test_retrieve_cli_defaults_to_bge_and_prints_result(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_retrieve(question: str, **kwargs: object) -> list[dict]:
        captured.update(question=question, **kwargs)
        return [
            {
                "score": 0.81234,
                "payload": {
                    "chunk_id": "bmo-test",
                    "origin_filename": "report.pdf",
                    "pages": [7],
                    "headings": ["Capital"],
                    "text": "The retrieved passage.",
                },
            }
        ]

    monkeypatch.setattr("bmo_rag.cli.retrieve_hybrid_chunks", fake_retrieve)
    runner = CliRunner()
    result = runner.invoke(app, ["retrieve", "What is the ratio?", "--no-start-local"])

    assert result.exit_code == 0
    assert captured["model"] == "bge-m3"
    assert "BAAI/bge-m3" in result.stdout
    assert "Mode: hybrid + reranker" in result.stdout
    assert "score=0.8123" in result.stdout
    assert "report.pdf, p. 7" in result.stdout
    assert "The retrieved passage." in result.stdout
