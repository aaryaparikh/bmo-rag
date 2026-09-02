from __future__ import annotations

from typer.testing import CliRunner

from bmo_rag.cli import app
from bmo_rag.retrieval.semantic import citation


def test_citation_uses_filename_and_pages() -> None:
    assert citation({"origin_filename": "report.pdf", "pages": [4]}) == "report.pdf, p. 4"
    assert citation({"source_url": "https://example.test", "pages": [2, 3]}) == (
        "https://example.test, pp. 2, 3"
    )


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
