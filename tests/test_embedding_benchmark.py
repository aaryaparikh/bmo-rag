from __future__ import annotations

import json

from bmo_rag.evaluation.retrieval import metrics_at_k, metrics_by_facets
from bmo_rag.indexing.corpus import load_corpus, stable_chunk_id
from bmo_rag.indexing.embeddings import MODEL_SPECS, VllmEmbeddingProvider, resolve_model
from bmo_rag.indexing.qdrant_store import QdrantStore, collection_name


def _record(record_id: str, gold: list[str], split: str = "test") -> dict:
    return {
        "id": record_id,
        "question": "question?",
        "split": split,
        "expected_chunks": [{"chunk_id": chunk_id} for chunk_id in gold],
    }


def test_metrics_compute_macro_precision_recall_and_mrr() -> None:
    records = [_record("one", ["a"]), _record("two", ["b", "c"]), _record("empty", [])]
    rankings = {
        "one": ["x", "a", "z"],
        "two": ["b", "x", "c"],
    }

    result = metrics_at_k(records, rankings, (2,))

    assert result["evaluated_answerable_count"] == 2
    assert result["excluded_empty_gold_count"] == 1
    assert result["cutoffs"]["2"] == {
        "precision": 0.5,
        "recall": 0.75,
        "mrr": 0.75,
        "hit_rate": 1.0,
    }


def test_metrics_include_query_difficulty_and_edge_case_facets() -> None:
    factual = _record("factual", ["a"])
    factual.update(query_type="factual", difficulty="easy", edge_case="standard")
    table = _record("table", ["b"])
    table.update(query_type="table_lookup", difficulty="hard", edge_case="citation_required")
    empty = _record("empty", [])
    empty.update(query_type="unanswerable", difficulty="easy", edge_case="future_period")
    rankings = {"factual": ["a", "x"], "table": ["x", "b"]}

    result = metrics_by_facets([factual, table, empty], rankings, (2,))

    assert result["by_query_type"]["factual"]["cutoffs"]["2"]["recall"] == 1.0
    assert result["by_query_type"]["table_lookup"]["cutoffs"]["2"]["mrr"] == 0.5
    assert result["by_difficulty"]["easy"]["excluded_empty_gold_count"] == 1
    assert result["by_edge_case"]["future_period"]["cutoffs"] == {}
    assert "undefined" in result["by_edge_case"]["future_period"]["not_evaluated_reason"]


def test_corpus_loader_matches_golden_chunk_id_algorithm(tmp_path) -> None:
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    raw = {
        "text": "  A   normalized passage. ",
        "meta": {
            "headings": [" Section  One "],
            "doc_items": [{"prov": [{"page_no": 7}]}],
            "origin": {"filename": "sample.pdf"},
        },
    }
    (chunk_dir / "sample.chunks.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")

    (chunk,) = load_corpus(chunk_dir)

    assert chunk.chunk_id == stable_chunk_id("sample", "A normalized passage.")
    assert chunk.embedding_text == "Section One\n\nA normalized passage."
    assert chunk.pages == (7,)


def test_open_model_registry_has_native_dimensions_and_task_prefixes() -> None:
    assert set(MODEL_SPECS) == {
        "qwen3-embedding-8b",
        "qwen3-embedding-4b",
        "bge-m3",
        "nomic-embed-v1.5",
    }
    assert resolve_model("Qwen/Qwen3-Embedding-8B").dimension == 4096
    qwen = resolve_model("qwen3-embedding-4b")
    assert qwen.prepare("question", input_type="query").startswith("Instruct:")
    assert qwen.prepare("document", input_type="document") == "document"
    nomic = resolve_model("nomic-embed-text-v1.5")
    assert nomic.prepare("question", input_type="query") == "search_query: question"
    assert nomic.prepare("passage", input_type="document") == "search_document: passage"


def test_vllm_provider_sends_prepared_input_and_model(monkeypatch) -> None:
    spec = resolve_model("nomic-embed-v1.5")
    provider = VllmEmbeddingProvider(spec, base_url="http://localhost:8000/v1")
    captured: dict = {}

    def fake_post(body: dict) -> dict:
        captured.update(body)
        return {"data": [{"index": 0, "embedding": [0.0] * spec.dimension}]}

    monkeypatch.setattr(provider, "_post", fake_post)
    provider.embed(["where?"], input_type="query")

    assert captured["model"] == "nomic-ai/nomic-embed-text-v1.5"
    assert captured["input"] == ["search_query: where?"]


def test_qdrant_ids_and_collection_names_are_stable() -> None:
    assert QdrantStore.point_id("bmo-abc") == QdrantStore.point_id("bmo-abc")
    assert collection_name("bge-m3", 1024) == "bmo_chunks_bge-m3_d1024"
