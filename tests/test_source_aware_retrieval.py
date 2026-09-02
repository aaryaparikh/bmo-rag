from __future__ import annotations

from bmo_rag.retrieval.semantic import retrieve_source_aware_hybrid_chunks
from bmo_rag.retrieval.sources import SourceConstraint


def _point(chunk: str, source: str, score: float) -> dict:
    return {
        "id": chunk,
        "score": score,
        "payload": {"chunk_id": chunk, "source_id": source, "text": chunk},
    }


def test_explicit_sources_receive_guaranteed_results(monkeypatch) -> None:
    global_points = [_point(f"global-{index}", "quarterly", 1 - index / 100) for index in range(8)]
    annual_points = [_point("annual-1", "annual", 0.1), _point("annual-2", "annual", 0.09)]

    class Provider:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def embed_queries(self, questions: list[str]) -> list[list[float]]:
            return [[0.1, 0.2]]

    class Store:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def hybrid_search_points(self, *args: object, **kwargs: object) -> list[dict]:
            return annual_points if kwargs.get("source_ids") else global_points

    class Reranker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def rerank(self, question: str, points: list[dict], *, top_k: int) -> list[dict]:
            return sorted(points, key=lambda item: item["score"], reverse=True)[:top_k]

    monkeypatch.setattr("bmo_rag.retrieval.semantic.VllmEmbeddingProvider", Provider)
    monkeypatch.setattr("bmo_rag.retrieval.semantic.QdrantStore", Store)
    monkeypatch.setattr("bmo_rag.retrieval.semantic.VllmReranker", Reranker)

    result = retrieve_source_aware_hybrid_chunks(
        "Compare the annual report",
        source_constraints=[SourceConstraint("Annual Report", ("annual",))],
        top_k=4,
    )

    assert sum(item["payload"]["source_id"] == "annual" for item in result) == 2
    assert len(result) == 4
