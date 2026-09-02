"""Dense retrieval against the per-model Qdrant collections."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from bmo_rag.indexing.embeddings import VllmEmbeddingProvider, resolve_model
from bmo_rag.indexing.qdrant_store import (
    QdrantStore,
    collection_name,
    hybrid_collection_name,
)
from bmo_rag.retrieval.reranker import VllmReranker
from bmo_rag.retrieval.sources import SourceConstraint

if TYPE_CHECKING:
    from bmo_rag.observability.store import Trace


def citation(payload: dict[str, Any]) -> str:
    """Build a readable source citation from stored chunk metadata."""
    source = payload.get("source_url") or payload.get("origin_filename") or payload.get("source_id")
    pages = payload.get("pages") or []
    if not pages:
        return str(source or "unknown source")
    label = "p." if len(pages) == 1 else "pp."
    return f"{source}, {label} {', '.join(str(page) for page in pages)}"


def retrieve_chunks(
    question: str,
    *,
    model: str = "bge-m3",
    top_k: int = 5,
    base_url: str = "http://127.0.0.1:8000/v1",
    qdrant_url: str = "http://127.0.0.1:6333",
    exact: bool = True,
) -> list[dict[str, Any]]:
    """Embed a question and return full ranked Qdrant points."""
    spec = resolve_model(model)
    provider = VllmEmbeddingProvider(spec, base_url=base_url, batch_size=1)
    vector = provider.embed_queries([question])[0]
    store = QdrantStore(qdrant_url)
    return store.search_points(
        collection_name(spec.slug, spec.dimension),
        vector,
        top_k=top_k,
        exact=exact,
    )


def retrieve_hybrid_chunks(
    question: str,
    *,
    model: str = "bge-m3",
    top_k: int = 5,
    candidate_k: int = 30,
    base_url: str = "http://127.0.0.1:8000/v1",
    qdrant_url: str = "http://127.0.0.1:6333",
    rerank: bool = True,
    reranker_url: str = "http://127.0.0.1:8001",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    exact: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve with dense+BM25 RRF fusion and optional cross-encoder reranking."""
    spec = resolve_model(model)
    provider = VllmEmbeddingProvider(spec, base_url=base_url, batch_size=1)
    vector = provider.embed_queries([question])[0]
    store = QdrantStore(qdrant_url)
    candidates = store.hybrid_search_points(
        hybrid_collection_name(spec.slug, spec.dimension),
        vector,
        question,
        top_k=candidate_k,
        candidate_k=candidate_k,
        exact=exact,
    )
    if not rerank:
        return candidates[:top_k]
    return VllmReranker(model=reranker_model, base_url=reranker_url).rerank(
        question, candidates, top_k=top_k
    )


def retrieve_source_aware_hybrid_chunks(
    question: str,
    *,
    source_constraints: list[SourceConstraint],
    model: str = "bge-m3",
    top_k: int = 8,
    candidate_k: int = 30,
    source_candidate_k: int = 10,
    source_min_results: int = 2,
    base_url: str = "http://127.0.0.1:8000/v1",
    qdrant_url: str = "http://127.0.0.1:6333",
    reranker_url: str = "http://127.0.0.1:8001",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    exact: bool = True,
    trace: Trace | None = None,
) -> list[dict[str, Any]]:
    """Combine global retrieval with constrained lanes for explicitly named sources."""
    spec = resolve_model(model)
    with trace.stage("query_embedding") if trace else nullcontext():
        provider = VllmEmbeddingProvider(spec, base_url=base_url, batch_size=1)
        vector = provider.embed_queries([question])[0]
    store = QdrantStore(qdrant_url)
    collection = hybrid_collection_name(spec.slug, spec.dimension)
    with trace.stage("global_hybrid_retrieval") if trace else nullcontext():
        candidates = store.hybrid_search_points(
            collection,
            vector,
            question,
            top_k=candidate_k,
            candidate_k=candidate_k,
            exact=exact,
        )
    if trace:
        trace.record_chunks("retrieved", candidates, lane="global")

    by_id = {_point_key(point): point for point in candidates}
    for constraint in source_constraints:
        with (
            trace.stage(
                "source_filtered_retrieval",
                metadata={"label": constraint.label, "source_ids": constraint.source_ids},
            )
            if trace
            else nullcontext()
        ):
            constrained = store.hybrid_search_points(
                collection,
                vector,
                question,
                top_k=source_candidate_k,
                candidate_k=source_candidate_k,
                exact=exact,
                source_ids=list(constraint.source_ids),
            )
        if trace:
            trace.record_chunks("retrieved", constrained, lane=constraint.label)
        for point in constrained:
            by_id.setdefault(_point_key(point), point)

    with trace.stage("cross_encoder_reranking") if trace else nullcontext():
        reranked = VllmReranker(model=reranker_model, base_url=reranker_url).rerank(
            question, list(by_id.values()), top_k=len(by_id)
        )
    if trace:
        trace.record_chunks("reranked", reranked, lane="merged")
    if not source_constraints:
        selected = reranked[:top_k]
        if trace:
            trace.record_chunks("selected_seed", selected, lane="final")
        return selected

    with trace.stage("source_quota_selection") if trace else nullcontext():
        quota = max(1, min(source_min_results, top_k // len(source_constraints)))
        required: set[str] = set()
        for constraint in source_constraints:
            matches = [
                point
                for point in reranked
                if (point.get("payload") or {}).get("source_id") in constraint.source_ids
            ]
            required.update(_point_key(point) for point in matches[:quota])

        selected_ids = set(required)
        for point in reranked:
            if len(selected_ids) >= top_k:
                break
            selected_ids.add(_point_key(point))
        selected = [point for point in reranked if _point_key(point) in selected_ids][:top_k]
    if trace:
        trace.record_chunks("selected_seed", selected, lane="final")
    return selected


def _point_key(point: dict[str, Any]) -> str:
    payload = point.get("payload") or {}
    return str(payload.get("chunk_id") or point.get("id"))
