"""Dense retrieval against the per-model Qdrant collections."""

from __future__ import annotations

from typing import Any

from bmo_rag.indexing.embeddings import VllmEmbeddingProvider, resolve_model
from bmo_rag.indexing.qdrant_store import (
    QdrantStore,
    collection_name,
    hybrid_collection_name,
)
from bmo_rag.retrieval.reranker import VllmReranker


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
