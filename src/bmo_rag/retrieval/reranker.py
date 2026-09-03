"""Cross-encoder reranking through vLLM's Cohere-compatible endpoint."""

from __future__ import annotations

import time
from typing import Any

import requests

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class RerankerError(RuntimeError):
    """Raised when the reranking endpoint cannot score candidates."""


class VllmReranker:
    def __init__(
        self,
        *,
        model: str = DEFAULT_RERANKER_MODEL,
        base_url: str = "http://127.0.0.1:8001",
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def rerank(
        self, question: str, points: list[dict[str, Any]], *, top_k: int
    ) -> list[dict[str, Any]]:
        """Return points ordered by cross-encoder relevance."""
        if not points:
            return []
        documents = []
        for point in points:
            payload = point.get("payload") or {}
            source = payload.get("origin_filename") or payload.get("source_id") or ""
            headings = " > ".join(payload.get("headings") or [])
            text = str(payload.get("text") or "")
            prefix = "\n".join(value for value in [str(source), headings] if value)
            documents.append(f"{prefix}\n\n{text}" if prefix else text)
        body: dict[str, Any] = {
            "model": self.model,
            "query": question,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
            "return_documents": False,
        }
        response: requests.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/rerank", json=body, timeout=self.timeout
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise RerankerError(
                        f"Cannot reach reranker at {self.base_url}: {exc}"
                    ) from exc
                time.sleep(min(2**attempt, 10))
                continue
            if response.status_code < 400:
                break
            if attempt == self.max_retries or response.status_code < 500:
                raise RerankerError(
                    f"Reranker returned {response.status_code}: {response.text[:1000]}"
                )
            time.sleep(min(2**attempt, 10))
        if response is None:
            raise AssertionError("unreachable")

        reranked: list[dict[str, Any]] = []
        for result in response.json().get("results", []):
            source = points[int(result["index"])]
            reranked.append(
                {
                    **source,
                    "fusion_score": source.get("score"),
                    "score": float(result["relevance_score"]),
                    "rerank_score": float(result["relevance_score"]),
                }
            )
        return reranked
