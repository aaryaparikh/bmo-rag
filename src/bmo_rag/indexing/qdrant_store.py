"""Small Qdrant REST adapter used by the embedding benchmark."""

from __future__ import annotations

import re
import uuid
from typing import Any
from urllib.parse import quote

import requests

from bmo_rag.indexing.embeddings import EmbeddedBatch, Vector


class QdrantError(RuntimeError):
    pass


def collection_name(model: str, dimension: int, prefix: str = "bmo_chunks") -> str:
    safe_model = re.sub(r"[^a-zA-Z0-9_-]+", "_", model).strip("_").lower()
    return f"{prefix}_{safe_model}_d{dimension}"


def hybrid_collection_name(model: str, dimension: int, prefix: str = "bmo_chunks") -> str:
    """Return a separate collection name so dense benchmark indexes remain intact."""
    return collection_name(model, dimension, prefix=f"{prefix}_hybrid")


class QdrantStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["api-key"] = api_key

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        allowed: set[int] | None = None,
    ) -> requests.Response:
        try:
            response = self.session.request(
                method,
                f"{self.url}{path}",
                headers=self.headers,
                json=body,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise QdrantError(f"Cannot reach Qdrant at {self.url}: {exc}") from exc
        accepted = allowed or {200}
        if response.status_code not in accepted:
            raise QdrantError(
                f"Qdrant {method} {path} returned {response.status_code}: {response.text[:500]}"
            )
        return response

    def ensure_collection(self, name: str, dimension: int, *, recreate: bool = False) -> None:
        encoded = quote(name, safe="")
        response = self._request("GET", f"/collections/{encoded}", allowed={200, 404})
        if response.status_code == 200 and recreate:
            self._request("DELETE", f"/collections/{encoded}")
            response = self._request("GET", f"/collections/{encoded}", allowed={200, 404})
        if response.status_code == 404:
            self._request(
                "PUT",
                f"/collections/{encoded}",
                body={"vectors": {"size": dimension, "distance": "Cosine"}},
            )
            return
        vectors = response.json()["result"]["config"]["params"]["vectors"]
        actual = vectors.get("size") if isinstance(vectors, dict) else None
        if actual != dimension:
            raise QdrantError(
                f"Collection {name} has dimension {actual}, expected {dimension}; use --reindex"
            )

    def ensure_hybrid_collection(
        self, name: str, dimension: int, *, recreate: bool = False
    ) -> None:
        """Create a named dense + native BM25 sparse collection."""
        encoded = quote(name, safe="")
        response = self._request("GET", f"/collections/{encoded}", allowed={200, 404})
        if response.status_code == 200 and recreate:
            self._request("DELETE", f"/collections/{encoded}")
            response = self._request("GET", f"/collections/{encoded}", allowed={200, 404})
        if response.status_code == 404:
            self._request(
                "PUT",
                f"/collections/{encoded}",
                body={
                    "vectors": {"dense": {"size": dimension, "distance": "Cosine"}},
                    "sparse_vectors": {"bm25": {"modifier": "idf"}},
                },
            )
            return
        config = response.json()["result"]["config"]["params"]
        dense = config.get("vectors", {}).get("dense", {})
        sparse = config.get("sparse_vectors", {}).get("bm25", {})
        if dense.get("size") != dimension or not sparse:
            raise QdrantError(
                f"Collection {name} is not a {dimension}-dimensional dense+BM25 collection; "
                "rebuild it with --reindex"
            )

    def count(self, name: str) -> int:
        response = self._request(
            "POST",
            f"/collections/{quote(name, safe='')}/points/count",
            body={"exact": True},
        )
        return int(response.json()["result"]["count"])

    def existing_chunk_ids(self, name: str, *, page_size: int = 256) -> set[str]:
        """Return indexed corpus IDs so interrupted local runs can resume safely."""
        found: set[str] = set()
        offset: str | int | None = None
        while True:
            body: dict[str, Any] = {
                "limit": page_size,
                "with_payload": ["chunk_id"],
                "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            response = self._request(
                "POST", f"/collections/{quote(name, safe='')}/points/scroll", body=body
            )
            result = response.json()["result"]
            found.update(
                point["payload"]["chunk_id"]
                for point in result["points"]
                if point.get("payload", {}).get("chunk_id")
            )
            offset = result.get("next_page_offset")
            if offset is None:
                return found

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"bmo-rag:{chunk_id}"))

    def upsert(self, name: str, embedded: EmbeddedBatch) -> None:
        points = [
            {
                "id": self.point_id(chunk.chunk_id),
                "vector": vector,
                "payload": chunk.payload(),
            }
            for chunk, vector in embedded
        ]
        self._request(
            "PUT",
            f"/collections/{quote(name, safe='')}/points?wait=true",
            body={"points": points},
        )

    def upsert_hybrid(self, name: str, embedded: EmbeddedBatch) -> None:
        """Store dense vectors and let Qdrant generate native BM25 sparse vectors."""
        points = [
            {
                "id": self.point_id(chunk.chunk_id),
                "vector": {
                    "dense": vector,
                    "bm25": {"text": chunk.embedding_text, "model": "qdrant/bm25"},
                },
                "payload": chunk.payload(),
            }
            for chunk, vector in embedded
        ]
        self._request(
            "PUT",
            f"/collections/{quote(name, safe='')}/points?wait=true",
            body={"points": points},
        )

    def upsert_in_batches(
        self, name: str, embedded: EmbeddedBatch, *, batch_size: int = 128
    ) -> None:
        for offset in range(0, len(embedded), batch_size):
            self.upsert(name, embedded[offset : offset + batch_size])

    def search(self, name: str, vector: Vector, *, top_k: int, exact: bool = True) -> list[str]:
        return [
            point["payload"]["chunk_id"]
            for point in self.search_points(name, vector, top_k=top_k, exact=exact)
        ]

    def search_points(
        self, name: str, vector: Vector, *, top_k: int, exact: bool = True
    ) -> list[dict[str, Any]]:
        """Return scored points and full payloads for retrieval audits."""
        body: dict[str, Any] = {
            "query": vector,
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
        }
        if exact:
            body["params"] = {"exact": True}
        response = self._request(
            "POST", f"/collections/{quote(name, safe='')}/points/query", body=body
        )
        return response.json()["result"]["points"]

    def hybrid_search_points(
        self,
        name: str,
        dense_vector: Vector,
        query_text: str,
        *,
        top_k: int,
        candidate_k: int = 30,
        exact: bool = True,
    ) -> list[dict[str, Any]]:
        """Fuse dense and native BM25 candidates with Qdrant RRF."""
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        dense_prefetch: dict[str, Any] = {
            "query": dense_vector,
            "using": "dense",
            "limit": candidate_k,
        }
        if exact:
            dense_prefetch["params"] = {"exact": True}
        body = {
            "prefetch": [
                dense_prefetch,
                {
                    "query": {"text": query_text, "model": "qdrant/bm25"},
                    "using": "bm25",
                    "limit": candidate_k,
                },
            ],
            "query": {"rrf": {}},
            "limit": top_k,
            "with_payload": True,
            "with_vector": False,
        }
        response = self._request(
            "POST", f"/collections/{quote(name, safe='')}/points/query", body=body
        )
        return response.json()["result"]["points"]
