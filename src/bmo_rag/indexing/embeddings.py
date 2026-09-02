"""Open-source embedding models served by a local vLLM endpoint."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import requests

from bmo_rag.indexing.corpus import CorpusChunk

Vector = list[float]
EmbeddedBatch = list[tuple[CorpusChunk, Vector]]

QWEN_RETRIEVAL_INSTRUCTION = (
    "Given a question about BMO financial and corporate documents, retrieve relevant "
    "passages that answer the question"
)


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model_id: str
    dimension: int
    query_prefix: str = ""
    document_prefix: str = ""
    max_model_len: int = 2048
    recommended_batch_size: int = 32
    quantize_4bit: bool = False
    trust_remote_code: bool = False
    hf_overrides: str | None = None
    pooler_config: str | None = None

    def prepare(self, text: str, *, input_type: str) -> str:
        prefix = self.query_prefix if input_type == "query" else self.document_prefix
        return f"{prefix}{text}"


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen3-embedding-8b": ModelSpec(
        slug="qwen3-embedding-8b",
        model_id="Qwen/Qwen3-Embedding-8B",
        dimension=4096,
        query_prefix=f"Instruct: {QWEN_RETRIEVAL_INSTRUCTION}\nQuery: ",
        recommended_batch_size=8,
        quantize_4bit=True,
    ),
    "qwen3-embedding-4b": ModelSpec(
        slug="qwen3-embedding-4b",
        model_id="Qwen/Qwen3-Embedding-4B",
        dimension=2560,
        query_prefix=f"Instruct: {QWEN_RETRIEVAL_INSTRUCTION}\nQuery: ",
        recommended_batch_size=16,
        quantize_4bit=True,
    ),
    "bge-m3": ModelSpec(
        slug="bge-m3",
        model_id="BAAI/bge-m3",
        dimension=1024,
        hf_overrides='{"architectures":["BgeM3EmbeddingModel"]}',
        pooler_config='{"task":"embed"}',
    ),
    "nomic-embed-v1.5": ModelSpec(
        slug="nomic-embed-v1.5",
        model_id="nomic-ai/nomic-embed-text-v1.5",
        dimension=768,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        trust_remote_code=True,
    ),
}

MODEL_ALIASES = {
    **{key: key for key in MODEL_SPECS},
    **{spec.model_id.casefold(): key for key, spec in MODEL_SPECS.items()},
    "nomic-embed-text-v1.5": "nomic-embed-v1.5",
}


class EmbeddingError(RuntimeError):
    pass


def resolve_model(value: str) -> ModelSpec:
    try:
        return MODEL_SPECS[MODEL_ALIASES[value.casefold()]]
    except KeyError as exc:
        choices = ", ".join(MODEL_SPECS)
        raise EmbeddingError(f"Unknown model {value!r}. Choose one of: {choices}") from exc


class EmbeddingProvider(ABC):
    def __init__(self, spec: ModelSpec, batch_size: int = 32) -> None:
        self.spec = spec
        self.model = spec.model_id
        self.dimension = spec.dimension
        self.batch_size = batch_size

    @abstractmethod
    def embed(self, texts: Sequence[str], *, input_type: str) -> list[Vector]:
        """Embed query or document texts in input order."""

    def embed_documents(self, chunks: Sequence[CorpusChunk]) -> Iterator[EmbeddedBatch]:
        for offset in range(0, len(chunks), self.batch_size):
            batch = list(chunks[offset : offset + self.batch_size])
            vectors = self.embed([chunk.embedding_text for chunk in batch], input_type="document")
            yield list(zip(batch, vectors, strict=True))

    def embed_queries(self, queries: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for offset in range(0, len(queries), self.batch_size):
            vectors.extend(self.embed(queries[offset : offset + self.batch_size], input_type="query"))
        return vectors

    def _validate(self, vectors: list[Vector], expected_count: int) -> list[Vector]:
        if len(vectors) != expected_count:
            raise EmbeddingError(
                f"{self.model} returned {len(vectors)} vectors for {expected_count} inputs"
            )
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingError(
                    f"{self.model} returned dimension {len(vector)}; expected {self.dimension}"
                )
        return vectors


class VllmEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        spec: ModelSpec,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "local-vllm",
        batch_size: int = 32,
        timeout: float = 300.0,
        max_retries: int = 5,
    ) -> None:
        super().__init__(spec, batch_size)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise EmbeddingError(f"Cannot reach vLLM at {self.base_url}: {exc}") from exc
                time.sleep(min(2**attempt, 20))
                continue
            if response.status_code < 400:
                return response.json()
            if response.status_code not in {408, 409, 429} and response.status_code < 500:
                raise EmbeddingError(
                    f"vLLM returned {response.status_code}: {response.text[:1000]}"
                )
            if attempt == self.max_retries:
                raise EmbeddingError(
                    f"vLLM failed after retries ({response.status_code}): {response.text[:1000]}"
                )
            time.sleep(min(2**attempt, 20))
        raise AssertionError("unreachable")

    def embed(self, texts: Sequence[str], *, input_type: str) -> list[Vector]:
        prepared = [self.spec.prepare(text, input_type=input_type) for text in texts]
        response = self._post(
            {
                "input": prepared,
                "model": self.spec.model_id,
                "encoding_format": "float",
            }
        )
        ordered = sorted(response["data"], key=lambda item: item["index"])
        return self._validate([item["embedding"] for item in ordered], len(texts))


def create_provider(
    model: str,
    *,
    batch_size: int = 32,
    api_key: str | None = None,
    base_url: str = "http://127.0.0.1:8000/v1",
) -> EmbeddingProvider:
    return VllmEmbeddingProvider(
        resolve_model(model),
        batch_size=batch_size,
        api_key=api_key or "local-vllm",
        base_url=base_url,
    )
