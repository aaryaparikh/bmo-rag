from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    source: str | None = None


class VectorStore:
    """Interface for vector store implementations."""

    def add_texts(self, texts: list[str], sources: list[str] | None = None) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError
