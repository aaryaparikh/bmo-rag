from bmo_rag.indexing.vector_store import RetrievedChunk, VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, top_k: int = 5) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return self.vector_store.search(question, top_k=self.top_k)
