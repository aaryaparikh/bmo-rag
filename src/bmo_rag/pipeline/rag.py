from bmo_rag.indexing.vector_store import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(chunk.text for chunk in chunks)
