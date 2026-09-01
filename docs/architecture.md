# Architecture

This project is organized around a simple RAG pipeline:

1. **Ingestion** reads documents from local files or external systems.
2. **Processing** cleans text and splits it into retrieval-sized chunks.
3. **Indexing** creates embeddings and persists searchable vectors.
4. **Retrieval** selects relevant chunks for a question.
5. **Generation** produces an answer grounded in retrieved context.

The current baseline keeps provider-specific choices behind module boundaries so the project can start locally and move to managed services later.
