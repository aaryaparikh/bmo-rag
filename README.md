# BMO RAG

Baseline Python project layout for a retrieval-augmented generation system.

## Structure

```text
.
├── config/                 # Runtime configuration templates
├── data/                   # Local documents, transformed chunks, and vector indexes
├── docs/                   # Architecture notes and project documentation
├── notebooks/              # Exploration notebooks
├── scripts/                # CLI-friendly project utilities
├── src/bmo_rag/            # Application package
└── tests/                  # Unit and integration tests
```

## First Steps

1. Create a virtual environment.
2. Install dependencies with `pip install -e ".[dev]"`.
3. Copy `.env.example` to `.env` and fill in provider keys.
4. Add source documents under `data/raw/`.
5. Run ingestion, indexing, and query scripts as they are implemented.

## Core Pipeline

1. Load documents from configured sources.
2. Split and normalize content into chunks.
3. Embed chunks and store them in a vector index.
4. Retrieve relevant context for a user question.
5. Generate an answer with citations or source references.
