"""Build a named dense + Qdrant-native BM25 hybrid collection."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

corpus_module = import_module("bmo_rag.indexing.corpus")
embedding_module = import_module("bmo_rag.indexing.embeddings")
server_module = import_module("bmo_rag.indexing.local_vllm")
qdrant_module = import_module("bmo_rag.indexing.qdrant_store")


def log(message: str) -> None:
    local_time = datetime.now(UTC).astimezone().strftime("%H:%M:%S")
    print(f"[{local_time}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--chunk-dir", type=Path, default=ROOT / "data/processed/docling")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--no-start-local", action="store_true")
    args = parser.parse_args()

    spec = embedding_module.resolve_model(args.model)
    if not args.no_start_local:
        server_module.ensure_local_embedding_services(
            spec,
            project_root=ROOT,
            base_url=args.base_url,
            gpu_memory_utilization=0.35,
            progress=log,
        )
    chunks = corpus_module.load_corpus(args.chunk_dir)
    store = qdrant_module.QdrantStore(args.qdrant_url)
    name = qdrant_module.hybrid_collection_name(spec.slug, spec.dimension)
    store.ensure_hybrid_collection(name, spec.dimension, recreate=args.reindex)
    existing = store.existing_chunk_ids(name)
    pending = [chunk for chunk in chunks if chunk.chunk_id not in existing]
    log(f"Hybrid collection {name}: {len(existing)} existing, {len(pending)} pending")
    provider = embedding_module.VllmEmbeddingProvider(
        spec, base_url=args.base_url, batch_size=args.batch_size
    )
    completed = len(existing)
    for embedded in provider.embed_documents(pending):
        store.upsert_hybrid(name, embedded)
        completed += len(embedded)
        log(f"Indexed {completed}/{len(chunks)} chunks")
    final_count = store.count(name)
    if final_count != len(chunks):
        raise RuntimeError(f"Qdrant count mismatch: {final_count} != {len(chunks)}")
    log(f"Hybrid index ready: {name} ({final_count} points)")


if __name__ == "__main__":
    main()
