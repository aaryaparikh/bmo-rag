"""Index and compare embedding models on retrieval_golden_200."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bmo_rag.evaluation.retrieval import load_golden, metrics_by_facets
from bmo_rag.indexing.corpus import load_corpus
from bmo_rag.indexing.embeddings import EmbeddingError, create_provider, resolve_model
from bmo_rag.indexing.qdrant_store import QdrantError, QdrantStore, collection_name

DEFAULT_K = (5, 10, 20, 30)


def log(message: str) -> None:
    stamp = datetime.now(UTC).astimezone().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def citation(payload: dict[str, Any]) -> str:
    source = payload.get("source_url") or payload.get("origin_filename") or payload["source_id"]
    pages = payload.get("pages") or []
    if not pages:
        return str(source)
    label = "p." if len(pages) == 1 else "pp."
    return f"{source}, {label} {', '.join(str(page) for page in pages)}"


def expected_citation(chunk: dict[str, Any]) -> str:
    source = chunk.get("source_locator") or chunk.get("source_id") or "unknown source"
    pages = chunk.get("pages") or []
    if not pages:
        return str(source)
    label = "p." if len(pages) == 1 else "pp."
    return f"{source}, {label} {', '.join(str(page) for page in pages)}"


def write_query_details(
    output_dir: Path,
    model_slug: str,
    model_id: str,
    records: list[dict[str, Any]],
    retrieved_by_id: dict[str, list[dict[str, Any]]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_slug}.jsonl"
    rows: list[dict[str, Any]] = []
    for record in records:
        expected_ids = {item["chunk_id"] for item in record["expected_chunks"]}
        returned = []
        for rank, point in enumerate(retrieved_by_id[record["id"]], start=1):
            payload = point["payload"]
            returned.append(
                {
                    "rank": rank,
                    "score": round(float(point["score"]), 8),
                    "is_expected": payload["chunk_id"] in expected_ids,
                    **payload,
                    "citation": citation(payload),
                }
            )
        expected = [
            {**item, "citation": expected_citation(item)} for item in record["expected_chunks"]
        ]
        first_relevant = next(
            (item["rank"] for item in returned if item["is_expected"]), None
        )
        rows.append(
            {
                "model": model_slug,
                "model_id": model_id,
                "question_id": record["id"],
                "question": record["question"],
                "query_type": record.get("query_type"),
                "difficulty": record.get("difficulty"),
                "edge_case": record.get("edge_case"),
                "split": record.get("split"),
                "expected_behavior": record.get("expected_behavior"),
                "expected_answer": record.get("expected_answer"),
                "expected_chunks": expected,
                "returned_chunks": returned,
                "first_relevant_rank": first_relevant,
                "hit_at_5": first_relevant is not None and first_relevant <= 5,
                "hit_at_10": first_relevant is not None and first_relevant <= 10,
                "hit_at_20": first_relevant is not None and first_relevant <= 20,
                "hit_at_30": first_relevant is not None and first_relevant <= 30,
            }
        )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        help="One local model slug or Hugging Face model ID. Use the orchestrator for all models.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--upsert-batch-size", type=int, default=128)
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K))
    parser.add_argument("--chunk-dir", type=Path, default=ROOT / "data/processed/docling")
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/golden/retrieval_golden_200.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/benchmarks/embedding_model_comparison/summary.json",
    )
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--collection-prefix", default="bmo_chunks")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete and recreate each selected model collection before embedding.",
    )
    parser.add_argument(
        "--skip-index", action="store_true", help="Evaluate already complete collections only."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate corpus/golden alignment without contacting providers or Qdrant.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible embeddings URL (for one vLLM-hosted open-weight model).",
    )
    parser.add_argument("--api-key", help="Optional API key if the local vLLM server requires one.")
    parser.add_argument(
        "--append-report",
        action="store_true",
        help="Preserve compatible model results already present in the output report.",
    )
    parser.add_argument(
        "--details-output-dir",
        type=Path,
        default=ROOT / "outputs/benchmarks/embedding_model_comparison/query_details",
        help="Write one query-level JSONL audit dataset per evaluated model.",
    )
    args = parser.parse_args()

    if not args.validate_only and not args.models:
        parser.error("--models is required unless --validate-only is used")
    if args.models and len(args.models) != 1:
        parser.error(
            "One vLLM server hosts one model at a time. Pass one model, or run "
            "scripts/evaluation/run_local_embedding_benchmark.py to orchestrate all four."
        )
    if any(k <= 0 for k in args.k):
        parser.error("All k values must be positive")

    chunks = load_corpus(args.chunk_dir)
    records = load_golden(args.dataset)
    corpus_ids = {chunk.chunk_id for chunk in chunks}
    if len(corpus_ids) != len(chunks):
        raise RuntimeError("The corpus contains duplicate chunk IDs")
    gold_ids = {
        item["chunk_id"] for record in records for item in record["expected_chunks"]
    }
    missing_gold = sorted(gold_ids - corpus_ids)
    if missing_gold:
        raise RuntimeError(
            f"Golden dataset references {len(missing_gold)} missing chunk IDs; rebuild it "
            f"against the completed corpus. First missing ID: {missing_gold[0]}"
        )
    fingerprint = hashlib.sha256(
        "\n".join(chunk.chunk_id for chunk in chunks).encode()
    ).hexdigest()
    manifest_path = args.dataset.with_name(f"{args.dataset.stem}.manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("corpus_chunk_count") != len(chunks):
            raise RuntimeError(
                f"Golden manifest corpus count is {manifest.get('corpus_chunk_count')}, "
                f"but the current corpus has {len(chunks)} chunks"
            )
        if manifest.get("corpus_fingerprint_sha256") != fingerprint:
            raise RuntimeError("Golden manifest fingerprint does not match the current corpus")
    answerable = [record for record in records if record["expected_chunks"]]
    store = QdrantStore(args.qdrant_url, api_key=args.qdrant_api_key)
    new_report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset),
        "corpus_chunk_count": len(chunks),
        "corpus_fingerprint_sha256": fingerprint,
        "record_count": len(records),
        "distance": "cosine",
        "qdrant_search": "exact",
        "k_values": args.k,
        "empty_gold_policy": (
            "Excluded from precision, recall and MRR because fixed top-k retrieval has no "
            "abstention threshold."
        ),
        "models": {},
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "corpus_chunk_count": len(chunks),
                    "corpus_fingerprint_sha256": fingerprint,
                    "record_count": len(records),
                    "answerable_count": len(answerable),
                    "empty_gold_count": len(records) - len(answerable),
                },
                indent=2,
            )
        )
        return

    report = new_report
    if args.append_report and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        compatible = (
            existing.get("corpus_fingerprint_sha256") == fingerprint
            and existing.get("k_values") == args.k
            and existing.get("distance") == "cosine"
        )
        if not compatible:
            raise RuntimeError(
                f"Cannot append to incompatible benchmark report {args.output}; move it or "
                "rerun without --append-report"
            )
        report = existing
        report["updated_at"] = datetime.now(UTC).isoformat()

    for model in args.models:
        spec = resolve_model(model)
        provider = create_provider(
            model,
            batch_size=args.batch_size,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        name = collection_name(spec.slug, spec.dimension, args.collection_prefix)
        log(f"Preparing {spec.model_id} in Qdrant collection {name}")
        store.ensure_collection(name, spec.dimension, recreate=args.reindex)
        existing_count = store.count(name)
        if existing_count > len(chunks):
            raise RuntimeError(
                f"Collection {name} has {existing_count} points for a {len(chunks)}-chunk "
                "corpus; use --reindex to rebuild it."
            )
        if args.skip_index and existing_count != len(chunks):
            raise RuntimeError(
                f"Cannot --skip-index: {name} has {existing_count}/{len(chunks)} points"
            )

        index_started = time.perf_counter()
        if existing_count < len(chunks) and not args.skip_index:
            existing_ids = store.existing_chunk_ids(name) if existing_count else set()
            pending_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
            log(
                f"{spec.slug}: resuming with {len(pending_chunks)} chunks pending "
                f"({len(existing_ids)} already indexed)"
            )
            indexed = len(existing_ids)
            for embedded in provider.embed_documents(pending_chunks):
                store.upsert_in_batches(name, embedded, batch_size=args.upsert_batch_size)
                indexed += len(embedded)
                log(f"{spec.slug}: indexed {indexed}/{len(chunks)} chunks")
            final_count = store.count(name)
            if final_count != len(chunks):
                raise RuntimeError(
                    f"Qdrant count mismatch for {name}: {final_count} != {len(chunks)}"
                )
        else:
            log(f"{spec.slug}: reusing complete collection with {existing_count} points")
        index_seconds = time.perf_counter() - index_started

        query_started = time.perf_counter()
        queries = [record["question"] for record in records]
        query_vectors = provider.embed_queries(queries)
        max_k = max(args.k)
        rankings: dict[str, list[str]] = {}
        retrieved_by_id: dict[str, list[dict[str, Any]]] = {}
        for index, (record, vector) in enumerate(zip(records, query_vectors, strict=True), 1):
            points = store.search_points(name, vector, top_k=max_k, exact=True)
            retrieved_by_id[record["id"]] = points
            rankings[record["id"]] = [point["payload"]["chunk_id"] for point in points]
            if index % 25 == 0 or index == len(records):
                log(f"{spec.slug}: evaluated {index}/{len(records)} queries")
        evaluation_seconds = time.perf_counter() - query_started
        details_path = None
        if args.details_output_dir:
            details_path = write_query_details(
                args.details_output_dir,
                spec.slug,
                spec.model_id,
                records,
                retrieved_by_id,
            )
        model_report = {
            "model_id": spec.model_id,
            "dimension": spec.dimension,
            "runtime": "vllm",
            "weight_precision": "bitsandbytes-4bit" if spec.quantize_4bit else "float16",
            "max_model_len": spec.max_model_len,
            "collection": name,
            "index_seconds_this_run": round(index_seconds, 3),
            "evaluation_seconds": round(evaluation_seconds, 3),
            "metrics": metrics_by_facets(records, rankings, args.k),
        }
        if details_path:
            model_report["query_details"] = str(details_path)
        report["models"][spec.slug] = model_report
        write_report(args.output, report)
        log(f"{spec.slug}: complete; checkpointed report to {args.output}")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except QdrantError as exc:
        raise SystemExit(
            "Qdrant is not available. Start Docker Desktop, run "
            "`docker compose up -d qdrant`, and retry.\n"
            f"Details: {exc}"
        ) from None
    except EmbeddingError as exc:
        raise SystemExit(
            "The local vLLM embedding server is unavailable or returned an invalid response. "
            "Start the requested model with the local benchmark orchestrator and retry.\n"
            f"Details: {exc}"
        ) from None
