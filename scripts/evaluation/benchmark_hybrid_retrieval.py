"""Benchmark BGE dense baseline versus Qdrant hybrid RRF and cross-encoder reranking."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

evaluation_module = import_module("bmo_rag.evaluation.retrieval")
corpus_module = import_module("bmo_rag.indexing.corpus")
embedding_module = import_module("bmo_rag.indexing.embeddings")
server_module = import_module("bmo_rag.indexing.local_vllm")
qdrant_module = import_module("bmo_rag.indexing.qdrant_store")
reranker_module = import_module("bmo_rag.retrieval.reranker")
semantic_module = import_module("bmo_rag.retrieval.semantic")

K_VALUES = (5, 10, 20, 30)


def log(message: str) -> None:
    stamp = datetime.now(UTC).astimezone().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


def point_row(point: dict[str, Any], rank: int) -> dict[str, Any]:
    payload = point.get("payload") or {}
    return {
        "rank": rank,
        "score": round(float(point.get("score", 0.0)), 8),
        "fusion_score": (
            round(float(point["fusion_score"]), 8)
            if point.get("fusion_score") is not None
            else None
        ),
        **payload,
        "citation": semantic_module.citation(payload),
    }


def expected_row(chunk: dict[str, Any]) -> dict[str, Any]:
    source = chunk.get("source_locator") or chunk.get("source_id") or "unknown source"
    pages = chunk.get("pages") or []
    if pages:
        label = "p." if len(pages) == 1 else "pp."
        citation = f"{source}, {label} {', '.join(str(page) for page in pages)}"
    else:
        citation = str(source)
    return {**chunk, "citation": citation}


def load_dense_baseline(path: Path, *, corpus_fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("corpus_fingerprint_sha256") != corpus_fingerprint:
        log(f"Skipping stale dense baseline with a different corpus fingerprint: {path}")
        return None
    model = report.get("models", {}).get("bge-m3")
    if not model:
        return None
    return {
        "method": "dense",
        "model_id": model.get("model_id", "BAAI/bge-m3"),
        "collection": model.get("collection"),
        "metrics": model["metrics"],
        "source_report": str(path),
    }


def write_facet_metrics(path: Path, methods: dict[str, Any]) -> None:
    """Write a flat review table from the exact metrics stored in the JSON report."""
    fields = [
        "Facet",
        "Group",
        "Method",
        "Answerable",
        "ExcludedEmptyGold",
        "K",
        "Precision",
        "EvidenceGroupRecall",
        "MRR",
        "HitRate",
        "ExactChunkRecall",
        "RelevantResultRedundancy",
    ]
    rows: list[dict[str, Any]] = []
    for method_name, method in methods.items():
        metrics = method["metrics"]
        metric_groups = [
            ("Split", name, value)
            for name, value in metrics.items()
            if name in {"all", "development", "test"}
        ]
        metric_groups.extend(
            (facet.removeprefix("by_"), group, value)
            for facet, groups in metrics.items()
            if facet.startswith("by_")
            for group, value in groups.items()
        )
        for facet, group, value in metric_groups:
            for k, cutoff in value.get("cutoffs", {}).items():
                rows.append(
                    {
                        "Facet": facet,
                        "Group": group,
                        "Method": method_name,
                        "Answerable": value["evaluated_answerable_count"],
                        "ExcludedEmptyGold": value["excluded_empty_gold_count"],
                        "K": k,
                        "Precision": cutoff["precision"],
                        "EvidenceGroupRecall": cutoff["recall"],
                        "MRR": cutoff["mrr"],
                        "HitRate": cutoff["hit_rate"],
                        "ExactChunkRecall": cutoff["exact_chunk_recall"],
                        "RelevantResultRedundancy": cutoff["relevant_result_redundancy"],
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/golden/retrieval_golden_200.jsonl"
    )
    parser.add_argument("--chunk-dir", type=Path, default=ROOT / "data/processed/docling")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--reranker-url", default="http://127.0.0.1:8001")
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/benchmarks/hybrid_retrieval_comparison/summary.json",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=ROOT / "outputs/benchmarks/hybrid_retrieval_comparison/query_details.jsonl",
    )
    parser.add_argument(
        "--facet-output",
        type=Path,
        default=ROOT / "outputs/benchmarks/hybrid_retrieval_comparison/facet_metrics.csv",
    )
    parser.add_argument(
        "--dense-baseline-report",
        type=Path,
        default=ROOT / "outputs/benchmarks/embedding_model_comparison/summary.json",
    )
    parser.add_argument("--no-start-local", action="store_true")
    args = parser.parse_args()
    if args.candidate_k < max(K_VALUES):
        parser.error(f"--candidate-k must be at least {max(K_VALUES)}")

    spec = embedding_module.resolve_model("bge-m3")
    if not args.no_start_local:
        server_module.ensure_local_embedding_services(
            spec,
            project_root=ROOT,
            base_url=args.base_url,
            gpu_memory_utilization=0.35,
            progress=log,
        )
        server_module.ensure_local_reranker(
            model=reranker_module.DEFAULT_RERANKER_MODEL,
            project_root=ROOT,
            base_url=args.reranker_url,
            progress=log,
        )

    records = evaluation_module.load_golden(args.dataset)
    chunks = corpus_module.load_corpus(args.chunk_dir)
    manifest_path = args.dataset.with_name(f"{args.dataset.stem}.manifest.json")
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    dataset_hash = evaluation_module.validate_golden_dataset_hash(args.dataset, manifest)
    fingerprint = evaluation_module.validate_golden_alignment(
        records, [chunk.chunk_id for chunk in chunks], manifest
    )
    provider = embedding_module.VllmEmbeddingProvider(spec, base_url=args.base_url, batch_size=32)
    store = qdrant_module.QdrantStore(args.qdrant_url)
    collection = qdrant_module.hybrid_collection_name(spec.slug, spec.dimension)
    if store.count(collection) == 0:
        raise RuntimeError(f"Hybrid collection {collection} is empty")
    reranker = reranker_module.VllmReranker(
        model=reranker_module.DEFAULT_RERANKER_MODEL,
        base_url=args.reranker_url,
    )

    started = time.perf_counter()
    vectors = provider.embed_queries([record["question"] for record in records])
    embedding_seconds = time.perf_counter() - started
    rrf_points: dict[str, list[dict[str, Any]]] = {}
    reranked_points: dict[str, list[dict[str, Any]]] = {}
    retrieval_seconds = 0.0
    rerank_seconds = 0.0

    for position, (record, vector) in enumerate(zip(records, vectors, strict=True), start=1):
        query_started = time.perf_counter()
        candidates = store.hybrid_search_points(
            collection,
            vector,
            record["question"],
            top_k=args.candidate_k,
            candidate_k=args.candidate_k,
            exact=True,
        )
        retrieval_seconds += time.perf_counter() - query_started
        rrf_points[record["id"]] = candidates

        rerank_started = time.perf_counter()
        reranked_points[record["id"]] = reranker.rerank(
            record["question"], candidates, top_k=args.candidate_k
        )
        rerank_seconds += time.perf_counter() - rerank_started
        if position % 10 == 0 or position == len(records):
            log(f"Evaluated {position}/{len(records)} queries")

    rrf_rankings = {
        record_id: [point["payload"]["chunk_id"] for point in points]
        for record_id, points in rrf_points.items()
    }
    reranked_rankings = {
        record_id: [point["payload"]["chunk_id"] for point in points]
        for record_id, points in reranked_points.items()
    }
    diversified_points = {
        record_id: semantic_module.deduplicate_retrieved_points(points)
        for record_id, points in reranked_points.items()
    }
    diversified_rankings = {
        record_id: [point["payload"]["chunk_id"] for point in points]
        for record_id, points in diversified_points.items()
    }
    methods: dict[str, Any] = {}
    dense = load_dense_baseline(
        args.dense_baseline_report, corpus_fingerprint=fingerprint
    )
    if dense:
        methods["bge-m3-dense"] = dense
    methods["bge-m3-hybrid-rrf"] = {
        "method": "qdrant_dense_bm25_rrf",
        "model_id": spec.model_id,
        "collection": collection,
        "candidate_k": args.candidate_k,
        "metrics": evaluation_module.metrics_by_facets(records, rrf_rankings, K_VALUES),
    }
    methods["bge-m3-hybrid-rrf-reranked"] = {
        "method": "qdrant_dense_bm25_rrf_then_cross_encoder",
        "model_id": spec.model_id,
        "reranker_model": reranker_module.DEFAULT_RERANKER_MODEL,
        "collection": collection,
        "candidate_k": args.candidate_k,
        "metrics": evaluation_module.metrics_by_facets(records, reranked_rankings, K_VALUES),
    }
    methods["bge-m3-hybrid-rrf-reranked-diversified"] = {
        "method": (
            "qdrant_dense_bm25_rrf_then_cross_encoder_then_exact_or_contained_"
            "passage_deduplication"
        ),
        "model_id": spec.model_id,
        "reranker_model": reranker_module.DEFAULT_RERANKER_MODEL,
        "collection": collection,
        "candidate_k": args.candidate_k,
        "metrics": evaluation_module.metrics_by_facets(records, diversified_rankings, K_VALUES),
    }

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_hash,
        "record_count": len(records),
        "answerable_count": sum(bool(record["expected_chunks"]) for record in records),
        "corpus_chunk_count": len(chunks),
        "corpus_fingerprint_sha256": fingerprint,
        "k_values": list(K_VALUES),
        "empty_gold_policy": (
            "Excluded from precision, recall and MRR because fixed top-k retrieval has no "
            "abstention threshold."
        ),
        "timings_seconds": {
            "dense_query_embedding": round(embedding_seconds, 3),
            "qdrant_hybrid_rrf": round(retrieval_seconds, 3),
            "cross_encoder_reranking": round(rerank_seconds, 3),
        },
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_facet_metrics(args.facet_output, methods)

    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    with args.details_output.open("w", encoding="utf-8") as handle:
        for record in records:
            expected_ids = evaluation_module.relevant_chunk_ids(record)
            canonical_ids = {chunk["chunk_id"] for chunk in record["expected_chunks"]}
            rrf_rows = [
                {
                    **point_row(point, rank),
                    "is_expected": point["payload"]["chunk_id"] in expected_ids,
                    "is_canonical_expected": point["payload"]["chunk_id"] in canonical_ids,
                    "matched_evidence_groups": evaluation_module.matched_evidence_group_indexes(
                        record, point["payload"]["chunk_id"]
                    ),
                }
                for rank, point in enumerate(rrf_points[record["id"]], start=1)
            ]
            reranked_rows = [
                {
                    **point_row(point, rank),
                    "is_expected": point["payload"]["chunk_id"] in expected_ids,
                    "is_canonical_expected": point["payload"]["chunk_id"] in canonical_ids,
                    "matched_evidence_groups": evaluation_module.matched_evidence_group_indexes(
                        record, point["payload"]["chunk_id"]
                    ),
                }
                for rank, point in enumerate(reranked_points[record["id"]], start=1)
            ]
            diversified_rows = [
                {
                    **point_row(point, rank),
                    "is_expected": point["payload"]["chunk_id"] in expected_ids,
                    "is_canonical_expected": point["payload"]["chunk_id"] in canonical_ids,
                    "matched_evidence_groups": evaluation_module.matched_evidence_group_indexes(
                        record, point["payload"]["chunk_id"]
                    ),
                    "duplicate_chunk_ids": point.get("duplicate_chunk_ids", []),
                    "duplicate_sources": point.get("duplicate_sources", []),
                }
                for rank, point in enumerate(diversified_points[record["id"]], start=1)
            ]
            handle.write(
                json.dumps(
                    {
                        "question_id": record["id"],
                        "question": record["question"],
                        "query_type": record.get("query_type"),
                        "difficulty": record.get("difficulty"),
                        "edge_case": record.get("edge_case"),
                        "split": record.get("split"),
                        "expected_behavior": record.get("expected_behavior"),
                        "expected_answer": record.get("expected_answer"),
                        "expected_chunks": [expected_row(chunk) for chunk in record["expected_chunks"]],
                        "hybrid_rrf_chunks": rrf_rows,
                        "reranked_chunks": reranked_rows,
                        "diversified_reranked_chunks": diversified_rows,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    log(f"Report: {args.output}")
    log(f"Query details: {args.details_output}")
    log(f"Facet metrics: {args.facet_output}")


if __name__ == "__main__":
    main()
