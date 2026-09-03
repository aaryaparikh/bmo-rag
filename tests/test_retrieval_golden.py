import hashlib
import json
from pathlib import Path

from bmo_rag.evaluation.retrieval import (
    validate_golden_alignment,
    validate_golden_dataset_hash,
)
from bmo_rag.indexing.corpus import load_corpus

DATASET = Path("data/golden/retrieval_golden_200.jsonl")
MANIFEST = Path("data/golden/retrieval_golden_200.manifest.json")


def test_retrieval_gold_has_200_traceable_unique_records() -> None:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert len(rows) == manifest["record_count"] == 200
    assert len({row["id"] for row in rows}) == 200
    assert len({row["question"].casefold() for row in rows}) == 200
    assert {row["split"] for row in rows} == {"development", "test"}
    assert sum(row["split"] == "test" for row in rows) == 40

    for row in rows:
        assert row["question"]
        assert len(row["hard_negative_chunk_ids"]) <= 3
        for gold in row["expected_chunks"]:
            assert gold["relevance"] == 3
            assert gold["chunk_id"].startswith("bmo-")
            assert gold["source_locator"]
            assert gold["evidence"]
            assert len(gold["text_sha256"]) == 64
            for equivalent in gold.get("equivalent_chunks", []):
                assert equivalent["chunk_id"].startswith("bmo-")
                assert equivalent["chunk_id"] != gold["chunk_id"]
                assert equivalent["match_method"] in {
                    "normalized_exact",
                    "full_passage_containment",
                    "near_duplicate_4gram",
                }


def test_manifest_matches_dataset_bytes_and_source_coverage() -> None:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for row in rows:
        for gold in row["expected_chunks"]:
            source = gold["source_id"]
            counts[source] = counts.get(source, 0) + 1

    assert counts == manifest["source_distribution"]
    corpus = load_corpus(Path("data/processed/docling"))
    fingerprint = validate_golden_alignment(
        rows, [chunk.chunk_id for chunk in corpus], manifest
    )
    dataset_hash = validate_golden_dataset_hash(DATASET, manifest)

    assert manifest["corpus_source_count"] == len({chunk.source_id for chunk in corpus}) == 26
    assert manifest["corpus_chunk_count"] == len(corpus)
    assert fingerprint == manifest["corpus_fingerprint_sha256"]
    assert dataset_hash == manifest["dataset_sha256"]
    assert hashlib.sha256(Path("data/golden/retrieval_golden_200.csv").read_bytes()).hexdigest() == (
        manifest["csv_sha256"]
    )
    assert len(fingerprint) == hashlib.sha256().digest_size * 2
