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
                    "manually_verified_answer_equivalent",
                }


def test_manifest_matches_dataset_bytes_and_source_coverage() -> None:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    acceptable_counts: dict[str, int] = {}
    for row in rows:
        for gold in row["expected_chunks"]:
            source = gold["source_id"]
            counts[source] = counts.get(source, 0) + 1
            for item in [gold, *gold.get("equivalent_chunks", [])]:
                source = item["source_id"]
                acceptable_counts[source] = acceptable_counts.get(source, 0) + 1

    assert counts == manifest["source_distribution"]
    assert acceptable_counts == manifest["acceptable_source_distribution"]
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


def test_important_questions_use_verified_evidence_groups() -> None:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    important = {row["id"]: row for row in rows if row["id"] >= "bmo-retrieval-181"}

    expected_group_sizes = {
        "bmo-retrieval-181": [3],
        "bmo-retrieval-182": [3],
        "bmo-retrieval-183": [1, 3],
        "bmo-retrieval-184": [2, 1],
        "bmo-retrieval-185": [1, 1, 1],
        "bmo-retrieval-186": [1, 2, 1],
        "bmo-retrieval-187": [1, 1],
    }
    for record_id, group_sizes in expected_group_sizes.items():
        record = important[record_id]
        assert record["preferred_source_ids"]
        actual_sizes = [
            1 + len(group.get("equivalent_chunks", []))
            for group in record["expected_chunks"]
        ]
        assert actual_sizes == group_sizes
        assert all(group.get("requirement") for group in record["expected_chunks"])

        labelled_ids = [
            item["chunk_id"]
            for group in record["expected_chunks"]
            for item in [group, *group.get("equivalent_chunks", [])]
        ]
        assert len(labelled_ids) == len(set(labelled_ids))
