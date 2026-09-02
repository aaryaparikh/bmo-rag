import hashlib
import json
from pathlib import Path

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


def test_manifest_matches_dataset_bytes_and_source_coverage() -> None:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for row in rows:
        for gold in row["expected_chunks"]:
            source = gold["source_id"]
            counts[source] = counts.get(source, 0) + 1

    assert counts == manifest["source_distribution"]
    assert manifest["corpus_source_count"] == 26
    assert manifest["corpus_chunk_count"] > 6_000
    assert len(manifest["corpus_fingerprint_sha256"]) == hashlib.sha256().digest_size * 2
