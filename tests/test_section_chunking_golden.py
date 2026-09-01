import hashlib
import json
from collections import Counter
from pathlib import Path


DATASET = Path("data/golden/section_aware_chunking_golden.jsonl")
MANIFEST = Path("data/golden/section_aware_chunking_golden.manifest.json")


def load_rows() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]


def test_golden_dataset_has_100_valid_traceable_records() -> None:
    rows = load_rows()
    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100
    keys = {
        (row["source_document"], tuple(row["source_pages"]), row["chunk_sha256"])
        for row in rows
    }
    assert len(keys) == 100

    for row in rows:
        assert (Path("data/raw") / row["source_document"]).is_file()
        assert row["chunk"].strip()
        assert row["chunk_sha256"] == hashlib.sha256(row["chunk"].encode("utf-8")).hexdigest()
        contexts = row["expected"]["section_contexts"]
        assert contexts
        for context in contexts:
            ancestry = context["ancestry"]
            assert context["parent_section"] == (ancestry[-1] if ancestry else None)
            assert context["annotation_method"] != "layout_inference_review_required"


def test_golden_dataset_covers_documents_and_corner_cases() -> None:
    rows = load_rows()
    source_pdfs = {path.name for path in Path("data/raw").glob("*.pdf")}
    assert {row["source_document"] for row in rows} == source_pdfs

    counts = Counter(tag for row in rows for tag in row["corner_cases"])
    assert counts["multiple_sections"] >= 8
    assert counts["deep_ancestry"] >= 10
    assert counts["table_or_numeric_dense"] >= 12
    assert counts["short_chunk"] >= 5
    assert counts["long_chunk"] >= 10

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["record_count"] == len(rows)
    assert manifest["corner_case_counts"] == dict(sorted(counts.items()))
