import hashlib
import json
import re
from pathlib import Path


DATASET = Path("data/golden/docling_hierarchy_golden_100.jsonl")
MANIFEST = Path("data/golden/docling_hierarchy_golden_100.manifest.json")


def load_rows() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]


def test_gold_labels_actual_docling_chunks_with_independent_pdf_hierarchy() -> None:
    rows = load_rows()
    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100

    for row in rows:
        assert (Path("data/raw") / row["source_document"]).is_file()
        assert row["source_pages"]
        assert row["chunk"]
        assert row["expected_section_heading"] == row["expected_hierarchy"][-1]
        assert row["annotation_method"] in {"pdf_outline", "visual_reviewed"}
        assert row["chunk_sha256"] == hashlib.sha256(row["chunk"].encode()).hexdigest()
        chunk_file = Path("data/processed/docling") / (
            Path(row["source_document"]).stem + ".chunks.jsonl"
        )
        source_lines = chunk_file.read_text(encoding="utf-8").splitlines()
        source_row = json.loads(source_lines[row["chunk_index"]])
        source_text = re.sub(r"\s+", " ", source_row["text"]).strip()
        assert row["chunk"] == source_text


def test_manifest_matches_hierarchy_dataset() -> None:
    rows = load_rows()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["record_count"] == len(rows) == 100
    assert manifest["covered_pdf_count"] >= 10
    assert manifest["annotation_methods"]["pdf_outline"] > 0
    assert manifest["annotation_methods"]["visual_reviewed"] > 0
