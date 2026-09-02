from bmo_rag.ingestion.chunking import normalize_chunk_sizes
from bmo_rag.ingestion.postprocessing import add_native_chunk_ids, deduplicate_chunks


def test_native_chunk_ids_are_stable_and_scoped_to_the_source() -> None:
    chunks = [{"text": "BMO reported\n strong capital levels.", "meta": {}}]

    first = add_native_chunk_ids(chunks, source_id="annual-report")
    whitespace_variant = add_native_chunk_ids(
        [{"text": "BMO reported strong capital levels.", "meta": {}}],
        source_id="annual-report",
    )
    other_source = add_native_chunk_ids(chunks, source_id="quarterly-report")

    assert first[0]["chunk_id"].startswith("bmo-")
    assert len(first[0]["chunk_id"]) == 24
    assert first[0]["chunk_id"] == whitespace_variant[0]["chunk_id"]
    assert first[0]["chunk_id"] != other_source[0]["chunk_id"]
    assert "chunk_id" not in chunks[0]


def test_deduplication_ignores_case_unicode_and_whitespace_differences() -> None:
    chunks = [
        {
            "text": "BMO\u00a0reported strong capital levels.",
            "meta": {"headings": ["Capital"]},
        },
        {
            "text": "  bmo reported   STRONG capital levels. ",
            "meta": {
                "headings": ["Capital"],
                "doc_items": [{"prov": [{"page_no": 7}]}],
            },
        },
    ]

    deduplicated, summary = deduplicate_chunks(chunks)

    assert len(deduplicated) == 1
    assert summary["duplicates_removed"] == 1
    audit = deduplicated[0]["meta"]["deduplication"]
    assert audit["duplicate_count"] == 1
    assert audit["occurrences"] == [
        {"source_chunk_index": 1, "pages": [7], "headings": ["Capital"]}
    ]


def test_small_chunks_merge_only_within_the_same_section() -> None:
    chunks = [
        {"text": "Small introduction.", "meta": {"headings": ["Capital"]}},
        {"text": "More capital details.", "meta": {"headings": ["Capital"]}},
        {"text": "Tiny liquidity note.", "meta": {"headings": ["Liquidity"]}},
    ]

    normalized = normalize_chunk_sizes(
        chunks,
        min_chars=50,
        max_chars=500,
        overlap_chars=50,
        preserve_sections=True,
    )

    assert len(normalized) == 2
    assert normalized[0]["text"] == "Small introduction.\n\nMore capital details."
    assert normalized[0]["meta"]["headings"] == ["Capital"]
    assert normalized[1]["meta"]["headings"] == ["Liquidity"]
