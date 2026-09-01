import json
from pathlib import Path

from bmo_rag.ingestion.chunking import normalize_chunk_sizes
from bmo_rag.ingestion.html_sections import parse_html_sections, parse_markdown_sections
from bmo_rag.ingestion.loaders import DoclingIngestionPipeline
from bmo_rag.ingestion.sources import list_raw_sources


class FakeDoclingDocument:
    def export_to_dict(self, confid_precision: int = 4) -> dict:
        return {
            "schema_name": "DoclingDocument",
            "texts": [
                {
                    "label": "section_header",
                    "text": "Management's Discussion and Analysis",
                    "confidence": 0.98,
                },
                {
                    "label": "text",
                    "text": "BMO reported strong capital levels.",
                    "confidence": 0.92,
                },
            ],
        }


class FakeConversionResult:
    status = "success"
    document = FakeDoclingDocument()


class FakeConverter:
    def __init__(self) -> None:
        self.converted: list[str] = []
        self.headers_seen: list[dict[str, str] | None] = []

    def convert(
        self,
        source: str,
        headers: dict[str, str] | None = None,
    ) -> FakeConversionResult:
        self.converted.append(source)
        self.headers_seen.append(headers)
        return FakeConversionResult()


class FakeChunk:
    def model_dump(self, mode: str, by_alias: bool, exclude_none: bool) -> dict:
        return {
            "text": "Management's Discussion and Analysis\nBMO reported strong capital levels.",
            "meta": {
                "headings": ["Management's Discussion and Analysis"],
                "doc_items": [{"self_ref": "#/texts/1"}],
            },
        }


class FakeHierarchicalChunker:
    def chunk(self, dl_doc: FakeDoclingDocument) -> list[FakeChunk]:
        return [FakeChunk()]


def test_lists_pdfs_urls_and_marks_transcripts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "report.pdf").write_bytes(b"%PDF-1.4")
    (raw_dir / "transcript_Q3.pdf").write_bytes(b"%PDF-1.4")
    (raw_dir / "urls.txt").write_text(
        "https://example.com/investor-presentation.pdf\n"
        "https://example.com/transcript_event.pdf\n",
        encoding="utf-8",
    )

    sources = list_raw_sources(raw_dir)

    assert [source.document_kind for source in sources] == [
        "document",
        "transcript",
        "url",
        "transcript",
    ]


def test_docling_ingestion_writes_native_document_metadata_and_chunks(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    (raw_dir / "transcript_example.pdf").write_bytes(b"%PDF-1.4")

    pipeline = DoclingIngestionPipeline(
        raw_dir=raw_dir,
        output_dir=output_dir,
        converter=FakeConverter(),
        chunker=FakeHierarchicalChunker(),
    )

    results = pipeline.run()

    assert len(results) == 1
    result = results[0]
    assert result.status == "success"
    assert result.source.document_kind == "transcript"

    docling_document = json.loads(result.document_path.read_text(encoding="utf-8"))
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    chunks = [
        json.loads(line)
        for line in result.chunks_path.read_text(encoding="utf-8").splitlines()
    ]

    assert docling_document["schema_name"] == "DoclingDocument"
    assert docling_document["texts"][0]["label"] == "section_header"
    assert metadata["confidence"]["conversion"] == 1.0
    assert metadata["confidence"]["extraction"] == 0.95
    assert metadata["source"]["document_kind"] == "transcript"
    assert metadata["accelerator"] == {"device": "auto", "num_threads": 4}
    assert metadata["docling_options"] == {
        "do_ocr": False,
        "force_backend_text": True,
        "do_table_structure": False,
    }
    assert metadata["chunk_size_options"] == {
        "min_chars": 300,
        "max_chars": 1500,
        "overlap_chars": 150,
    }
    assert chunks[0]["meta"]["headings"] == ["Management's Discussion and Analysis"]


def test_docling_ingestion_reuses_converter_for_multiple_sources(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    (raw_dir / "first.pdf").write_bytes(b"%PDF-1.4")
    (raw_dir / "second.pdf").write_bytes(b"%PDF-1.4")
    converter = FakeConverter()

    pipeline = DoclingIngestionPipeline(
        raw_dir=raw_dir,
        output_dir=output_dir,
        converter=converter,
        chunker=FakeHierarchicalChunker(),
    )

    results = pipeline.run()

    assert len(results) == 2
    assert converter.converted == [str(raw_dir / "first.pdf"), str(raw_dir / "second.pdf")]


def test_docling_ingestion_reports_intermediate_progress(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    (raw_dir / "report.pdf").write_bytes(b"%PDF-1.4")
    messages: list[str] = []

    pipeline = DoclingIngestionPipeline(
        raw_dir=raw_dir,
        output_dir=output_dir,
        converter=FakeConverter(),
        chunker=FakeHierarchicalChunker(),
        progress=messages.append,
    )

    pipeline.run()

    assert messages[0] == f"Scanning for source files in {raw_dir}"
    assert "Found 1 source(s) to ingest" in messages
    assert any("[1/1] report.pdf: converting document" in message for message in messages)
    assert any("[1/1] report.pdf: generated 1 raw chunk(s)" in message for message in messages)
    assert any("[1/1] report.pdf: completed" in message for message in messages)
    assert messages[-1] == "Ingestion finished: 1 succeeded, 0 failed"


def test_remote_pdf_ingestion_uses_browser_headers(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    url = "https://www.bmo.com/investor-relations/report.pdf"
    (raw_dir / "urls.txt").write_text(f"{url}\n", encoding="utf-8")
    converter = FakeConverter()

    pipeline = DoclingIngestionPipeline(
        raw_dir=raw_dir,
        output_dir=output_dir,
        converter=converter,
        chunker=FakeHierarchicalChunker(),
    )

    result = pipeline.run()[0]

    assert result.status == "success"
    assert converter.converted == [url]
    assert converter.headers_seen[0] is not None
    assert converter.headers_seen[0]["User-Agent"].startswith("Mozilla/5.0")


def test_html_parser_uses_article_title_and_bold_section_labels() -> None:
    html = b"""
    <html><head><title>Newsroom shell</title></head><body>
      <main><h1>News Releases</h1>
        <div id="wd_printable_content">
          <div class="wd_title">BMO Reports Second Quarter Results</div>
          <div class="wd_body wd_news_body">
            <p>Opening article summary.</p>
            <p><b>Financial Results Highlights</b></p>
            <p>Reported net income increased.</p>
            <p><b>Canadian P&amp;C</b></p>
            <p>Canadian banking performed well.</p>
          </div>
        </div>
      </main>
    </body></html>
    """

    document, chunks = parse_html_sections(
        html,
        source_url="https://example.com/release",
        fallback_title="release",
    )

    assert document["title"] == "BMO Reports Second Quarter Results"
    assert [chunk["meta"]["headings"] for chunk in chunks] == [
        ["BMO Reports Second Quarter Results"],
        ["BMO Reports Second Quarter Results", "Financial Results Highlights"],
        ["BMO Reports Second Quarter Results", "Canadian P&C"],
    ]
    assert all("News Releases" not in chunk["meta"]["headings"] for chunk in chunks)


def test_html_chunk_normalization_does_not_merge_sections() -> None:
    chunks = [
        {"text": "Short A", "meta": {"headings": ["Page", "Section A"]}},
        {"text": "Short B", "meta": {"headings": ["Page", "Section B"]}},
    ]

    normalized = normalize_chunk_sizes(
        chunks,
        min_chars=300,
        max_chars=1500,
        overlap_chars=150,
        preserve_sections=True,
    )

    assert len(normalized) == 2
    assert normalized[0]["meta"]["headings"] == ["Page", "Section A"]
    assert normalized[1]["meta"]["headings"] == ["Page", "Section B"]


def test_reader_markdown_parser_skips_navigation_and_footer() -> None:
    markdown = """
Title: Financial Information | BMO
Markdown Content:
Navigation that must not be ingested.
# Financial Information
Introductory content.
## Quarterly Results
Quarterly results content.
## Medium Term Financial Objectives
Objectives content.
## Looking for BMO U.S.?
Footer content that must not be ingested.
"""

    document, chunks = parse_markdown_sections(
        markdown,
        source_url="https://example.com/financial-information",
        fallback_title="financial-information",
    )

    assert document["title"] == "Financial Information"
    assert [chunk["meta"]["headings"] for chunk in chunks] == [
        ["Financial Information"],
        ["Financial Information", "Quarterly Results"],
        ["Financial Information", "Medium Term Financial Objectives"],
    ]
    combined = " ".join(chunk["text"] for chunk in chunks)
    assert "Navigation" not in combined
    assert "Footer" not in combined


def test_reader_markdown_parser_infers_sections_without_hash_headings() -> None:
    markdown = """
Title: 40-F
URL Source: https://www.sec.gov/example.htm
Markdown Content:
U.S. Securities and Exchange Commission
Form 40-F
BANK OF MONTREAL
Cover-page information.
DISCLOSURE CONTROLS AND PROCEDURES
Disclosure-control details.
INTERNAL CONTROL OVER FINANCIAL REPORTING
a. Management's annual report on internal control over financial reporting
Management report details.
b. Auditor's attestation report on internal control over financial reporting
Auditor report details.
"""

    document, chunks = parse_markdown_sections(
        markdown,
        source_url="https://www.sec.gov/example.htm",
        fallback_title="example",
    )

    assert document["title"] == "40-F"
    assert [chunk["meta"]["headings"] for chunk in chunks] == [
        ["40-F"],
        ["40-F", "BANK OF MONTREAL"],
        ["40-F", "DISCLOSURE CONTROLS AND PROCEDURES"],
        [
            "40-F",
            "INTERNAL CONTROL OVER FINANCIAL REPORTING",
            "a. Management's annual report on internal control over financial reporting",
        ],
        [
            "40-F",
            "INTERNAL CONTROL OVER FINANCIAL REPORTING",
            "b. Auditor's attestation report on internal control over financial reporting",
        ],
    ]


def test_docling_ingestion_records_table_structure_for_regular_documents(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()
    (raw_dir / "annual_report.pdf").write_bytes(b"%PDF-1.4")

    pipeline = DoclingIngestionPipeline(
        raw_dir=raw_dir,
        output_dir=output_dir,
        converter=FakeConverter(),
        chunker=FakeHierarchicalChunker(),
    )

    result = pipeline.run()[0]

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["docling_options"]["do_table_structure"] is True


def test_normalize_chunk_sizes_merges_small_chunks_within_same_section() -> None:
    chunks = [
        {"text": "Short intro.", "meta": {"headings": ["Capital"], "doc_items": [{"self_ref": "#/1"}]}},
        {
            "text": "More detail about capital ratios.",
            "meta": {"headings": ["Capital"], "doc_items": [{"self_ref": "#/2"}]},
        },
        {
            "text": "Different section with enough substance to stand by itself.",
            "meta": {"headings": ["Liquidity"]},
        },
    ]

    normalized = normalize_chunk_sizes(chunks, min_chars=40, max_chars=500, overlap_chars=50)

    assert len(normalized) == 2
    assert normalized[0]["text"] == "Short intro.\n\nMore detail about capital ratios."
    assert normalized[0]["meta"]["headings"] == ["Capital"]
    assert normalized[0]["meta"]["chunk_normalization"]["strategy"] == "merged_small"
    assert normalized[0]["meta"]["chunk_normalization"]["source_chunk_count"] == 2


def test_normalize_chunk_sizes_merges_isolated_small_chunks_across_sections() -> None:
    chunks = [
        {"text": "Tiny title.", "meta": {"headings": ["Cover"]}},
        {"text": "Larger overview text that gives the tiny title enough context.", "meta": {"headings": ["Overview"]}},
    ]

    normalized = normalize_chunk_sizes(chunks, min_chars=50, max_chars=500, overlap_chars=50)

    assert len(normalized) == 1
    assert normalized[0]["meta"]["headings"] == ["Cover", "Overview"]
    assert normalized[0]["meta"]["chunk_normalization"]["strategy"] == "merged_small_cross_section"


def test_normalize_chunk_sizes_splits_large_chunks_and_preserves_headings() -> None:
    chunks = [
        {
            "text": " ".join(f"word{i}" for i in range(80)),
            "meta": {"headings": ["Risk management"], "doc_items": [{"self_ref": "#/risk"}]},
        }
    ]

    normalized = normalize_chunk_sizes(chunks, min_chars=0, max_chars=120, overlap_chars=20)

    assert len(normalized) > 1
    assert all(len(chunk["text"]) <= 120 for chunk in normalized)
    assert all(chunk["meta"]["headings"] == ["Risk management"] for chunk in normalized)
    assert normalized[0]["meta"]["chunk_normalization"]["strategy"] == "split_large"
    assert normalized[0]["meta"]["chunk_normalization"]["split_count"] == len(normalized)
