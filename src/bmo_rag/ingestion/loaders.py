from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any
from urllib.parse import urldefrag

from bmo_rag.ingestion.chunking import normalize_chunk_sizes, validate_chunk_size_options
from bmo_rag.ingestion.docling_adapter import (
    build_docling_converter,
    build_hierarchical_chunker,
    export_docling_document,
    serialize_chunk,
    status_name,
)
from bmo_rag.ingestion.html_sections import (
    is_html_source,
    parse_html_sections,
    parse_markdown_sections,
)
from bmo_rag.ingestion.metadata import build_metadata, confidence_summary
from bmo_rag.ingestion.postprocessing import add_native_chunk_ids, deduplicate_chunks
from bmo_rag.ingestion.sources import RawSource, list_raw_documents, list_raw_sources

__all__ = [
    "DoclingIngestionPipeline",
    "IngestionResult",
    "RawSource",
    "list_raw_documents",
    "list_raw_sources",
    "normalize_chunk_sizes",
    "parse_html_sections",
    "parse_markdown_sections",
]

DEFAULT_URL_ATTEMPTS = 2
URL_CONNECT_TIMEOUT_SECONDS = 5
URL_READ_TIMEOUT_SECONDS = 10
READER_READ_TIMEOUT_SECONDS = 30
MAX_HTML_BYTES = 20 * 1024 * 1024
READER_FALLBACK_PREFIX = "https://r.jina.ai/"
DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-CA,en;q=0.9",
}


def _elapsed_seconds(started_at: float) -> float:
    return time.perf_counter() - started_at


@dataclass(frozen=True)
class IngestionResult:
    """Files written for one ingested source."""

    source: RawSource
    document_path: Path | None
    metadata_path: Path
    chunks_path: Path | None
    status: str
    confidence: dict[str, Any]


class DoclingIngestionPipeline:
    """Convert raw BMO corpus files into Docling documents and hierarchical chunks."""

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        *,
        device: str = "auto",
        num_threads: int = 4,
        do_ocr: bool = False,
        force_backend_text: bool = True,
        do_table_structure: bool = True,
        disable_table_structure_for_transcripts: bool = True,
        min_chunk_chars: int = 300,
        max_chunk_chars: int = 1500,
        chunk_overlap_chars: int = 150,
        deduplicate: bool = True,
        converter: Any | None = None,
        chunker: Any | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.device = device
        self.num_threads = num_threads
        self.do_ocr = do_ocr
        self.force_backend_text = force_backend_text
        self.do_table_structure = do_table_structure
        self.disable_table_structure_for_transcripts = disable_table_structure_for_transcripts
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.deduplicate = deduplicate
        self.converter = converter
        self._converter_cache: dict[bool, Any] = {}
        self.chunker = chunker
        self.progress = progress
        validate_chunk_size_options(
            min_chunk_chars=min_chunk_chars,
            max_chunk_chars=max_chunk_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        )

    def run(self, *, limit: int | None = None) -> list[IngestionResult]:
        self._report(f"Scanning for source files in {self.raw_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sources = list_raw_sources(self.raw_dir)
        if limit is not None:
            sources = sources[:limit]

        self._report(f"Found {len(sources)} source(s) to ingest")
        results = [
            self.ingest_source(source, position=index, total=len(sources))
            for index, source in enumerate(sources, start=1)
        ]
        self._report("Writing ingestion manifest")
        self._write_manifest(results)
        succeeded = sum(result.status.lower() == "success" for result in results)
        self._report(
            f"Ingestion finished: {succeeded} succeeded, {len(results) - succeeded} failed"
        )
        return results

    def ingest_source(
        self,
        source: RawSource,
        *,
        position: int | None = None,
        total: int | None = None,
    ) -> IngestionResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        document_path = self.output_dir / f"{source.source_id}.docling.json"
        metadata_path = self.output_dir / f"{source.source_id}.metadata.json"
        chunks_path = self.output_dir / f"{source.source_id}.chunks.jsonl"
        prefix = f"[{position}/{total}] " if position is not None and total is not None else ""
        label = f"{prefix}{source.display_name}"

        try:
            self._report(f"{label}: starting ({source.location})")
            preserve_sections = is_html_source(source)
            if preserve_sections:
                step_started = time.perf_counter()
                self._report(f"{label}: downloading HTML page")
                page_content, final_url, page_format = self._download_html(source, label=label)
                self._report(
                    f"{label}: downloaded {len(page_content):,} bytes in "
                    f"{_elapsed_seconds(step_started):.1f}s"
                )
                step_started = time.perf_counter()
                self._report(f"{label}: extracting main content and section hierarchy")
                parser = (
                    parse_markdown_sections
                    if page_format == "markdown"
                    else parse_html_sections
                )
                document_dict, raw_chunks = parser(
                    page_content, source_url=final_url, fallback_title=source.display_name
                )
                document_dict["fetch_method"] = page_format
                status = "success"
                self._report(
                    f"{label}: extracted {len(raw_chunks)} section(s) in "
                    f"{_elapsed_seconds(step_started):.1f}s"
                )
            else:
                step_started = time.perf_counter()
                self._report(f"{label}: preparing Docling converter and models")
                with self._heartbeat(f"{label}: still preparing converter and models"):
                    converter = self._converter_for(source)
                self._report(
                    f"{label}: converter ready in {_elapsed_seconds(step_started):.1f}s"
                )

                if self.chunker is None:
                    step_started = time.perf_counter()
                    self._report(f"{label}: preparing hierarchical chunker")
                    with self._heartbeat(f"{label}: still preparing hierarchical chunker"):
                        self.chunker = build_hierarchical_chunker()
                    self._report(
                        f"{label}: chunker ready in {_elapsed_seconds(step_started):.1f}s"
                    )

                step_started = time.perf_counter()
                self._report(f"{label}: converting document (this is usually the longest step)")
                with self._heartbeat(f"{label}: still converting document"):
                    conversion_result = self._convert_source(converter, source, label=label)
                status = status_name(getattr(conversion_result, "status", "success"))
                self._report(
                    f"{label}: conversion completed with status={status} "
                    f"in {_elapsed_seconds(step_started):.1f}s"
                )
                document = getattr(conversion_result, "document", None)
                if document is None:
                    raise RuntimeError("Docling did not return a document")

                step_started = time.perf_counter()
                self._report(f"{label}: exporting Docling document")
                document_dict = export_docling_document(document)

                step_started = time.perf_counter()
                self._report(f"{label}: generating hierarchical chunks")
                with self._heartbeat(f"{label}: still generating hierarchical chunks"):
                    raw_chunks = [
                        serialize_chunk(chunk) for chunk in self.chunker.chunk(dl_doc=document)
                    ]
                self._report(
                    f"{label}: generated {len(raw_chunks)} raw chunk(s) "
                    f"in {_elapsed_seconds(step_started):.1f}s"
                )

            _write_json(document_path, document_dict)
            self._report(f"{label}: wrote {document_path}")

            postprocessing: dict[str, Any] = {
                "deduplication": {"enabled": self.deduplicate},
                "preserve_sections": True,
            }
            step_started = time.perf_counter()
            self._report(f"{label}: normalizing chunk sizes")
            chunks = normalize_chunk_sizes(
                raw_chunks,
                min_chars=self.min_chunk_chars,
                max_chars=self.max_chunk_chars,
                overlap_chars=self.chunk_overlap_chars,
                preserve_sections=True,
            )
            self._report(
                f"{label}: normalized to {len(chunks)} chunk(s) "
                f"in {_elapsed_seconds(step_started):.1f}s"
            )

            if self.deduplicate:
                step_started = time.perf_counter()
                self._report(f"{label}: deduplicating normalized chunks")
                chunks, deduplication_summary = deduplicate_chunks(chunks)
                postprocessing["deduplication"] = deduplication_summary
                self._report(
                    f"{label}: removed "
                    f"{deduplication_summary['duplicates_removed']} duplicate chunk(s) in "
                    f"{_elapsed_seconds(step_started):.1f}s"
                )

            step_started = time.perf_counter()
            self._report(f"{label}: assigning native chunk IDs")
            chunks = add_native_chunk_ids(chunks, source_id=source.source_id)
            postprocessing["chunk_ids"] = {
                "strategy": "source_text_sha256",
                "prefix": "bmo-",
                "digest_chars": 20,
            }
            self._report(
                f"{label}: assigned {len(chunks)} native chunk ID(s) "
                f"in {_elapsed_seconds(step_started):.1f}s"
            )

            step_started = time.perf_counter()
            self._report(f"{label}: writing chunks and metadata")
            _write_jsonl(chunks_path, chunks)

            confidence = confidence_summary(
                document_dict=document_dict,
                conversion_status=status,
                chunk_count=len(chunks),
                failed=False,
            )
            metadata = build_metadata(
                source,
                status,
                confidence,
                started_at,
                len(chunks),
                device=self.device,
                num_threads=self.num_threads,
                do_ocr=self.do_ocr,
                force_backend_text=self.force_backend_text,
                do_table_structure=self._do_table_structure_for(source),
                min_chunk_chars=self.min_chunk_chars,
                max_chunk_chars=self.max_chunk_chars,
                chunk_overlap_chars=self.chunk_overlap_chars,
                postprocessing=postprocessing,
            )
            _write_json(metadata_path, metadata)
            self._report(
                f"{label}: completed in {_elapsed_seconds(started_at):.1f}s; "
                f"wrote {chunks_path} and {metadata_path}"
            )

            return IngestionResult(
                source=source,
                document_path=document_path,
                metadata_path=metadata_path,
                chunks_path=chunks_path,
                status=status,
                confidence=confidence,
            )
        except Exception as exc:  # noqa: BLE001 - one source failure must not stop the corpus
            confidence = confidence_summary(
                document_dict={},
                conversion_status="failure",
                chunk_count=0,
                failed=True,
            )
            metadata = build_metadata(
                source,
                "failure",
                confidence,
                started_at,
                0,
                device=self.device,
                num_threads=self.num_threads,
                do_ocr=self.do_ocr,
                force_backend_text=self.force_backend_text,
                do_table_structure=self._do_table_structure_for(source),
                min_chunk_chars=self.min_chunk_chars,
                max_chunk_chars=self.max_chunk_chars,
                chunk_overlap_chars=self.chunk_overlap_chars,
                error=str(exc),
            )
            _write_json(metadata_path, metadata)
            self._report(
                f"{label}: FAILED after {_elapsed_seconds(started_at):.1f}s: "
                f"{type(exc).__name__}: {exc}"
            )
            return IngestionResult(
                source=source,
                document_path=None,
                metadata_path=metadata_path,
                chunks_path=None,
                status="failure",
                confidence=confidence,
            )

    def _report(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def _convert_source(self, converter: Any, source: RawSource, *, label: str) -> Any:
        """Convert a source, retrying transient URL download failures with browser headers."""
        if source.source_type != "url":
            return converter.convert(str(source.location))

        for attempt in range(1, DEFAULT_URL_ATTEMPTS + 1):
            self._report(f"{label}: URL download attempt {attempt}/{DEFAULT_URL_ATTEMPTS}")
            try:
                return converter.convert(
                    str(source.location),
                    headers=DEFAULT_HTTP_HEADERS,
                )
            except Exception as exc:
                if attempt == DEFAULT_URL_ATTEMPTS:
                    raise
                retry_delay = 2 ** (attempt - 1)
                self._report(
                    f"{label}: URL attempt {attempt} failed: {type(exc).__name__}: {exc}; "
                    f"retrying in {retry_delay}s"
                )
                time.sleep(retry_delay)

        raise RuntimeError("URL conversion attempts exhausted")

    def _download_html(self, source: RawSource, *, label: str) -> tuple[bytes, str, str]:
        """Download one HTML page with strict timeouts and bounded retries."""
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is required for HTML ingestion") from exc

        try:
            response = requests.get(
                str(source.location),
                headers=DEFAULT_HTTP_HEADERS,
                timeout=(URL_CONNECT_TIMEOUT_SECONDS, URL_READ_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            content = response.content
            if len(content) > MAX_HTML_BYTES:
                raise ValueError(
                    f"HTML response is {len(content):,} bytes; limit is {MAX_HTML_BYTES:,}"
                )
            return content, response.url, "html"
        except (requests.RequestException, ValueError) as direct_error:
            self._report(
                f"{label}: direct HTTP fetch failed: {type(direct_error).__name__}: "
                f"{direct_error}; trying public reader fallback"
            )

        source_without_fragment = urldefrag(str(source.location)).url
        reader_url = f"{READER_FALLBACK_PREFIX}{source_without_fragment}"
        response = requests.get(
            reader_url,
            timeout=(URL_CONNECT_TIMEOUT_SECONDS, READER_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        content = response.content
        if len(content) > MAX_HTML_BYTES:
            raise ValueError(
                f"Reader response is {len(content):,} bytes; limit is {MAX_HTML_BYTES:,}"
            )
        self._report(f"{label}: public reader fallback succeeded")
        return content, str(source.location), "markdown"

    @contextmanager
    def _heartbeat(self, message: str, *, interval_seconds: float = 30.0) -> Iterator[None]:
        """Report elapsed time periodically while a blocking Docling call is running."""
        if self.progress is None:
            yield
            return

        stopped = Event()
        started_at = time.perf_counter()

        def report_until_stopped() -> None:
            while not stopped.wait(interval_seconds):
                self._report(f"{message} ({_elapsed_seconds(started_at):.0f}s elapsed)")

        reporter = Thread(target=report_until_stopped, name="ingestion-progress", daemon=True)
        reporter.start()
        try:
            yield
        finally:
            stopped.set()
            reporter.join(timeout=1.0)

    def _write_manifest(self, results: list[IngestionResult]) -> None:
        manifest = {
            "source_count": len(results),
            "succeeded": sum(result.status.lower() == "success" for result in results),
            "failed": sum(result.status.lower() != "success" for result in results),
            "results": [
                {
                    "source_id": result.source.source_id,
                    "status": result.status,
                    "document": str(result.document_path) if result.document_path else None,
                    "metadata": str(result.metadata_path),
                    "chunks": str(result.chunks_path) if result.chunks_path else None,
                    "confidence": result.confidence,
                }
                for result in results
            ],
        }
        _write_json(self.output_dir / "ingestion_manifest.json", manifest)

    def _do_table_structure_for(self, source: RawSource) -> bool:
        return not (
            self.disable_table_structure_for_transcripts
            and source.document_kind == "transcript"
        ) and self.do_table_structure

    def _converter_for(self, source: RawSource) -> Any:
        if self.converter is not None:
            return self.converter

        do_table_structure = self._do_table_structure_for(source)
        if do_table_structure not in self._converter_cache:
            self._converter_cache[do_table_structure] = build_docling_converter(
                device=self.device,
                num_threads=self.num_threads,
                do_ocr=self.do_ocr,
                force_backend_text=self.force_backend_text,
                do_table_structure=do_table_structure,
            )
        return self._converter_cache[do_table_structure]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
