from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from bmo_rag.config import Settings
from bmo_rag.ingestion.loaders import DoclingIngestionPipeline

app = typer.Typer(help="BMO RAG command line tools.")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/processed/docling")


def _show_progress(message: str) -> None:
    """Print timestamped progress immediately, even when stdout is redirected."""
    local_time = datetime.now(UTC).astimezone()
    typer.echo(f"[{local_time.strftime('%H:%M:%S')}] {message}", err=True)


@app.command()
def health() -> None:
    """Print basic project configuration."""
    settings = Settings()
    typer.echo(f"{settings.app_name} is ready")


@app.command()
def ingest(
    raw_dir: Annotated[
        Path,
        typer.Option(help="Directory containing PDFs and urls.txt."),
    ] = DEFAULT_RAW_DIR,
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for Docling JSON, metadata, chunks, and manifest."),
    ] = DEFAULT_OUTPUT_DIR,
    limit: int | None = typer.Option(None, help="Limit number of sources for smoke tests."),
    device: str = typer.Option(
        "auto",
        help="Docling accelerator device: auto, cpu, cuda, cuda:0, mps, or xpu.",
    ),
    num_threads: int = typer.Option(4, help="CPU threads for Docling model inference."),
    do_ocr: bool = typer.Option(
        False,
        help="Enable OCR for scanned/image-based PDFs. Off by default for born-digital PDFs.",
    ),
    force_backend_text: bool = typer.Option(
        True,
        help="Prefer embedded PDF text over OCR/layout text detection.",
    ),
    do_table_structure: bool = typer.Option(
        True,
        "--table-structure/--no-table-structure",
        help="Enable Docling table structure reconstruction.",
    ),
    disable_table_structure_for_transcripts: bool = typer.Option(
        True,
        "--disable-table-structure-for-transcripts/--table-structure-for-transcripts",
        help="Skip table reconstruction for transcript-prefixed PDFs.",
    ),
    min_chunk_chars: int = typer.Option(300, help="Merge section-local chunks below this size."),
    max_chunk_chars: int = typer.Option(1500, help="Split chunks above this size."),
    chunk_overlap_chars: int = typer.Option(150, help="Character overlap when splitting large chunks."),
    deduplicate: bool = typer.Option(
        True,
        "--deduplicate/--no-deduplicate",
        help="Remove normalized exact duplicate chunks within each source.",
    ),
) -> None:
    """Load raw documents and prepare them for indexing."""
    results = DoclingIngestionPipeline(
        raw_dir=raw_dir,
        output_dir=output_dir,
        device=device,
        num_threads=num_threads,
        do_ocr=do_ocr,
        force_backend_text=force_backend_text,
        do_table_structure=do_table_structure,
        disable_table_structure_for_transcripts=disable_table_structure_for_transcripts,
        min_chunk_chars=min_chunk_chars,
        max_chunk_chars=max_chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        deduplicate=deduplicate,
        progress=_show_progress,
    ).run(limit=limit)
    succeeded = sum(result.status == "success" for result in results)
    typer.echo(f"Ingested {succeeded}/{len(results)} sources into {output_dir}")


@app.command()
def index() -> None:
    """Build or refresh the vector index."""
    typer.echo("Indexing pipeline placeholder")


@app.command()
def ask(question: str) -> None:
    """Ask a question against the indexed knowledge base."""
    typer.echo(f"Question received: {question}")
