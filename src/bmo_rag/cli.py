from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from bmo_rag.config import Settings
from bmo_rag.indexing.embeddings import EmbeddingError, resolve_model
from bmo_rag.indexing.local_vllm import (
    LocalVllmError,
    ensure_local_embedding_services,
    ensure_local_reranker,
)
from bmo_rag.indexing.qdrant_store import QdrantError
from bmo_rag.ingestion.loaders import DoclingIngestionPipeline
from bmo_rag.retrieval.reranker import DEFAULT_RERANKER_MODEL, RerankerError
from bmo_rag.retrieval.semantic import citation, retrieve_chunks, retrieve_hybrid_chunks

app = typer.Typer(help="BMO RAG command line tools.")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/processed/docling")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


@app.command("retrieve")
def retrieve_command(
    question: Annotated[
        str | None,
        typer.Argument(help="Question to retrieve for; omit it for an interactive prompt."),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1, max=30)] = 5,
    model: Annotated[str, typer.Option(help="Embedding model slug or Hugging Face ID.")] = "bge-m3",
    base_url: Annotated[str, typer.Option(help="OpenAI-compatible vLLM base URL.")] = (
        "http://127.0.0.1:8000/v1"
    ),
    qdrant_url: Annotated[str, typer.Option(help="Qdrant HTTP URL.")] = (
        "http://127.0.0.1:6333"
    ),
    start_local: Annotated[
        bool,
        typer.Option(
            "--start-local/--no-start-local",
            help="Start/reuse local Docker Qdrant and vLLM services automatically.",
        ),
    ] = True,
    hybrid: Annotated[
        bool,
        typer.Option(
            "--hybrid/--dense",
            help="Use Qdrant native dense+BM25 RRF fusion or dense-only retrieval.",
        ),
    ] = True,
    rerank: Annotated[
        bool,
        typer.Option(
            "--rerank/--no-rerank",
            help="Rerank hybrid candidates with the BGE cross-encoder.",
        ),
    ] = True,
    candidate_k: Annotated[
        int,
        typer.Option(help="Hybrid candidates passed to the reranker.", min=1, max=100),
    ] = 30,
    reranker_url: Annotated[str, typer.Option(help="vLLM reranker URL.")] = (
        "http://127.0.0.1:8001"
    ),
    reranker_model: Annotated[str, typer.Option(help="Cross-encoder reranker model.")] = (
        DEFAULT_RERANKER_MODEL
    ),
    max_chars: Annotated[
        int,
        typer.Option(help="Maximum chunk text characters printed per result.", min=100),
    ] = 900,
) -> None:
    """Retrieve ranked source chunks for a question (BGE-M3 by default)."""
    selected_question = question or typer.prompt("Question")
    try:
        spec = resolve_model(model)
        if start_local:
            ensure_local_embedding_services(
                spec,
                project_root=PROJECT_ROOT,
                base_url=base_url,
                gpu_memory_utilization=0.35 if hybrid and rerank else 0.80,
                progress=lambda message: typer.echo(message, err=True),
            )
            if hybrid and rerank:
                ensure_local_reranker(
                    model=reranker_model,
                    project_root=PROJECT_ROOT,
                    base_url=reranker_url,
                    progress=lambda message: typer.echo(message, err=True),
                )
        if hybrid:
            if candidate_k < top_k:
                raise ValueError("--candidate-k must be greater than or equal to --top-k")
            points = retrieve_hybrid_chunks(
                selected_question,
                model=spec.slug,
                top_k=top_k,
                candidate_k=candidate_k,
                base_url=base_url,
                qdrant_url=qdrant_url,
                rerank=rerank,
                reranker_url=reranker_url,
                reranker_model=reranker_model,
            )
        else:
            points = retrieve_chunks(
                selected_question,
                model=spec.slug,
                top_k=top_k,
                base_url=base_url,
                qdrant_url=qdrant_url,
            )
    except (EmbeddingError, LocalVllmError, QdrantError, RerankerError, ValueError) as exc:
        typer.echo(f"Retrieval failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"\nQuestion: {selected_question}")
    mode = "hybrid + reranker" if hybrid and rerank else "hybrid RRF" if hybrid else "dense"
    typer.echo(f"Model: {spec.model_id} | Mode: {mode} | Results: {len(points)}\n")
    for rank, point in enumerate(points, start=1):
        payload = point.get("payload") or {}
        headings = " > ".join(payload.get("headings") or [])
        text = str(payload.get("text") or "")
        if len(text) > max_chars:
            text = f"{text[: max_chars - 3].rstrip()}..."
        if "rerank_score" in point:
            typer.echo(
                f"[{rank}] rerank={float(point['rerank_score']):.4f} "
                f"fusion={float(point.get('fusion_score', 0.0)):.4f}"
            )
        else:
            typer.echo(f"[{rank}] score={float(point.get('score', 0.0)):.4f}")
        typer.echo(f"    chunk_id: {payload.get('chunk_id', 'unknown')}")
        typer.echo(f"    citation: {citation(payload)}")
        if headings:
            typer.echo(f"    section: {headings}")
        typer.echo(f"    text: {text}\n")
