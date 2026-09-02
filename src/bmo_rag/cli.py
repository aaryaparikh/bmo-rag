import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from bmo_rag.config import Settings
from bmo_rag.generation.memory import ConversationMemory
from bmo_rag.generation.openai_responses import OpenAIError, OpenAIResponsesClient
from bmo_rag.indexing.embeddings import EmbeddingError, resolve_model
from bmo_rag.indexing.local_vllm import (
    LocalVllmError,
    ensure_local_embedding_services,
    ensure_local_reranker,
)
from bmo_rag.indexing.qdrant_store import QdrantError
from bmo_rag.ingestion.loaders import DoclingIngestionPipeline
from bmo_rag.observability.store import SQLiteObservabilityStore
from bmo_rag.pipeline.chat import ChatAnswer, RAGChatbot
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


def _print_chat_answer(result: ChatAnswer) -> None:
    typer.echo(f"\n{result.answer}\n")
    if result.requested_sources:
        resolved = "; ".join(
            f"{group.label} -> {', '.join(group.source_ids)}"
            for group in result.requested_sources
        )
        typer.echo(f"Source constraints: {resolved}")
    typer.echo("Sources:")
    for source in result.sources:
        section = " > ".join(source.headings)
        suffix = f" — {section}" if section else ""
        typer.echo(f"  [{source.label}] {source.citation}{suffix}")
    if result.context_truncated:
        typer.echo("  (Some lower-priority context was omitted to stay within the token budget.)")
    if result.trace_id:
        typer.echo(f"Trace: {result.trace_id}")


@app.command()
def ask(
    question: Annotated[
        str | None,
        typer.Argument(help="Question to answer; omit it for a memory-enabled chat session."),
    ] = None,
    llm_model: Annotated[str, typer.Option(help="OpenAI answer model.")] = "gpt-5",
    embedding_model: Annotated[
        str, typer.Option(help="Local embedding model slug or Hugging Face ID.")
    ] = "bge-m3",
    candidate_k: Annotated[
        int, typer.Option(help="Hybrid candidates passed to the reranker.", min=8, max=100)
    ] = 30,
    seed_k: Annotated[
        int, typer.Option(help="Reranked seed chunks before expansion.", min=1, max=20)
    ] = 8,
    max_answer_tokens: Annotated[
        int,
        typer.Option(
            help="Maximum GPT-5 output-token budget, including reasoning tokens.",
            min=512,
            max=32000,
        ),
    ] = 5000,
    base_url: Annotated[str, typer.Option(help="OpenAI-compatible vLLM embedding URL.")] = (
        "http://127.0.0.1:8000/v1"
    ),
    qdrant_url: Annotated[str, typer.Option(help="Qdrant HTTP URL.")] = (
        "http://127.0.0.1:6333"
    ),
    reranker_url: Annotated[str, typer.Option(help="vLLM reranker URL.")] = (
        "http://127.0.0.1:8001"
    ),
    start_local: Annotated[
        bool,
        typer.Option(
            "--start-local/--no-start-local",
            help="Start/reuse local Qdrant, BGE-M3, and reranker services.",
        ),
    ] = True,
    observe: Annotated[
        bool,
        typer.Option(
            "--observe/--no-observe",
            help="Persist full local latency, token, retrieval, prompt, and response traces.",
        ),
    ] = True,
    observability_db: Annotated[
        Path | None,
        typer.Option(help="SQLite path for local RAG observability traces."),
    ] = None,
) -> None:
    """Answer questions with GPT-5 over hybrid, reranked, selectively expanded evidence."""
    try:
        settings = Settings()
        spec = resolve_model(embedding_model)
        if candidate_k < seed_k:
            raise ValueError("--candidate-k must be greater than or equal to --seed-k")
        if start_local:
            ensure_local_embedding_services(
                spec,
                project_root=PROJECT_ROOT,
                base_url=base_url,
                gpu_memory_utilization=0.35,
                progress=lambda message: typer.echo(message, err=True),
            )
            ensure_local_reranker(
                model=DEFAULT_RERANKER_MODEL,
                project_root=PROJECT_ROOT,
                base_url=reranker_url,
                progress=lambda message: typer.echo(message, err=True),
            )
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        chatbot = RAGChatbot(
            llm=OpenAIResponsesClient(model=llm_model, api_key=key),
            memory=ConversationMemory(max_turns=6),
            embedding_model=spec.slug,
            embedding_url=base_url,
            qdrant_url=qdrant_url,
            reranker_url=reranker_url,
            candidate_k=candidate_k,
            seed_k=seed_k,
            max_answer_tokens=max_answer_tokens,
            observability_store=(
                SQLiteObservabilityStore(observability_db or settings.observability_db)
                if observe
                else None
            ),
        )
        if question is not None:
            _print_chat_answer(chatbot.answer(question))
            return
        typer.echo("Memory-enabled BMO chat. Type 'exit' or 'quit' to stop.")
        while True:
            selected = typer.prompt("You").strip()
            if selected.casefold() in {"exit", "quit"}:
                return
            _print_chat_answer(chatbot.answer(selected))
    except (
        EmbeddingError,
        LocalVllmError,
        OpenAIError,
        QdrantError,
        RerankerError,
        ValueError,
    ) as exc:
        typer.echo(f"Chat failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def monitor(
    trace_id: Annotated[
        str | None, typer.Option(help="Show the complete record for one trace ID.")
    ] = None,
    limit: Annotated[
        int, typer.Option(help="Number of recent traces to show.", min=1, max=500)
    ] = 20,
    observability_db: Annotated[
        Path | None, typer.Option(help="SQLite observability database path.")
    ] = None,
) -> None:
    """Inspect recent RAG latency/token summaries or one complete trace."""
    database_path = observability_db or Settings().observability_db
    store = SQLiteObservabilityStore(database_path)
    if trace_id:
        detail = store.trace_detail(trace_id)
        if detail is None:
            typer.echo(f"Trace not found: {trace_id}", err=True)
            raise typer.Exit(code=1)
        typer.echo(json.dumps(detail, indent=2, ensure_ascii=False))
        return
    rows = store.recent(limit=limit)
    if not rows:
        typer.echo(f"No traces in {database_path}")
        return
    summary = store.summary(limit=limit)
    typer.echo(
        f"Last {summary['trace_count']} traces | failures={summary['failure_count']} | "
        f"latency avg/p50/p95/max={summary['average_latency_ms']:.2f}/"
        f"{summary['p50_latency_ms']:.2f}/{summary['p95_latency_ms']:.2f}/"
        f"{summary['max_latency_ms']:.2f} ms | tokens in/out/total="
        f"{summary['input_tokens']}/{summary['output_tokens']}/{summary['total_tokens']}"
    )
    for row in rows:
        query = str(row["original_query"]).replace("\n", " ")
        if len(query) > 80:
            query = f"{query[:77]}..."
        typer.echo(
            f"{row['started_at']}  {row['status']:<9}  {row['duration_ms']:>9.2f} ms  "
            f"tokens={row['input_tokens']}/{row['output_tokens']}  "
            f"evidence={row['evidence_tokens']}  {row['trace_id']}  {query}"
        )


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


if __name__ == "__main__":
    app()
