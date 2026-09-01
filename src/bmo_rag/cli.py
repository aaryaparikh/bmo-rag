import typer

from bmo_rag.config import Settings

app = typer.Typer(help="BMO RAG command line tools.")


@app.command()
def health() -> None:
    """Print basic project configuration."""
    settings = Settings()
    typer.echo(f"{settings.app_name} is ready")


@app.command()
def ingest() -> None:
    """Load raw documents and prepare them for indexing."""
    typer.echo("Ingestion pipeline placeholder")


@app.command()
def index() -> None:
    """Build or refresh the vector index."""
    typer.echo("Indexing pipeline placeholder")


@app.command()
def ask(question: str) -> None:
    """Ask a question against the indexed knowledge base."""
    typer.echo(f"Question received: {question}")
