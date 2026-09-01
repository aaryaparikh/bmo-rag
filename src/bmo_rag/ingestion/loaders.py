from pathlib import Path


def list_raw_documents(raw_dir: Path) -> list[Path]:
    """Return files that are candidates for ingestion."""
    if not raw_dir.exists():
        return []

    return [path for path in raw_dir.rglob("*") if path.is_file()]
