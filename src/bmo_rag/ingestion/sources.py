from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

URLS_FILENAME = "urls.txt"
TRANSCRIPT_PREFIX = "transcript_"


@dataclass(frozen=True)
class RawSource:
    """A local file or remote URL that can be ingested."""

    location: Path | str
    source_type: str
    source_id: str
    document_kind: str
    display_name: str


def list_raw_documents(raw_dir: Path) -> list[Path]:
    """Return local PDF files that are candidates for ingestion."""
    if not raw_dir.exists():
        return []

    return [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.name != URLS_FILENAME and path.suffix.lower() == ".pdf"
    ]


def list_raw_sources(raw_dir: Path) -> list[RawSource]:
    """Return PDF files and URLs declared in urls.txt."""
    sources = [_source_from_file(path) for path in list_raw_documents(raw_dir)]
    sources.extend(_source_from_url(url) for url in _read_urls(raw_dir / URLS_FILENAME))
    return sources


def _source_from_file(path: Path) -> RawSource:
    name = path.name
    document_kind = "transcript" if name.lower().startswith(TRANSCRIPT_PREFIX) else "document"
    display_name = name[len(TRANSCRIPT_PREFIX) :] if document_kind == "transcript" else name
    return RawSource(
        location=path,
        source_type="file",
        source_id=_safe_source_id(path.stem),
        document_kind=document_kind,
        display_name=display_name,
    )


def _source_from_url(url: str) -> RawSource:
    parsed = urlparse(url)
    name = Path(parsed.path).name or parsed.netloc or "url"
    document_kind = "transcript" if name.lower().startswith(TRANSCRIPT_PREFIX) else "url"
    display_name = name[len(TRANSCRIPT_PREFIX) :] if document_kind == "transcript" else name
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return RawSource(
        location=url,
        source_type="url",
        source_id=_safe_source_id(f"{Path(name).stem}-{digest}"),
        document_kind=document_kind,
        display_name=display_name,
    )


def _read_urls(path: Path) -> list[str]:
    if not path.exists():
        return []

    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            urls.append(stripped)
    return urls


def _safe_source_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "source"
