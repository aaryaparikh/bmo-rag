"""Profile Docling chunks against practical vector-ingestion quality gates."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

from build_retrieval_gold import OUT_DIR, load_corpus


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WEB_NOISE_RE = re.compile(
    r"Explore our services|Ways to Bank|Customer support|Branch locator|"
    r"Report Lost/Stolen|apps\.apple\.com|play\.google\.com",
    re.I,
)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    corpus, by_source = load_corpus()
    normalized_counts = Counter(row["text"].casefold() for row in corpus)
    per_source = {}
    for source, rows in sorted(by_source.items()):
        lengths = [len(row["text"]) for row in rows]
        is_url = bool(rows[0]["source_url"])
        per_source[source] = {
            "source_type": "url" if is_url else "pdf",
            "chunks": len(rows),
            "characters": {
                "minimum": min(lengths),
                "p50": round(statistics.median(lengths)),
                "p95": percentile(lengths, 0.95),
                "maximum": max(lengths),
            },
            "under_300_chars": sum(length < 300 for length in lengths),
            "missing_headings": sum(not row["headings"] for row in rows),
            "missing_page_provenance": None if is_url else sum(not row["pages"] for row in rows),
            "web_navigation_or_footer_noise": sum(bool(WEB_NOISE_RE.search(row["text"])) for row in rows) if is_url else 0,
            "control_character_chunks": sum(bool(CONTROL_RE.search(row["text"])) for row in rows),
        }

    lengths = [len(row["text"]) for row in corpus]
    short = sum(length < 300 for length in lengths)
    missing_headings = sum(not row["headings"] for row in corpus)
    pdf_rows = [row for row in corpus if not row["source_url"]]
    missing_pages = sum(not row["pages"] for row in pdf_rows)
    noise = sum(bool(row["source_url"] and WEB_NOISE_RE.search(row["text"])) for row in corpus)
    controls = sum(bool(CONTROL_RE.search(row["text"])) for row in corpus)
    duplicate_instances = sum(count - 1 for count in normalized_counts.values() if count > 1)
    gates = {
        "json_schema_parseable": True,
        "non_empty_text": all(bool(row["text"]) for row in corpus),
        "maximum_1500_chars": max(lengths) <= 1500,
        "source_schema_contains_stable_chunk_id": False,
        "stable_content_ids_derivable": True,
        "all_chunks_have_hierarchy": missing_headings == 0,
        "all_pdf_chunks_have_page_provenance": missing_pages == 0,
        "web_boilerplate_removed": noise == 0,
        "under_300_char_rate_at_most_10_percent": short / len(corpus) <= 0.10,
        "control_characters_removed": controls == 0,
    }
    report = {
        "corpus": {
            "sources": len(by_source),
            "pdf_sources": sum(not rows[0]["source_url"] for rows in by_source.values()),
            "url_sources": sum(bool(rows[0]["source_url"]) for rows in by_source.values()),
            "chunks": len(corpus),
        },
        "global": {
            "character_length": {
                "minimum": min(lengths),
                "p05": percentile(lengths, 0.05),
                "p50": round(statistics.median(lengths)),
                "p95": percentile(lengths, 0.95),
                "maximum": max(lengths),
            },
            "under_300_chars": short,
            "under_300_chars_rate": round(short / len(corpus), 4),
            "missing_headings": missing_headings,
            "pdf_chunks_missing_page_provenance": missing_pages,
            "web_navigation_or_footer_noise": noise,
            "control_character_chunks": controls,
            "normalized_exact_duplicate_instances": duplicate_instances,
            "normalized_exact_duplicate_rate": round(duplicate_instances / len(corpus), 4),
        },
        "quality_gates": gates,
        "vector_db_recommendation": "conditional_after_cleanup" if not all(gates.values()) else "ready",
        "per_source": per_source,
        "notes": [
            "Character count is a proxy; enforce the embedding model's token limit at indexing time.",
            "Source JSONL records contain text and meta but no native chunk_id; the gold tooling derives a stable content hash.",
            "Exact duplicates across separate quarterly filings may be intentional but should share a canonical-content hash.",
            "The web-noise detector is conservative and only flags known navigation/footer markers.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "chunk_quality_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"global": report["global"], "quality_gates": gates, "recommendation": report["vector_db_recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
