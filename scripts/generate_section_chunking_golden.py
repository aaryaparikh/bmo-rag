"""Build a reproducible golden set for section-aware chunking evaluation.

The source of truth is the PDF corpus in ``data/raw``. Native PDF outlines are
preferred for section paths. For PDFs without useful outlines, a conservative
page-title heuristic is used and the annotation method is recorded so these
examples remain auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


DEFAULT_SEED = 20260831
TARGET_SIZE = 100
MAX_CHUNK_CHARS = 1_200

# Explicit adjudications for the selected examples from PDFs without native
# outlines. These page/section labels were checked against rendered source pages.
REVIEWED_SECTION_PATHS: dict[tuple[str, int], list[str]] = {
    ("Bail_In_TLAC_Disclosure.pdf", 3): ["Compensation Regime"],
    ("LCR_CY26Q1.pdf", 8): ["6. Forward Looking Information"],
    ("LCR_CY26Q2.pdf", 1): [
        "BMO Financial Corp.",
        "Liquidity Coverage Ratio Disclosure",
    ],
    ("LCR_CY26Q2.pdf", 2): ["Table of Contents"],
    ("NSFRCY26Q2.pdf", 1): [
        "BMO Financial Corp.",
        "Net Stable Funding Ratio Disclosure",
    ],
    ("NSFRCY26Q2.pdf", 2): ["Table of Contents"],
    ("Q126_EarningsRelease.pdf", 3): [
        "Non-GAAP and Other Financial Measures",
        "Adjusting Items",
    ],
    ("Q226_EarningsRelease.pdf", 6): ["Caution Regarding Forward-Looking Statements"],
    ("Q326_EarningsRelease.pdf", 5): [
        "Summary of Reported and Adjusted Results by Operating Segment"
    ],
    ("transcript_2026BMOInvestoDayTranscript.pdf", 2): ["Presentation", "Darryl White"],
}


@dataclass(frozen=True)
class Section:
    title: str
    level: int
    page: int


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\ufffd", "'")
    value = value.replace("\u00ad", "").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def flatten_outline(reader: PdfReader) -> list[Section]:
    sections: list[Section] = []

    def visit(items: list[Any], level: int) -> None:
        for item in items:
            if isinstance(item, list):
                visit(item, level + 1)
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:
                continue
            title = clean_text(str(getattr(item, "title", item)))
            if title:
                sections.append(Section(title=title, level=level, page=page))

    try:
        visit(reader.outline, 0)
    except Exception:
        return []
    return sections


def outline_paths(page_count: int, sections: list[Section]) -> dict[int, list[Section]]:
    by_page: dict[int, list[Section]] = {}
    stack: list[Section] = []
    cursor = 0
    for page in range(1, page_count + 1):
        while cursor < len(sections) and sections[cursor].page <= page:
            section = sections[cursor]
            stack = stack[: section.level]
            stack.append(section)
            cursor += 1
        by_page[page] = list(stack)
    return by_page


def inferred_page_title(text: str, page: int) -> str | None:
    """Conservatively infer a heading from the first non-boilerplate lines."""
    lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]
    for line in lines[:10]:
        if len(line) < 3 or len(line) > 120:
            continue
        if re.fullmatch(r"(?:page\s+)?\d+", line, re.IGNORECASE):
            continue
        if re.search(r"bmo financial group|bank of montreal", line, re.IGNORECASE) and len(line) < 35:
            continue
        words = line.split()
        letters = sum(character.isalpha() for character in line)
        if letters < 3 or len(words) > 15:
            continue
        if line.endswith(('.', ';', ',')) and len(words) > 7:
            continue
        return line
    return f"Page {page}" if lines else None


def split_page_text(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    lines = [line for line in text.splitlines() if line]
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for line in lines:
        projected = current_length + len(line) + (1 if current else 0)
        if current and projected > MAX_CHUNK_CHARS:
            chunks.append("\n".join(current))
            current = []
            current_length = 0
        current.append(line)
        current_length += len(line) + (1 if current_length else 0)
    if current:
        chunks.append("\n".join(current))
    return chunks


def section_context(path: list[Section], method: str) -> dict[str, Any]:
    if not path:
        return {"parent_section": None, "ancestry": [], "annotation_method": method}
    return {
        "parent_section": path[-1].title,
        "ancestry": [section.title for section in path],
        "annotation_method": method,
    }


def tags_for(text: str, path: list[Section], chunk_index: int, chunk_count: int) -> list[str]:
    tags: set[str] = set()
    if not path:
        tags.add("no_section_parent")
    if len(path) >= 3:
        tags.add("deep_ancestry")
    if len(text) < 220:
        tags.add("short_chunk")
    if len(text) > 900:
        tags.add("long_chunk")
    numeric = sum(character.isdigit() for character in text) / max(len(text), 1)
    if numeric >= 0.10 or text.count("$") >= 4 or text.count("%") >= 4:
        tags.add("table_or_numeric_dense")
    list_lines = sum(
        bool(re.match(r"^(?:[•▪●\-*]|\(?\d+[.)])\s+", line)) for line in text.splitlines()
    )
    if list_lines >= 2:
        tags.add("list_or_bullets")
    if chunk_count > 1:
        tags.add("multi_chunk_page")
    if chunk_index == 0:
        tags.add("page_opening")
    if chunk_index == chunk_count - 1:
        tags.add("page_closing")
    return sorted(tags)


def make_candidates(raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    page_counts: dict[str, int] = {}
    for pdf_path in sorted(raw_dir.glob("*.pdf"), key=lambda path: path.name.lower()):
        reader = PdfReader(pdf_path)
        page_counts[pdf_path.name] = len(reader.pages)
        outline = flatten_outline(reader)
        paths = outline_paths(len(reader.pages), outline)
        has_outline = bool(outline)
        extracted: dict[int, list[str]] = {}
        inferred_titles: dict[int, str | None] = {}
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = clean_text(page.extract_text(extraction_mode="layout") or "")
            page_chunks = split_page_text(page_text)
            extracted[page_number] = page_chunks
            inferred_titles[page_number] = inferred_page_title(page_text, page_number)
            if has_outline:
                path = paths[page_number]
                method = "pdf_outline"
            else:
                reviewed = REVIEWED_SECTION_PATHS.get((pdf_path.name, page_number))
                if reviewed:
                    path = [
                        Section(title, level, page_number)
                        for level, title in enumerate(reviewed)
                    ]
                    method = "visual_reviewed"
                else:
                    title = inferred_titles[page_number]
                    path = [Section(title, 0, page_number)] if title else []
                    method = "layout_inference_review_required" if title else "none"
            for chunk_index, chunk in enumerate(page_chunks):
                candidates.append(
                    {
                        "source_document": pdf_path.name,
                        "source_pages": [page_number],
                        "chunk": chunk,
                        "expected": {"section_contexts": [section_context(path, method)]},
                        "corner_cases": tags_for(chunk, path, chunk_index, len(page_chunks)),
                    }
                )

        # Add one boundary-spanning candidate per document when adjacent pages
        # actually have distinct section paths.
        for page_number in range(1, len(reader.pages)):
            left_path = paths[page_number] if has_outline else []
            right_path = paths[page_number + 1] if has_outline else []
            if has_outline and [s.title for s in left_path] == [s.title for s in right_path]:
                continue
            left = extracted.get(page_number, [])
            right = extracted.get(page_number + 1, [])
            if not left or not right:
                continue
            if not has_outline:
                left_reviewed = REVIEWED_SECTION_PATHS.get((pdf_path.name, page_number))
                right_reviewed = REVIEWED_SECTION_PATHS.get((pdf_path.name, page_number + 1))
                if left_reviewed and right_reviewed:
                    left_path = [
                        Section(title, level, page_number)
                        for level, title in enumerate(left_reviewed)
                    ]
                    right_path = [
                        Section(title, level, page_number + 1)
                        for level, title in enumerate(right_reviewed)
                    ]
                    method = "visual_reviewed"
                else:
                    left_title = inferred_titles[page_number]
                    right_title = inferred_titles[page_number + 1]
                    if not left_title or not right_title or left_title == right_title:
                        continue
                    left_path = [Section(left_title, 0, page_number)]
                    right_path = [Section(right_title, 0, page_number + 1)]
                    method = "layout_inference_review_required"
            else:
                method = "pdf_outline"
            text = clean_text(left[-1][-500:] + "\n" + right[0][:700])
            candidates.append(
                {
                    "source_document": pdf_path.name,
                    "source_pages": [page_number, page_number + 1],
                    "chunk": text,
                    "expected": {
                        "section_contexts": [
                            section_context(left_path, method),
                            section_context(right_path, method),
                        ]
                    },
                    "corner_cases": sorted(
                        set(tags_for(text, right_path, 0, 1)) | {"cross_page", "multiple_sections"}
                    ),
                }
            )
            break
    return candidates, page_counts


def sample_candidates(
    candidates: list[dict[str, Any]], target_size: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, tuple[int, ...], str]] = set()

    def key(row: dict[str, Any]) -> tuple[str, tuple[int, ...], str]:
        digest = hashlib.sha256(row["chunk"].encode("utf-8")).hexdigest()
        return row["source_document"], tuple(row["source_pages"]), digest

    def add(row: dict[str, Any]) -> None:
        row_key = key(row)
        if row_key not in selected_keys and len(selected) < target_size:
            selected.append(row)
            selected_keys.add(row_key)

    by_document: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_document.setdefault(row["source_document"], []).append(row)

    # Guarantee broad corpus coverage before satisfying edge-case quotas. A
    # single row is enough for documents whose hierarchy must be inferred;
    # native-outline documents provide the bulk of the strict gold records.
    for document in sorted(by_document):
        pool = list(by_document[document])
        rng.shuffle(pool)
        has_native_outline = any(
            context["annotation_method"] == "pdf_outline"
            for row in pool
            for context in row["expected"]["section_contexts"]
        )
        minimum = 3 if has_native_outline else 1
        for row in pool[:minimum]:
            add(row)

    quotas = {
        "multiple_sections": 8,
        "deep_ancestry": 10,
        "table_or_numeric_dense": 12,
        "list_or_bullets": 5,
        "short_chunk": 5,
        "long_chunk": 10,
    }
    for tag, quota in quotas.items():
        current = sum(tag in row["corner_cases"] for row in selected)
        pool = [row for row in candidates if tag in row["corner_cases"]]
        rng.shuffle(pool)
        pool.sort(
            key=lambda row: all(
                context["annotation_method"] != "pdf_outline"
                for context in row["expected"]["section_contexts"]
            )
        )
        for row in pool:
            if current >= quota:
                break
            before = len(selected)
            add(row)
            current += int(len(selected) > before)

    remainder = list(candidates)
    rng.shuffle(remainder)
    remainder.sort(
        key=lambda row: all(
            context["annotation_method"] != "pdf_outline"
            for context in row["expected"]["section_contexts"]
        )
    )
    for row in remainder:
        add(row)
        if len(selected) == target_size:
            break
    if len(selected) != target_size:
        raise RuntimeError(f"Only {len(selected)} unique examples available; need {target_size}")

    rng.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["id"] = f"section-aware-{index:03d}"
        row["chunk_sha256"] = hashlib.sha256(row["chunk"].encode("utf-8")).hexdigest()
    return selected


def write_outputs(
    rows: list[dict[str, Any]], page_counts: dict[str, int], output_dir: Path, seed: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "section_aware_chunking_golden.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    methods = Counter(
        context["annotation_method"]
        for row in rows
        for context in row["expected"]["section_contexts"]
    )
    tags = Counter(tag for row in rows for tag in row["corner_cases"])
    documents = Counter(row["source_document"] for row in rows)
    manifest = {
        "schema_version": "1.0.0",
        "dataset": jsonl_path.name,
        "record_count": len(rows),
        "random_seed": seed,
        "source_directory": "data/raw",
        "source_pdf_count": len(page_counts),
        "source_page_count": sum(page_counts.values()),
        "documents": dict(sorted(documents.items())),
        "annotation_methods": dict(sorted(methods.items())),
        "corner_case_counts": dict(sorted(tags.items())),
        "generator": "scripts/generate_section_chunking_golden.py",
    }
    (output_dir / "section_aware_chunking_golden.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/golden"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--size", type=int, default=TARGET_SIZE)
    args = parser.parse_args()
    candidates, page_counts = make_candidates(args.raw_dir)
    rows = sample_candidates(candidates, args.size, args.seed)
    write_outputs(rows, page_counts, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
