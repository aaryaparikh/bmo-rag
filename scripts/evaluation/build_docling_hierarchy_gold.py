"""Create a 100-record gold set for evaluating Docling section headings.

Every ``chunk`` is copied verbatim from ``data/processed/docling/*.chunks.jsonl``.
Expected headings come from PDF bookmarks when the chunk is safely inside a bookmark
interval, or from explicit text-anchored visual adjudications for PDFs whose bookmarks
are missing or merely say ``Page N``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


SEED = 20260901
TARGET = 100
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "docling"
OUTPUT = ROOT / "data" / "golden"


@dataclass(frozen=True)
class Section:
    title: str
    level: int
    page: int


@dataclass(frozen=True)
class VisualRule:
    source: str
    page: int
    text_contains: str
    expected_heading: str
    expected_path: tuple[str, ...]
    evidence: str


# Each rule was checked against a rendered source page. Text anchors prevent the
# rule from labelling a different region on pages containing multiple headings.
VISUAL_RULES = [
    VisualRule("Bail_In_TLAC_Disclosure", 3, "The CDIC Act provides for a compensation process", "Compensation Regime", ("Compensation Regime",), "Printed heading on PDF page 3."),
    VisualRule("CorporateFactSheet", 1, "BMO Financial Group is the eighth largest bank", "About BMO", ("About BMO",), "Printed heading on PDF page 1."),
    VisualRule("LCR_CY26Q1", 4, "three months ended March 31, 2026", "3. Liquidity Coverage Ratio", ("3. Liquidity Coverage Ratio",), "Printed heading on PDF page 4."),
    VisualRule("LCR_CY26Q2", 4, "three months ended June 30, 2026", "3. Liquidity Coverage Ratio", ("3. Liquidity Coverage Ratio",), "Printed heading on PDF page 4."),
    VisualRule("MainFeaturesTemplateQ326", 1, "Included in both regulatory capital and TLAC", "Main Features Of Regulatory Capital Instruments", ("Main Features Of Regulatory Capital Instruments",), "Printed table title on PDF page 1."),
    VisualRule("NSFRCY26Q2", 4, "The following table summarizes the BFC average NSFR", "3. Net Stable Funding Ratio", ("3. Net Stable Funding Ratio",), "Printed heading on PDF page 4."),
    VisualRule("Q126_EarningsRelease", 3, "Adjusted results in the current quarter", "Adjusting Items", ("Non-GAAP and Other Financial Measures", "Adjusting Items"), "Printed heading on PDF page 3."),
    VisualRule("Q226_EarningsRelease", 6, "Bank of Montreal's public communications often include", "Caution Regarding Forward-Looking Statements", ("Caution Regarding Forward-Looking Statements",), "Printed heading on PDF page 6."),
    VisualRule("Q326_EarningsRelease", 5, "Q3-2026 Reported net income", "Summary of Reported and Adjusted Results by Operating Segment", ("Summary of Reported and Adjusted Results by Operating Segment",), "Printed heading on PDF page 5."),
    VisualRule("RegSuppQ326", 3, "supplementary information contained in this package", "Use of this Document", ("Notes to Users", "Use of this Document"), "Printed heading on PDF page 3."),
    VisualRule("Suppq126", 3, "BMO reports financial results for its four operating segments", "Operating Segment Results", ("Notes to Users", "Operating Segment Results"), "Printed heading on PDF page 3."),
    VisualRule("Suppq226", 3, "BMO reports financial results for its four operating segments", "Operating Segment Results", ("Notes to Users", "Operating Segment Results"), "Printed heading on PDF page 3."),
    VisualRule("SuppQ326", 3, "BMO reports financial results for its four operating segments", "Operating Segment Results", ("Notes to Users", "Operating Segment Results"), "Printed heading on PDF page 3."),
    VisualRule("transcript_2026BMOInvestoDayTranscript", 2, "Thank you, Christine, and good morning", "Darryl White - Bank of Montreal - CEO", ("Presentation", "Darryl White - Bank of Montreal - CEO"), "Printed speaker heading on PDF page 2."),
    VisualRule("BMOInvestorPresentationEN", 5, "Premium commercial banking franchise", "Diversified and competitively positioned businesses that deliver resilient and robust earnings", ("Reasons to invest in BMO", "Diversified and competitively positioned businesses that deliver resilient and robust earnings"), "Printed subheading on PDF page 5."),
]


def clean(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\ufffd", "'")
    return re.sub(r"\s+", " ", value).strip()


def canonical_heading(value: str | None) -> str:
    if not value:
        return ""
    value = clean(value).casefold().replace("&", "and")
    value = re.sub(r"^\d+(?:\.\d+)*[.)]?\s+", "", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def flatten_outline(reader: PdfReader) -> list[Section]:
    result: list[Section] = []

    def visit(items: list[Any], level: int) -> None:
        for item in items:
            if isinstance(item, list):
                visit(item, level + 1)
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:
                continue
            title = clean(str(getattr(item, "title", item)))
            if title:
                result.append(Section(title, level, page))

    try:
        visit(reader.outline, 0)
    except Exception:
        return []
    return result


def outline_states(sections: list[Section], page_count: int) -> dict[int, tuple[list[Section], Section | None]]:
    states: dict[int, tuple[list[Section], Section | None]] = {}
    stack: list[Section] = []
    cursor = 0
    active: Section | None = None
    for page in range(1, page_count + 1):
        while cursor < len(sections) and sections[cursor].page <= page:
            active = sections[cursor]
            stack = stack[: active.level]
            stack.append(active)
            cursor += 1
        states[page] = (list(stack), active)
    return states


def useful(title: str) -> bool:
    title = clean(title)
    return not bool(re.match(r"^(?:page\s+\d+|slide number\s+\d+|cover|index|title page)$", title, re.I)) and "template (f2" not in title.casefold()


def chunk_pages(meta: dict[str, Any]) -> list[int]:
    pages: set[int] = set()
    for item in meta.get("doc_items", []):
        for prov in item.get("prov", []):
            if isinstance(prov.get("page_no"), int):
                pages.add(prov["page_no"])
    return sorted(pages)


def load_chunks() -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(PROCESSED.glob("*.chunks.jsonl")):
        source = path.name.removesuffix(".chunks.jsonl")
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            meta = row.get("meta", {})
            filename = (meta.get("origin") or {}).get("filename")
            if not filename:
                continue  # URLs are intentionally excluded.
            by_source[source].append({
                "source": source,
                "source_document": filename,
                "chunk_index": index,
                "chunk": row["text"],
                "pages": chunk_pages(meta),
                "docling_path": [clean(str(h)) for h in meta.get("headings", []) if clean(str(h))],
            })
    return by_source


def make_record(chunk: dict[str, Any], expected_path: list[str], method: str, evidence: str) -> dict[str, Any]:
    expected_heading = expected_path[-1]
    predicted_path = chunk["docling_path"]
    text = clean(chunk["chunk"])
    required_present = canonical_heading(expected_heading) in {
        canonical_heading(heading) for heading in predicted_path
    }
    placement = "leaf" if method == "visual_reviewed" else "anywhere_in_heading_path"
    expected_canonical = [canonical_heading(heading) for heading in expected_path]
    predicted_canonical = [canonical_heading(heading) for heading in predicted_path]
    if predicted_canonical == expected_canonical:
        verdict = "exact_hierarchy"
    elif all(heading in predicted_canonical for heading in expected_canonical):
        verdict = "expected_path_present_with_additional_context"
    elif required_present:
        verdict = "expected_parent_present_in_incomplete_path"
    else:
        verdict = "expected_parent_missing"
    public_method = "visual_reviewed" if method == "visual_reviewed" else "pdf_outline"
    return {
        "source_document": chunk["source_document"],
        "chunk_index": chunk["chunk_index"],
        "source_chunk_index": chunk["chunk_index"],
        "source_pages": chunk["pages"],
        "chunk": text,
        "chunk_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "observed_docling_hierarchy": predicted_path,
        "observed_docling_heading": predicted_path[-1] if predicted_path else None,
        "expected_hierarchy": expected_path,
        "expected_section_heading": expected_heading,
        "annotation_method": public_method,
        "hierarchy_verdict": verdict,
        "docling": {
            "heading_path": predicted_path,
            "section_heading": predicted_path[-1] if predicted_path else None,
        },
        "expected": {
            "heading_path": expected_path,
            "section_heading": expected_heading,
            "placement": placement,
            "annotation_method": public_method,
            "confidence": "high",
            "evidence": evidence,
        },
        "evaluation": {
            "required_heading_present": required_present,
            "leaf_match": (
                canonical_heading(predicted_path[-1] if predicted_path else None)
                == canonical_heading(expected_heading)
                if placement == "leaf"
                else None
            ),
        },
    }


def candidates(by_source: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()

    for rule in VISUAL_RULES:
        matches = [
            row for row in by_source[rule.source]
            if rule.page in row["pages"] and rule.text_contains.casefold() in clean(row["chunk"]).casefold()
        ]
        if not matches:
            raise RuntimeError(f"Visual rule did not resolve: {rule.source}: {rule.text_contains}")
        row = min(matches, key=lambda item: item["chunk_index"])
        result.append(make_record(row, list(rule.expected_path), "visual_reviewed", rule.evidence))
        used.add((rule.source, row["chunk_index"]))

    for pdf_path in sorted(RAW.glob("*.pdf")):
        source = pdf_path.stem
        if source not in by_source:
            continue
        reader = PdfReader(pdf_path)
        sections = flatten_outline(reader)
        states = outline_states(sections, len(reader.pages))
        for row in by_source[source]:
            if (source, row["chunk_index"]) in used or not row["pages"] or len(clean(row["chunk"])) < 80:
                continue
            page_states = [states[page] for page in row["pages"] if page in states]
            if len(page_states) != len(row["pages"]):
                continue
            active = [state[1] for state in page_states]
            if not active or any(section is None for section in active):
                continue
            assert all(section is not None for section in active)
            if len({(section.title, section.page) for section in active if section}) != 1:
                continue
            section = active[0]
            assert section is not None
            if not useful(section.title):
                continue
            # On a transition page, accept only chunks that actually contain the
            # printed bookmark title; otherwise the preceding section may continue.
            if section.page in row["pages"] and canonical_heading(section.title) not in canonical_heading(row["chunk"]):
                continue
            path = [item.title for item in page_states[0][0] if useful(item.title)]
            if not path:
                continue
            result.append(make_record(row, path, "pdf_bookmark_interval", f"PDF bookmark '{section.title}' begins on page {section.page}; chunk pages remain within that interval."))
    return result


def select(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_doc[row["source_document"]].append(row)
    source_pdfs = {path.name for path in RAW.glob("*.pdf")}
    missing = source_pdfs - set(by_doc)
    if missing:
        raise RuntimeError(f"No high-confidence candidate for: {sorted(missing)}")

    for pool in by_doc.values():
        rng.shuffle(pool)
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            buckets[row["expected"]["section_heading"]].append(row)
        ordered: list[dict[str, Any]] = []
        headings = sorted(buckets)
        while any(buckets.values()):
            for heading in headings:
                if buckets[heading]:
                    ordered.append(buckets[heading].pop())
        pool[:] = sorted(
            ordered,
            key=lambda row: row["expected"]["annotation_method"] != "visual_reviewed",
        )

    selected: list[dict[str, Any]] = []
    keys: set[tuple[str, int]] = set()
    docs = sorted(by_doc)
    # First guarantee all-PDF coverage, then round-robin to prevent large reports
    # from dominating the gold set.
    while len(selected) < size:
        progress = False
        for document in docs:
            pool = by_doc[document]
            while pool:
                row = pool.pop(0)
                key = (document, row["source_chunk_index"])
                if key not in keys:
                    selected.append(row)
                    keys.add(key)
                    progress = True
                    break
            if len(selected) == size:
                break
        if not progress:
            raise RuntimeError(f"Only {len(selected)} high-confidence chunks available; need {size}")

    rng.shuffle(selected)
    for number, row in enumerate(selected, 1):
        row["id"] = f"docling-hierarchy-{number:03d}"
    return selected


def write(rows: list[dict[str, Any]], output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "docling_hierarchy_golden_100.jsonl"
    dataset.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    methods = Counter(row["annotation_method"] for row in rows)
    docs = Counter(row["source_document"] for row in rows)
    verdicts = Counter(row["hierarchy_verdict"] for row in rows)
    required_matches = sum(row["evaluation"]["required_heading_present"] for row in rows)
    visual_rows = [row for row in rows if row["expected"]["placement"] == "leaf"]
    visual_leaf_matches = sum(bool(row["evaluation"]["leaf_match"]) for row in visual_rows)
    manifest = {
        "schema_version": "1.0.0",
        "dataset": dataset.name,
        "record_count": len(rows),
        "random_seed": seed,
        "source_scope": "PDFs only; chunks copied from data/processed/docling",
        "source_pdf_count": len(docs),
        "covered_pdf_count": len(docs),
        "documents": dict(sorted(docs.items())),
        "annotation_methods": dict(sorted(methods.items())),
        "hierarchy_verdicts": dict(sorted(verdicts.items())),
        "docling_baseline": {
            "required_heading_recall": round(required_matches / len(rows), 4),
            "matched_required_headings": required_matches,
            "visual_leaf_accuracy": round(visual_leaf_matches / len(visual_rows), 4),
            "visual_leaf_examples": len(visual_rows),
        },
        "generator": "scripts/evaluation/build_docling_hierarchy_gold.py",
    }
    (output_dir / "docling_hierarchy_golden_100.manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = select(candidates(load_chunks()), args.size, args.seed)
    write(rows, args.output_dir, args.seed)
    print(f"Wrote {len(rows)} hierarchy labels to {args.output_dir}")


if __name__ == "__main__":
    main()
