"""Compact search over Docling JSONL chunks for gold-set curation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "data" / "processed" / "docling"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("--source")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    regex = re.compile(args.pattern, re.I)
    shown = 0
    for path in sorted(CHUNK_DIR.glob("*.chunks.jsonl")):
        source = path.name.removesuffix(".chunks.jsonl")
        if args.source and source != args.source:
            continue
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            headings = " > ".join(row.get("meta", {}).get("headings", []))
            haystack = headings + "\n" + row.get("text", "")
            if not regex.search(haystack):
                continue
            text = re.sub(r"\s+", " ", row.get("text", "")).strip()
            pages = sorted({
                prov["page_no"]
                for item in row.get("meta", {}).get("doc_items", [])
                for prov in item.get("prov", [])
                if isinstance(prov.get("page_no"), int)
            })
            print(f"{source}:{index} pages={pages} headings={headings!r}")
            print(text[:700])
            print()
            shown += 1
            if shown >= args.limit:
                return


if __name__ == "__main__":
    main()
