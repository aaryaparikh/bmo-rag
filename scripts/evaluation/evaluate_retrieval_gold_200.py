"""Evaluate a lexical baseline on the 200-question retrieval gold set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_retrieval_gold import load_corpus
from evaluate_retrieval_gold import bm25_rank, build_index


ROOT = Path(__file__).resolve().parents[2]


def evaluate(records: list[dict], corpus: list[dict], include_headings: bool) -> dict:
    answerable = [row for row in records if row["expected_chunks"]]
    totals = {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr@10": 0.0}
    index = build_index(corpus, include_headings)
    for record in answerable:
        ranked = bm25_rank(corpus, record["question"], index)
        gold = {item["chunk_id"] for item in record["expected_chunks"]}
        first = next((rank for rank, cid in enumerate(ranked[:10], 1) if cid in gold), None)
        if first:
            totals["mrr@10"] += 1 / first
        for cutoff in (1, 3, 5, 10):
            totals[f"hit@{cutoff}"] += float(bool(gold.intersection(ranked[:cutoff])))
    return {
        "evaluated_answerable_count": len(answerable),
        "excluded_empty_gold_count": len(records) - len(answerable),
        **{key: round(value / len(answerable), 4) for key, value in totals.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/golden/retrieval_golden_200.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/golden/retrieval_golden_200_baseline.json")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines()]
    corpus, _ = load_corpus()
    report = {
        "dataset": args.dataset.name,
        "record_count": len(records),
        "note": "Empty-gold abstention cases are excluded because an always-ranking BM25 retriever has no abstention threshold.",
        "baselines": {
            "bm25_body_only": evaluate(records, corpus, False),
            "bm25_headings_plus_body": evaluate(records, corpus, True),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
