"""Run reproducible lexical retrieval baselines on the 100-query gold set."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_retrieval_gold import OUT_DIR, load_corpus


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)?", re.I)


def tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def build_index(corpus: list[dict[str, Any]], include_headings: bool) -> tuple[list[list[str]], dict[str, list[tuple[int, int]]], float]:
    documents: list[list[str]] = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, row in enumerate(corpus):
        prefix = " > ".join(row["headings"]) + " " if include_headings else ""
        doc_tokens = tokens(prefix + row["text"])
        documents.append(doc_tokens)
        for term, frequency in Counter(doc_tokens).items():
            postings[term].append((index, frequency))
    average_length = sum(map(len, documents)) / len(documents)
    return documents, postings, average_length


def bm25_rank(
    corpus: list[dict[str, Any]],
    query: str,
    index: tuple[list[list[str]], dict[str, list[tuple[int, int]]], float],
) -> list[str]:
    documents, postings, average_length = index
    scores: dict[int, float] = defaultdict(float)
    k1, b = 1.5, 0.75
    for term, query_frequency in Counter(tokens(query)).items():
        entries = postings.get(term, [])
        if not entries:
            continue
        document_frequency = len(entries)
        idf = math.log(1 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5))
        for index, term_frequency in entries:
            length_norm = 1 - b + b * len(documents[index]) / average_length
            scores[index] += query_frequency * idf * (
                term_frequency * (k1 + 1) / (term_frequency + k1 * length_norm)
            )
    ranked = sorted(range(len(corpus)), key=lambda index: (-scores.get(index, 0.0), index))
    return [corpus[index]["chunk_id"] for index in ranked]


def evaluate(records: list[dict[str, Any]], corpus: list[dict[str, Any]], include_headings: bool) -> dict[str, float]:
    totals: Counter[str] = Counter()
    index = build_index(corpus, include_headings)
    for record in records:
        ranking = bm25_rank(corpus, record["query"], index)
        grades = {entry["chunk_id"]: entry["relevance"] for entry in record["gold"]}
        direct = {chunk_id for chunk_id, grade in grades.items() if grade == 3}
        first_rank = next((rank for rank, chunk_id in enumerate(ranking[:10], 1) if chunk_id in direct), None)
        if first_rank:
            totals["mrr@10"] += 1 / first_rank
        for cutoff in (1, 3, 5, 10):
            hits = len(direct.intersection(ranking[:cutoff]))
            totals[f"hit@{cutoff}"] += float(hits > 0)
            totals[f"recall@{cutoff}"] += hits / len(direct)
        gains = [(2 ** grades.get(chunk_id, 0) - 1) / math.log2(rank + 1) for rank, chunk_id in enumerate(ranking[:10], 1)]
        ideal = sorted((2 ** grade - 1 for grade in grades.values()), reverse=True)[:10]
        idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
        totals["ndcg@10"] += sum(gains) / idcg if idcg else 0
    return {name: round(value / len(records), 4) for name, value in sorted(totals.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=OUT_DIR / "retrieval_golden_100.jsonl")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "retrieval_baseline_metrics.json")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines()]
    corpus, _ = load_corpus()
    report = {
        "dataset": args.dataset.name,
        "record_count": len(records),
        "corpus_chunk_count": len(corpus),
        "relevance_threshold": 3,
        "baselines": {
            "bm25_body_only": evaluate(records, corpus, include_headings=False),
            "bm25_hierarchy_enriched": evaluate(records, corpus, include_headings=True),
        },
        "metric_definitions": {
            "hit@k": "fraction of queries with direct-answer evidence in the top k",
            "recall@k": "mean fraction of direct-answer chunks found in the top k",
            "mrr@10": "mean reciprocal rank of the first direct-answer chunk, capped at 10",
            "ndcg@10": "graded ranking quality using relevance 3 for direct and 1 for supporting chunks",
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
