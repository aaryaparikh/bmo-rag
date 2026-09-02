"""Retrieval metrics for qrel-style golden datasets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def load_golden(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def metrics_at_k(
    records: Sequence[dict[str, Any]],
    rankings: Mapping[str, Sequence[str]],
    k_values: Sequence[int] = (5, 10, 20, 30),
) -> dict[str, Any]:
    answerable = [record for record in records if record["expected_chunks"]]
    if not answerable:
        raise ValueError("The evaluation set has no answerable records")
    cutoffs: dict[str, dict[str, float]] = {}
    for k in k_values:
        precision = recall = reciprocal_rank = hit_rate = 0.0
        for record in answerable:
            gold = {item["chunk_id"] for item in record["expected_chunks"]}
            retrieved = list(rankings[record["id"]][:k])
            relevant_count = len(gold.intersection(retrieved))
            precision += relevant_count / k
            recall += relevant_count / len(gold)
            first_rank = next(
                (rank for rank, chunk_id in enumerate(retrieved, start=1) if chunk_id in gold),
                None,
            )
            if first_rank is not None:
                reciprocal_rank += 1 / first_rank
                hit_rate += 1
        count = len(answerable)
        cutoffs[str(k)] = {
            "precision": round(precision / count, 6),
            "recall": round(recall / count, 6),
            "mrr": round(reciprocal_rank / count, 6),
            "hit_rate": round(hit_rate / count, 6),
        }
    return {
        "evaluated_answerable_count": len(answerable),
        "excluded_empty_gold_count": len(records) - len(answerable),
        "cutoffs": cutoffs,
    }


def metrics_by_split(
    records: Sequence[dict[str, Any]],
    rankings: Mapping[str, Sequence[str]],
    k_values: Sequence[int] = (5, 10, 20, 30),
) -> dict[str, Any]:
    splits = sorted({record.get("split", "unspecified") for record in records})
    return {
        "all": metrics_at_k(records, rankings, k_values),
        **{
            split: metrics_at_k(
                [record for record in records if record.get("split", "unspecified") == split],
                rankings,
                k_values,
            )
            for split in splits
        },
    }


def metrics_by_facets(
    records: Sequence[dict[str, Any]],
    rankings: Mapping[str, Sequence[str]],
    k_values: Sequence[int] = (5, 10, 20, 30),
    facet_fields: Sequence[str] = ("query_type", "difficulty", "edge_case"),
) -> dict[str, Any]:
    """Compute backward-compatible split metrics plus categorical breakdowns."""

    result = metrics_by_split(records, rankings, k_values)
    for field in facet_fields:
        groups: dict[str, Any] = {}
        values = sorted({str(record.get(field, "unspecified")) for record in records})
        for value in values:
            subset = [
                record for record in records if str(record.get(field, "unspecified")) == value
            ]
            if any(record["expected_chunks"] for record in subset):
                groups[value] = metrics_at_k(subset, rankings, k_values)
            else:
                groups[value] = {
                    "evaluated_answerable_count": 0,
                    "excluded_empty_gold_count": len(subset),
                    "cutoffs": {},
                    "not_evaluated_reason": (
                        "No records in this group have gold chunks; fixed top-k retrieval "
                        "metrics are undefined."
                    ),
                }
        result[f"by_{field}"] = groups
    return result
