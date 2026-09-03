"""Retrieval metrics for qrel-style golden datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def load_golden(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evidence_groups(record: Mapping[str, Any]) -> list[set[str]]:
    """Return one acceptable-ID set for each independently required evidence item.

    ``chunk_id`` remains the canonical, human-reviewed anchor.  ``equivalent_chunks``
    contains alternate corpus chunks that carry the same evidence.  Keeping the
    alternatives inside the group avoids the two common scoring errors: treating a
    valid alternate chunk as irrelevant and treating ten copies of one fact as ten
    separately required facts.
    """
    groups: list[set[str]] = []
    for item in record.get("expected_chunks", []):
        ids = {str(item["chunk_id"])}
        ids.update(
            str(equivalent["chunk_id"])
            for equivalent in item.get("equivalent_chunks", [])
        )
        groups.append(ids)
    return groups


def relevant_chunk_ids(record: Mapping[str, Any]) -> set[str]:
    return set().union(*evidence_groups(record)) if record.get("expected_chunks") else set()


def matched_evidence_group_indexes(
    record: Mapping[str, Any], chunk_id: str
) -> list[int]:
    return [
        index
        for index, group in enumerate(evidence_groups(record), start=1)
        if chunk_id in group
    ]


def validate_golden_alignment(
    records: Sequence[dict[str, Any]],
    corpus_chunk_ids: Sequence[str],
    manifest: Mapping[str, Any] | None = None,
) -> str:
    """Fail loudly when canonical or equivalent labels drift from the corpus."""
    corpus_ids = list(corpus_chunk_ids)
    unique_ids = set(corpus_ids)
    if len(unique_ids) != len(corpus_ids):
        raise RuntimeError("The corpus contains duplicate chunk IDs")
    labelled_ids = set().union(*(relevant_chunk_ids(record) for record in records))
    missing = sorted(labelled_ids - unique_ids)
    if missing:
        raise RuntimeError(
            f"Golden dataset references {len(missing)} missing canonical/equivalent chunk IDs; "
            f"rebuild it against the completed corpus. First missing ID: {missing[0]}"
        )
    fingerprint = hashlib.sha256("\n".join(corpus_ids).encode()).hexdigest()
    if manifest is not None:
        if manifest.get("corpus_chunk_count") != len(corpus_ids):
            raise RuntimeError(
                f"Golden manifest corpus count is {manifest.get('corpus_chunk_count')}, "
                f"but the current corpus has {len(corpus_ids)} chunks"
            )
        if manifest.get("corpus_fingerprint_sha256") != fingerprint:
            raise RuntimeError("Golden manifest fingerprint does not match the current corpus")
    return fingerprint


def validate_golden_dataset_hash(path: Path, manifest: Mapping[str, Any] | None) -> str:
    """Verify that the manifest describes the exact JSONL bytes being evaluated."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest is not None and manifest.get("dataset_sha256") != digest:
        raise RuntimeError("Golden manifest dataset hash does not match the JSONL dataset")
    return digest


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
        exact_recall = redundancy_rate = 0.0
        for record in answerable:
            groups = evidence_groups(record)
            relevant = set().union(*groups)
            canonical = {item["chunk_id"] for item in record["expected_chunks"]}
            retrieved = list(rankings[record["id"]][:k])
            relevant_retrieved = [chunk_id for chunk_id in retrieved if chunk_id in relevant]
            relevant_count = len(relevant_retrieved)
            covered_groups = {
                group_index
                for chunk_id in retrieved
                for group_index, group in enumerate(groups)
                if chunk_id in group
            }
            precision += relevant_count / k
            recall += len(covered_groups) / len(groups)
            exact_recall += len(canonical.intersection(retrieved)) / len(canonical)
            if relevant_count:
                redundancy_rate += max(0, relevant_count - len(covered_groups)) / relevant_count
            first_rank = next(
                (rank for rank, chunk_id in enumerate(retrieved, start=1) if chunk_id in relevant),
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
            "exact_chunk_recall": round(exact_recall / count, 6),
            "relevant_result_redundancy": round(redundancy_rate / count, 6),
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
