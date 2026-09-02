"""Resolve explicit report names in questions to indexed source IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceConstraint:
    label: str
    source_ids: tuple[str, ...]


def resolve_source_constraints(
    question: str, catalog: dict[str, str | None]
) -> list[SourceConstraint]:
    """Return source groups explicitly named by the user, without guessing a source."""
    normalized_question = _words(question)
    compact_question = _compact(question)
    groups: list[SourceConstraint] = []

    annual_years = {
        match.group(1) or match.group(2)
        for match in re.finditer(
            r"(?:\b(20\d{2})\s+annual\s+report\b|\bannual\s+report\s+(20\d{2})\b)",
            normalized_question,
        )
    }
    for year in sorted(annual_years):
        matches = tuple(
            source_id
            for source_id, filename in catalog.items()
            if _is_annual_report(source_id, filename, year)
        )
        _append_group(groups, f"{year} Annual Report", matches)

    if "corporate fact sheet" in normalized_question:
        matches = tuple(
            source_id
            for source_id, filename in catalog.items()
            if "corporatefactsheet" in _compact(f"{source_id} {filename or ''}")
        )
        _append_group(groups, "Corporate Fact Sheet", matches)

    investor_day = re.search(r"\binvestor\s+day(?:\s+(20\d{2}))?\b", normalized_question)
    if investor_day:
        year = investor_day.group(1)
        matches = tuple(
            source_id
            for source_id, filename in catalog.items()
            if _is_investor_day_source(source_id, filename)
            and (not year or year in f"{source_id} {filename or ''}" or "presentation" in source_id.lower())
        )
        label = f"BMO Investor Day {year}" if year else "BMO Investor Day"
        _append_group(groups, label, matches)

    # Generic exact-name matching supports filenames such as Q226_EarningsRelease.pdf.
    already_grouped = {source_id for group in groups for source_id in group.source_ids}
    for source_id, filename in catalog.items():
        if source_id in already_grouped:
            continue
        aliases = _aliases(source_id, filename)
        matched = next(
            (
                alias
                for alias in aliases
                if len(_compact(alias)) >= 8 and _compact(alias) in compact_question
            ),
            None,
        )
        if matched:
            _append_group(groups, matched, (source_id,))
    return groups


def _append_group(
    groups: list[SourceConstraint], label: str, source_ids: tuple[str, ...]
) -> None:
    unique = tuple(dict.fromkeys(source_ids))
    if unique and not any(set(unique) == set(group.source_ids) for group in groups):
        groups.append(SourceConstraint(label=label, source_ids=unique))


def _is_annual_report(source_id: str, filename: str | None, year: str) -> bool:
    value = _compact(f"{source_id} {filename or ''}")
    return year in value and ("annualreport" in value or f"ar{year}" in value)


def _is_investor_day_source(source_id: str, filename: str | None) -> bool:
    value = _compact(f"{source_id} {filename or ''}")
    return "investorday" in value or "investoday" in value or "investorpresentation" in value


def _aliases(source_id: str, filename: str | None) -> set[str]:
    values = {source_id, filename or ""}
    aliases: set[str] = set()
    for value in values:
        stem = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
        readable = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
        readable = re.sub(r"[_-]+", " ", readable)
        aliases.add(_words(readable))
        quarter = re.search(r"\bq([1-4])(\d{2})\b", _words(readable))
        if quarter:
            aliases.add(
                _words(readable).replace(
                    quarter.group(0), f"q{quarter.group(1)} 20{quarter.group(2)}"
                )
            )
    return {alias for alias in aliases if alias}


def _words(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.casefold())).strip()


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())
