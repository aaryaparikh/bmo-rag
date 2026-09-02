from __future__ import annotations

from bmo_rag.retrieval.sources import resolve_source_constraints

CATALOG = {
    "CorporateFactSheet": "CorporateFactSheet.pdf",
    "bmo_ar2025": "bmo_ar2025.pdf",
    "transcript_2026BMOInvestoDayTranscript": (
        "transcript_2026BMOInvestoDayTranscript.pdf"
    ),
    "BMOInvestorPresentationEN": "BMOInvestorPresentationEN.pdf",
    "Q226_EarningsRelease": "Q226_EarningsRelease.pdf",
}


def test_resolves_corporate_fact_sheet() -> None:
    result = resolve_source_constraints(
        "As of Q2 2026, what CET1 ratio is shown in BMO's Corporate Fact Sheet?",
        CATALOG,
    )
    assert result[0].label == "Corporate Fact Sheet"
    assert result[0].source_ids == ("CorporateFactSheet",)


def test_resolves_annual_report_and_investor_day_as_separate_groups() -> None:
    result = resolve_source_constraints(
        "Compare the 2025 Annual Report with BMO Investor Day 2026.", CATALOG
    )
    assert [group.label for group in result] == [
        "2025 Annual Report",
        "BMO Investor Day 2026",
    ]
    assert result[0].source_ids == ("bmo_ar2025",)
    assert set(result[1].source_ids) == {
        "transcript_2026BMOInvestoDayTranscript",
        "BMOInvestorPresentationEN",
    }


def test_resolves_generic_filename_spelling() -> None:
    result = resolve_source_constraints(
        "Use Q226 Earnings Release for this answer.", CATALOG
    )
    assert result[0].source_ids == ("Q226_EarningsRelease",)
