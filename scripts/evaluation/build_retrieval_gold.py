"""Build and validate a 100-query retrieval gold set from Docling chunks.

The curation rules are intentionally explicit.  Each seed names the source, a natural
query, and either a heading or text anchor.  Anchors are resolved against the current
corpus and the emitted qrels use content-derived IDs, so line reordering is harmless and
content drift fails loudly.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CHUNK_DIR = ROOT / "data" / "processed" / "docling"
OUT_DIR = ROOT / "data" / "golden"


# source, query, heading regex, text regex. Four cases per PDF and three per URL.
SEEDS: list[tuple[str, str, str | None, str | None]] = [
    ("2026BFCBBNAStressTest", "What entities are included when the stress-test disclosure refers to the Companies?", "Overview", "collectively referred"),
    ("2026BFCBBNAStressTest", "How severe is the housing and commercial-real-estate shock in the 2026 severely adverse scenario?", "Scenario Overview", "30% drop in housing"),
    ("2026BFCBBNAStressTest", "What were BFC's actual and minimum stressed CET1 capital ratios through Q1 2028?", "Scenario Estimates", "14.0% 12.0% 12.0%"),
    ("2026BFCBBNAStressTest", "Which major risk types are captured in the Companies' capital adequacy stress test?", "Material Risks", "credit and counterparty risk"),
    ("Bail_In_TLAC_Disclosure", "What powers can CDIC exercise if BMO has ceased or is about to cease to be viable?", "Canadian Bank Resolution", "temporary control"),
    ("Bail_In_TLAC_Disclosure", "Who determines the timing and terms of a bail-in conversion?", "Bail-in Conversion", "CDIC determines"),
    ("Bail_In_TLAC_Disclosure", "Which holders can use the CDIC Act compensation process after a bail-in conversion?", "Compensation Regime", "immediately prior"),
    ("Bail_In_TLAC_Disclosure", "What loss-absorbing capacity must BMO maintain under OSFI's TLAC Guideline?", "TLAC Guideline", "minimum capacity"),
    ("bmo_AIF2025", "When did BMO complete the Bank of the West acquisition and what was the cash purchase price?", "Three-Year History", "US$13.8 billion"),
    ("bmo_AIF2025", "Where are Bank of Montreal's head office and executive offices located?", "CORPORATE STRUCTURE", "129 rue Saint Jacques"),
    ("bmo_AIF2025", "How many full-time-equivalent employees and bank branches did BMO have at October 31, 2025?", "Business", "53,000"),
    ("bmo_AIF2025", "How does the 2025 AIF describe competition in Canadian financial services?", "Competition", None),
    ("bmo_ar2025", "How large was BMO by assets and approximately how many clients did it serve in 2025?", "Canadian Personal", "1.5 trillion"),
    ("bmo_ar2025", "What products and services does BMO Wealth Management offer?", "Wealth Management", "planning, growing"),
    ("bmo_ar2025", "How does BMO describe its digital-first, future-ready strategy?", "OUR STRATEGY", "future-ready"),
    ("bmo_ar2025", "What are BMO's four integrated operating segments?", "Canadian Personal", "four integrated operating segments"),
    ("BMOInvestorPresentationEN", "What are the main components of BMO's strategy in the Q3 2026 investor presentation?", "Our Strategy", "digital-first"),
    ("BMOInvestorPresentationEN", "What recognition and employee-giving results did BMO cite for stronger communities?", "For Stronger Communities", "36 million"),
    ("BMOInvestorPresentationEN", "What business strengths does BMO say support resilient and robust earnings?", "Diversified and competitively", "commercial banking franchise"),
    ("BMOInvestorPresentationEN", "What were BMO's Q3 2026 financial highlights?", "Q3'26 Financial Highlights", None),
    ("CorporateFactSheet", "How does BMO describe its size, age and client base in the Q3 2026 corporate fact sheet?", "About BMO", "thirteen million"),
    ("CorporateFactSheet", "What assets, CET1 ratio, employee count and deposits are listed for Q3 2026?", "Key Metrics", "Assets ($B)"),
    ("CorporateFactSheet", "What strategic priorities are listed in BMO's Q3 2026 corporate fact sheet?", "Strategic Priorities", "AI-powered"),
    ("CorporateFactSheet", "What dividend, dividend yield and market capitalization are shown in shareholder information?", "Shareholder Information", "175.3B"),
    ("LCR_CY26Q1", "Which institutions and regulators are covered by BFC's Q1 2026 liquidity disclosure?", "Introduction", "Intermediate Holding"),
    ("LCR_CY26Q1", "How does BFC's liquidity risk framework address commitments during stress?", "Liquidity Risk Management", "times of stress"),
    ("LCR_CY26Q1", "What tailored LCR factor applied to BFC for the quarter ended March 31, 2026?", "Liquidity Coverage Ratio", "85%"),
    ("LCR_CY26Q1", "What do unweighted and weighted amounts represent in BFC's Q1 LCR quantitative disclosure?", "LCR Quantitative", "unweighted amounts"),
    ("LCR_CY26Q2", "Which institutions and regulators are covered by BFC's Q2 2026 liquidity disclosure?", "Introduction", "Intermediate Holding"),
    ("LCR_CY26Q2", "How does BFC optimize liquidity for current and future needs?", "Liquidity Risk Management", "optimized"),
    ("LCR_CY26Q2", "What tailored LCR factor applied to BFC for the quarter ended June 30, 2026?", "Liquidity Coverage Ratio", "85%"),
    ("LCR_CY26Q2", "What do unweighted and weighted amounts represent in BFC's Q2 LCR quantitative disclosure?", "LCR Quantitative", "unweighted amounts"),
    ("MainFeaturesTemplateQ326", "What is the identifier and governing law for Bank of Montreal common shares?", None, "063671101"),
    ("MainFeaturesTemplateQ326", "How are Bank of Montreal common shares treated under Basel III and TLAC?", None, "Common Equity Tier 1"),
    ("MainFeaturesTemplateQ326", "What does convertible mean in the regulatory capital main-features table?", "Main Features", "interpreted to mean"),
    ("MainFeaturesTemplateQ326", "Which fields does BMO's main-features template use to describe a regulatory capital instrument?", "Main Features", "Unique identifier"),
    ("NSFRCY26Q2", "Which quarters are covered by BFC's 2026 Net Stable Funding Ratio disclosure?", "Net Stable Funding Ratio Disclosure", "March 31, 2026"),
    ("NSFRCY26Q2", "How does BFC's liquidity framework manage needs across entities, businesses and currencies?", "Liquidity Risk Management", "across legal entities"),
    ("NSFRCY26Q2", "What tailored NSFR factor applies to BFC?", "Net Stable Funding Ratio", "85%"),
    ("NSFRCY26Q2", "What are BFC's primary sources of available stable funding?", "Net Stable Funding Ratio", "ASF primarily"),
    ("Q126_EarningsRelease", "What reported and adjusted net income did BMO post in Q1 2026?", "Financial Results Highlights", "2,489"),
    ("Q126_EarningsRelease", "What were BMO's reported and adjusted EPS in Q1 2026?", "Financial Results Highlights", "3.39"),
    ("Q126_EarningsRelease", "How did Canadian P&C net income change in Q1 2026?", "Canadian P&C", "948 million"),
    ("Q126_EarningsRelease", "How did U.S. Banking perform in Q1 2026 in Canadian and U.S. dollars?", "U.S. Banking", "742 million"),
    ("Q126_ReportToShareholders", "For what period were BMO's Q1 2026 interim financial statements prepared?", "REPORT TO SHAREHOLDERS", "January 31, 2026"),
    ("Q126_ReportToShareholders", "What were BMO's Q1 2026 PCL, ROE and CET1 ratio?", "Financial Results Highlights", "746 million"),
    ("Q126_ReportToShareholders", "What sources comprise the Enhanced Disclosure Task Force index for Q1 2026?", "Enhanced Disclosure", "Supplemental Financial"),
    ("Q126_ReportToShareholders", "As of what date is the Q1 2026 MD&A commentary?", "Management's Discussion", "February 25, 2026"),
    ("Q226_EarningsRelease", "What reported and adjusted net income did BMO post in Q2 2026?", "Financial Results Highlights", "2,630"),
    ("Q226_EarningsRelease", "What dividend did BMO declare in Q2 2026 and how much did it increase?", "Financial Results Highlights", "1.71"),
    ("Q226_EarningsRelease", "What were BMO's year-to-date 2026 reported and adjusted EPS?", "Year-to-Date", "6.92"),
    ("Q226_EarningsRelease", "How did Canadian P&C net income change in Q2 2026?", "Canadian P&C", "884 million"),
    ("Q226_ReportToShareholders", "For what period were BMO's Q2 2026 interim financial statements prepared?", "REPORT TO SHAREHOLDERS", "April 30, 2026"),
    ("Q226_ReportToShareholders", "What were BMO's Q2 2026 reported EPS, adjusted EPS and CET1 ratio?", "Financial Results Highlights", "3.53"),
    ("Q226_ReportToShareholders", "What were BMO's year-to-date 2026 net income and PCL?", "Year-to-Date", "5,119"),
    ("Q226_ReportToShareholders", "What sources comprise the Enhanced Disclosure Task Force index for Q2 2026?", "Enhanced Disclosure", "Supplemental Regulatory"),
    ("Q326_EarningsRelease", "What reported and adjusted net income did BMO post in Q3 2026?", "Financial Results Highlights", "1,750"),
    ("Q326_EarningsRelease", "What were BMO's reported and adjusted EPS in Q3 2026?", "Financial Results Highlights", "2.38"),
    ("Q326_EarningsRelease", "How did Canadian P&C net income change in Q3 2026?", "Canadian P&C", "980 million"),
    ("Q326_EarningsRelease", "How did U.S. Banking perform in Q3 2026 in Canadian and U.S. dollars?", "U.S. Banking", "868 million"),
    ("Q326_ReportToShareholders", "For what period were BMO's Q3 2026 interim financial statements prepared?", "REPORT TO SHAREHOLDERS", "July 31, 2026"),
    ("Q326_ReportToShareholders", "What were BMO's Q3 2026 PCL, ROE and CET1 ratio?", "Financial Results Highlights", "722 million"),
    ("Q326_ReportToShareholders", "What deposits, loans and premium are involved in the planned sale of 138 U.S. branches?", "Divestitures", "US$5.3 billion"),
    ("Q326_ReportToShareholders", "What business did BMO agree to acquire from Euroz Hartleys in June 2026?", "Acquisitions", "metals and mining"),
    ("RegSuppQ326", "How should BMO's Q3 2026 supplementary regulatory capital information be used?", "Use of this Document", "used in conjunction"),
    ("RegSuppQ326", "Which frameworks and OSFI guidelines determine BMO's regulatory capital disclosures?", "Regulatory Framework", "Basel III"),
    ("RegSuppQ326", "Which insurance subsidiaries are excluded from BMO's regulatory-scope balance sheet?", "CC2", "BMO Life Insurance"),
    ("RegSuppQ326", "What products are offered by BMO Life Insurance Company?", "CC2", "critical illness"),
    ("Suppq126", "What four operating segments does BMO use in its Q1 2026 supplemental information?", "Operating Segment Results", "four operating segments"),
    ("Suppq126", "Why does BMO's Q1 2026 supplemental package present both reported and adjusted results?", "Adjusted measures", "assessing underlying"),
    ("Suppq126", "What businesses are excluded from the net interest margin measure in BMO's Q1 2026 supplement?", "Net Interest Margin", "Global Markets"),
    ("Suppq126", "Which report and annual report should be read with the Q1 supplemental package?", "Return on Equity", "First Quarter 2026"),
    ("Suppq226", "What four operating segments does BMO use in its Q2 2026 supplemental information?", "Operating Segment Results", "four operating segments"),
    ("Suppq226", "Why does BMO's Q2 2026 supplemental package present both reported and adjusted results?", "Adjusted measures", "assessing underlying"),
    ("Suppq226", "What businesses are excluded from the net interest margin measure in BMO's Q2 2026 supplement?", "Net Interest Margin", "Global Markets"),
    ("Suppq226", "Which report and annual report should be read with the Q2 supplemental package?", "Return on Equity", "Second Quarter 2026"),
    ("SuppQ326", "What four operating segments does BMO use in its Q3 2026 supplemental information?", "Operating Segment Results", "four operating segments"),
    ("SuppQ326", "Why does BMO's Q3 2026 supplemental package present both reported and adjusted results?", "Adjusted measures", "assessing underlying"),
    ("SuppQ326", "What businesses are excluded from the net interest margin measure in BMO's Q3 2026 supplement?", "Net Interest Margin", "Global Markets"),
    ("SuppQ326", "Which report and annual report should be read with the Q3 supplemental package?", "Return on Equity", "Third Quarter 2026"),
    ("sustainability_and_climate_report_interactive", "How does BMO connect its sustainability priorities to growth and risk management?", "Executive Committee Sponsor", "opportunities for growth"),
    ("sustainability_and_climate_report_interactive", "What does BMO disclose about responsible artificial intelligence?", "Responsible artificial intelligence", None),
    ("sustainability_and_climate_report_interactive", "What does BMO report about Indigenous partnerships?", "Indigenous partnerships", None),
    ("sustainability_and_climate_report_interactive", "How does BMO describe its financed-emissions measurement and targets?", "Financed emissions", None),
    ("transcript_2026BMOInvestoDayTranscript", "Who welcomed attendees to BMO's 2026 Investor Day and what anniversary did she mention?", "Christine Viau", "30th anniversary"),
    ("transcript_2026BMOInvestoDayTranscript", "Where did BMO host its 2026 Investor Day?", "Darryl White", "downtown Toronto"),
    ("transcript_2026BMOInvestoDayTranscript", "What sustainable ROE objective did Darryl White describe at Investor Day?", "March 26, 2026", "15% plus"),
    ("transcript_2026BMOInvestoDayTranscript", "How does BMO define the time horizon and basis for its medium-term targets?", "Non-GAAP Measures", "Fiscal 2028"),
    ("2026-05-27-BMO-Financial-Group-Reports-Second-Quarter-2026-Results-ed105dc24d", "What reported and adjusted net income did BMO announce for Q2 2026?", "Second Quarter 2026 compared", "2,630"),
    ("2026-05-27-BMO-Financial-Group-Reports-Second-Quarter-2026-Results-ed105dc24d", "How did BMO's U.S. Banking business perform in Q2 2026?", "U.S. Banking", None),
    ("2026-05-27-BMO-Financial-Group-Reports-Second-Quarter-2026-Results-ed105dc24d", "When was BMO's Q2 2026 conference call and webcast scheduled?", "Quarterly Conference", None),
    ("financial-information-c5dc87e0c5", "Which quarterly investor materials does BMO's 2026 financial-information page provide?", "Quarterly Results", "Earnings Release"),
    ("financial-information-c5dc87e0c5", "What historical IFRS information did BMO release on February 7, 2014?", "Quarterly Results", "Supplementary Financial Information"),
    ("financial-information-c5dc87e0c5", "Where does BMO direct investors for quarterly reports, presentations and transcripts?", "Financial Information", None),
    ("investor-day-2026-609c901e12", "When and where was BMO Investor Day 2026 held?", "Toronto, ON", None),
    ("investor-day-2026-609c901e12", "Who presented the U.S. Banking session at BMO Investor Day 2026?", "Agenda", "Aron Levine"),
    ("investor-day-2026-609c901e12", "Who presented the Risk and Financial Overview at BMO Investor Day 2026?", "Agenda", "Piyush Agrawal"),
    ("d938207d40f-e11c68d11f", "What fiscal year and SEC commission file number are shown on BMO's 2025 Form 40-F?", "40-F", "001-13354"),
    ("d938207d40f-e11c68d11f", "What is Bank of Montreal's U.S. agent-for-service address?", "BANK OF MONTREAL", "320 S. Canal"),
    ("d938207d40f-e11c68d11f", "How many BMO common shares were outstanding at the end of the Form 40-F period?", "BANK OF MONTREAL", "708,905,679"),
]


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def chunk_id(source: str, text: str) -> str:
    digest = hashlib.sha256(f"{source}\0{norm(text)}".encode()).hexdigest()[:20]
    return f"bmo-{digest}"


def pages(meta: dict[str, Any]) -> list[int]:
    found: set[int] = set()
    for item in meta.get("doc_items", []):
        for prov in item.get("prov", []):
            if isinstance(prov.get("page_no"), int):
                found.add(prov["page_no"])
    return sorted(found)


def load_corpus() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    corpus: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(CHUNK_DIR.glob("*.chunks.jsonl")):
        source = path.name.removesuffix(".chunks.jsonl")
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            raw = json.loads(line)
            meta = raw.get("meta", {})
            item = {
                "chunk_id": chunk_id(source, raw["text"]),
                "source_id": source,
                "chunk_index": index,
                "text": norm(raw["text"]),
                "headings": [norm(str(h)) for h in meta.get("headings", []) if norm(str(h))],
                "pages": pages(meta),
                "source_url": meta.get("source_url"),
                "origin_filename": (meta.get("origin") or {}).get("filename"),
            }
            corpus.append(item)
            by_source[source].append(item)
    return corpus, by_source


def resolve(seed: tuple[str, str, str | None, str | None], rows: list[dict[str, Any]]) -> dict[str, Any]:
    source, query, heading_pattern, text_pattern = seed
    candidates = rows
    if heading_pattern:
        rx = re.compile(heading_pattern, re.I)
        candidates = [row for row in candidates if rx.search(" > ".join(row["headings"]))]
    if text_pattern:
        candidates = [
            row for row in candidates
            if text_pattern.casefold() in row["text"].casefold()
        ]
    if not candidates:
        raise ValueError(f"Unresolved seed: {source}: {query}")
    # Prefer a self-contained passage; deterministic tie breaking by corpus order.
    selected = max(candidates, key=lambda row: (min(len(row["text"]), 1200), -row["chunk_index"]))
    return selected


def quote(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 300 else limit] + "…"


def build() -> None:
    if len(SEEDS) != 100:
        raise ValueError(f"Expected exactly 100 seeds, found {len(SEEDS)}")
    corpus, by_source = load_corpus()
    records: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for number, seed in enumerate(SEEDS, 1):
        source, query, _, _ = seed
        chosen = resolve(seed, by_source[source])
        source_counts[source] += 1
        same_heading = [
            row for row in by_source[source]
            if row["chunk_id"] != chosen["chunk_id"] and row["headings"] == chosen["headings"]
        ][:3]
        negatives = [
            row for row in by_source[source]
            if row["chunk_id"] != chosen["chunk_id"] and row["headings"] != chosen["headings"]
        ]
        negatives.sort(key=lambda row: abs(len(row["text"]) - len(chosen["text"])))
        split = "test" if number % 5 == 0 else "development"
        records.append({
            "id": f"bmo-retrieval-{number:03d}",
            "query": query,
            "query_type": "table_lookup" if re.search(r"what (?:were|was|is)|how many|how large|which quarters", query, re.I) and re.search(r"\d", chosen["text"]) else "factual",
            "difficulty": "hard" if same_heading else "medium",
            "split": split,
            "gold": [{
                "chunk_id": chosen["chunk_id"],
                "relevance": 3,
                "source_id": source,
                "source_type": "url" if chosen["source_url"] else "pdf",
                "source_locator": chosen["source_url"] or f"data/raw/{chosen['origin_filename']}",
                "pages": chosen["pages"],
                "headings": chosen["headings"],
                "evidence": quote(chosen["text"]),
                "text_sha256": hashlib.sha256(chosen["text"].encode()).hexdigest(),
            }] + [{
                "chunk_id": row["chunk_id"], "relevance": 1, "source_id": source,
                "source_type": "url" if row["source_url"] else "pdf",
                "source_locator": row["source_url"] or f"data/raw/{row['origin_filename']}",
                "pages": row["pages"], "headings": row["headings"],
                "evidence": quote(row["text"], 240),
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            } for row in same_heading],
            "hard_negative_chunk_ids": [row["chunk_id"] for row in negatives[:3]],
            "annotation": {"method": "manual_query_and_anchor_with_deterministic_resolution", "status": "gold"},
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = OUT_DIR / "retrieval_golden_100.jsonl"
    dataset_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    corpus_fingerprint = hashlib.sha256("\n".join(row["chunk_id"] for row in corpus).encode()).hexdigest()
    manifest = {
        "schema_version": "1.0.0",
        "dataset": dataset_path.name,
        "record_count": len(records),
        "corpus_chunk_count": len(corpus),
        "corpus_source_count": len(by_source),
        "corpus_fingerprint_sha256": corpus_fingerprint,
        "source_distribution": dict(sorted(source_counts.items())),
        "split_distribution": dict(Counter(r["split"] for r in records)),
        "query_type_distribution": dict(Counter(r["query_type"] for r in records)),
        "relevance_scale": {"3": "direct answer evidence", "1": "same-section supporting context", "0": "not relevant"},
        "generator": "scripts/evaluation/build_retrieval_gold.py",
    }
    (OUT_DIR / "retrieval_golden_100.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} records from {len(by_source)} sources and {len(corpus)} chunks")


if __name__ == "__main__":
    build()
