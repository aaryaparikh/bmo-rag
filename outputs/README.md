# Output guide

This directory contains generated results, grouped by what question the result answers.

## `benchmarks/embedding_model_comparison/`

Use this to compare embedding models before hybrid retrieval or reranking.

- `summary.json`: aggregate equivalent-aware Precision, evidence-group Recall, MRR, Hit Rate,
  strict exact-chunk recall, and relevant-result redundancy by model and cutoff.
- `query_details/*.jsonl`: one record per question, including expected and retrieved chunks.
- `comparison_snapshots/*.json`: saved partial/model-subset runs kept for comparison.

## `benchmarks/hybrid_retrieval_comparison/`

Use this to compare retrieval strategies for BGE-M3.

- `summary.json`: aggregate dense vs. hybrid RRF vs. hybrid-plus-reranker metrics using evidence
  equivalence groups rather than only one canonical chunk ID.
- `query_details.jsonl`: question-level candidates before and after reranking.
- `facet_metrics.csv`: metrics broken down by query type, difficulty, and edge case.

## `audits/three_model_retrieval_audit/`

This is the human-review package comparing BGE-M3, Nomic Embed v1.5, and Qwen3 Embedding 0.6B.

- `reports/`: the main Excel workbook and source JSONL.
- `tables/`: analysis-ready CSV exports.
- `figures/`: rendered previews of important workbook sheets.
- `diagnostics/`: inspection metadata and error scans.

Start with `reports/retrieval_audit.xlsx` for manual review.

## `gold_exports/retrieval_golden_200/`

- `review_workbook.xlsx`: a spreadsheet export of the 200-question golden retrieval set. The
  canonical machine-readable fixtures remain under `data/golden/`.

## Legacy build workspace

The opaque-ID directory contains only the local Node dependency junction and assembly script used
to build the three-model audit workbook. It is kept in place so that runtime dependency resolution
continues to work; all user-facing results have been moved into the named folders above.
