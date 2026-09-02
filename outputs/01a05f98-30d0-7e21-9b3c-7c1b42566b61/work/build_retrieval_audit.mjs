import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(process.cwd(), "..", "..", "..");
const sourceDir = path.join(root, "outputs", "embedding_benchmark", "query_details");
const reportPath = path.join(root, "outputs", "embedding_benchmark", "bge_nomic_qwen06_facets.json");
const outputDir = path.join(root, "outputs", "01a05f98-30d0-7e21-9b3c-7c1b42566b61");
const supportDir = path.join(outputDir, "support");
const modelFiles = [
  ["bge-m3", "bge-m3.jsonl"],
  ["nomic-embed-v1.5", "nomic-embed-v1.5.jsonl"],
  ["qwen3-embedding-0.6b", "qwen3-embedding-0.6b.jsonl"],
];

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(supportDir, { recursive: true });

function parseJsonl(text) {
  return text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function joined(value, separator = " | ") {
  if (value === null || value === undefined) return "";
  return Array.isArray(value) ? value.join(separator) : String(value);
}

function cleanCell(value) {
  if (typeof value !== "string") return value;
  return value.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F]/g, "").slice(0, 32700);
}

function cleanMatrix(rows) {
  return rows.map((row) => row.map(cleanCell));
}

function csvCell(value) {
  const cleaned = cleanCell(value);
  const text = cleaned === null || cleaned === undefined ? "" : String(cleaned);
  return `"${text.replaceAll('"', '""')}"`;
}

async function writeCsv(filename, rows) {
  const text = rows.map((row) => row.map(csvCell).join(",")).join("\r\n") + "\r\n";
  await fs.writeFile(path.join(outputDir, filename), text, "utf8");
}

const report = JSON.parse(await fs.readFile(reportPath, "utf8"));
const records = [];
for (const [model, filename] of modelFiles) {
  const parsed = parseJsonl(await fs.readFile(path.join(sourceDir, filename), "utf8"));
  for (const record of parsed) records.push({ ...record, model });
}

await fs.writeFile(
  path.join(outputDir, "retrieval_audit_3_models.jsonl"),
  records.map((record) => JSON.stringify(record)).join("\n") + "\n",
  "utf8",
);

const questionHeaders = [
  "Model", "Question ID", "Question", "Query Type", "Difficulty", "Edge Case", "Split",
  "Expected Behavior", "Expected Answer", "Expected Chunk IDs", "Expected Citations", "Expected Pages",
  "First Relevant Rank", "Hit@5", "Hit@10", "Hit@20", "Hit@30",
  "Top-1 Chunk ID", "Top-1 Score", "Top-1 Expected?", "Top-1 Citation", "Top-1 Pages", "Top-1 Text",
];
const questionRows = records.map((record) => {
  const expected = record.expected_chunks ?? [];
  const top = record.returned_chunks?.[0] ?? {};
  return [
    record.model, record.question_id, record.question, record.query_type, record.difficulty, record.edge_case,
    record.split, record.expected_behavior, record.expected_answer,
    joined(expected.map((chunk) => chunk.chunk_id)), joined(expected.map((chunk) => chunk.citation)),
    joined(expected.flatMap((chunk) => chunk.pages ?? []), ", "), record.first_relevant_rank,
    record.hit_at_5, record.hit_at_10, record.hit_at_20, record.hit_at_30,
    top.chunk_id ?? "", top.score ?? null, top.is_expected ?? false, top.citation ?? "",
    joined(top.pages, ", "), top.text ?? "",
  ];
});

const expectedHeaders = [
  "Model", "Question ID", "Question", "Expected Chunk ID", "Relevance", "Source ID", "Chunk Index",
  "Citation", "Pages", "Headings", "Source Locator", "Expected Evidence", "Expected Answer",
];
const expectedRows = records.flatMap((record) => (record.expected_chunks ?? []).map((chunk) => [
  record.model, record.question_id, record.question, chunk.chunk_id, chunk.relevance ?? null,
  chunk.source_id ?? "", chunk.chunk_index ?? null, chunk.citation ?? "", joined(chunk.pages, ", "),
  joined(chunk.headings), chunk.source_locator ?? "", chunk.evidence ?? "", record.expected_answer ?? "",
]));

const retrievedHeaders = [
  "Model", "Question ID", "Question", "Rank", "Score", "Expected?", "Chunk ID", "Source ID",
  "Chunk Index", "Citation", "Pages", "Headings", "Source URL", "Origin Filename", "Returned Text",
];
const retrievedRows = records.flatMap((record) => (record.returned_chunks ?? []).map((chunk) => [
  record.model, record.question_id, record.question, chunk.rank, chunk.score, chunk.is_expected,
  chunk.chunk_id, chunk.source_id, chunk.chunk_index, chunk.citation ?? "", joined(chunk.pages, ", "),
  joined(chunk.headings), chunk.source_url ?? "", chunk.origin_filename ?? "", chunk.text ?? "",
]));

await writeCsv("question_model_comparison.csv", [questionHeaders, ...questionRows]);
await writeCsv("expected_chunks.csv", [expectedHeaders, ...expectedRows]);
await writeCsv("retrieved_chunks_top30.csv", [retrievedHeaders, ...retrievedRows]);

const facetHeaders = ["Facet", "Group", "Model", "Answerable", "Excluded Empty Gold", "K", "Precision", "Recall", "MRR", "Hit Rate"];
const facetRows = [];
for (const [model, modelResult] of Object.entries(report.models)) {
  for (const [facetName, reportKey] of [["Query Type", "by_query_type"], ["Difficulty", "by_difficulty"], ["Edge Case", "by_edge_case"]]) {
    const groups = modelResult.metrics[reportKey] ?? {};
    for (const [group, result] of Object.entries(groups)) {
      for (const k of [5, 10, 20, 30]) {
        const score = result.cutoffs?.[String(k)] ?? {};
        facetRows.push([facetName, group, model, result.evaluated_answerable_count ?? 0, result.excluded_empty_gold_count ?? 0,
          k, score.precision ?? null, score.recall ?? null, score.mrr ?? null, score.hit_rate ?? null]);
      }
    }
  }
}

const metricHeaders = ["Model", "Model ID", "Dimension", "K", "Precision", "Recall", "MRR", "Hit Rate"];
const metricRows = [];
for (const [model, result] of Object.entries(report.models)) {
  for (const k of [5, 10, 20, 30]) {
    const score = result.metrics.all.cutoffs[String(k)];
    metricRows.push([model, result.model_id, result.dimension, k, score.precision, score.recall, score.mrr, score.hit_rate]);
  }
}

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const questions = workbook.worksheets.add("Question Comparison");
const expected = workbook.worksheets.add("Expected Chunks");
const retrieved = workbook.worksheets.add("Retrieved Chunks");
const facets = workbook.worksheets.add("Facet Metrics");

const navy = "#0B1F3A";
const blue = "#DCEAF7";
const pale = "#F5F8FC";
const green = "#E2F0D9";
const red = "#FCE4D6";
const white = "#FFFFFF";
const border = "#CBD5E1";

function styleTable(sheet, range, tableName, headerCols, percentCols = []) {
  const table = sheet.tables.add(range, true, tableName);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
  sheet.getRange(`A1:${headerCols}1`).format = {
    fill: navy, font: { bold: true, color: white }, wrapText: true,
    verticalAlignment: "center", borders: { preset: "all", style: "thin", color: border },
  };
  sheet.getRange(`A1:${headerCols}1`).format.rowHeight = 32;
  for (const col of percentCols) sheet.getRange(`${col}2:${col}${range.match(/\d+$/)[0]}`).format.numberFormat = "0.0%";
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  return table;
}

summary.getRange("A1:H1").merge();
summary.getRange("A1").values = [["BMO Retrieval Audit — BGE-M3, Nomic Embed v1.5, Qwen3 Embedding 0.6B"]];
summary.getRange("A1:H1").format = { fill: navy, font: { bold: true, color: white, size: 16 }, verticalAlignment: "center" };
summary.getRange("A1:H1").format.rowHeight = 34;
summary.getRange("A3:B9").values = [
  ["Dataset", "retrieval_golden_200"],
  ["Questions", report.record_count],
  ["Answerable questions", report.models["bge-m3"].metrics.all.evaluated_answerable_count],
  ["Empty-gold questions", report.models["bge-m3"].metrics.all.excluded_empty_gold_count],
  ["Corpus chunks", report.corpus_chunk_count],
  ["Models", modelFiles.length],
  ["Returned ranks per model/query", 30],
];
summary.getRange("A3:A9").format = { fill: blue, font: { bold: true }, borders: { preset: "all", style: "thin", color: border } };
summary.getRange("B3:B9").format = { fill: pale, borders: { preset: "all", style: "thin", color: border } };
summary.getRange("A11:H11").values = [metricHeaders];
summary.getRange(`A12:H${11 + metricRows.length}`).values = metricRows;
styleTable(summary, `A11:H${11 + metricRows.length}`, "OverallMetricsTable", "H", ["E", "F", "G", "H"]);
summary.getRange(`D12:D${11 + metricRows.length}`).format.numberFormat = "0";
summary.getRange("A29:H29").merge();
summary.getRange("A29").values = [["Reading guide: Question Comparison is one row per model/question. Expected Chunks is the gold evidence. Retrieved Chunks contains every rank 1–30 result, including citation and page metadata. Facet Metrics provides query-type, difficulty, and edge-case breakdowns."]];
summary.getRange("A29:H29").format = { fill: blue, font: { italic: true }, wrapText: true };
summary.getRange("A29:H29").format.rowHeight = 48;
summary.getRange("A1:H29").format.verticalAlignment = "top";
summary.getRange("A:A").format.columnWidth = 25;
summary.getRange("B:B").format.columnWidth = 42;
summary.getRange("C:H").format.columnWidth = 14;
summary.freezePanes.freezeRows(1);
summary.showGridLines = false;

questions.getRange(`A1:W${questionRows.length + 1}`).values = cleanMatrix([questionHeaders, ...questionRows]);
styleTable(questions, `A1:W${questionRows.length + 1}`, "QuestionComparisonTable", "W");
questions.freezePanes.freezeColumns(2);
questions.getRange(`M2:M${questionRows.length + 1}`).format.numberFormat = "0";
questions.getRange(`S2:S${questionRows.length + 1}`).format.numberFormat = "0.0000";
questions.getRange(`N2:Q${questionRows.length + 1}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "TRUE", format: { fill: green } });
questions.getRange(`N2:Q${questionRows.length + 1}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "FALSE", format: { fill: red } });
questions.getRange(`A2:W${questionRows.length + 1}`).format.verticalAlignment = "top";
questions.getRange("A:A").format.columnWidth = 24;
questions.getRange("B:B").format.columnWidth = 20;
questions.getRange("C:C").format.columnWidth = 48;
questions.getRange("D:G").format.columnWidth = 18;
questions.getRange("H:I").format.columnWidth = 55;
questions.getRange("J:L").format.columnWidth = 38;
questions.getRange("M:V").format.columnWidth = 16;
questions.getRange("W:W").format.columnWidth = 70;

expected.getRange(`A1:M${expectedRows.length + 1}`).values = cleanMatrix([expectedHeaders, ...expectedRows]);
styleTable(expected, `A1:M${expectedRows.length + 1}`, "ExpectedChunksTable", "M");
expected.freezePanes.freezeColumns(2);
expected.getRange(`A2:M${expectedRows.length + 1}`).format.verticalAlignment = "top";
expected.getRange("A:B").format.columnWidth = 24;
expected.getRange("C:C").format.columnWidth = 48;
expected.getRange("D:K").format.columnWidth = 22;
expected.getRange("L:M").format.columnWidth = 70;

retrieved.getRange(`A1:O${retrievedRows.length + 1}`).values = cleanMatrix([retrievedHeaders, ...retrievedRows]);
styleTable(retrieved, `A1:O${retrievedRows.length + 1}`, "RetrievedChunksTable", "O");
retrieved.freezePanes.freezeColumns(2);
retrieved.getRange(`D2:D${retrievedRows.length + 1}`).format.numberFormat = "0";
retrieved.getRange(`E2:E${retrievedRows.length + 1}`).format.numberFormat = "0.0000";
retrieved.getRange(`F2:F${retrievedRows.length + 1}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "TRUE", format: { fill: green, font: { bold: true } } });
retrieved.getRange(`A2:O${retrievedRows.length + 1}`).format.verticalAlignment = "top";
retrieved.getRange("A:B").format.columnWidth = 24;
retrieved.getRange("C:C").format.columnWidth = 48;
retrieved.getRange("D:F").format.columnWidth = 12;
retrieved.getRange("G:N").format.columnWidth = 22;
retrieved.getRange("O:O").format.columnWidth = 75;

facets.getRange(`A1:J${facetRows.length + 1}`).values = [facetHeaders, ...facetRows];
styleTable(facets, `A1:J${facetRows.length + 1}`, "FacetMetricsTable", "J", ["G", "H", "I", "J"]);
facets.freezePanes.freezeColumns(3);
facets.getRange("A:C").format.columnWidth = 25;
facets.getRange("D:F").format.columnWidth = 16;
facets.getRange("G:J").format.columnWidth = 14;

for (const [sheetName, range, filename] of [
  ["Summary", "A1:H29", "summary.png"],
  ["Question Comparison", "A1:W14", "question_comparison.png"],
  ["Expected Chunks", "A1:M14", "expected_chunks.png"],
  ["Retrieved Chunks", "A1:O14", "retrieved_chunks.png"],
  ["Facet Metrics", "A1:J18", "facet_metrics.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 0.8, format: "png" });
  await fs.writeFile(path.join(supportDir, filename), new Uint8Array(await preview.arrayBuffer()));
}

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 10000,
  tableMaxRows: 4,
  tableMaxCols: 8,
  tableMaxCellChars: 80,
});
await fs.writeFile(path.join(supportDir, "inspection.json"), inspection.ndjson ?? JSON.stringify(inspection, null, 2));
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 200 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(supportDir, "error_scan.json"), errors.ndjson ?? JSON.stringify(errors, null, 2));

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(path.join(outputDir, "bmo_retrieval_audit_3_models.xlsx"));

console.log(JSON.stringify({
  questions: questionRows.length,
  expectedChunks: expectedRows.length,
  retrievedChunks: retrievedRows.length,
  facetRows: facetRows.length,
  outputDir,
}, null, 2));
