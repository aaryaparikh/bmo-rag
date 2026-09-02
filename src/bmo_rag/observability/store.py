"""SQLite-backed traces for latency, tokens, retrieval, prompts, and responses."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bmo_rag.generation.openai_responses import ResponseTelemetry


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class StageTiming:
    sequence: int
    name: str
    category: str
    started_at: str
    completed_at: str
    duration_ns: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChunkSnapshot:
    stage: str
    lane: str
    rank: int
    chunk_id: str
    source_id: str | None
    chunk_index: int | None
    text: str
    headings: tuple[str, ...]
    pages: tuple[int, ...]
    scores: dict[str, float]
    context_role: str | None


@dataclass
class Trace:
    session_id: str
    original_query: str
    model: str
    embedding_model: str
    candidate_k: int
    seed_k: int
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(default_factory=utc_now)
    started_ns: int = field(default_factory=time.perf_counter_ns)
    status: str = "running"
    completed_at: str | None = None
    duration_ns: int | None = None
    retrieval_query: str | None = None
    system_prompt: str | None = None
    llm_input: str | None = None
    evidence_text: str | None = None
    evidence_tokens: int | None = None
    evidence_token_method: str | None = None
    final_response: str | None = None
    source_constraints: list[dict[str, Any]] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    stages: list[StageTiming] = field(default_factory=list)
    llm_calls: list[ResponseTelemetry] = field(default_factory=list)
    chunks: list[ChunkSnapshot] = field(default_factory=list)

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        category: str = "micro",
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        started_at = utc_now()
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            completed_ns = time.perf_counter_ns()
            self.stages.append(
                StageTiming(
                    sequence=len(self.stages) + 1,
                    name=name,
                    category=category,
                    started_at=started_at,
                    completed_at=utc_now(),
                    duration_ns=completed_ns - started_ns,
                    metadata=metadata or {},
                )
            )

    def record_llm_call(self, telemetry: ResponseTelemetry) -> None:
        self.llm_calls.append(telemetry)

    def record_chunks(self, stage: str, points: list[dict[str, Any]], *, lane: str) -> None:
        for rank, point in enumerate(points, start=1):
            payload = point.get("payload") or {}
            scores = {
                key: float(value)
                for key, value in point.items()
                if "score" in key and isinstance(value, int | float)
            }
            chunk_index = payload.get("chunk_index")
            self.chunks.append(
                ChunkSnapshot(
                    stage=stage,
                    lane=lane,
                    rank=rank,
                    chunk_id=str(payload.get("chunk_id") or point.get("id") or "unknown"),
                    source_id=(
                        str(payload["source_id"]) if payload.get("source_id") is not None else None
                    ),
                    chunk_index=chunk_index if isinstance(chunk_index, int) else None,
                    text=str(payload.get("text") or ""),
                    headings=tuple(str(value) for value in payload.get("headings") or []),
                    pages=tuple(
                        int(value) for value in payload.get("pages") or [] if isinstance(value, int)
                    ),
                    scores=scores,
                    context_role=(
                        str(point["context_role"]) if point.get("context_role") else None
                    ),
                )
            )

    def complete(self, response: str) -> None:
        self.status = "completed"
        self.final_response = response
        self._finish_clock()

    def fail(self, exc: BaseException) -> None:
        self.status = "failed"
        self.error_type = type(exc).__name__
        self.error_message = str(exc)
        self._finish_clock()

    def _finish_clock(self) -> None:
        self.completed_at = utc_now()
        self.duration_ns = time.perf_counter_ns() - self.started_ns


class SQLiteObservabilityStore:
    def __init__(self, path: Path | str = Path("data/observability/rag_observability.sqlite3")):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def new_trace(
        self,
        *,
        session_id: str,
        original_query: str,
        model: str,
        embedding_model: str,
        candidate_k: int,
        seed_k: int,
    ) -> Trace:
        return Trace(
            session_id=session_id,
            original_query=original_query,
            model=model,
            embedding_model=embedding_model,
            candidate_k=candidate_k,
            seed_k=seed_k,
        )

    def save(self, trace: Trace) -> None:
        if trace.duration_ns is None:
            raise ValueError("trace must be completed or failed before it is saved")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO traces (
                    trace_id, session_id, started_at, completed_at, status,
                    duration_ns, duration_us, duration_ms, duration_seconds,
                    model, embedding_model, candidate_k, seed_k,
                    original_query, retrieval_query, system_prompt, llm_input,
                    evidence_text, evidence_tokens, evidence_token_method,
                    input_tokens, output_tokens, total_tokens, cached_input_tokens,
                    reasoning_tokens, source_constraints_json, final_response,
                    error_type, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.session_id,
                    trace.started_at,
                    trace.completed_at,
                    trace.status,
                    trace.duration_ns,
                    trace.duration_ns // 1_000,
                    trace.duration_ns / 1_000_000,
                    trace.duration_ns / 1_000_000_000,
                    trace.model,
                    trace.embedding_model,
                    trace.candidate_k,
                    trace.seed_k,
                    trace.original_query,
                    trace.retrieval_query,
                    trace.system_prompt,
                    trace.llm_input,
                    trace.evidence_text,
                    trace.evidence_tokens,
                    trace.evidence_token_method,
                    sum(call.input_tokens for call in trace.llm_calls),
                    sum(call.output_tokens for call in trace.llm_calls),
                    sum(call.total_tokens for call in trace.llm_calls),
                    sum(call.cached_input_tokens for call in trace.llm_calls),
                    sum(call.reasoning_tokens for call in trace.llm_calls),
                    json.dumps(trace.source_constraints, ensure_ascii=False),
                    trace.final_response,
                    trace.error_type,
                    trace.error_message,
                ),
            )
            connection.executemany(
                """
                INSERT INTO stage_timings (
                    trace_id, sequence, name, category, started_at, completed_at,
                    duration_ns, duration_us, duration_ms, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        trace.trace_id,
                        stage.sequence,
                        stage.name,
                        stage.category,
                        stage.started_at,
                        stage.completed_at,
                        stage.duration_ns,
                        stage.duration_ns // 1_000,
                        stage.duration_ns / 1_000_000,
                        json.dumps(stage.metadata, ensure_ascii=False),
                    )
                    for stage in trace.stages
                ],
            )
            connection.executemany(
                """
                INSERT INTO llm_calls (
                    trace_id, operation, response_id, model, system_prompt, input_text,
                    output_text, input_tokens, output_tokens, total_tokens,
                    cached_input_tokens, reasoning_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        trace.trace_id,
                        call.operation,
                        call.response_id,
                        call.model,
                        call.instructions,
                        call.input_text,
                        call.output_text,
                        call.input_tokens,
                        call.output_tokens,
                        call.total_tokens,
                        call.cached_input_tokens,
                        call.reasoning_tokens,
                    )
                    for call in trace.llm_calls
                ],
            )
            connection.executemany(
                """
                INSERT INTO retrieved_chunks (
                    trace_id, stage, lane, rank, chunk_id, source_id, chunk_index,
                    chunk_text, headings_json, pages_json, scores_json, context_role
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        trace.trace_id,
                        chunk.stage,
                        chunk.lane,
                        chunk.rank,
                        chunk.chunk_id,
                        chunk.source_id,
                        chunk.chunk_index,
                        chunk.text,
                        json.dumps(chunk.headings, ensure_ascii=False),
                        json.dumps(chunk.pages),
                        json.dumps(chunk.scores),
                        chunk.context_role,
                    )
                    for chunk in trace.chunks
                ],
            )

    def recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT trace_id, started_at, status, duration_ms, input_tokens,
                       output_tokens, evidence_tokens, original_query
                FROM traces ORDER BY started_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def summary(self, *, limit: int = 1000) -> dict[str, int | float]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT status, duration_ms, input_tokens, output_tokens, total_tokens,
                       evidence_tokens
                FROM traces ORDER BY started_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        if not rows:
            return {
                "trace_count": 0,
                "failure_count": 0,
                "average_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "evidence_tokens": 0,
            }
        durations = sorted(float(row["duration_ms"]) for row in rows)
        return {
            "trace_count": len(rows),
            "failure_count": sum(row["status"] == "failed" for row in rows),
            "average_latency_ms": sum(durations) / len(durations),
            "p50_latency_ms": _percentile(durations, 0.50),
            "p95_latency_ms": _percentile(durations, 0.95),
            "max_latency_ms": durations[-1],
            "input_tokens": sum(int(row["input_tokens"]) for row in rows),
            "output_tokens": sum(int(row["output_tokens"]) for row in rows),
            "total_tokens": sum(int(row["total_tokens"]) for row in rows),
            "evidence_tokens": sum(int(row["evidence_tokens"] or 0) for row in rows),
        }

    def trace_detail(self, trace_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            trace = connection.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if trace is None:
                return None
            stages = connection.execute(
                "SELECT * FROM stage_timings WHERE trace_id = ? ORDER BY sequence", (trace_id,)
            ).fetchall()
            calls = connection.execute(
                "SELECT * FROM llm_calls WHERE trace_id = ? ORDER BY id", (trace_id,)
            ).fetchall()
            chunks = connection.execute(
                "SELECT * FROM retrieved_chunks WHERE trace_id = ? ORDER BY id", (trace_id,)
            ).fetchall()
        return {
            "trace": dict(trace),
            "stages": [dict(row) for row in stages],
            "llm_calls": [dict(row) for row in calls],
            "retrieved_chunks": [dict(row) for row in chunks],
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    duration_ns INTEGER NOT NULL,
                    duration_us INTEGER NOT NULL,
                    duration_ms REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    model TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    candidate_k INTEGER NOT NULL,
                    seed_k INTEGER NOT NULL,
                    original_query TEXT NOT NULL,
                    retrieval_query TEXT,
                    system_prompt TEXT,
                    llm_input TEXT,
                    evidence_text TEXT,
                    evidence_tokens INTEGER,
                    evidence_token_method TEXT,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    source_constraints_json TEXT NOT NULL DEFAULT '[]',
                    final_response TEXT,
                    error_type TEXT,
                    error_message TEXT
                );
                CREATE TABLE IF NOT EXISTS stage_timings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    duration_ns INTEGER NOT NULL,
                    duration_us INTEGER NOT NULL,
                    duration_ms REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
                    operation TEXT NOT NULL,
                    response_id TEXT,
                    model TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    input_text TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cached_input_tokens INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS retrieved_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    lane TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    source_id TEXT,
                    chunk_index INTEGER,
                    chunk_text TEXT NOT NULL,
                    headings_json TEXT NOT NULL,
                    pages_json TEXT NOT NULL,
                    scores_json TEXT NOT NULL,
                    context_role TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_traces_started_at ON traces(started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_stages_trace ON stage_timings(trace_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_chunks_trace ON retrieved_chunks(trace_id, stage, rank);
                """
            )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    index = max(0, min(len(sorted_values) - 1, round((len(sorted_values) - 1) * fraction)))
    return sorted_values[index]
