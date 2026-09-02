from __future__ import annotations

from typer.testing import CliRunner

from bmo_rag.cli import app
from bmo_rag.observability.store import SQLiteObservabilityStore


def test_monitor_lists_and_inspects_trace(tmp_path) -> None:
    path = tmp_path / "traces.sqlite3"
    store = SQLiteObservabilityStore(path)
    trace = store.new_trace(
        session_id="session",
        original_query="What is CET1?",
        model="gpt-5",
        embedding_model="bge-m3",
        candidate_k=30,
        seed_k=8,
    )
    trace.complete("13.0% [S1]")
    store.save(trace)

    runner = CliRunner()
    listing = runner.invoke(app, ["monitor", "--observability-db", str(path)])
    assert listing.exit_code == 0
    assert "latency avg/p50/p95/max=" in listing.stdout
    assert trace.trace_id in listing.stdout
    assert "What is CET1?" in listing.stdout

    detail = runner.invoke(
        app,
        ["monitor", "--observability-db", str(path), "--trace-id", trace.trace_id],
    )
    assert detail.exit_code == 0
    assert '"original_query": "What is CET1?"' in detail.stdout
    assert '"final_response": "13.0% [S1]"' in detail.stdout
