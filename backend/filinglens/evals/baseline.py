"""Export/import one eval run as JSON, so a baseline can live in git and CI can compare against it."""
import json
import sqlite3
from pathlib import Path

from ..db import write_lock


def export_run(conn: sqlite3.Connection, run_id: str, path) -> int:
    """Write the run and its per-question results. Returns the number of results."""
    run = conn.execute("SELECT * FROM eval_runs WHERE id=?", (run_id,)).fetchone()
    if not run:
        raise ValueError(f"unknown run: {run_id}")
    results = [dict(r) for r in conn.execute(
        "SELECT * FROM eval_results WHERE run_id=? ORDER BY question_id", (run_id,))]
    Path(path).write_text(json.dumps({"version": 1, "run": dict(run), "results": results}, indent=1) + "\n")
    return len(results)


def import_run(conn: sqlite3.Connection, path, as_baseline: bool = True) -> str:
    """Load an exported run (replacing any run with the same id). Returns the run id."""
    data = json.loads(Path(path).read_text())
    if data.get("version") != 1:
        raise ValueError(f"unsupported baseline file version: {data.get('version')!r}")
    run, results = data["run"], data["results"]
    if as_baseline:
        run["is_baseline"] = 1
    with write_lock:
        conn.execute("DELETE FROM eval_results WHERE run_id=?", (run["id"],))
        conn.execute("INSERT OR REPLACE INTO eval_runs (id, label, git_sha, pipeline_config, started_at, "
                     "is_baseline, summary_metrics) VALUES (:id, :label, :git_sha, :pipeline_config, "
                     ":started_at, :is_baseline, :summary_metrics)", run)
        for r in results:
            conn.execute("INSERT INTO eval_results (run_id, question_id, retrieved_chunk_ids, answer, "
                         "retrieval_scores, judge_scores, latency_ms, cost_usd, e2e_ms, refused) VALUES "
                         "(:run_id, :question_id, :retrieved_chunk_ids, :answer, :retrieval_scores, "
                         ":judge_scores, :latency_ms, :cost_usd, :e2e_ms, :refused)", r)
        conn.commit()
    return run["id"]
