"""Eval harness: verified golden questions + a pipeline config -> eval_runs / eval_results rows.

Modes: retrieval only (default, free and fast), `generate` (also answers each question through the
full RAG path), `judge` (also scores the answers with the LLM judge; implies generate).
"""
import json
import sqlite3
import subprocess
import time
import uuid
from datetime import datetime, timezone

from ..config import Settings
from ..db import write_lock
from ..pipeline import run_query
from ..retrieval.service import PipelineConfig, Retriever
from .. import llm
from .judge import judge_result
from .metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank

K = 5


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip() or None
    except Exception:
        return None


def stratified_subset(questions: list, n: int | None) -> list:
    """First `n` questions, taken round-robin across types so every type stays represented."""
    if n is None or n >= len(questions):
        return list(questions)
    by_type: dict[str, list] = {}
    for q in sorted(questions, key=lambda q: q["id"]):
        by_type.setdefault(q["type"], []).append(q)
    out: list = []
    while len(out) < n:
        progressed = False
        for t in sorted(by_type):
            if by_type[t] and len(out) < n:
                out.append(by_type[t].pop(0))
                progressed = True
        if not progressed:
            break
    return out


def _answer(retriever: Retriever, settings: Settings, config: PipelineConfig, question: str) -> dict:
    out = {"citations": [], "answer": "", "refused": False, "trace": {}}
    tokens = []
    for ev in run_query(retriever, settings, {"question": question, "pipeline": config.to_dict()}):
        if ev["type"] == "citations":
            out["citations"] = ev["citations"]
        elif ev["type"] == "token":
            tokens.append(ev["text"])
        elif ev["type"] == "refusal":
            out["refused"] = True
        elif ev["type"] == "done":
            out["trace"] = ev["trace"]
    out["answer"] = "".join(tokens)
    return out


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-question row dicts (see run_eval / rows_from_run) into run metrics."""
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    judged = [r["judge"] for r in rows if r.get("judge")]
    rate = lambda key: mean([float(j[key]) for j in judged if j.get(key) is not None])
    gen = [r for r in rows if r.get("e2e_ms") is not None]
    ref = [r for r in rows if r.get("refused") is not None]
    return {
        "questions": len(rows),
        "recall_at_5": mean([recall_at_k(r["retrieved"], r["relevant"], K) for r in answerable]),
        "mrr": mean([reciprocal_rank(r["retrieved"], r["relevant"]) for r in answerable]),
        "ndcg_at_5": mean([ndcg_at_k(r["retrieved"], r["relevant"], K) for r in answerable]),
        "p95_latency_ms": round(percentile([r["latency_ms"] for r in rows], 95), 1),
        "p95_e2e_ms": round(percentile([r["e2e_ms"] for r in gen], 95), 1) if gen else None,
        "cost_per_query_usd": round(sum(r["cost_usd"] or 0 for r in gen) / len(gen), 6) if gen else 0.0,
        "refusal_accuracy": mean([float(r["refused"] == (r["type"] == "unanswerable")) for r in ref]),
        "faithfulness": rate("faithfulness"),
        "answer_correctness": rate("correctness"),
        "citation_accuracy": rate("citations"),
        "judge_errors": sum(j.get("errors", 0) for j in judged),
        "generation_errors": sum(1 for r in rows if r.get("error")),
    }


def rows_from_run(conn: sqlite3.Connection, run_id: str, question_ids: set[str] | None = None) -> list[dict]:
    """Rebuild summarize() input from stored results (to compare runs on the same questions)."""
    out = []
    for r in conn.execute(
            "SELECT e.*, g.type, g.relevant_chunk_ids FROM eval_results e "
            "JOIN golden_questions g ON g.id = e.question_id WHERE e.run_id=?", (run_id,)):
        if question_ids is not None and r["question_id"] not in question_ids:
            continue
        out.append({"id": r["question_id"], "type": r["type"],
                    "relevant": set(json.loads(r["relevant_chunk_ids"])),
                    "retrieved": json.loads(r["retrieved_chunk_ids"]), "latency_ms": r["latency_ms"] or 0.0,
                    "e2e_ms": r["e2e_ms"], "cost_usd": r["cost_usd"],
                    "refused": None if r["refused"] is None else bool(r["refused"]),
                    "judge": json.loads(r["judge_scores"]) if r["judge_scores"] else None})
    return out


def run_eval(conn: sqlite3.Connection, retriever: Retriever, settings: Settings, config: PipelineConfig,
             label: str, is_baseline: bool = False, generate: bool = False, judge: bool = False,
             subset: int | None = None) -> dict:
    generate = generate or judge
    if generate:
        llm.require_anthropic(settings)  # fail before doing any work if no real model is configured
    questions = stratified_subset(
        conn.execute("SELECT * FROM golden_questions WHERE status='verified' ORDER BY id").fetchall(), subset)
    if not questions:
        raise ValueError("no verified golden questions: generate and review some (scripts/golden.py)")

    run_id = "run_" + uuid.uuid4().hex[:6]
    rows, db_rows = [], []
    judge_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for q in questions:
        relevant = set(json.loads(q["relevant_chunk_ids"]))
        row = {"id": q["id"], "type": q["type"], "relevant": relevant, "retrieved": [], "latency_ms": 0.0,
               "e2e_ms": None, "cost_usd": None, "refused": None, "judge": None, "answer": None}
        scores_json = None
        if generate:
            t0 = time.perf_counter()
            try:
                res = _answer(retriever, settings, config, q["question"])
            except Exception as exc:  # one bad call must not sink a 150-question run
                row["error"], row["answer"] = True, f"[error] {exc}"
                res = None
            if res:
                tr = res["trace"]
                row.update(retrieved=[c["chunk_id"] for c in res["citations"]], answer=res["answer"],
                           refused=res["refused"], e2e_ms=tr.get("total_ms", (time.perf_counter() - t0) * 1000),
                           latency_ms=round(sum(s["ms"] for s in tr.get("stages", []) if s["name"] != "llm"), 2),
                           cost_usd=tr.get("cost_usd"))
                if judge:
                    row["judge"], used = judge_result(settings, q["type"], q["question"], q["expected_answer"],
                                                      res["answer"], res["citations"], res["refused"])
                    for k in judge_usage:
                        judge_usage[k] += used[k]
                    scores_json = json.dumps(row["judge"])
                row["scores"] = [c["scores"] for c in res["citations"]]
        else:
            t0 = time.perf_counter()
            cands, _ = retriever.retrieve(q["question"], None, config)
            row.update(latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                       retrieved=[c["chunk_id"] for c in cands], scores=[c["scores"] for c in cands])
        rows.append(row)
        db_rows.append((run_id, q["id"], json.dumps(row["retrieved"]), row["answer"],
                        json.dumps(row.get("scores", [])), scores_json, row["latency_ms"], row["cost_usd"],
                        row["e2e_ms"], None if row["refused"] is None else int(row["refused"])))

    summary = summarize(rows)
    if judge:
        summary["judge_cost_usd"] = round((judge_usage["prompt_tokens"] * settings.llm_price_in
                                           + judge_usage["completion_tokens"] * settings.llm_price_out) / 1e6, 6)
    summary["mode"] = "judge" if judge else "generate" if generate else "retrieval"
    with write_lock:
        conn.execute("INSERT INTO eval_runs VALUES (?,?,?,?,?,?,?)",
                     (run_id, label, _git_sha(), json.dumps(config.to_dict()),
                      datetime.now(timezone.utc).isoformat(timespec="seconds"), int(is_baseline),
                      json.dumps(summary)))
        conn.executemany("INSERT INTO eval_results (run_id, question_id, retrieved_chunk_ids, answer, "
                         "retrieval_scores, judge_scores, latency_ms, cost_usd, e2e_ms, refused) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?)", db_rows)
        conn.commit()
    return {"id": run_id, "summary_metrics": summary}
