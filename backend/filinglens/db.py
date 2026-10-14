"""SQLite schema (stand-in for Postgres) and connection helper.

Tables follow the system design. Embeddings are float32 BLOBs; `embedding_model`
is stored per chunk so a model change forces a clean re-index.
"""
import sqlite3
import threading
from pathlib import Path

# Serialises writers (ingestion, eval runs) across server threads.
write_lock = threading.RLock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,
    ticker       TEXT NOT NULL,
    company      TEXT NOT NULL,
    fiscal_year  INTEGER NOT NULL,
    source_url   TEXT,
    content_hash TEXT,
    ingested_at  TEXT,
    UNIQUE (ticker, fiscal_year)
);
CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES documents(id),
    section         TEXT,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    token_count     INTEGER NOT NULL,
    embedding       BLOB,
    embedding_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE TABLE IF NOT EXISTS golden_questions (
    id                 TEXT PRIMARY KEY,
    question           TEXT NOT NULL,
    type               TEXT NOT NULL CHECK (type IN
                       ('factual','table','multi-hop','comparison','unanswerable')),
    expected_answer    TEXT,
    relevant_chunk_ids TEXT NOT NULL DEFAULT '[]',
    status             TEXT NOT NULL DEFAULT 'verified'
                       CHECK (status IN ('candidate','verified','rejected'))
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id               TEXT PRIMARY KEY,
    label            TEXT NOT NULL,
    git_sha          TEXT,
    pipeline_config  TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    is_baseline      INTEGER NOT NULL DEFAULT 0,
    summary_metrics  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_results (
    run_id              TEXT NOT NULL REFERENCES eval_runs(id),
    question_id         TEXT NOT NULL REFERENCES golden_questions(id),
    retrieved_chunk_ids TEXT NOT NULL,
    answer              TEXT,
    retrieval_scores    TEXT,
    judge_scores        TEXT,
    latency_ms          REAL,
    cost_usd            REAL,
    e2e_ms              REAL,
    refused             INTEGER,
    PRIMARY KEY (run_id, question_id)
);
CREATE TABLE IF NOT EXISTS judge_labels (
    run_id      TEXT NOT NULL REFERENCES eval_runs(id),
    question_id TEXT NOT NULL REFERENCES golden_questions(id),
    metric      TEXT NOT NULL,
    human_pass  INTEGER NOT NULL,
    judge_pass  INTEGER NOT NULL,
    PRIMARY KEY (run_id, question_id, metric)
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _add_column(conn, "golden_questions", "status", "TEXT NOT NULL DEFAULT 'verified'")
    _add_column(conn, "eval_results", "e2e_ms", "REAL")
    _add_column(conn, "eval_results", "refused", "INTEGER")
    return conn


def _add_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """Upgrade a database created before `column` existed."""
    if column not in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        conn.commit()
