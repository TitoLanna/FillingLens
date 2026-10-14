"""Ingestion: fetch -> hash -> parse -> chunk -> embed -> write. Idempotent per document."""
import hashlib
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from .. import embeddings
from ..db import write_lock
from ..text import tokenize
from . import edgar
from .chunking import chunk_blocks
from .parse import parse_blocks


def ingest_document(conn: sqlite3.Connection, meta: dict, html: str) -> str:
    """Returns 'skipped' (unchanged), 'ingested' (new) or 'replaced' (changed).

    Parsing and embedding (slow, may hit the network) happen before any write, and the
    write is one transaction, so a failure leaves the previous version intact.
    """
    digest = hashlib.sha256(html.encode()).hexdigest()
    model = embeddings.model_name()
    with write_lock:
        row = conn.execute("SELECT id, content_hash FROM documents WHERE ticker=? AND fiscal_year=?",
                           (meta["ticker"], meta["fiscal_year"])).fetchone()
        if row and row["content_hash"] == digest and not conn.execute(
                "SELECT 1 FROM chunks WHERE document_id=? AND embedding_model IS NOT ? LIMIT 1",
                (row["id"], model)).fetchone():
            return "skipped"  # unchanged text and vectors already from the current model

    pieces = [(section, piece) for section, blocks in parse_blocks(html)
              for piece in chunk_blocks(blocks)]
    vectors = embeddings.embed_documents([p for _, p in pieces])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with write_lock:
        try:
            if row:
                doc_id, status = row["id"], "replaced"
                conn.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
                conn.execute("UPDATE documents SET company=?, source_url=?, content_hash=?, ingested_at=? "
                             "WHERE id=?", (meta["company"], meta.get("url"), digest, now, doc_id))
            else:
                status = "ingested"
                doc_id = conn.execute(
                    "INSERT INTO documents (ticker, company, fiscal_year, source_url, content_hash, "
                    "ingested_at) VALUES (?,?,?,?,?,?)",
                    (meta["ticker"], meta["company"], meta["fiscal_year"], meta.get("url"), digest,
                     now)).lastrowid
            for index, ((section, piece), vec) in enumerate(zip(pieces, vectors)):
                conn.execute(
                    "INSERT INTO chunks (id, document_id, section, chunk_index, text, token_count, "
                    "embedding, embedding_model) VALUES (?,?,?,?,?,?,?,?)",
                    (f"{meta['ticker'].lower()}-{meta['fiscal_year']}-{index:04d}", doc_id, section,
                     index, piece, len(tokenize(piece)), embeddings.to_blob(vec), model))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return status


def ingest_url(conn: sqlite3.Connection, meta: dict, user_agent: str, raw_dir=None) -> str:
    html, _ = edgar.fetch_filing(meta["url"], user_agent, raw_dir)
    return ingest_document(conn, meta, html)


class Jobs:
    """In-memory job table; each job runs in its own thread."""

    def __init__(self, conn, user_agent: str, on_done, raw_dir=None):
        self.conn, self.user_agent, self.on_done, self.raw_dir = conn, user_agent, on_done, raw_dir
        self.jobs: dict[str, dict] = {}

    def submit(self, documents: list[dict] | None = None, tickers: list[str] | None = None,
               years: list[int] | None = None) -> str:
        """Ingest explicit documents, and/or discover 10-Ks on EDGAR for tickers x years."""
        job_id = "job_" + uuid.uuid4().hex[:8]
        self.jobs[job_id] = {"id": job_id, "status": "running", "total": len(documents or []),
                             "counts": {"ingested": 0, "replaced": 0, "skipped": 0, "failed": 0},
                             "errors": []}
        threading.Thread(target=self._run, args=(job_id, documents or [], tickers or [], years or []),
                         daemon=True).start()
        return job_id

    def _run(self, job_id: str, documents: list[dict], tickers: list[str], years: list[int]) -> None:
        job = self.jobs[job_id]
        documents = list(documents)
        for ticker in tickers:
            try:
                found = edgar.find_10k_filings(ticker, years, self.user_agent)
                missing = sorted(set(years) - {d["fiscal_year"] for d in found})
                if missing:
                    job["errors"].append({"ticker": ticker, "error": f"no 10-K found for {missing}"})
                documents.extend(found)
            except Exception as exc:
                job["counts"]["failed"] += 1
                job["errors"].append({"ticker": ticker, "error": str(exc)})
        job["total"] = len(documents)
        for doc in documents:
            try:
                job["counts"][ingest_url(self.conn, doc, self.user_agent, self.raw_dir)] += 1
            except Exception as exc:
                job["counts"]["failed"] += 1
                job["errors"].append({"ticker": doc.get("ticker"), "error": str(exc)})
        job["status"] = "done"
        self.on_done()
