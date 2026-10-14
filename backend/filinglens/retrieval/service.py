"""Retrieval service: config-driven stages (dense / bm25 / rrf), per-stage timing.

This is the part to load-test; /v1/retrieve exposes it directly.
"""
import sqlite3
import time
from array import array
from dataclasses import dataclass, asdict

from .. import embeddings
from ..embeddings import from_blob
from .bm25 import BM25
from .dense import DenseIndex
from .fusion import rrf


@dataclass(frozen=True)
class PipelineConfig:
    dense: bool = True
    bm25: bool = False
    fusion: str = "rrf"        # used only when both retrievers are on
    candidates: int = 50       # per retriever
    top_k: int = 6
    dense_mode: str = "auto"   # exact | binary | auto (exact for small scans, else binary + rescoring)
    oversample: int = 10       # binary mode rescored candidates = candidates * oversample

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "baseline": PipelineConfig(dense=True, bm25=False),
    "hybrid": PipelineConfig(dense=True, bm25=True),
    # exact dense search, to measure what binary quantisation costs in recall (slow on big corpora)
    "baseline-exact": PipelineConfig(dense=True, bm25=False, dense_mode="exact"),
    "hybrid-exact": PipelineConfig(dense=True, bm25=True, dense_mode="exact"),
}


def config_from(value, default: PipelineConfig) -> PipelineConfig:
    """Accept a preset name, a partial dict, or None."""
    if value is None:
        return default
    if isinstance(value, str):
        if value not in PRESETS:
            raise ValueError(f"unknown pipeline preset: {value}")
        return PRESETS[value]
    fields = {k: v for k, v in value.items() if k in PipelineConfig.__dataclass_fields__}
    return PipelineConfig(**{**default.to_dict(), **fields})


class Retriever:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.reload()

    def reload(self) -> None:
        rows = self.conn.execute(
            "SELECT c.id, c.section, c.text, c.embedding, c.embedding_model, "
            "d.ticker, d.company, d.fiscal_year, d.source_url "
            "FROM chunks c JOIN documents d ON d.id = c.document_id ORDER BY d.id, c.chunk_index").fetchall()
        self.meta = {r["id"]: {k: r[k] for k in r.keys() if k not in ("id", "embedding", "embedding_model")}
                     for r in rows}
        self.n = len(rows)
        # Rows are ordered by document, so each (ticker, year) owns one contiguous slice [start, end).
        # Filters become slice selections instead of per-chunk lookups.
        self.ranges: dict[tuple[str, int], tuple[int, int]] = {}
        for i, r in enumerate(rows):
            key = (r["ticker"], r["fiscal_year"])
            start = self.ranges.get(key, (i, i))[0]
            self.ranges[key] = (start, i + 1)
        self.bm25 = BM25({r["id"]: r["text"] for r in rows})
        current = embeddings.model_name()
        valid = [r["embedding"] if r["embedding"] and r["embedding_model"] == current else None for r in rows]
        first = next((from_blob(b) for b in valid if b), None)
        if first is None:
            self.dense = DenseIndex([], [])
        else:
            blank = array("f", [0.0]) * len(first)   # chunks embedded by another model score 0
            self.dense = DenseIndex([r["id"] for r in rows], [from_blob(b) if b else blank for b in valid])

    def _select(self, filters: dict | None):
        """Return (ranges, mask) for the filters; (None, None) means no restriction."""
        filters = filters or {}
        tickers = {t.upper() for t in filters.get("tickers") or []}
        years = set(filters.get("fiscal_years") or [])
        if not tickers and not years:
            return None, None
        picked = sorted(span for (ticker, year), span in self.ranges.items()
                        if (not tickers or ticker in tickers) and (not years or year in years))
        merged: list[list[int]] = []
        for s, e in picked:
            if merged and merged[-1][1] == s:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        mask = bytearray(self.n)
        for s, e in merged:
            mask[s:e] = b"\x01" * (e - s)
        return [(s, e) for s, e in merged], mask

    def retrieve(self, query: str, filters: dict | None, config: PipelineConfig):
        """Return (candidates, stage_ms). Candidates are best first, each with per-stage scores."""
        ranges, mask = self._select(filters)
        stage_ms: list[dict] = []

        def timed(name, fn):
            t0 = time.perf_counter()
            out = fn()
            stage_ms.append({"name": name, "ms": round((time.perf_counter() - t0) * 1000, 2)})
            return out

        bm25_hits = timed("bm25", lambda: self.bm25.search(query, config.candidates, mask=mask)) \
            if config.bm25 else []
        dense_hits = timed("dense", lambda: self.dense.search(query, config.candidates, ranges=ranges,
                                                              mode=config.dense_mode, oversample=config.oversample)) \
            if config.dense else []
        bm25_s, dense_s = dict(bm25_hits), dict(dense_hits)

        if config.bm25 and config.dense:
            fused = timed("rrf", lambda: rrf([[c for c, _ in bm25_hits], [c for c, _ in dense_hits]]))
        else:
            fused = [(c, 0.0) for c, _ in (bm25_hits or dense_hits)]

        out = []
        for cid, rrf_score in fused[:config.top_k]:
            out.append({"chunk_id": cid, **self.meta[cid],
                        "scores": {"bm25": bm25_s.get(cid), "dense": dense_s.get(cid),
                                   "rrf": rrf_score if (config.bm25 and config.dense) else None,
                                   "rerank": None}})
        return out, stage_ms
