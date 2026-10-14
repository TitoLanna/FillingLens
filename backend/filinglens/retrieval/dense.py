"""Dense search: exact cosine, or sign-bit (binary) quantisation with exact rescoring.

Exact search costs one dot product per vector, which is too slow in pure Python beyond a few
thousand vectors. Binary mode keeps one bit per dimension (the sign) as a Python int, ranks
every vector by Hamming distance with a single C-level `bit_count()`, then rescores only the
best few hundred candidates with the exact dot product. Cheap, needs no training, and loses
some recall, which is why `exact` stays selectable and can be compared in the eval harness.
"""
import heapq
from array import array
from operator import mul

from ..embeddings import embed_query

try:
    from math import sumprod as _dot   # Python 3.12+, C speed
except ImportError:                    # 3.10 / 3.11
    def _dot(a, b):
        return sum(map(mul, a, b))

EXACT_BELOW = 5000    # in "auto" mode, scan this few or fewer vectors exactly


def to_bits(vec) -> int:
    """Sign bits of a vector as an int (bit j set when component j > 0)."""
    out = 0
    for j, x in enumerate(vec):
        if x > 0:
            out |= 1 << j
    return out


class DenseIndex:
    def __init__(self, ids: list[str], vectors: list[array]):
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must align")
        self.ids, self.vectors = ids, vectors
        self.bits = [to_bits(v) for v in vectors]

    def search(self, query: str, top_k: int = 10, allowed: set[str] | None = None,
               ranges: list[tuple[int, int]] | None = None, mode: str = "auto", oversample: int = 10):
        """Return [(id, cosine)] best first.

        `ranges` restricts the scan to [start, end) slices of the index (the fast path used by the
        retriever); `allowed` does it by id set. `mode`: exact | binary | auto.
        """
        if mode not in ("auto", "exact", "binary"):
            raise ValueError(f"unknown dense mode: {mode}")
        n = len(self.ids)
        if allowed is not None:
            ranges = [(i, i + 1) for i, doc_id in enumerate(self.ids) if doc_id in allowed]
        if ranges is None:
            ranges = [(0, n)]
        total = sum(e - s for s, e in ranges)
        if not total:
            return []
        q = embed_query(query)
        if mode == "exact" or (mode == "auto" and total <= EXACT_BELOW):
            scored = ((_dot(q, self.vectors[i]), i) for s, e in ranges for i in range(s, e))
            return [(self.ids[i], sc) for sc, i in heapq.nlargest(top_k, scored)]

        qb = to_bits(q)
        want = min(total, max(top_k * oversample, 100))
        cand: list[tuple[int, int]] = []
        for s, e in ranges:
            seg = self.bits[s:e]
            dist = [(qb ^ b).bit_count() for b in seg]
            cand.extend((dist[r], s + r) for r in heapq.nsmallest(want, range(len(seg)), key=dist.__getitem__))
        cand = heapq.nsmallest(want, cand)
        rescored = ((_dot(q, self.vectors[i]), i) for _, i in cand)
        return [(self.ids[i], sc) for sc, i in heapq.nlargest(top_k, rescored)]
