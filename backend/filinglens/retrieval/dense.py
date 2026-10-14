"""Brute-force dense search over stored embeddings."""
from array import array

from ..embeddings import cosine, embed_query


class DenseIndex:
    def __init__(self, vectors: dict[str, array]):
        self.vectors = vectors

    def search(self, query: str, top_k: int = 10, allowed: set[str] | None = None):
        q = embed_query(query)
        scored = [(cid, cosine(q, v)) for cid, v in self.vectors.items()
                  if allowed is None or cid in allowed]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
