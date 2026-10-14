"""Okapi BM25, written from scratch."""
import math
from collections import Counter

from ..text import tokenize


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: dict[str, Counter] = {}
        self.length: dict[str, int] = {}
        df: Counter = Counter()
        for doc_id, text in docs.items():
            tokens = tokenize(text)
            self.tf[doc_id] = Counter(tokens)
            self.length[doc_id] = len(tokens)
            df.update(set(tokens))
        n = len(docs)
        self.avg_len = (sum(self.length.values()) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 10, allowed: set[str] | None = None):
        """Return [(doc_id, score)] best first. Zero-score docs are dropped."""
        terms = tokenize(query)
        scores = []
        for doc_id, tf in self.tf.items():
            if allowed is not None and doc_id not in allowed:
                continue
            norm = 1 - self.b + self.b * self.length[doc_id] / (self.avg_len or 1)
            score = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm)
            if score > 0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
