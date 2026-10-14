"""Okapi BM25 over an inverted index, written from scratch.

Postings are two parallel typed arrays per term (document index, term frequency), so a query only
touches the documents that contain its terms instead of scanning the whole corpus.
"""
import heapq
import math
from array import array
from collections import Counter
from operator import itemgetter

from ..text import tokenize

# A term present in more than this share of documents adds little signal but the most work.
# It is skipped, unless that would leave the query with nothing to match.
MAX_DF_RATIO = 0.5


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.ids = list(docs)
        self.pos = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.k1 = k1
        self.postings: dict[str, tuple[array, array]] = {}
        lengths = array("I")
        for i, text in enumerate(docs.values()):
            tokens = tokenize(text)
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                entry = self.postings.get(term)
                if entry is None:
                    entry = self.postings[term] = (array("I"), array("H"))
                entry[0].append(i)
                entry[1].append(min(tf, 65535))
        n = len(self.ids)
        avg = (sum(lengths) / n) if n else 0.0
        # per-document length normalisation, computed once: k1 * (1 - b + b * len / avg)
        self.norm = array("f", (k1 * (1 - b + b * ln / (avg or 1)) for ln in lengths))
        self.idf = {t: math.log(1 + (n - len(p[0]) + 0.5) / (len(p[0]) + 0.5)) for t, p in self.postings.items()}
        self.n = n

    def search(self, query: str, top_k: int = 10, allowed: set[str] | None = None,
               mask: bytearray | None = None):
        """Return [(doc_id, score)] best first. Zero-score docs are dropped.

        `mask` is a bytearray with one byte per document (non-zero = eligible); `allowed` is the
        same restriction as a set of ids. Pass at most one.
        """
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self.postings]
        if len(terms) > 1:
            rare = [t for t in terms if len(self.postings[t][0]) <= self.n * MAX_DF_RATIO]
            terms = rare or terms
        if allowed is not None:
            mask = bytearray(self.n)
            for doc_id in allowed:
                i = self.pos.get(doc_id)
                if i is not None:
                    mask[i] = 1
        scores: dict[int, float] = {}
        k1p1, norm = self.k1 + 1, self.norm
        for term in terms:
            idf = self.idf[term]
            ids, tfs = self.postings[term]
            if mask is None:
                for i, f in zip(ids, tfs):
                    scores[i] = scores.get(i, 0.0) + idf * f * k1p1 / (f + norm[i])
            else:
                for i, f in zip(ids, tfs):
                    if mask[i]:
                        scores[i] = scores.get(i, 0.0) + idf * f * k1p1 / (f + norm[i])
        best = heapq.nlargest(top_k, scores.items(), key=itemgetter(1))
        return [(self.ids[i], s) for i, s in best]
