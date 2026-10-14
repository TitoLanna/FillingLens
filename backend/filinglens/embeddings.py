"""Embedding providers: offline feature hashing, or Voyage AI over raw HTTP.

`configure(settings)` picks the provider once at startup. Every stored chunk records
`model_name()`, and retrieval/ingest treat vectors from any other model as stale.
"""
import hashlib
import math
from array import array

from . import http
from .text import tokenize

HASH_DIM = 256
VOYAGE_BATCH = 64  # inputs per request, well under the API limit


def _normalise(vec) -> array:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return array("f", (x / norm for x in vec))


class HashEmbedder:
    name = f"hash-{HASH_DIM}-v1"

    def embed(self, texts: list[str], kind: str) -> list[array]:
        out = []
        for text in texts:
            vec = [0.0] * HASH_DIM
            for tok in tokenize(text):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                vec[h % HASH_DIM] += 1.0 if (h >> 63) & 1 else -1.0
            out.append(_normalise(vec))
        return out


class VoyageEmbedder:
    URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("EMBEDDING_PROVIDER=voyage requires VOYAGE_API_KEY")
        self.api_key, self.model = api_key, model
        self.name = f"voyage:{model}"

    def embed(self, texts: list[str], kind: str) -> list[array]:
        out: list[array] = []
        for i in range(0, len(texts), VOYAGE_BATCH):
            body = {"input": texts[i:i + VOYAGE_BATCH], "model": self.model,
                    "input_type": "query" if kind == "query" else "document"}
            resp = http.post_json(self.URL, {"Authorization": f"Bearer {self.api_key}"}, body)
            rows = sorted(resp["data"], key=lambda d: d["index"])
            out.extend(_normalise(r["embedding"]) for r in rows)
        return out


_provider = HashEmbedder()


def configure(settings) -> None:
    global _provider
    if settings.embedding_provider == "voyage":
        _provider = VoyageEmbedder(settings.voyage_api_key, settings.voyage_model)
    elif settings.embedding_provider == "hash":
        _provider = HashEmbedder()
    else:
        raise ValueError(f"unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")


def model_name() -> str:
    return _provider.name


def embed_documents(texts: list[str]) -> list[array]:
    return _provider.embed(texts, "document")


def embed_query(text: str) -> array:
    return _provider.embed([text], "query")[0]


def to_blob(vec: array) -> bytes:
    return vec.tobytes()


def from_blob(blob: bytes) -> array:
    out = array("f")
    out.frombytes(blob)
    return out


def cosine(a: array, b: array) -> float:
    """Vectors are L2-normalised, so cosine is the dot product."""
    return sum(x * y for x, y in zip(a, b))
