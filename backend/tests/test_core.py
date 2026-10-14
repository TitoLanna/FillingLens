import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filinglens import db  # noqa: E402
from filinglens.embeddings import cosine, embed_query as embed  # noqa: E402
from filinglens.evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from filinglens.ingestion.parse import parse_sections  # noqa: E402
from filinglens.ingestion.worker import ingest_document  # noqa: E402
from filinglens.retrieval.bm25 import BM25  # noqa: E402
from filinglens.retrieval.fusion import rrf  # noqa: E402
from filinglens.retrieval.service import PRESETS, Retriever  # noqa: E402
from filinglens import embeddings, http, llm  # noqa: E402
from filinglens.config import load  # noqa: E402
from filinglens.pipeline import _generate  # noqa: E402
from filinglens.text import chunk_words, tokenize  # noqa: E402
from filinglens.trace import Trace  # noqa: E402
from unittest import mock  # noqa: E402

HTML = ("<p>Item 1A. Risk Factors</p><p>We depend on three GPU suppliers.</p>"
        "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>")
META = {"ticker": "T", "company": "Tee", "fiscal_year": 2024, "url": "u"}


class CoreTests(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("The GPU suppliers of Contoso"), ["gpu", "supplier", "contoso"])

    def test_chunk_overlap(self):
        text = " ".join(str(i) for i in range(10))
        self.assertEqual(chunk_words(text, size=4, overlap=2),
                         ["0 1 2 3", "2 3 4 5", "4 5 6 7", "6 7 8 9"])

    def test_bm25(self):
        hits = BM25({"a": "gpu supply chain risk", "b": "revenue grew"}).search("gpu supply")
        self.assertEqual([h[0] for h in hits], ["a"])

    def test_rrf_prefers_agreement(self):
        self.assertEqual(rrf([["a", "b"], ["b", "c"]])[0][0], "b")

    def test_embedding_similarity(self):
        q, near, far = embed("gpu supplier risk"), embed("gpu suppliers risk"), embed("revenue margin")
        self.assertGreater(cosine(q, near), cosine(q, far))

    def test_metrics(self):
        self.assertEqual(recall_at_k(["a", "b"], {"b", "z"}, 2), 0.5)
        self.assertEqual(reciprocal_rank(["a", "b"], {"b"}), 0.5)
        self.assertAlmostEqual(ndcg_at_k(["a"], {"a"}, 5), 1.0)

    def test_parse_sections_and_tables(self):
        secs = dict(parse_sections(HTML))
        self.assertIn("Item 1A. Risk Factors", secs)
        self.assertIn("| A | B |", secs["Item 1A. Risk Factors"])

    def test_ingest_is_idempotent_and_replaces(self):
        conn = db.connect(Path(":memory:"))
        self.assertEqual(ingest_document(conn, META, HTML), "ingested")
        self.assertEqual(ingest_document(conn, META, HTML), "skipped")
        self.assertEqual(ingest_document(conn, META, HTML + "<p>more</p>"), "replaced")
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 1)

    def test_baseline_vs_hybrid_retrieve(self):
        conn = db.connect(Path(":memory:"))
        ingest_document(conn, META, HTML)
        r = Retriever(conn)
        base, _ = r.retrieve("GPU suppliers", None, PRESETS["baseline"])
        hyb, stages = r.retrieve("GPU suppliers", None, PRESETS["hybrid"])
        self.assertIsNone(base[0]["scores"]["bm25"])
        self.assertIsNotNone(hyb[0]["scores"]["rrf"])
        self.assertEqual([s["name"] for s in stages], ["bm25", "dense", "rrf"])


    def test_xbrl_header_skipped(self):
        secs = parse_sections("<ix:header><div>hidden-meta</div></ix:header><p>visible text</p>")
        self.assertEqual(secs, [("Preamble", "visible text")])

    def test_voyage_embedder_batches_and_normalises(self):
        calls = []

        def fake_post(url, headers, body, timeout=60):
            calls.append(body)
            return {"data": [{"index": i, "embedding": [3.0, 4.0]} for i in reversed(range(len(body["input"])))]}

        emb = embeddings.VoyageEmbedder("k", "voyage-3.5")
        with mock.patch.object(http, "post_json", fake_post):
            vecs = emb.embed(["x"] * 70, "document")
        self.assertEqual([len(c["input"]) for c in calls], [64, 6])
        self.assertEqual(calls[0]["input_type"], "document")
        self.assertAlmostEqual(vecs[0][0], 0.6, places=5)
        self.assertEqual(emb.name, "voyage:voyage-3.5")

    def test_anthropic_stream_parsing_and_cost(self):
        events = [("message_start", {"message": {"usage": {"input_tokens": 1000}}}),
                  ("content_block_delta", {"delta": {"type": "text_delta", "text": "Revenue rose [1]."}}),
                  ("message_delta", {"usage": {"output_tokens": 50}})]
        settings = load().__class__(**{**load().__dict__, "llm_provider": "anthropic",
                                       "anthropic_api_key": "k", "llm_price_in": 3.0, "llm_price_out": 15.0})
        trace = Trace()
        with mock.patch.object(http, "post_sse", lambda *a, **k: iter(events)):
            out = list(_generate(settings, "q", [{"id": 1, "company": "C", "fiscal_year": 2024,
                                                  "section": "S", "text": "t"}], trace))
        self.assertEqual(out, [{"type": "token", "text": "Revenue rose [1]."}])
        self.assertEqual((trace.prompt_tokens, trace.completion_tokens), (1000, 50))
        self.assertAlmostEqual(trace.cost_usd, 0.00375)

    def test_not_found_becomes_refusal(self):
        for chunks in (["NOT_FOUND"], ["NOT", "_FOUND"], ["NOT_FOUND. The passages lack it."]):
            with mock.patch.object(llm, "stream_answer", lambda *a, c=chunks: iter(c)):
                out = list(_generate(load(), "q", [], Trace()))
            self.assertEqual([e["type"] for e in out], ["refusal"], chunks)

    def test_answer_starting_like_sentinel_is_not_swallowed(self):
        with mock.patch.object(llm, "stream_answer", lambda *a: iter(["NOT", "ably, revenue grew [1]."])):
            out = list(_generate(load(), "q", [], Trace()))
        self.assertEqual("".join(e["text"] for e in out), "NOTably, revenue grew [1].")


if __name__ == "__main__":
    unittest.main()
