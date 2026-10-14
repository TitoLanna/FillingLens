import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filinglens import db  # noqa: E402
from filinglens.embeddings import cosine, embed_query as embed  # noqa: E402
from filinglens.evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from filinglens.ingestion import edgar  # noqa: E402
from filinglens.ingestion.chunking import chunk_blocks, chunk_table  # noqa: E402
from filinglens.ingestion.parse import clean_row, parse_blocks  # noqa: E402
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

    def test_bm25_mask_matches_allowed(self):
        docs = {"a": "gpu supply chain", "b": "gpu demand", "c": "revenue grew"}
        idx = BM25(docs)
        self.assertEqual([h[0] for h in idx.search("gpu", allowed={"b"})], ["b"])
        self.assertEqual([h[0] for h in idx.search("gpu", mask=bytearray([0, 1, 0]))], ["b"])
        self.assertEqual(idx.search("gpu", allowed=set()), [])

    def test_bm25_skips_common_terms_unless_nothing_else(self):
        idx = BM25({"a": "chip gpu", "b": "chip cpu", "c": "chip ram"})
        self.assertEqual([h[0] for h in idx.search("chip gpu")], ["a"])   # "chip" is in every doc: skipped
        self.assertEqual(len(idx.search("chip")), 3)                      # but kept when it is all there is

    def test_dense_binary_agrees_with_exact_and_respects_ranges(self):
        from filinglens.retrieval.dense import DenseIndex
        texts = [f"filing {i} about topic{i % 7} risk{i}" for i in range(60)]
        idx = DenseIndex([str(i) for i in range(60)], [embed(t) for t in texts])
        q = texts[13]
        exact = idx.search(q, 5, mode="exact")
        self.assertEqual(exact[0][0], "13")
        self.assertEqual(idx.search(q, 5, mode="binary", oversample=20)[0][0], "13")
        self.assertEqual(idx.search(q, 5, mode="auto"), exact)
        ranged = idx.search(q, 5, ranges=[(20, 30)], mode="binary")
        self.assertTrue(all(20 <= int(i) < 30 for i, _ in ranged))
        self.assertEqual(idx.search(q, 5, ranges=[]), [])
        with self.assertRaises(ValueError):
            idx.search(q, mode="bogus")

    def test_retriever_filters_by_ticker_and_year(self):
        conn = db.connect(Path(":memory:"))
        for ticker, year in [("AAA", 2023), ("BBB", 2023), ("AAA", 2024)]:
            ingest_document(conn, {**META, "ticker": ticker, "fiscal_year": year, "url": f"u{ticker}{year}"},
                            HTML.replace("three", ticker))
        r = Retriever(conn)
        for name in ("hybrid", "hybrid-exact"):
            hits, _ = r.retrieve("GPU suppliers", {"tickers": ["aaa"], "fiscal_years": [2024]}, PRESETS[name])
            self.assertTrue(hits)
            self.assertTrue(all(h["ticker"] == "AAA" and h["fiscal_year"] == 2024 for h in hits))
        hits, _ = r.retrieve("GPU suppliers", {"tickers": ["AAA", "BBB"]}, PRESETS["hybrid"])
        self.assertEqual({h["ticker"] for h in hits}, {"AAA", "BBB"})
        self.assertEqual(r.retrieve("GPU suppliers", {"tickers": ["ZZZ"]}, PRESETS["hybrid"])[0], [])

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

    def test_clean_row_collapses_sec_sparse_cells(self):
        self.assertEqual(clean_row(["Products", "$", "294,866", "", "", "$", "298,085"]), ["Products", "$294,866", "$298,085"])
        self.assertEqual(clean_row(["Other", "(565", ")", "", "12", "%"]), ["Other", "(565)", "12%"])
        self.assertEqual(clean_row(["", " ", ""]), [])

    def test_table_cells_do_not_glue_words(self):
        html = "<table><tr><td>September 28,<br/>2024</td><td>x</td></tr><tr><td>a</td><td>1</td></tr></table>"
        (title, blocks), = parse_blocks(html)
        self.assertEqual(blocks[0][1][0][0], "September 28, 2024")

    def test_layout_tables_become_prose(self):
        html = "<table><tr><td>/s/ Jane Doe</td></tr><tr><td>Chief Executive Officer</td></tr></table>"
        (_, blocks), = parse_blocks(html)
        self.assertEqual([k for k, _ in blocks], ["text"])

    def table_html(self, n):
        rows = "".join(f"<tr><td>Item {i}</td><td>$</td><td>{i * 100:,}</td><td>{i * 90:,}</td></tr>" for i in range(n))
        return ("<p>Item 8. Financial Statements</p><p>CONSOLIDATED BALANCE SHEETS</p><p>(In millions)</p>"
                "<table><tr><td></td><td>2024</td><td>2023</td></tr>" + rows + "</table><p>After the table.</p>")

    def test_small_table_is_one_chunk_with_caption_and_aligned_header(self):
        (_, blocks), = parse_blocks(self.table_html(5))
        chunks = chunk_blocks(blocks)
        table = [c for c in chunks if "| --- |" in c]
        self.assertEqual(len(table), 1)
        self.assertTrue(table[0].startswith("CONSOLIDATED BALANCE SHEETS\n(In millions)\n"))
        self.assertIn("|  | 2024 | 2023 |", table[0].replace("| --- |", ""))
        self.assertIn("| Item 3 | $300 | 270 |", table[0])
        self.assertTrue(any("After the table." in c for c in chunks))

    def test_long_table_splits_between_rows_and_repeats_header(self):
        (_, blocks), = parse_blocks(self.table_html(120))
        chunks = [c for c in chunk_blocks(blocks, table_words=120) if "| --- |" in c]
        self.assertGreater(len(chunks), 3)
        seen = []
        for c in chunks:
            self.assertTrue(c.startswith("CONSOLIDATED BALANCE SHEETS"))     # caption repeated
            self.assertIn("|  | 2024 | 2023 |", c)                            # header repeated
            for line in c.splitlines():
                if line.startswith("| Item "):
                    self.assertEqual(line.count("|"), 4)                        # 3 cells = 4 pipes: rows are never cut in half
                    seen.append(line)
        self.assertEqual(len(seen), 120)                                        # every row exactly once

    def test_chunk_table_without_header(self):
        self.assertEqual(len(chunk_table([["a", "1"], ["b", "2"]], "", 100)), 1)

    def test_find_10k_walks_history_pages_and_extra_ciks(self):
        def block(rows):
            return {"form": [r[0] for r in rows], "reportDate": [r[1] for r in rows],
                    "accessionNumber": [r[2] for r in rows], "primaryDocument": [r[3] for r in rows]}
        pages = {
            "CIK0000000100.json": {"name": "NEW CO", "filings": {"recent": block([("4", "", "x-1", "a.xml"), ("10-K", "2024-12-31", "0001-24-1", "k24.htm")]),
                                                                 "files": [{"name": "hist-1.json"}]}},
            "hist-1.json": block([("424B2", "", "x-2", "b.htm"), ("10-K", "2022-12-31", "0001-22-1", "k22.htm"), ("10-K/A", "2023-12-31", "0001-23-9", "a.htm")]),
            "CIK0000000200.json": {"name": "OLD CO", "filings": {"recent": block([("10-K", "2023-12-31", "0002-23-1", "k23.htm")]), "files": []}},
        }
        fake = lambda url, ua: next(v for k, v in pages.items() if url.endswith(k))
        with mock.patch.object(edgar, "_get", fake), \
                mock.patch.object(edgar, "lookup_ticker", lambda t, ua: {"cik_str": 100, "title": "T"}):
            without = edgar.find_10k_filings("zz", [2022, 2023, 2024], "ua")
            both = edgar.find_10k_filings("zz", [2022, 2023, 2024], "ua", extra_ciks=[200])
        self.assertEqual([d["fiscal_year"] for d in without], [2022, 2024])  # history page found 2022; the 10-K/A is ignored
        self.assertEqual([d["fiscal_year"] for d in both], [2022, 2023, 2024])
        self.assertTrue(both[1]["url"].endswith("/200/0002231/k23.htm"))
        self.assertEqual({d["ticker"] for d in both}, {"ZZ"})
        self.assertEqual(both[0]["company"], "NEW CO")

    def test_raw_filing_cache(self):
        import tempfile
        url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
        calls = []
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(http, "get_text", lambda u, h, timeout=60: calls.append(u) or "<html>x</html>"):
            self.assertEqual(edgar.raw_path(url, Path(d)).name, "320193_000032019324000123_aapl-20240928.htm")
            self.assertEqual(edgar.fetch_filing(url, "ua", Path(d)), ("<html>x</html>", False))
            self.assertEqual(edgar.fetch_filing(url, "ua", Path(d)), ("<html>x</html>", True))
            self.assertEqual(len(calls), 1)  # second read came from disk
            self.assertEqual([p.name for p in Path(d).iterdir()], ["320193_000032019324000123_aapl-20240928.htm"])
        with self.assertRaises(ValueError):
            edgar.raw_path("https://www.sec.gov/", Path("/tmp"))

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
