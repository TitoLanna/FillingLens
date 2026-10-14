import json
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filinglens import db, llm  # noqa: E402
from filinglens.config import load  # noqa: E402
from filinglens.evals import baseline, golden, report  # noqa: E402
from filinglens.evals.calibration import agreement  # noqa: E402
from filinglens.evals.judge import check_citation_markers, judge_result  # noqa: E402
from filinglens.evals.runner import rows_from_run, run_eval, stratified_subset, summarize  # noqa: E402
from filinglens.ingestion.worker import ingest_document  # noqa: E402
from filinglens.retrieval.service import PRESETS, Retriever  # noqa: E402

SETTINGS = replace(load(), llm_provider="anthropic", anthropic_api_key="k", llm_price_in=1.0, llm_price_out=2.0)
FILLER = " ".join(["word"] * 45)
HTML = {
    ("AAA", 2024): f"<p>Item 1A. Risk Factors</p><p>Aaa depends on three GPU suppliers. {FILLER}</p>"
                   f"<p>Item 7. MD&amp;A</p><p>Aaa revenue grew 21 percent. {FILLER}</p>",
    ("BBB", 2024): f"<p>Item 1A. Risk Factors</p><p>Bbb relies on one foundry. {FILLER}</p>",
}


def make_db():
    conn = db.connect(Path(":memory:"))
    for (ticker, year), html in HTML.items():
        ingest_document(conn, {"ticker": ticker, "company": ticker.title(), "fiscal_year": year, "url": "u"}, html)
    return conn


def add_q(conn, qid, qtype, relevant, expected="21 percent", status="verified", q=None):
    conn.execute("INSERT INTO golden_questions (id, question, type, expected_answer, relevant_chunk_ids, status) "
                 "VALUES (?,?,?,?,?,?)", (qid, q or f"question {qid} GPU suppliers", qtype, expected,
                                           json.dumps(relevant), status))


class GoldenTests(unittest.TestCase):
    def test_sample_sources_by_type(self):
        pool = golden.load_pool(make_db())
        rng = random.Random(1)
        self.assertEqual(len(golden.sample_sources(pool, "factual", rng)), 1)
        self.assertIsNone(golden.sample_sources(pool, "table", rng))  # no tables in this corpus
        hops = [h for h in (golden.sample_sources(pool, "multi-hop", rng) for _ in range(20)) if h]
        self.assertTrue(hops)  # None is legal when the draw lands on a one-chunk document
        hop = hops[0]
        self.assertEqual(hop[0]["doc_id"], hop[1]["doc_id"])
        self.assertNotEqual(hop[0]["id"], hop[1]["id"])
        for _ in range(20):  # comparison needs a different document, same section
            a, b = golden.sample_sources(pool, "comparison", rng) or (None, None)
            if a:
                self.assertNotEqual(a["doc_id"], b["doc_id"])
                self.assertEqual(a["section"], b["section"])

    def test_generate_candidates(self):
        conn = make_db()
        reply = '```json\n{"question": "What does Aaa depend on?", "expected_answer": "GPU suppliers"}\n```'
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        with mock.patch.object(llm, "complete", return_value=(reply, usage)):
            out = golden.generate_candidates(conn, SETTINGS, {"factual": 2}, seed=3)
        self.assertEqual(out["created"], 1)  # second identical question is de-duplicated
        row = conn.execute("SELECT * FROM golden_questions").fetchone()
        self.assertEqual(row["status"], "candidate")
        self.assertEqual(len(json.loads(row["relevant_chunk_ids"])), 1)

    def test_unanswerable_has_no_sources_and_no_answer(self):
        conn = make_db()
        reply = '{"question": "What is the CEO salary of Aaa?", "expected_answer": null}'
        with mock.patch.object(llm, "complete", return_value=(reply, {"prompt_tokens": 1, "completion_tokens": 1})):
            golden.generate_candidates(conn, SETTINGS, {"unanswerable": 1})
        row = conn.execute("SELECT * FROM golden_questions").fetchone()
        self.assertEqual((row["expected_answer"], row["relevant_chunk_ids"]), (None, "[]"))

    def test_provider_misconfiguration_fails_loudly(self):
        with self.assertRaises(ValueError):
            golden.generate_candidates(make_db(), load(), {"factual": 1})  # stub provider

    def test_review_edits_and_export_roundtrip(self):
        conn = make_db()
        cid = conn.execute("SELECT id FROM chunks").fetchone()["id"]
        add_q(conn, "g1", "factual", [cid], status="candidate")
        with self.assertRaises(ValueError):
            golden.update_fields(conn, "g1", relevant_chunk_ids=["nope"])
        golden.update_fields(conn, "g1", question="Edited?")
        golden.set_status(conn, "g1", "verified")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "g.jsonl"
            self.assertEqual(golden.export_verified(conn, path), 1)
            conn.execute("DELETE FROM golden_questions")
            self.assertEqual(golden.import_verified(conn, path), 1)
        self.assertEqual(conn.execute("SELECT question, status FROM golden_questions").fetchone()[:], ("Edited?", "verified"))


class JudgeTests(unittest.TestCase):
    def test_citation_marker_precheck(self):
        self.assertIn("no citation", check_citation_markers("plain", 3))
        self.assertIn("out of range", check_citation_markers("x [4]", 3))
        self.assertIsNone(check_citation_markers("x [1][3]", 3))

    def judge(self, replies, qtype="factual", refused=False, answer="Revenue grew [1]."):
        seq = iter(replies)
        with mock.patch.object(llm, "complete", lambda *a, **k: (next(seq), {"prompt_tokens": 5, "completion_tokens": 2})):
            return judge_result(SETTINGS, qtype, "q", "ref", answer, [{"id": 1, "text": "t"}], refused)

    def test_all_pass_and_fail_mix(self):
        ok, bad = '{"reason": "r", "pass": true}', 'Sure! {"reason": "r", "pass": false}'
        scores, usage = self.judge([ok, ok, bad])  # faithfulness, citations, correctness
        self.assertEqual((scores["faithfulness"], scores["citations"], scores["correctness"]), (True, True, False))
        self.assertEqual(usage["prompt_tokens"], 15)

    def test_judge_error_is_none_not_fail(self):
        scores, _ = self.judge(['{"reason": "r"}', '{"pass": true, "reason": "r"}', "not json"])
        self.assertIsNone(scores["faithfulness"])
        self.assertIsNone(scores["correctness"])
        self.assertEqual(scores["errors"], 2)

    def test_refusal_cases_need_no_model(self):
        with mock.patch.object(llm, "complete", side_effect=AssertionError("no model call expected")):
            s1, _ = judge_result(SETTINGS, "unanswerable", "q", None, "", [], True)
            s2, _ = judge_result(SETTINGS, "factual", "q", "ref", "", [], True)
        self.assertTrue(s1["refusal"])
        self.assertFalse(s2["refusal"])
        self.assertFalse(s2["correctness"])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.conn = make_db()
        self.gpu = next(r["id"] for r in self.conn.execute("SELECT id, text FROM chunks") if "GPU" in r["text"])
        add_q(self.conn, "q1", "factual", [self.gpu], q="Aaa GPU suppliers")
        add_q(self.conn, "q2", "unanswerable", [], expected=None, q="zebra migration patterns")
        add_q(self.conn, "q3", "factual", [self.gpu], status="candidate")  # must be ignored
        self.retriever = Retriever(self.conn)

    def fake_stream(self, settings, question, passages, usage):
        usage.update(prompt_tokens=1000, completion_tokens=100)
        yield f"Aaa depends on GPU suppliers [{passages[0]['id']}]."

    def fake_complete(self, settings, system, user, **kw):
        return '{"reason": "fine", "pass": true}', {"prompt_tokens": 10, "completion_tokens": 5}

    def test_retrieval_only_ignores_unverified(self):
        out = run_eval(self.conn, self.retriever, SETTINGS, PRESETS["hybrid"], "r")
        s = out["summary_metrics"]
        self.assertEqual((s["questions"], s["recall_at_5"], s["mode"]), (2, 1.0, "retrieval"))
        self.assertIsNone(s["faithfulness"])

    def test_generate_and_judge(self):
        with mock.patch.object(llm, "stream_answer", self.fake_stream), \
                mock.patch.object(llm, "complete", self.fake_complete):
            out = run_eval(self.conn, self.retriever, SETTINGS, PRESETS["hybrid"], "full", generate=True, judge=True)
        s = out["summary_metrics"]
        self.assertEqual(s["mode"], "judge")
        self.assertEqual((s["faithfulness"], s["citation_accuracy"], s["answer_correctness"]), (1.0, 1.0, 1.0))
        self.assertEqual(s["refusal_accuracy"], 1.0)  # q2 refused (no lexical/dense support), q1 answered
        self.assertAlmostEqual(s["cost_per_query_usd"], 0.0006)  # (1000*1 + 100*2)/1e6 for q1, 0 for the refusal, averaged
        self.assertGreater(s["judge_cost_usd"], 0)
        res = {r["question_id"]: r for r in self.conn.execute("SELECT * FROM eval_results WHERE run_id=?", (out["id"],))}
        self.assertEqual(set(res), {"q1", "q2"})
        self.assertIn("GPU", res["q1"]["answer"])
        self.assertEqual(res["q2"]["refused"], 1)

    def test_summary_rebuilds_from_stored_rows(self):
        out = run_eval(self.conn, self.retriever, SETTINGS, PRESETS["baseline"], "r")
        again = summarize(rows_from_run(self.conn, out["id"]))
        for k in ("recall_at_5", "mrr", "ndcg_at_5", "questions"):
            self.assertEqual(again[k], out["summary_metrics"][k])
        self.assertEqual(summarize(rows_from_run(self.conn, out["id"], {"q1"}))["questions"], 1)

    def test_baseline_export_import_into_fresh_db(self):
        with mock.patch.object(llm, "stream_answer", self.fake_stream), \
                mock.patch.object(llm, "complete", self.fake_complete):
            out = run_eval(self.conn, self.retriever, SETTINGS, PRESETS["hybrid"], "v0", generate=True, judge=True)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "baseline.json"
            self.assertEqual(baseline.export_run(self.conn, out["id"], path), 2)
            fresh = make_db()                       # what CI has: corpus + golden set, no runs
            for r in self.conn.execute("SELECT * FROM golden_questions"):
                fresh.execute("INSERT OR REPLACE INTO golden_questions (id, question, type, expected_answer, "
                              "relevant_chunk_ids, status) VALUES (?,?,?,?,?,?)",
                              (r["id"], r["question"], r["type"], r["expected_answer"], r["relevant_chunk_ids"], r["status"]))
            self.assertEqual(baseline.import_run(fresh, path), out["id"])
            self.assertEqual(baseline.import_run(fresh, path), out["id"])  # idempotent
        self.assertEqual(fresh.execute("SELECT is_baseline FROM eval_runs").fetchone()[0], 1)
        self.assertEqual(fresh.execute("SELECT COUNT(*) FROM eval_results").fetchone()[0], 2)
        same = summarize(rows_from_run(fresh, out["id"]))
        self.assertEqual(same["faithfulness"], out["summary_metrics"]["faithfulness"])
        with self.assertRaises(ValueError):
            baseline.export_run(self.conn, "run_missing", "/tmp/x.json")

    def test_errors(self):
        with self.assertRaises(ValueError):  # no real model configured
            run_eval(self.conn, self.retriever, load(), PRESETS["hybrid"], "x", generate=True)
        self.conn.execute("UPDATE golden_questions SET status='rejected'")
        with self.assertRaises(ValueError):
            run_eval(self.conn, self.retriever, SETTINGS, PRESETS["hybrid"], "x")

    def test_generation_failure_is_isolated(self):
        def boom(*a, **k):
            raise RuntimeError("upstream down")
            yield
        with mock.patch.object(llm, "stream_answer", boom):
            out = run_eval(self.conn, self.retriever, SETTINGS, PRESETS["hybrid"], "x", generate=True)
        self.assertEqual(out["summary_metrics"]["generation_errors"], 1)

    def test_stratified_subset(self):
        qs = [{"id": f"{t}{i}", "type": t} for t in ("a", "b") for i in range(4)]
        self.assertEqual(sorted(q["type"] for q in stratified_subset(qs, 4)), ["a", "a", "b", "b"])
        self.assertEqual(len(stratified_subset(qs, 99)), 8)


class ReportTests(unittest.TestCase):
    def test_compare_gate_and_markdown(self):
        base = {"recall_at_5": 0.80, "faithfulness": 0.90, "p95_latency_ms": 100.0}
        new = {"recall_at_5": 0.70, "faithfulness": 0.92, "p95_latency_ms": 300.0, "mrr": 0.5}
        rows = report.compare(base, new)
        bad = report.regressions(rows, 0.03)
        self.assertEqual([r["key"] for r in bad], ["recall_at_5"])  # latency is informational, mrr has no baseline
        md = report.to_markdown(rows, "T")
        self.assertIn("⚠️", md)
        self.assertIn("| MRR | – | 0.500 | – |", md)

    def test_agreement_edge_cases(self):
        self.assertEqual(agreement([])["n"], 0)
        self.assertEqual(agreement([(True, True), (False, False)])["kappa"], 1.0)


if __name__ == "__main__":
    unittest.main()
