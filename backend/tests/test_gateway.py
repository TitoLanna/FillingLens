import http.client
import json
import sys
import threading
import time
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gateway.config import GatewaySettings  # noqa: E402
from gateway.ratelimit import RateLimiter  # noqa: E402
from gateway.server import make_server  # noqa: E402
from gateway.validate import ValidationError, clean_query  # noqa: E402


class RateLimiterTests(unittest.TestCase):
    def test_burst_then_refill(self):
        now = [0.0]
        rl = RateLimiter(60, clock=lambda: now[0])  # 1 token/second, burst 60
        self.assertTrue(all(rl.allow("a")[0] for _ in range(60)))
        ok, retry = rl.allow("a")
        self.assertEqual((ok, retry), (False, 1))
        self.assertTrue(rl.allow("b")[0])  # other clients unaffected
        now[0] += 2.0
        self.assertTrue(rl.allow("a")[0])

    def test_eviction_keeps_memory_bounded(self):
        now = [0.0]
        rl = RateLimiter(6, clock=lambda: now[0], max_keys=5)
        for i in range(5):
            rl.allow(f"k{i}")
        now[0] += 1000
        rl.allow("new")
        self.assertLessEqual(len(rl._buckets), 2)


class ValidateTests(unittest.TestCase):
    def test_accepts_and_strips_unknown_fields(self):
        out = clean_query({"question": "  hi  ", "filters": {"tickers": ["AAPL"], "fiscal_years": [2024], "x": 1},
                           "top_k": 5, "pipeline": {"dense": False}, "evil": 1})
        self.assertEqual(out, {"question": "hi", "filters": {"tickers": ["AAPL"], "fiscal_years": [2024]}, "top_k": 5})

    def test_rejects_bad_input(self):
        bad = [None, [], {}, {"question": "  "}, {"question": 5}, {"question": "x" * 1001},
               {"question": "q", "top_k": 0}, {"question": "q", "top_k": True}, {"question": "q", "top_k": 21},
               {"question": "q", "filters": []}, {"question": "q", "filters": {"tickers": ["a b"]}},
               {"question": "q", "filters": {"tickers": ["A"] * 11}}, {"question": "q", "filters": {"fiscal_years": ["2024"]}},
               {"question": "q", "filters": {"fiscal_years": [1800]}}]
        for body in bad:
            with self.subTest(body=body), self.assertRaises(ValidationError):
                clean_query(body)


class FakeBackend(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list = []
    release = threading.Event()
    client_gone = threading.Event()
    extra_events = 0   # additional slow events after release, to exercise mid-stream cancellation
    sent = 0

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")  # gateway must strip this
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeBackend.seen.append(("GET", self.path, None, dict(self.headers)))
        if self.path == "/healthz":
            return self._json(200, {"status": "ok"})
        if self.path == "/v1/corpus":
            return self._json(200, {"companies": [], "fiscal_years": []})
        if self.path == "/v1/boom":
            return self._json(500, {"error": "Traceback: secret internals"})
        self._json(200, {"path": self.path})

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        FakeBackend.seen.append(("POST", self.path, json.loads(body), dict(self.headers)))
        if self.path != "/v1/query":
            return self._json(200, {"ok": True})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(b'data: {"type":"citations","citations":[]}\n\n')
            self.wfile.flush()
            if FakeBackend.release.wait(5):
                for _ in range(1 + FakeBackend.extra_events):
                    self.wfile.write(b'data: {"type":"token","text":"x"}\n\n')
                    self.wfile.flush()
                    FakeBackend.sent += 1
                    if FakeBackend.extra_events:
                        time.sleep(0.05)
            # detect the gateway closing its side (cancel propagation)
            self.connection.settimeout(0.3)
            try:
                if self.connection.recv(1) == b"":
                    FakeBackend.client_gone.set()
            except OSError:
                FakeBackend.client_gone.set()
        except (BrokenPipeError, ConnectionResetError):
            FakeBackend.client_gone.set()

    def log_message(self, *a):
        pass


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


class GatewayTests(unittest.TestCase):
    ORIGIN = "http://localhost:5173"

    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackend)
        cls.backend.daemon_threads = True
        cls.backend_port = serve(cls.backend)
        cls.base = GatewaySettings(
            host="127.0.0.1", port=0, upstream_host="127.0.0.1", upstream_port=cls.backend_port,
            allowed_origins=(cls.ORIGIN,), admin_token="s3cret", rate_query_per_min=3, rate_read_per_min=100,
            max_inflight_queries=2, max_body_bytes=512, upstream_timeout=5, trust_proxy=False)

    @classmethod
    def tearDownClass(cls):
        cls.backend.shutdown()
        cls.backend.server_close()

    def setUp(self):
        FakeBackend.seen.clear()
        FakeBackend.release.clear()
        FakeBackend.client_gone.clear()
        FakeBackend.extra_events, FakeBackend.sent = 0, 0
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.shutdown()
            s.server_close()

    def gateway(self, **overrides):
        cfg = replace(self.base, **overrides)
        server = make_server(cfg, log=lambda *_: None)
        self.servers.append(server)
        return serve(server)

    def call(self, port, method, path, body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
        payload = raw if raw is not None else (json.dumps(body) if body is not None else None)
        h = {"Content-Type": "application/json", **(headers or {})} if payload is not None else dict(headers or {})
        conn.request(method, path, body=payload, headers=h)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, resp, (json.loads(data) if data and resp.getheader("Content-Type") == "application/json" else data)

    def test_public_read_route_and_cors(self):
        port = self.gateway()
        status, resp, body = self.call(port, "GET", "/v1/corpus", headers={"Origin": self.ORIGIN})
        self.assertEqual(status, 200)
        self.assertEqual(resp.getheader("Access-Control-Allow-Origin"), self.ORIGIN)  # not the backend's "*"
        self.assertTrue(resp.getheader("X-Request-ID"))
        self.assertEqual(self.call(port, "GET", "/v1/corpus", headers={"Origin": "http://evil.example"})[0], 403)

    def test_preflight(self):
        port = self.gateway()
        status, resp, _ = self.call(port, "OPTIONS", "/v1/query", headers={"Origin": self.ORIGIN})
        self.assertEqual(status, 204)
        self.assertIn("POST", resp.getheader("Access-Control-Allow-Methods"))
        self.assertEqual(self.call(port, "OPTIONS", "/v1/query", headers={"Origin": "http://evil.example"})[0], 403)

    def test_unknown_and_internal_routes_hidden(self):
        port = self.gateway()
        self.assertEqual(self.call(port, "GET", "/v1/secret")[0], 404)
        self.assertEqual(self.call(port, "GET", "/")[0], 404)
        self.assertEqual(self.call(port, "DELETE", "/v1/corpus")[0], 404)

    def test_admin_routes(self):
        port = self.gateway()
        self.assertEqual(self.call(port, "POST", "/v1/ingest", {"tickers": ["A"]})[0], 401)
        self.assertEqual(self.call(port, "POST", "/v1/ingest", {"tickers": ["A"]}, {"Authorization": "Bearer nope"})[0], 401)
        status, _, body = self.call(port, "POST", "/v1/ingest", {"tickers": ["A"]}, {"Authorization": "Bearer s3cret"})
        self.assertEqual((status, body["ok"]), (200, True))
        off = self.gateway(admin_token="")
        self.assertEqual(self.call(off, "POST", "/v1/ingest", {"x": 1}, {"Authorization": "Bearer s3cret"})[0], 404)

    def test_query_is_validated_and_rebuilt(self):
        port = self.gateway(rate_query_per_min=50)
        self.assertEqual(self.call(port, "POST", "/v1/query", {"question": ""})[0], 400)
        self.assertEqual(self.call(port, "POST", "/v1/query", raw="{not json")[0], 400)
        self.assertEqual(self.call(port, "POST", "/v1/query", raw="{}", headers={"Content-Type": "text/plain"})[0], 415)
        FakeBackend.release.set()
        status, _, _ = self.call(port, "POST", "/v1/query", {"question": "q", "pipeline": {"dense": False}, "top_k": 3})
        self.assertEqual(status, 200)
        forwarded = [s for s in FakeBackend.seen if s[1] == "/v1/query"][0]
        self.assertEqual(forwarded[2], {"question": "q", "top_k": 3})  # `pipeline` stripped
        self.assertTrue(forwarded[3].get("X-Request-ID"))

    def test_body_limit(self):
        port = self.gateway(rate_query_per_min=50)
        status, _, body = self.call(port, "POST", "/v1/query", {"question": "x" * 600})
        self.assertEqual(status, 413)

    def test_rate_limit_returns_429_with_retry_after(self):
        port = self.gateway(rate_query_per_min=3)
        FakeBackend.release.set()
        codes = [self.call(port, "POST", "/v1/query", {"question": "q"})[0] for _ in range(4)]
        self.assertEqual(codes, [200, 200, 200, 429])
        _, resp, _ = self.call(port, "POST", "/v1/query", {"question": "q"})
        self.assertGreaterEqual(int(resp.getheader("Retry-After")), 1)
        self.assertEqual(self.call(port, "GET", "/v1/corpus")[0], 200)  # separate bucket for reads

    def test_keepalive_survives_a_rejected_post(self):
        port = self.gateway(rate_query_per_min=1)
        FakeBackend.release.set()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
        for expect in (200, 429, 429):
            conn.request("POST", "/v1/query", body=json.dumps({"question": "q"}), headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            resp.read()
            self.assertEqual(resp.status, expect)  # a leftover body must not corrupt later requests
        conn.close()

    def test_sse_streams_incrementally(self):
        port = self.gateway()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
        conn.request("POST", "/v1/query", body=json.dumps({"question": "q"}), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.getheader("Content-Type"), "text/event-stream")
        first = resp.read1(4096)  # backend is still blocked: this proves nothing is buffered
        self.assertIn(b"citations", first)
        FakeBackend.release.set()
        rest = b""
        while chunk := resp.read1(4096):
            rest += chunk
        self.assertIn(b'"token"', rest)
        conn.close()

    def test_client_disconnect_cancels_upstream(self):
        FakeBackend.extra_events = 59            # a long answer: 60 events, 50 ms apart
        port = self.gateway()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
        conn.request("POST", "/v1/query", body=json.dumps({"question": "q"}), headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read1(4096)
        FakeBackend.release.set()
        resp.read1(4096)
        resp.close()
        conn.close()                             # browser tab closed mid-answer (both ends of the socket)
        self.assertTrue(FakeBackend.client_gone.wait(5), "upstream connection was not closed")
        self.assertLess(FakeBackend.sent, 30, "upstream kept generating after the client left")

    def test_concurrency_cap(self):
        port = self.gateway(max_inflight_queries=1, rate_query_per_min=50)
        held = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
        held.request("POST", "/v1/query", body=json.dumps({"question": "q"}), headers={"Content-Type": "application/json"})
        held.getresponse().read1(4096)     # occupies the only slot
        status, resp, _ = self.call(port, "POST", "/v1/query", {"question": "q"})
        self.assertEqual(status, 503)
        self.assertEqual(resp.getheader("Retry-After"), "2")
        FakeBackend.release.set()
        held.close()

    def test_upstream_errors_are_masked_or_mapped(self):
        port = self.gateway()
        status, _, body = self.call(port, "GET", "/v1/evals/runs/../../v1/boom")  # not a route
        self.assertEqual(status, 404)
        dead = self.gateway(upstream_port=1)  # nothing listens there
        self.assertEqual(self.call(dead, "GET", "/v1/corpus")[0], 502)
        status, _, body = self.call(dead, "GET", "/healthz")
        self.assertEqual((status, body["upstream"]), (503, False))
        status, _, body = self.call(port, "GET", "/healthz")
        self.assertEqual((status, body["upstream"]), (200, True))

    def test_forwarded_for_only_when_trusted(self):
        port = self.gateway(trust_proxy=False, rate_read_per_min=2)
        for ip in ("1.1.1.1", "2.2.2.2", "3.3.3.3"):
            self.call(port, "GET", "/v1/corpus", headers={"X-Forwarded-For": ip})
        self.assertEqual(self.call(port, "GET", "/v1/corpus", headers={"X-Forwarded-For": "9.9.9.9"})[0], 429)  # spoofing doesn't help
        trusted = self.gateway(trust_proxy=True, rate_read_per_min=1)
        self.assertEqual(self.call(trusted, "GET", "/v1/corpus", headers={"X-Forwarded-For": "spoof, 5.5.5.5"})[0], 200)
        self.assertEqual(self.call(trusted, "GET", "/v1/corpus", headers={"X-Forwarded-For": "other, 5.5.5.5"})[0], 429)
        self.assertEqual(self.call(trusted, "GET", "/v1/corpus", headers={"X-Forwarded-For": "other, 6.6.6.6"})[0], 200)


if __name__ == "__main__":
    unittest.main()
