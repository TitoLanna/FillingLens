"""API gateway: the single public entry point in front of the FilingLens backend.

Responsibilities: route allowlist, admin auth, CORS, rate limiting, request validation,
concurrency cap on LLM queries, unbuffered SSE pass-through, masking of upstream 5xx details.
"""
import hmac
import http.client
import json
import re
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import GatewaySettings
from .ratelimit import RateLimiter
from .validate import ValidationError, clean_query

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailers", "transfer-encoding", "upgrade", "content-length"}

# (method, path regex, kind). kind: "query" = rate-limited, validated, concurrency-capped;
# "read" = cheap read; "admin" = needs ADMIN_TOKEN.
ROUTES = [
    ("POST", r"/v1/query", "query"),
    ("GET", r"/v1/corpus", "read"),
    ("GET", r"/v1/evals/runs", "read"),
    ("POST", r"/v1/retrieve", "admin"),
    ("POST", r"/v1/ingest", "admin"),
    ("GET", r"/v1/ingest/[\w-]+", "admin"),
    ("POST", r"/v1/evals/runs", "admin"),
    ("GET", r"/v1/evals/runs/[\w-]+", "admin"),
]


def make_handler(cfg: GatewaySettings, log=print):
    query_limit = RateLimiter(cfg.rate_query_per_min)
    read_limit = RateLimiter(cfg.rate_read_per_min)
    inflight = threading.BoundedSemaphore(cfg.max_inflight_queries)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30  # seconds to wait on a slow or idle client socket
        body_consumed = True  # False while a POST body is still unread on the socket

        # --- helpers ---------------------------------------------------
        def _client_ip(self) -> str:
            if cfg.trust_proxy:
                fwd = self.headers.get("X-Forwarded-For", "")
                if fwd:
                    return fwd.split(",")[-1].strip()  # last hop = the proxy we trust appended it
            return self.client_address[0]

        def _origin_headers(self) -> list[tuple[str, str]]:
            origin = self.headers.get("Origin")
            if not origin:
                return []
            if "*" in cfg.allowed_origins:
                return [("Access-Control-Allow-Origin", "*")]
            if origin in cfg.allowed_origins:
                return [("Access-Control-Allow-Origin", origin), ("Vary", "Origin")]
            return []

        def _common_headers(self) -> None:
            self.send_header("X-Request-ID", self.req_id)
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in self._origin_headers():
                self.send_header(k, v)

        def _reply(self, status: int, payload: dict, extra: dict | None = None) -> None:
            self.status = status
            body = json.dumps({**payload, "request_id": self.req_id}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            if not self.body_consumed:
                # Unread request bytes would be parsed as the next request on a keep-alive socket.
                self.send_header("Connection", "close")
                self.close_connection = True
            self._common_headers()
            self.end_headers()
            self.wfile.write(body)

        def _origin_blocked(self) -> bool:
            origin = self.headers.get("Origin")
            return bool(origin) and "*" not in cfg.allowed_origins and origin not in cfg.allowed_origins

        def _is_admin(self) -> bool:
            auth = self.headers.get("Authorization", "")
            return bool(cfg.admin_token) and auth.startswith("Bearer ") and hmac.compare_digest(
                auth[7:].encode(), cfg.admin_token.encode())

        def _read_body(self) -> bytes | None:
            """Return the body, or None after sending an error response."""
            length = self.headers.get("Content-Length")
            if length is None or not length.isdigit():
                self._reply(411, {"error": "Content-Length required"})
                return None
            if int(length) > cfg.max_body_bytes:
                self._reply(413, {"error": f"body too large (max {cfg.max_body_bytes} bytes)"})
                return None
            if "application/json" not in self.headers.get("Content-Type", "").lower():
                self._reply(415, {"error": "Content-Type must be application/json"})
                return None
            self.body_consumed = True
            return self.rfile.read(int(length))

        # --- request flow ----------------------------------------------
        def _handle(self, method: str) -> None:
            self.req_id = uuid.uuid4().hex[:12]
            self.status = 0
            self.body_consumed = method != "POST"
            t0 = time.perf_counter()
            try:
                self._route(method)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                self.close_connection = True
            except Exception as exc:  # never leak internals through the gateway
                log(f"[{self.req_id}] gateway error: {exc!r}")
                try:
                    self._reply(500, {"error": "internal error"})
                except OSError:
                    pass
            finally:
                log(f"[{self.req_id}] {self._client_ip()} {method} {self.path.split('?')[0]} "
                    f"-> {self.status} {(time.perf_counter() - t0) * 1000:.0f}ms")

        def _route(self, method: str) -> None:
            path = self.path.split("?")[0].rstrip("/") or "/"
            if path == "/healthz" and method == "GET":
                return self._health()
            kind = next((k for m, p, k in ROUTES if m == method and re.fullmatch(p, path)), None)
            if kind is None:
                return self._reply(404, {"error": "not found"})
            if self._origin_blocked():
                return self._reply(403, {"error": "origin not allowed"})
            if kind == "admin" and not cfg.admin_token:
                return self._reply(404, {"error": "not found"})  # disabled: don't advertise it
            if kind == "admin" and not self._is_admin():
                return self._reply(401, {"error": "admin token required"}, {"WWW-Authenticate": "Bearer"})

            limiter = query_limit if kind == "query" else read_limit
            ok, retry = limiter.allow(f"{kind}:{self._client_ip()}")
            if not ok:
                return self._reply(429, {"error": "rate limit exceeded, slow down"}, {"Retry-After": str(retry)})

            body = None
            if method == "POST":
                body = self._read_body()
                if body is None:
                    return
                if kind == "query":
                    try:
                        body = json.dumps(clean_query(json.loads(body or b"null"))).encode()
                    except json.JSONDecodeError:
                        return self._reply(400, {"error": "invalid JSON"})
                    except ValidationError as exc:
                        return self._reply(400, {"error": str(exc)})

            if kind == "query":
                if not inflight.acquire(blocking=False):
                    return self._reply(503, {"error": "server is busy, try again shortly"}, {"Retry-After": "2"})
                try:
                    self._proxy(method, self.path, body, mask_errors=True)
                finally:
                    inflight.release()
            else:
                self._proxy(method, self.path, body, mask_errors=kind != "admin")

        def _health(self) -> None:
            try:
                conn = http.client.HTTPConnection(cfg.upstream_host, cfg.upstream_port, timeout=3)
                conn.request("GET", "/healthz")
                up = conn.getresponse().status == 200
                conn.close()
            except OSError:
                up = False
            self._reply(200 if up else 503, {"status": "ok" if up else "degraded", "upstream": up})

        # --- proxying --------------------------------------------------
        def _proxy(self, method: str, target: str, body: bytes | None, mask_errors: bool) -> None:
            headers = {"X-Request-ID": self.req_id, "X-Forwarded-For": self._client_ip()}
            if body is not None:
                headers["Content-Type"] = "application/json"
            conn = http.client.HTTPConnection(cfg.upstream_host, cfg.upstream_port, timeout=cfg.upstream_timeout)
            try:
                conn.request(method, target, body=body, headers=headers)
                resp = conn.getresponse()
            except socket.timeout:
                conn.close()
                return self._reply(504, {"error": "upstream timed out"})
            except OSError:
                conn.close()
                return self._reply(502, {"error": "upstream unavailable"})

            try:
                if resp.status >= 500 and mask_errors:
                    log(f"[{self.req_id}] upstream {resp.status}: {resp.read(300)!r}")
                    return self._reply(502, {"error": "upstream error"})
                ctype = resp.getheader("Content-Type", "")
                self.status = resp.status
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP_BY_HOP and not k.lower().startswith("access-control-"):
                        self.send_header(k, v)
                self._common_headers()
                if ctype.startswith("text/event-stream"):
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")  # tell nginx-style proxies not to buffer
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    while True:
                        chunk = resp.read1(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    data = resp.read()
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            except socket.timeout:
                self.close_connection = True  # upstream stalled mid-stream; headers already sent
            finally:
                conn.close()  # on client disconnect this also cancels the upstream request

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

        # Other methods go through routing too, so they get the same 404 as any unknown route.
        def do_PUT(self):
            self._handle("PUT")

        def do_DELETE(self):
            self._handle("DELETE")

        def do_PATCH(self):
            self._handle("PATCH")

        def do_HEAD(self):
            self._handle("HEAD")

        def do_OPTIONS(self):
            self.req_id = uuid.uuid4().hex[:12]
            self.status = 204
            self.body_consumed = True
            if self._origin_blocked() or not self._origin_headers():
                return self._reply(403, {"error": "origin not allowed"})
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Max-Age", "600")
            self._common_headers()
            self.end_headers()

        def log_message(self, fmt, *args):  # replaced by the per-request line in _handle
            pass

    return Handler


def make_server(cfg: GatewaySettings, host: str | None = None, port: int | None = None, log=print):
    server = ThreadingHTTPServer((host or cfg.host, cfg.port if port is None else port), make_handler(cfg, log))
    server.daemon_threads = True
    return server
