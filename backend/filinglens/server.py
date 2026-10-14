"""HTTP server on http.server: route table, JSON, CORS, SSE."""
import json
import re
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import Settings
from .evals.runner import run_eval
from .ingestion.worker import Jobs
from .pipeline import run_query
from .retrieval.service import PRESETS, Retriever, config_from

UI_METRICS = ("recall_at_5", "mrr", "faithfulness", "p95_latency_ms", "cost_per_query_usd")


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _run_row(r: sqlite3.Row) -> dict:
    m = json.loads(r["summary_metrics"])
    return {"id": r["id"], "label": r["label"], "git_sha": r["git_sha"],
            "started_at": r["started_at"], "is_baseline": bool(r["is_baseline"]),
            "pipeline_config": json.loads(r["pipeline_config"]),
            "metrics": {k: m.get(k) for k in UI_METRICS}}


def make_handler(conn: sqlite3.Connection, retriever: Retriever, settings: Settings):
    jobs = Jobs(conn, settings.sec_user_agent, on_done=retriever.reload)
    default_config = PRESETS[settings.pipeline]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # --- plumbing -------------------------------------------------
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", settings.cors_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, status: int, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                data = json.loads(raw or b"{}")
            except (ValueError, json.JSONDecodeError):
                raise HttpError(400, "invalid JSON body")
            if not isinstance(data, dict):
                raise HttpError(400, "body must be a JSON object")
            return data

        def _sse(self, events):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.close_connection = True
            try:
                for event in events:
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # client cancelled
            except Exception as exc:  # contract: error event, then close
                try:
                    self.wfile.write(f'data: {json.dumps({"type": "error", "message": str(exc)})}\n\n'.encode())
                    self.wfile.flush()
                except OSError:
                    pass

        # --- handlers -------------------------------------------------
        def healthz(self):
            conn.execute("SELECT 1")
            return 200, {"status": "ok", "chunks": len(retriever.meta), "pipeline": settings.pipeline}

        def corpus(self):
            rows = conn.execute("SELECT DISTINCT ticker, company, fiscal_year FROM documents "
                                "ORDER BY ticker").fetchall()
            companies = {r["ticker"]: r["company"] for r in rows}
            return 200, {"companies": [{"ticker": t, "name": n} for t, n in companies.items()],
                         "fiscal_years": sorted({r["fiscal_year"] for r in rows})}

        def retrieve(self):
            req = self._body()
            if not str(req.get("query", "")).strip():
                raise HttpError(400, "query is required")
            try:
                config = config_from(req.get("pipeline"), default_config)
            except (ValueError, TypeError) as exc:
                raise HttpError(400, str(exc))
            cands, stages = retriever.retrieve(req["query"], req.get("filters"), config)
            return 200, {"config": config.to_dict(), "stages": stages, "results": cands}

        def ingest(self):
            req = self._body()
            docs, tickers, years = req.get("documents"), req.get("tickers"), req.get("years")
            need = ("ticker", "company", "fiscal_year", "url")
            if docs is not None and (not isinstance(docs, list) or any(
                    not isinstance(d, dict) or any(k not in d for k in need) for d in docs)):
                raise HttpError(400, f"documents must be a list of objects with {need}")
            if tickers is not None and (not isinstance(tickers, list) or not isinstance(years, list)
                                        or not years or not all(isinstance(y, int) for y in years)):
                raise HttpError(400, "tickers needs a non-empty integer list `years`")
            if not docs and not tickers:
                raise HttpError(400, "provide `documents` or `tickers` + `years`")
            return 202, {"job_id": jobs.submit(docs, tickers, years)}

        def ingest_status(self, job_id):
            if job_id not in jobs.jobs:
                raise HttpError(404, "unknown job")
            return 200, jobs.jobs[job_id]

        def evals_list(self):
            rows = conn.execute("SELECT * FROM eval_runs ORDER BY started_at DESC").fetchall()
            return 200, [_run_row(r) for r in rows]

        def evals_create(self):
            req = self._body()
            try:
                config = config_from(req.get("pipeline"), default_config)
            except (ValueError, TypeError) as exc:
                raise HttpError(400, str(exc))
            try:
                result = run_eval(conn, retriever, settings, config, req.get("label") or "unlabeled run",
                                  bool(req.get("is_baseline")), bool(req.get("generate")),
                                  bool(req.get("judge")), req.get("subset"))
            except ValueError as exc:
                raise HttpError(400, str(exc))
            return 201, result

        def evals_get(self, run_id):
            r = conn.execute("SELECT * FROM eval_runs WHERE id=?", (run_id,)).fetchone()
            if not r:
                raise HttpError(404, "unknown run")
            out = _run_row(r)
            out["summary_metrics"] = json.loads(r["summary_metrics"])
            base = conn.execute("SELECT * FROM eval_runs WHERE is_baseline=1 AND id!=? "
                                "ORDER BY started_at DESC LIMIT 1", (run_id,)).fetchone()
            if base:
                bm = json.loads(base["summary_metrics"])
                out["diff_vs_baseline"] = {"baseline_id": base["id"], **{
                    k: (None if out["summary_metrics"].get(k) is None or bm.get(k) is None
                        else round(out["summary_metrics"][k] - bm[k], 4)) for k in UI_METRICS}}
            out["results"] = [dict(x) for x in conn.execute(
                "SELECT * FROM eval_results WHERE run_id=?", (run_id,)).fetchall()]
            return 200, out

        # --- routing --------------------------------------------------
        ROUTES = [
            ("GET", r"/healthz", "healthz"),
            ("GET", r"/v1/corpus", "corpus"),
            ("POST", r"/v1/retrieve", "retrieve"),
            ("POST", r"/v1/ingest", "ingest"),
            ("GET", r"/v1/ingest/([\w-]+)", "ingest_status"),
            ("GET", r"/v1/evals/runs", "evals_list"),
            ("POST", r"/v1/evals/runs", "evals_create"),
            ("GET", r"/v1/evals/runs/([\w-]+)", "evals_get"),
        ]

        def _dispatch(self, method: str):
            path = self.path.split("?")[0].rstrip("/") or "/"
            try:
                if method == "POST" and path == "/v1/query":
                    req = self._body()
                    if not str(req.get("question", "")).strip():
                        raise HttpError(400, "question is required")
                    try:
                        config_from(req.get("pipeline"), default_config)
                    except (ValueError, TypeError) as exc:
                        raise HttpError(400, str(exc))
                    return self._sse(run_query(retriever, settings, req))
                for m, pattern, name in self.ROUTES:
                    hit = re.fullmatch(pattern, path)
                    if m == method and hit:
                        status, payload = getattr(self, name)(*hit.groups())
                        return self._json(status, payload)
                raise HttpError(404, "not found")
            except HttpError as exc:
                self._json(exc.status, {"error": exc.message})
            except Exception as exc:
                self._json(500, {"error": str(exc)})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self._cors()
            self.end_headers()

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}")

    return Handler


def serve(settings: Settings, conn: sqlite3.Connection) -> None:
    retriever = Retriever(conn)
    server = ThreadingHTTPServer((settings.host, settings.port), make_handler(conn, retriever, settings))
    print(f"FilingLens API on http://{settings.host}:{settings.port} (pipeline: {settings.pipeline})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
