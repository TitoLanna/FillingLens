# FilingLens backend

Standard-library Python only: no pip packages. Needs Python 3.10+. Follows
`RAG_Portfolio_Build_Plan_and_System_Design.pdf`; Postgres, pgvector and OpenSearch are
replaced by SQLite plus hand-written BM25 and brute-force cosine search, each behind a small
interface so they can be swapped later.

Project overview, architecture and status: [../README.md](../README.md). This file is the reference for the backend, the gateway, the eval workflow and CI.

## Run it

```bash
cp .env.example .env                                          # then edit: SEC_USER_AGENT, keys
python3 scripts/ingest.py --tickers AAPL MSFT --years 2023 2024   # real 10-Ks from EDGAR (or scripts/seed.py for fictional demo data)
python3 main.py        # backend on :8000 (internal: bind it to localhost, don't expose it)
python3 -m gateway     # gateway on :8080, the only thing the UI and the public talk to
python3 -m unittest discover tests
```

Then set `VITE_USE_MOCK=false` in the UI's `.env.local`.

## Gateway

`gateway/` is a from-scratch reverse proxy (stdlib `http.server` + `http.client`) in front of the backend.

| Concern | Behaviour |
| --- | --- |
| Route allowlist | Public: `POST /v1/query`, `GET /v1/corpus`, `GET /v1/evals/runs`, `GET /healthz`. Everything else is 404. |
| Admin routes | `/v1/ingest*`, `/v1/retrieve`, `POST /v1/evals/runs`, `GET /v1/evals/runs/{id}` need `Authorization: Bearer $ADMIN_TOKEN`; with no `ADMIN_TOKEN` set they return 404. |
| Validation | `/v1/query` is validated (question 1-1000 chars, ticker/year lists, `top_k` 1-20) and **rebuilt**: unknown fields such as `pipeline` are dropped before forwarding. Body cap 16 KB, JSON content type required. |
| Rate limits | Token bucket per client IP: `RATE_QUERY_PER_MIN` (10) for queries, `RATE_READ_PER_MIN` (120) for reads. `429` with `Retry-After`. |
| Load shedding | At most `MAX_INFLIGHT_QUERIES` (8) streaming queries at once, else `503` + `Retry-After`. |
| Streaming | SSE passes through unbuffered. If the browser disconnects, the upstream connection is closed too, so a cancelled question stops spending tokens. |
| CORS | Only `ALLOWED_ORIGINS`; the backend's own CORS headers are stripped. |
| Errors | Backend 5xx details are never forwarded to public callers (`502 upstream error`; `504` on timeout). Every response carries `X-Request-ID`, also logged and sent to the backend. |

Client IPs come from the socket. Set `TRUST_PROXY=true` only when a proxy you control (nginx, Fly, Railway) sits in front: the last `X-Forwarded-For` entry is then used. Behind such a proxy the gateway is plain HTTP; TLS ends at the proxy.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/query` | Streamed (SSE) cited answer or refusal; shape in [../API_CONTRACT.md](../API_CONTRACT.md). Optional `"pipeline"`: preset name or partial config |
| `POST /v1/retrieve` | `{query, filters?, pipeline?}` -> ranked chunks with per-stage scores and timings (the load-test target) |
| `POST /v1/ingest` | `{tickers:[...], years:[...]}` (EDGAR discovery) and/or `{documents:[{ticker, company, fiscal_year, url}]}` -> `job_id` |
| `GET /v1/ingest/{job_id}` | Job status and counts (ingested / replaced / skipped / failed) |
| `POST /v1/evals/runs` | `{label, pipeline?, is_baseline?, generate?, judge?, subset?}` -> runs the verified golden set (synchronous; `generate`/`judge` need `ANTHROPIC_API_KEY`) |
| `GET /v1/evals/runs` / `/{id}` | UI experiment log / full run with per-question results and diff vs baseline |
| `GET /v1/corpus`, `GET /healthz` | Filters for the UI, liveness |

## Layout

| Path | Role |
| --- | --- |
| `filinglens/server.py` | Route table, CORS, SSE |
| `filinglens/pipeline.py` | RAG API query path: retrieve, cite or refuse, generate |
| `filinglens/retrieval/service.py` | Config-driven retrieval (`baseline` = dense only, `hybrid` = BM25 + dense + RRF) |
| `filinglens/retrieval/{bm25,dense,fusion}.py` | BM25, cosine search, reciprocal rank fusion |
| `filinglens/embeddings.py` | Providers: `hash` (offline placeholder) or `voyage` (raw HTTP); float32 BLOBs, model name stored per chunk |
| `filinglens/http.py` | urllib JSON/SSE helpers with verified TLS |
| `filinglens/ingestion/` | `edgar.py` ticker + year to 10-K URL; `parse.py` HTML to sections and Markdown tables; `worker.py` hash-skip, embed, transactional replace, jobs |
| `filinglens/evals/` | `baseline.py` export/import a run as JSON; `metrics.py` Recall/MRR/nDCG; `golden.py` candidate generation + review helpers; `judge.py` binary LLM judges; `calibration.py` judge-vs-human agreement; `runner.py` harness; `report.py` baseline diff + regression gate |
| `filinglens/db.py` | Schema from the system design |
| `filinglens/llm.py` | Anthropic Messages API streamed over raw HTTP (or an offline stub); `NOT_FOUND` becomes a refusal |
| `gateway/` | API gateway: `server.py` proxy, `ratelimit.py`, `validate.py`, `config.py` |
| `scripts/` | `seed.py`, `ingest.py`, `golden.py`, `run_evals.py`, `baseline.py`, `calibrate_judge.py`, `ci_eval.py` |

## Going live: real embeddings and answers

Put these in `.env` (see `.env.example`):

```
EMBEDDING_PROVIDER=voyage   VOYAGE_API_KEY=...
ANTHROPIC_API_KEY=...       LLM_PRICE_IN=...  LLM_PRICE_OUT=...   # USD per million tokens
```

Then `python3 scripts/ingest.py ...` again: documents whose vectors came from another model are
re-embedded even though their text is unchanged. These two providers are covered by offline tests
with mocked HTTP only, not run against the live APIs yet.

## Eval workflow (weeks 3-4)

```bash
# 1. Build the golden set: generate candidates, then verify each by hand. Only verified ones count.
python3 scripts/golden.py generate --factual 50 --table 30 --multi-hop 30 --comparison 25 --unanswerable 25
python3 scripts/golden.py review          # a accept, r reject, eq/ea/ec edit, f find (check 'unanswerable')
python3 scripts/golden.py export data/golden.jsonl   # commit this so CI has the same questions

# 2. Record the baseline (freeze it with the v0-baseline git tag)
python3 scripts/run_evals.py --label "v0 baseline" --pipeline baseline --baseline --judge

# 3. Calibrate the judge: label ~30 answers yourself per metric, then read the agreement
python3 scripts/calibrate_judge.py label --run run_xxxxxx --metric faithfulness --n 30
python3 scripts/calibrate_judge.py report

# 4. CI: stratified subset, diff vs baseline on the same questions, exit 1 on regression
python3 scripts/ci_eval.py --subset 30 --pipeline hybrid --generate --judge --max-drop 0.03 --output diff.md
```

Judge verdicts are binary pass/fail with the reason written first. Faithfulness, correctness and citations
use the model; refusal correctness is deterministic. A judge error is recorded as "not scored", never as a
fail, and counted in `judge_errors`. Golden chunk ids look like `aapl-2024-0039`; re-parsing or re-chunking
changes them, so re-verify (or re-export) the golden set after any ingestion change.

## CI setup (GitHub Actions)

Workflows live in `../.github/workflows/` and assume this project folder is the repository root.

- `ci.yml` runs on every PR and push: backend unit tests on Python 3.11 and 3.13, frontend typecheck and build.
- `eval.yml` runs `scripts/ci_eval.py`: a 30-question stratified subset on every PR touching `backend/`, and the full golden set nightly or on manual dispatch. It posts the metrics diff as a PR comment (updating the same comment on each push), writes it to the job summary, and fails the job if Recall@5, MRR, faithfulness, correctness, citation accuracy or refusal accuracy drops by more than 0.03 against the baseline.

One-time setup for `eval.yml`:

1. Repository **secrets**: `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`. **Variables**: `SEC_USER_AGENT` (e.g. `FilingLens you@example.com`); optionally `LLM_PRICE_IN`, `LLM_PRICE_OUT`, `VOYAGE_MODEL`.
2. Commit three files in `backend/data/`: `corpus.json` (tickers and years; a starter is there), `golden.jsonl` (`scripts/golden.py export`) and `baseline.json` (`scripts/baseline.py export --run <id> --out data/baseline.json`).
3. Until all three exist and the keys are set, the eval job skips itself with a notice instead of failing. Pull requests from forks never receive secrets, so they skip it too.

The ingested corpus is cached, keyed on `corpus.json`, the ingestion/embedding code and the Voyage model, so only the first run (or a change to those) pays for EDGAR downloads and embeddings. The baseline is compared on the same questions as the subset being run, so a 30-question PR run is not judged against full-set numbers. Remember that changing the parser or chunker changes chunk ids: re-verify the golden set and re-export the baseline afterwards.

## Not built yet (plan order)
1. Tag `v0-baseline` once real embeddings and a reviewed golden set exist.
2. **Weeks 5-6:** cross-encoder reranker stage (fills `scores.rerank`), section-aware chunking, query rewriting.
3. **Week 7:** stage timeouts and fallbacks, embedding and query caches, OpenTelemetry-style traces, load test (rate limiting now lives in the gateway).
