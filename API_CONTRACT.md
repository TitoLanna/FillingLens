# API contract the UI expects

The UI talks to three endpoints, always through the API gateway (`backend/gateway`), never straight to the backend. All paths are relative to `VITE_API_BASE_URL` (empty by default; the Vite dev server proxies `/v1` to the gateway at `http://localhost:8080`).

The gateway exposes only the endpoints below. Besides the backend's own responses it can answer with `{"error": "..."}` and one of: `400` invalid request, `413` body too large, `429` rate limited (with `Retry-After` seconds), `502`/`504` backend unavailable, `503` busy (with `Retry-After`).

## `POST /v1/query` — ask a question (streamed)

Request body:

```json
{
  "question": "What supply chain risks did Contoso Cloud report?",
  "filters": { "tickers": ["CTSO"], "fiscal_years": [2024] },
  "top_k": 6
}
```

`filters` and its fields are optional.

Response: `Content-Type: text/event-stream`. Each event is one `data:` line holding JSON with a `type`, followed by a blank line. Send them in this order:

1. **citations** — once, as soon as retrieval and reranking finish (before the LLM call). Lets the UI show sources while the answer streams.
2. **token** — many, as the LLM streams.
   OR **refusal** — once, instead of tokens, when the context does not support an answer.
3. **done** — once, with the trace.
4. **error** — at any point, then close the stream.

```text
data: {"type":"citations","citations":[{"id":1,"chunk_id":"ctso-2024-1a-0012","company":"Contoso Cloud","ticker":"CTSO","fiscal_year":2024,"section":"Item 1A. Risk Factors","text":"We depend on a limited number of suppliers...","source_url":"https://www.sec.gov/...","scores":{"bm25":12.4,"dense":0.81,"rrf":0.032,"rerank":0.93}}]}

data: {"type":"token","text":"Contoso reported "}

data: {"type":"token","text":"reliance on three GPU suppliers [1]."}

data: {"type":"done","trace":{"trace_id":"tr_8f2a","total_ms":2140,"stages":[{"name":"bm25","ms":18},{"name":"dense","ms":24},{"name":"rrf","ms":1},{"name":"rerank","ms":118},{"name":"llm","ms":1979}],"prompt_tokens":2310,"completion_tokens":164,"cost_usd":0.0072}}
```

Rules:

- Citation `id` values are the numbers the answer uses inline as `[1]`, `[2]`… The UI turns those markers into links to the source cards.
- Any score in `scores` may be `null` (e.g. a chunk found only by BM25 has `dense: null`).
- Refusal event: `{"type":"refusal","reason":"The filings in scope don't mention ..."}`.
- Error event: `{"type":"error","message":"..."}`.

FastAPI sketch:

```python
from fastapi.responses import StreamingResponse
import json

def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"

@app.post("/v1/query")
async def query(req: QueryRequest):
    async def gen():
        chunks = await retrieve(req)                      # hybrid + rerank
        yield sse({"type": "citations", "citations": to_citations(chunks)})
        if not supports_answer(chunks):
            yield sse({"type": "refusal", "reason": "..."})
        else:
            async for piece in llm_stream(req.question, chunks):
                yield sse({"type": "token", "text": piece})
        yield sse({"type": "done", "trace": current_trace()})
    return StreamingResponse(gen(), media_type="text/event-stream")
```

## `GET /v1/corpus` — what's indexed (fills the filters)

```json
{
  "companies": [{ "ticker": "CTSO", "name": "Contoso Cloud" }],
  "fiscal_years": [2022, 2023, 2024]
}
```

One `SELECT DISTINCT` over the `documents` table.

## `GET /v1/evals/runs` — experiment log (Experiments tab)

Newest first. Mark the baseline run with `"is_baseline": true`; the UI shows every other run's delta against it.

```json
[
  {
    "id": "run_004",
    "label": "+ cross-encoder reranker",
    "git_sha": "a1b2c3d",
    "started_at": "2026-11-20T18:04:00Z",
    "is_baseline": false,
    "metrics": {
      "recall_at_5": 0.84,
      "mrr": 0.71,
      "faithfulness": 0.93,
      "p95_latency_ms": 285,
      "cost_per_query_usd": 0.0074
    }
  }
]
```

This is a list view over the `eval_runs` table from the System Design tab.
