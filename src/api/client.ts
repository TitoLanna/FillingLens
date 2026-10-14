import type { Corpus, EvalRun, QueryRequest, StreamEvent } from "./types";
import { mockCorpus, mockEvalRuns, mockStream } from "./mock";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

/** Mock mode is on unless VITE_USE_MOCK is explicitly "false". */
export const USE_MOCK = import.meta.env.VITE_USE_MOCK !== "false";

/** Builds a readable error from a failed response. The gateway replies {"error": "..."} and sets Retry-After on 429. */
async function httpError(res: Response, what: string): Promise<Error> {
  let detail = res.statusText;
  try {
    const body = (await res.json()) as { error?: string };
    if (body.error) detail = body.error;
  } catch {
    /* not JSON: keep the status text */
  }
  const retry = res.status === 429 ? res.headers.get("Retry-After") : null;
  return new Error(`${what}: ${detail}${retry ? ` (try again in ${retry}s)` : ""}`);
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { signal, headers: { Accept: "application/json" } });
  if (!res.ok) throw await httpError(res, `${path} failed`);
  return (await res.json()) as T;
}

export function fetchCorpus(signal?: AbortSignal): Promise<Corpus> {
  return USE_MOCK ? mockCorpus() : getJson<Corpus>("/v1/corpus", signal);
}

export function fetchEvalRuns(signal?: AbortSignal): Promise<EvalRun[]> {
  return USE_MOCK ? mockEvalRuns() : getJson<EvalRun[]>("/v1/evals/runs", signal);
}

/** Parses one SSE message block ("data: ..." lines) into an event. */
function parseSseBlock(block: string): StreamEvent | null {
  const data = block
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
  if (!data) return null;
  return JSON.parse(data) as StreamEvent;
}

/**
 * Streams a query. Yields events in contract order:
 * citations → token* | refusal → done (or error at any point).
 */
export async function* streamQuery(
  req: QueryRequest,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  if (USE_MOCK) {
    yield* mockStream(req, signal);
    return;
  }

  const res = await fetch(`${BASE}/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) throw await httpError(res, "Query failed");
  if (!res.body) throw new Error("Query failed: empty response");

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value.replace(/\r\n/g, "\n");
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseSseBlock(block);
        if (event) yield event;
      }
    }
    const tail = parseSseBlock(buffer.trim());
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}
