import type { QueryTrace } from "../api/types";

const STAGE_LABEL: Record<string, string> = {
  bm25: "BM25", dense: "Dense", rrf: "Fusion", rerank: "Rerank", llm: "LLM",
};

/** Per-stage latency as one proportional bar, plus tokens and cost. */
export function TraceBar({ trace }: { trace: QueryTrace }) {
  const sum = trace.stages.reduce((n, s) => n + s.ms, 0) || 1;
  return (
    <section className="trace" aria-label="Request trace">
      <div className="trace-row">
        <span className="trace-total">{(trace.total_ms / 1000).toFixed(2)} s</span>
        <span>{trace.prompt_tokens.toLocaleString()} in · {trace.completion_tokens.toLocaleString()} out</span>
        <span>${trace.cost_usd.toFixed(4)}</span>
        <code className="trace-id">{trace.trace_id}</code>
      </div>
      <div className="trace-bar" role="img" aria-label={trace.stages.map((s) => `${s.name} ${s.ms} ms`).join(", ")}>
        {trace.stages.map((s, i) => (
          <span
            key={s.name}
            className={`trace-seg seg-${i % 5}`}
            style={{ flexGrow: s.ms / sum }}
            title={`${STAGE_LABEL[s.name] ?? s.name}: ${s.ms} ms`}
          />
        ))}
      </div>
      <ul className="trace-legend">
        {trace.stages.map((s, i) => (
          <li key={s.name}>
            <span className={`swatch seg-${i % 5}`} />
            {STAGE_LABEL[s.name] ?? s.name} {s.ms} ms
          </li>
        ))}
      </ul>
    </section>
  );
}
