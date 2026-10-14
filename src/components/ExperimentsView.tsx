import { useEffect, useState } from "react";
import { fetchEvalRuns } from "../api/client";
import type { EvalMetrics, EvalRun } from "../api/types";

type Key = keyof EvalMetrics;

const COLUMNS: { key: Key; label: string; fmt: (v: number) => string; higherIsBetter: boolean }[] = [
  { key: "recall_at_5", label: "Recall@5", fmt: (v) => v.toFixed(2), higherIsBetter: true },
  { key: "mrr", label: "MRR", fmt: (v) => v.toFixed(2), higherIsBetter: true },
  { key: "faithfulness", label: "Faithfulness", fmt: (v) => v.toFixed(2), higherIsBetter: true },
  { key: "p95_latency_ms", label: "p95 latency (ms)", fmt: (v) => v.toFixed(0), higherIsBetter: false },
  { key: "cost_per_query_usd", label: "Cost / query ($)", fmt: (v) => v.toFixed(4), higherIsBetter: false },
];

function Delta({ value, base, higherIsBetter, fmt }: { value: number; base: number; higherIsBetter: boolean; fmt: (v: number) => string }) {
  const d = value - base;
  if (Math.abs(d) < 1e-9) return null;
  const good = higherIsBetter ? d > 0 : d < 0;
  return <span className={`delta ${good ? "is-good" : "is-bad"}`}>{d > 0 ? "+" : "−"}{fmt(Math.abs(d))}</span>;
}

/** Recall@5 per run, oldest to newest, with the baseline as a reference line. */
function RecallChart({ runs, baseline }: { runs: EvalRun[]; baseline: EvalRun }) {
  const ordered = [...runs].reverse();
  const W = 640, rowH = 34, left = 210, right = 56, top = 28;
  const H = top + ordered.length * rowH + 12;
  const x = (v: number) => left + v * (W - left - right);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`Recall at 5 by run: ${ordered.map((r) => `${r.label} ${r.metrics.recall_at_5.toFixed(2)}`).join("; ")}`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top - 6} y2={H - 8} className="grid" />
          <text x={x(t)} y={top - 12} textAnchor="middle" className="tick">{t.toFixed(2)}</text>
        </g>
      ))}
      <line x1={x(baseline.metrics.recall_at_5)} x2={x(baseline.metrics.recall_at_5)} y1={top - 6} y2={H - 8} className="baseline-line" />
      {ordered.map((r, i) => {
        const y = top + i * rowH;
        return (
          <g key={r.id}>
            <text x={left - 10} y={y + rowH / 2 + 4} textAnchor="end" className="row-label">{r.label}</text>
            <rect x={left} y={y + 7} width={Math.max(2, x(r.metrics.recall_at_5) - left)} height={rowH - 14} rx={4}
              className={r.is_baseline ? "bar bar-base" : "bar"} />
            <text x={x(r.metrics.recall_at_5) + 6} y={y + rowH / 2 + 4} className="bar-value">{r.metrics.recall_at_5.toFixed(2)}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function ExperimentsView() {
  const [runs, setRuns] = useState<EvalRun[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    fetchEvalRuns(ac.signal).then(setRuns).catch((e: Error) => {
      if (e.name !== "AbortError") setError(e.message);
    });
    return () => ac.abort();
  }, []);

  if (error) return <div className="notice notice-error" role="alert">Couldn't load eval runs: {error}</div>;
  if (!runs) return <p className="muted">Loading eval runs…</p>;
  if (runs.length === 0) return <p className="muted">No eval runs yet. Run the eval harness to populate this page.</p>;

  const baseline = runs.find((r) => r.is_baseline) ?? runs[runs.length - 1];
  const latest = runs[0];
  const lift = latest.metrics.recall_at_5 - baseline.metrics.recall_at_5;

  return (
    <div className="experiments">
      <header className="exp-head">
        <h2>Experiment log</h2>
        <p className="muted">
          Each row is one eval-harness run on the golden set. Deltas are against the baseline.
        </p>
      </header>

      <div className="kpis">
        <div className="kpi">
          <span className="kpi-label">Recall@5, latest</span>
          <span className="kpi-value">{latest.metrics.recall_at_5.toFixed(2)}</span>
          <span className={`kpi-sub ${lift >= 0 ? "is-good" : "is-bad"}`}>
            {lift >= 0 ? "+" : "−"}{Math.abs(lift).toFixed(2)} vs baseline
          </span>
        </div>
        <div className="kpi">
          <span className="kpi-label">Faithfulness, latest</span>
          <span className="kpi-value">{latest.metrics.faithfulness.toFixed(2)}</span>
          <span className="kpi-sub muted">baseline {baseline.metrics.faithfulness.toFixed(2)}</span>
        </div>
        <div className="kpi">
          <span className="kpi-label">p95 latency, latest</span>
          <span className="kpi-value">{latest.metrics.p95_latency_ms.toFixed(0)} ms</span>
          <span className="kpi-sub muted">baseline {baseline.metrics.p95_latency_ms.toFixed(0)} ms</span>
        </div>
      </div>

      <section className="card">
        <h3>Recall@5 by run</h3>
        <RecallChart runs={runs} baseline={baseline} />
      </section>

      <section className="card table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Config</th>
              {COLUMNS.map((c) => <th key={c.key} scope="col" className="num">{c.label}</th>)}
              <th scope="col">Commit</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id} className={r.is_baseline ? "is-baseline" : undefined}>
                <th scope="row">
                  {r.label}
                  {r.is_baseline && <span className="tag">baseline</span>}
                  <div className="muted small">{new Date(r.started_at).toLocaleDateString()}</div>
                </th>
                {COLUMNS.map((c) => (
                  <td key={c.key} className="num">
                    {c.fmt(r.metrics[c.key])}
                    {!r.is_baseline && (
                      <Delta value={r.metrics[c.key]} base={baseline.metrics[c.key]} higherIsBetter={c.higherIsBetter} fmt={c.fmt} />
                    )}
                  </td>
                ))}
                <td><code>{r.git_sha}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
