import { forwardRef } from "react";
import type { Citation } from "../api/types";

interface Props {
  citation: Citation;
  active: boolean;
  onSelect: (id: number) => void;
}

const fmt = (v: number | null | undefined, digits: number) =>
  v === null || v === undefined ? "—" : v.toFixed(digits);

export const SourceCard = forwardRef<HTMLElement, Props>(function SourceCard(
  { citation: c, active, onSelect },
  ref,
) {
  return (
    <article
      ref={ref}
      className={`source${active ? " is-active" : ""}`}
      onClick={() => onSelect(c.id)}
      aria-current={active ? "true" : undefined}
    >
      <header className="source-head">
        <span className="source-num">{c.id}</span>
        <div className="source-meta">
          <strong>{c.company}</strong>
          <span>
            {c.ticker} · FY{c.fiscal_year} · {c.section}
          </span>
        </div>
      </header>
      <p className="source-text">{c.text}</p>
      <footer className="source-foot">
        <dl className="scores" aria-label="Retrieval scores">
          <div><dt>BM25</dt><dd>{fmt(c.scores.bm25, 1)}</dd></div>
          <div><dt>Dense</dt><dd>{fmt(c.scores.dense, 2)}</dd></div>
          <div><dt>RRF</dt><dd>{fmt(c.scores.rrf, 3)}</dd></div>
          <div><dt>Rerank</dt><dd>{fmt(c.scores.rerank, 2)}</dd></div>
        </dl>
        <a href={c.source_url} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
          Open filing ↗
        </a>
      </footer>
    </article>
  );
});
