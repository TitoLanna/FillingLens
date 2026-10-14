import { useEffect, useMemo, useRef, useState } from "react";
import { fetchCorpus } from "../api/client";
import type { Corpus } from "../api/types";
import { useQueryStream } from "../hooks/useQueryStream";
import { AnswerText } from "./AnswerText";
import { SourceCard } from "./SourceCard";
import { TraceBar } from "./TraceBar";

const EXAMPLES = [
  "What supply chain risks did Northwind Semiconductor report?",
  "How fast did Contoso Cloud's cloud revenue grow?",
  "How exposed is Tailspin Air to jet fuel prices?",
  "What cybersecurity risks does Fabrikam Retail describe?",
];

const STATUS_TEXT: Record<string, string> = {
  retrieving: "Searching filings…",
  generating: "Writing answer…",
  cancelled: "Stopped.",
};

export function AskView() {
  const [question, setQuestion] = useState("");
  const [tickers, setTickers] = useState<string[]>([]);
  const [years, setYears] = useState<number[]>([]);
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [corpusError, setCorpusError] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const { state, run, cancel } = useQueryStream();
  const cardRefs = useRef(new Map<number, HTMLElement>());

  useEffect(() => {
    const ac = new AbortController();
    fetchCorpus(ac.signal).then(setCorpus).catch((e: Error) => {
      if (e.name !== "AbortError") setCorpusError(e.message);
    });
    return () => ac.abort();
  }, []);

  const busy = state.status === "retrieving" || state.status === "generating";
  const citationIds = useMemo(() => new Set(state.citations.map((c) => c.id)), [state.citations]);

  function submit(q = question) {
    const trimmed = q.trim();
    if (!trimmed || busy) return;
    setQuestion(trimmed);
    setActiveId(null);
    setHistory((h) => [trimmed, ...h.filter((x) => x !== trimmed)].slice(0, 6));
    run({
      question: trimmed,
      filters: {
        ...(tickers.length ? { tickers } : {}),
        ...(years.length ? { fiscal_years: years } : {}),
      },
      top_k: 4,
    });
  }

  function selectCitation(id: number) {
    setActiveId(id);
    cardRefs.current.get(id)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  const toggle = <T,>(list: T[], v: T) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  const hasResult = state.status !== "idle";

  return (
    <div className="ask">
      <form
        className="ask-form"
        onSubmit={(e) => { e.preventDefault(); submit(); }}
      >
        <label htmlFor="q" className="sr-only">Question</label>
        <textarea
          id="q"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
          }}
          placeholder="Ask about risks, revenue, margins, suppliers…"
          rows={2}
        />
        <div className="ask-actions">
          {busy ? (
            <button type="button" className="btn btn-secondary" onClick={cancel}>Stop</button>
          ) : (
            <button type="submit" className="btn btn-primary" disabled={!question.trim()}>Ask</button>
          )}
        </div>

        <fieldset className="filters">
          <legend className="sr-only">Filters</legend>
          {corpusError && <span className="muted">Filters unavailable: {corpusError}</span>}
          {corpus && (
            <>
              <div className="chip-group" role="group" aria-label="Companies">
                {corpus.companies.map((c) => (
                  <button
                    key={c.ticker}
                    type="button"
                    className={`chip${tickers.includes(c.ticker) ? " is-on" : ""}`}
                    aria-pressed={tickers.includes(c.ticker)}
                    onClick={() => setTickers((t) => toggle(t, c.ticker))}
                    title={c.name}
                  >
                    {c.ticker}
                  </button>
                ))}
              </div>
              <div className="chip-group" role="group" aria-label="Fiscal years">
                {corpus.fiscal_years.map((y) => (
                  <button
                    key={y}
                    type="button"
                    className={`chip${years.includes(y) ? " is-on" : ""}`}
                    aria-pressed={years.includes(y)}
                    onClick={() => setYears((ys) => toggle(ys, y))}
                  >
                    FY{y}
                  </button>
                ))}
              </div>
              {(tickers.length > 0 || years.length > 0) && (
                <button type="button" className="link-btn" onClick={() => { setTickers([]); setYears([]); }}>
                  Clear filters
                </button>
              )}
            </>
          )}
        </fieldset>
      </form>

      {!hasResult && (
        <section className="empty">
          <h2>Try a question</h2>
          <div className="examples">
            {EXAMPLES.map((ex) => (
              <button key={ex} type="button" className="example" onClick={() => { setQuestion(ex); submit(ex); }}>
                {ex}
              </button>
            ))}
          </div>
        </section>
      )}

      {hasResult && (
        <div className="result">
          <section className="answer" aria-live="polite">
            <h2 className="answer-q">{state.question}</h2>
            {STATUS_TEXT[state.status] && !state.answer && !state.refusal && (
              <p className="status"><span className="spinner" aria-hidden="true" /> {STATUS_TEXT[state.status]}</p>
            )}
            {state.answer && (
              <AnswerText
                text={state.answer}
                citationIds={citationIds}
                activeId={activeId}
                onCite={selectCitation}
                streaming={state.status === "generating"}
              />
            )}
            {state.refusal && (
              <div className="notice notice-warn" role="status">
                <strong>No supported answer.</strong> {state.refusal}
              </div>
            )}
            {state.error && (
              <div className="notice notice-error" role="alert">
                <strong>Something went wrong.</strong> {state.error}
              </div>
            )}
            {state.status === "cancelled" && <p className="muted">Stopped.</p>}
            {state.trace && <TraceBar trace={state.trace} />}

            {history.length > 1 && (
              <div className="history">
                <h3>Recent</h3>
                <ul>
                  {history.slice(1).map((h) => (
                    <li key={h}>
                      <button type="button" className="link-btn" onClick={() => submit(h)} disabled={busy}>{h}</button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </section>

          <aside className="sources" aria-label="Sources">
            <h2 className="sources-title">
              Sources {state.citations.length > 0 && <span className="count">{state.citations.length}</span>}
            </h2>
            {state.status === "retrieving" && <p className="muted">Retrieving…</p>}
            {state.status !== "retrieving" && state.citations.length === 0 && (
              <p className="muted">No passages matched.</p>
            )}
            {state.citations.map((c) => (
              <SourceCard
                key={c.chunk_id}
                ref={(el) => { if (el) cardRefs.current.set(c.id, el); else cardRefs.current.delete(c.id); }}
                citation={c}
                active={activeId === c.id}
                onSelect={setActiveId}
              />
            ))}
          </aside>
        </div>
      )}
    </div>
  );
}
