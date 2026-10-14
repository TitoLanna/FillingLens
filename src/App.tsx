import { useState } from "react";
import { USE_MOCK } from "./api/client";
import { AskView } from "./components/AskView";
import { ExperimentsView } from "./components/ExperimentsView";

type Tab = "ask" | "experiments";

export default function App() {
  const [tab, setTab] = useState<Tab>("ask");

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true">FL</span>
          <div>
            <h1>FilingLens</h1>
            <p className="tagline">Cited answers from SEC 10-K filings</p>
          </div>
        </div>
        <nav className="tabs" aria-label="Views">
          <button type="button" className={tab === "ask" ? "is-on" : ""} aria-current={tab === "ask" ? "page" : undefined} onClick={() => setTab("ask")}>
            Ask
          </button>
          <button type="button" className={tab === "experiments" ? "is-on" : ""} aria-current={tab === "experiments" ? "page" : undefined} onClick={() => setTab("experiments")}>
            Experiments
          </button>
        </nav>
      </header>

      {USE_MOCK && (
        <div className="mock-banner" role="note">
          Demo mode: fictional companies and sample data. Set <code>VITE_USE_MOCK=false</code> to use your API.
        </div>
      )}

      <main className="main">{tab === "ask" ? <AskView /> : <ExperimentsView />}</main>
    </div>
  );
}
