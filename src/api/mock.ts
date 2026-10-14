// In-browser stand-in for the backend. Companies and passages are fictional,
// so the demo never puts invented words in a real company's mouth.
import type { Citation, Corpus, EvalRun, QueryRequest, StreamEvent } from "./types";

interface Passage {
  chunk_id: string;
  ticker: string;
  company: string;
  fiscal_year: number;
  section: string;
  text: string;
  /** One-sentence answer fragment the mock "LLM" uses when it cites this passage. */
  claim: string;
}

const PASSAGES: Passage[] = [
  {
    chunk_id: "ctso-2024-1a-0012", ticker: "CTSO", company: "Contoso Cloud", fiscal_year: 2024,
    section: "Item 1A. Risk Factors",
    text: "We depend on a limited number of suppliers for the GPUs and networking equipment in our data centers. Supply constraints or price increases from these suppliers could delay capacity expansion and reduce our margins.",
    claim: "Contoso Cloud flags dependence on a small number of GPU and networking suppliers, warning that shortages could delay data center expansion and squeeze margins",
  },
  {
    chunk_id: "ctso-2024-7-0041", ticker: "CTSO", company: "Contoso Cloud", fiscal_year: 2024,
    section: "Item 7. MD&A",
    text: "Cloud services revenue grew 31% year over year to $18.2 billion, driven by AI infrastructure demand. Capital expenditures increased to $9.4 billion as we expanded data center capacity.",
    claim: "Contoso Cloud's cloud services revenue grew 31% to $18.2 billion in fiscal 2024 on AI infrastructure demand, while capital expenditures rose to $9.4 billion",
  },
  {
    chunk_id: "ctso-2023-7-0038", ticker: "CTSO", company: "Contoso Cloud", fiscal_year: 2023,
    section: "Item 7. MD&A",
    text: "Cloud services revenue grew 24% year over year to $13.9 billion. Operating margin for the segment was 29%.",
    claim: "In fiscal 2023, Contoso Cloud's cloud services revenue grew 24% to $13.9 billion at a 29% segment operating margin",
  },
  {
    chunk_id: "nwsc-2024-1a-0007", ticker: "NWSC", company: "Northwind Semiconductor", fiscal_year: 2024,
    section: "Item 1A. Risk Factors",
    text: "Substantially all of our wafers are manufactured by a single third-party foundry located in East Asia. Geopolitical tension, natural disasters, or export restrictions affecting that region could interrupt our supply.",
    claim: "Northwind Semiconductor relies on a single East Asian foundry for nearly all wafers and cites geopolitical and export-control risk to that supply",
  },
  {
    chunk_id: "nwsc-2024-7-0029", ticker: "NWSC", company: "Northwind Semiconductor", fiscal_year: 2024,
    section: "Item 7. MD&A",
    text: "Revenue increased 18% to $6.1 billion. Gross margin declined 2.3 points to 54.1% due to higher foundry pricing and inventory write-downs.",
    claim: "Northwind's revenue rose 18% to $6.1 billion while gross margin fell 2.3 points to 54.1% on higher foundry pricing",
  },
  {
    chunk_id: "nwsc-2023-1a-0009", ticker: "NWSC", company: "Northwind Semiconductor", fiscal_year: 2023,
    section: "Item 1A. Risk Factors",
    text: "New export control regulations may restrict our ability to sell certain advanced products to customers in some countries, which together represented 14% of revenue.",
    claim: "Northwind also warned that new export controls could restrict sales to markets that made up 14% of its revenue",
  },
  {
    chunk_id: "fbrk-2024-1a-0015", ticker: "FBRK", company: "Fabrikam Retail", fiscal_year: 2024,
    section: "Item 1A. Risk Factors",
    text: "A cybersecurity incident affecting our payment systems or customer data could result in regulatory penalties, litigation, and loss of customer trust.",
    claim: "Fabrikam Retail identifies cybersecurity incidents involving payment systems or customer data as a key risk, with regulatory and litigation exposure",
  },
  {
    chunk_id: "fbrk-2024-7-0033", ticker: "FBRK", company: "Fabrikam Retail", fiscal_year: 2024,
    section: "Item 7. MD&A",
    text: "Comparable store sales increased 3.2%, while e-commerce sales grew 19% and now represent 27% of total revenue. Inventory levels decreased 6% year over year.",
    claim: "Fabrikam's comparable store sales rose 3.2% as e-commerce grew 19% to 27% of revenue, and inventory fell 6%",
  },
  {
    chunk_id: "tlsp-2024-1a-0004", ticker: "TLSP", company: "Tailspin Air", fiscal_year: 2024,
    section: "Item 1A. Risk Factors",
    text: "Fuel is our second-largest operating expense. A $10 per barrel increase in jet fuel prices would increase our annual operating costs by approximately $210 million.",
    claim: "Tailspin Air estimates a $10 per barrel rise in jet fuel would add about $210 million in annual operating costs",
  },
  {
    chunk_id: "tlsp-2024-7-0021", ticker: "TLSP", company: "Tailspin Air", fiscal_year: 2024,
    section: "Item 7. MD&A",
    text: "Passenger revenue grew 9% to $11.7 billion on higher load factors. Labor costs increased 12% following new pilot and flight attendant agreements.",
    claim: "Tailspin's passenger revenue grew 9% to $11.7 billion while labor costs rose 12% under new pilot and flight attendant contracts",
  },
  {
    chunk_id: "tlsp-2023-1a-0006", ticker: "TLSP", company: "Tailspin Air", fiscal_year: 2023,
    section: "Item 1A. Risk Factors",
    text: "Delays in aircraft deliveries from manufacturers could constrain our planned capacity growth and require us to extend leases on older, less fuel-efficient aircraft.",
    claim: "Tailspin Air noted that aircraft delivery delays could limit capacity growth and force lease extensions on older aircraft",
  },
];

const STOPWORDS = new Set(
  "a an and are as at be by did do does for from has have how in is it its of on or our over the their this to was were what which who why with year years report reported reports company companies filing filings".split(" "),
);

function tokenize(s: string): string[] {
  return s.toLowerCase().replace(/[^a-z0-9$%. ]/g, " ").split(/\s+/)
    .map((w) => w.replace(/\.$/, ""))
    .filter((w) => w.length > 1 && !STOPWORDS.has(w));
}

/** Crude stemming so "risks" matches "risk" and "suppliers" matches "supplier". */
const stem = (w: string) => w.replace(/(ies|es|s)$/, "");

const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => { clearTimeout(t); reject(new DOMException("Aborted", "AbortError")); }, { once: true });
  });

const round = (n: number, d = 3) => Math.round(n * 10 ** d) / 10 ** d;

export async function mockCorpus(): Promise<Corpus> {
  const seen = new Map<string, string>();
  PASSAGES.forEach((p) => seen.set(p.ticker, p.company));
  return {
    companies: [...seen].map(([ticker, name]) => ({ ticker, name })).sort((a, b) => a.name.localeCompare(b.name)),
    fiscal_years: [...new Set(PASSAGES.map((p) => p.fiscal_year))].sort(),
  };
}

export async function* mockStream(req: QueryRequest, signal: AbortSignal): AsyncGenerator<StreamEvent> {
  const started = performance.now();
  const qTerms = new Set(tokenize(req.question).map(stem));
  const tickers = req.filters?.tickers ?? [];
  const years = req.filters?.fiscal_years ?? [];

  const candidates = PASSAGES.filter(
    (p) => (tickers.length === 0 || tickers.includes(p.ticker)) && (years.length === 0 || years.includes(p.fiscal_year)),
  );

  // Fake "hybrid" scoring: keyword overlap stands in for BM25; a company-name
  // bonus stands in for dense similarity.
  const scored = candidates
    .map((p) => {
      const pTerms = new Set(tokenize(`${p.company} ${p.section} ${p.text}`).map(stem));
      let overlap = 0;
      qTerms.forEach((t) => { if (pTerms.has(t)) overlap += 1; });
      const nameHit = tokenize(p.company).some((w) => qTerms.has(stem(w))) ? 1 : 0;
      return { p, overlap, nameHit, score: overlap + nameHit * 1.5 };
    })
    .filter((s) => s.overlap > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, req.top_k ?? 4);

  await sleep(350, signal); // retrieval + rerank latency

  const citations: Citation[] = scored.map((s, i) => ({
    id: i + 1,
    chunk_id: s.p.chunk_id,
    company: s.p.company,
    ticker: s.p.ticker,
    fiscal_year: s.p.fiscal_year,
    section: s.p.section,
    text: s.p.text,
    source_url: "https://www.sec.gov/edgar/search/",
    scores: {
      bm25: s.overlap > 0 ? round(4 + s.overlap * 2.7, 2) : null,
      dense: s.nameHit || s.overlap > 1 ? round(0.62 + Math.min(s.score, 6) * 0.05) : null,
      rrf: round(1 / (60 + i + 1) + 1 / (60 + i + 2), 4),
      rerank: round(Math.max(0.35, 0.96 - i * 0.11)),
    },
  }));
  yield { type: "citations", citations };

  const llmStart = performance.now();
  // If the question names a company, only that company's passages can support the answer.
  const named = scored.filter((s) => s.nameHit);
  const pool = named.length > 0 ? named : scored;
  const strong = pool.filter((s) => s.score >= 2);
  if (strong.length === 0) {
    await sleep(500, signal);
    yield {
      type: "refusal",
      reason: "The filings in scope don't contain enough information to answer this. Try naming a company, a topic like risks or revenue, or widening the year filter.",
    };
  } else {
    const answer = strong.map((s) => `${s.p.claim} [${scored.indexOf(s) + 1}].`).join(" ");
    for (const word of answer.match(/\S+\s*/g) ?? []) {
      await sleep(22, signal);
      yield { type: "token", text: word };
    }
  }

  const llmMs = Math.round(performance.now() - llmStart);
  const totalMs = Math.round(performance.now() - started);
  const promptTokens = 900 + citations.length * 280;
  const completionTokens = strong.length * 42 + 20;
  yield {
    type: "done",
    trace: {
      trace_id: `tr_${Math.random().toString(16).slice(2, 8)}`,
      total_ms: totalMs,
      stages: [
        { name: "bm25", ms: 18 },
        { name: "dense", ms: 26 },
        { name: "rrf", ms: 1 },
        { name: "rerank", ms: 121 },
        { name: "llm", ms: llmMs },
      ],
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      cost_usd: round(promptTokens * 3e-6 + completionTokens * 15e-6, 4),
    },
  };
}

export async function mockEvalRuns(): Promise<EvalRun[]> {
  const runs: EvalRun[] = [
    { id: "run_001", label: "v0 baseline (dense only)", git_sha: "4e1c9a0", started_at: "2026-11-02T17:10:00Z", is_baseline: true,
      metrics: { recall_at_5: 0.61, mrr: 0.48, faithfulness: 0.82, p95_latency_ms: 140, cost_per_query_usd: 0.0068 } },
    { id: "run_002", label: "+ BM25 hybrid (RRF)", git_sha: "9b77d21", started_at: "2026-11-09T16:42:00Z", is_baseline: false,
      metrics: { recall_at_5: 0.74, mrr: 0.59, faithfulness: 0.86, p95_latency_ms: 165, cost_per_query_usd: 0.0069 } },
    { id: "run_003", label: "+ cross-encoder reranker", git_sha: "c03f5be", started_at: "2026-11-14T19:05:00Z", is_baseline: false,
      metrics: { recall_at_5: 0.83, mrr: 0.72, faithfulness: 0.91, p95_latency_ms: 290, cost_per_query_usd: 0.0071 } },
    { id: "run_004", label: "+ section-aware chunking", git_sha: "e5a0d48", started_at: "2026-11-20T18:04:00Z", is_baseline: false,
      metrics: { recall_at_5: 0.87, mrr: 0.76, faithfulness: 0.93, p95_latency_ms: 285, cost_per_query_usd: 0.0066 } },
  ];
  return runs.reverse(); // newest first, as the API returns them
}
