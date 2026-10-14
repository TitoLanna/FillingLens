// Shapes shared with the backend. See API_CONTRACT.md.

export interface QueryFilters {
  tickers?: string[];
  fiscal_years?: number[];
}

export interface QueryRequest {
  question: string;
  filters?: QueryFilters;
  top_k?: number;
}

export interface CitationScores {
  bm25?: number | null;
  dense?: number | null;
  rrf?: number | null;
  rerank?: number | null;
}

export interface Citation {
  id: number;
  chunk_id: string;
  company: string;
  ticker: string;
  fiscal_year: number;
  section: string;
  text: string;
  source_url: string;
  scores: CitationScores;
}

export interface TraceStage {
  name: string;
  ms: number;
}

export interface QueryTrace {
  trace_id: string;
  total_ms: number;
  stages: TraceStage[];
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
}

export type StreamEvent =
  | { type: "citations"; citations: Citation[] }
  | { type: "token"; text: string }
  | { type: "refusal"; reason: string }
  | { type: "done"; trace: QueryTrace }
  | { type: "error"; message: string };

export interface Company {
  ticker: string;
  name: string;
}

export interface Corpus {
  companies: Company[];
  fiscal_years: number[];
}

export interface EvalMetrics {
  recall_at_5: number;
  mrr: number;
  faithfulness: number;
  p95_latency_ms: number;
  cost_per_query_usd: number;
}

export interface EvalRun {
  id: string;
  label: string;
  git_sha: string;
  started_at: string;
  is_baseline: boolean;
  metrics: EvalMetrics;
}
