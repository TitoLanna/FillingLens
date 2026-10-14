import { useCallback, useRef, useState } from "react";
import { streamQuery } from "../api/client";
import type { Citation, QueryRequest, QueryTrace } from "../api/types";

export type QueryStatus = "idle" | "retrieving" | "generating" | "done" | "error" | "cancelled";

export interface QueryState {
  status: QueryStatus;
  question: string;
  answer: string;
  citations: Citation[];
  refusal: string | null;
  trace: QueryTrace | null;
  error: string | null;
}

const EMPTY: QueryState = {
  status: "idle", question: "", answer: "", citations: [], refusal: null, trace: null, error: null,
};

/** Runs one streamed query at a time; a new run cancels the previous one. */
export function useQueryStream() {
  const [state, setState] = useState<QueryState>(EMPTY);
  const controller = useRef<AbortController | null>(null);

  const run = useCallback(async (req: QueryRequest) => {
    controller.current?.abort();
    const ac = new AbortController();
    controller.current = ac;
    setState({ ...EMPTY, status: "retrieving", question: req.question });

    try {
      for await (const ev of streamQuery(req, ac.signal)) {
        if (ac.signal.aborted) return;
        switch (ev.type) {
          case "citations":
            setState((s) => ({ ...s, citations: ev.citations, status: "generating" }));
            break;
          case "token":
            setState((s) => ({ ...s, answer: s.answer + ev.text, status: "generating" }));
            break;
          case "refusal":
            setState((s) => ({ ...s, refusal: ev.reason }));
            break;
          case "done":
            setState((s) => ({ ...s, trace: ev.trace, status: "done" }));
            break;
          case "error":
            setState((s) => ({ ...s, error: ev.message, status: "error" }));
            return;
        }
      }
      // Stream ended without a "done" event: treat what we have as final.
      setState((s) => (s.status === "done" || s.status === "error" ? s : { ...s, status: "done" }));
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        setState((s) => ({ ...s, status: "cancelled" }));
        return;
      }
      setState((s) => ({ ...s, status: "error", error: (err as Error).message }));
    }
  }, []);

  const cancel = useCallback(() => controller.current?.abort(), []);

  return { state, run, cancel };
}
