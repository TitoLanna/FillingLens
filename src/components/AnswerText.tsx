import { Fragment } from "react";

interface Props {
  text: string;
  citationIds: Set<number>;
  activeId: number | null;
  onCite: (id: number) => void;
  streaming: boolean;
}

/** Renders the answer, turning [n] markers into buttons that jump to source n. */
export function AnswerText({ text, citationIds, activeId, onCite, streaming }: Props) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="answer-text">
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+)\]$/);
        if (m && citationIds.has(Number(m[1]))) {
          const id = Number(m[1]);
          return (
            <button
              key={i}
              type="button"
              className={`cite${activeId === id ? " is-active" : ""}`}
              onClick={() => onCite(id)}
              aria-label={`Show source ${id}`}
            >
              {id}
            </button>
          );
        }
        return <Fragment key={i}>{part}</Fragment>;
      })}
      {streaming && <span className="caret" aria-hidden="true" />}
    </p>
  );
}
