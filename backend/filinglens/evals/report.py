"""Compare two metric summaries: markdown table for a PR comment, plus a regression gate."""

# (key, label, higher_is_better, gated)
METRICS = [
    ("recall_at_5", "Recall@5", True, True),
    ("mrr", "MRR", True, True),
    ("ndcg_at_5", "nDCG@5", True, False),
    ("faithfulness", "Faithfulness", True, True),
    ("answer_correctness", "Answer correctness", True, True),
    ("citation_accuracy", "Citation accuracy", True, True),
    ("refusal_accuracy", "Refusal accuracy", True, True),
    ("p95_latency_ms", "p95 retrieval ms", False, False),
    ("p95_e2e_ms", "p95 end-to-end ms", False, False),
    ("cost_per_query_usd", "Cost / query ($)", False, False),
]


def compare(base: dict, new: dict) -> list[dict]:
    rows = []
    for key, label, higher, gated in METRICS:
        b, n = base.get(key), new.get(key)
        if b is None and n is None:
            continue
        delta = None if b is None or n is None else round(n - b, 4)
        rows.append({"key": key, "label": label, "base": b, "new": n, "delta": delta,
                     "higher_is_better": higher, "gated": gated})
    return rows


def regressions(rows: list[dict], max_drop: float) -> list[dict]:
    """Gated quality metrics that fell by more than `max_drop` (absolute)."""
    return [r for r in rows if r["gated"] and r["delta"] is not None and r["delta"] < -max_drop]


def to_markdown(rows: list[dict], title: str, note: str = "") -> str:
    fmt = lambda v: "–" if v is None else (f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}")
    out = [f"### {title}", ""]
    if note:
        out += [note, ""]
    out += ["| Metric | Baseline | This run | Δ |", "| --- | --- | --- | --- |"]
    for r in rows:
        d = r["delta"]
        if d is None or d == 0:
            mark = ""
        else:
            mark = " ✅" if (d > 0) == r["higher_is_better"] else " ⚠️"
        out.append(f"| {r['label']} | {fmt(r['base'])} | {fmt(r['new'])} | "
                   f"{'–' if d is None else f'{d:+.3f}'}{mark} |")
    return "\n".join(out)
