"""How far to trust the judge: agreement between your labels and its verdicts."""
import sqlite3


def agreement(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs are (human_pass, judge_pass). Positive class = pass.

    tpr: of human-pass cases, how many the judge passed. tnr: of human-fail cases, how many it
    failed. kappa: agreement beyond chance (1 = perfect, 0 = chance). Fields are None when undefined.
    """
    n = len(pairs)
    if not n:
        return {"n": 0, "accuracy": None, "tpr": None, "tnr": None, "kappa": None}
    tp = sum(h and j for h, j in pairs)
    tn = sum((not h) and (not j) for h, j in pairs)
    fp = sum((not h) and j for h, j in pairs)
    fn = sum(h and (not j) for h, j in pairs)
    observed = (tp + tn) / n
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    ratio = lambda a, b: round(a / b, 3) if b else None
    return {"n": n, "accuracy": round(observed, 3), "tpr": ratio(tp, tp + fn), "tnr": ratio(tn, tn + fp),
            "kappa": None if expected == 1 else round((observed - expected) / (1 - expected), 3)}


def report(conn: sqlite3.Connection, run_id: str | None = None) -> dict:
    """{metric: agreement stats} over all stored labels (optionally for one run)."""
    sql = "SELECT metric, human_pass, judge_pass FROM judge_labels"
    rows = conn.execute(sql + (" WHERE run_id=?" if run_id else ""), (run_id,) if run_id else ()).fetchall()
    by_metric: dict[str, list] = {}
    for r in rows:
        by_metric.setdefault(r["metric"], []).append((bool(r["human_pass"]), bool(r["judge_pass"])))
    return {m: agreement(p) for m, p in sorted(by_metric.items())}
