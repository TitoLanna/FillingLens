"""Measure how far to trust the LLM judge against your own labels.

  python3 scripts/calibrate_judge.py label --run run_abc123 --metric faithfulness --n 30
  python3 scripts/calibrate_judge.py report [--run run_abc123]

`label` shows an answer and its passages WITHOUT the judge's verdict (to avoid anchoring you),
asks for your pass/fail, then stores both. Needs a run made with --judge.
"""
import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filinglens import config, db  # noqa: E402
from filinglens.evals import calibration  # noqa: E402
from filinglens.evals.judge import METRICS  # noqa: E402

DEFS = {
    "faithfulness": "PASS = every claim in the answer is supported by the passages.",
    "correctness": "PASS = the answer states the same key facts as the reference answer.",
    "citations": "PASS = every [n] marker is attached to a claim that passage n supports.",
}


def label(conn, run_id: str, metric: str, n: int, seed: int) -> None:
    rows = conn.execute(
        "SELECT e.*, g.question, g.expected_answer FROM eval_results e JOIN golden_questions g "
        "ON g.id=e.question_id WHERE e.run_id=? AND e.judge_scores IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM judge_labels l WHERE l.run_id=e.run_id AND l.question_id=e.question_id "
        "AND l.metric=?)", (run_id, metric)).fetchall()
    rows = [r for r in rows if json.loads(r["judge_scores"]).get(metric) is not None]
    random.Random(seed).shuffle(rows)
    print(f"{len(rows)} judged answers available for '{metric}'. {DEFS[metric]}\n")
    done = 0
    for r in rows[:n]:
        print("=" * 80, f"\nQ: {r['question']}")
        if metric == "correctness":
            print(f"REFERENCE: {r['expected_answer']}")
        print(f"ANSWER: {r['answer']}\n")
        for i, cid in enumerate(json.loads(r["retrieved_chunk_ids"]), 1):
            text = conn.execute("SELECT text FROM chunks WHERE id=?", (cid,)).fetchone()
            print(f"[{i}]", textwrap.fill((text["text"] if text else "(missing)")[:700], 100,
                                          subsequent_indent="    "))
        while True:
            a = input(f"\n{metric} pass? (y/n/s skip/q quit) > ").strip().lower()
            if a in ("y", "n", "s", "q"):
                break
        if a == "q":
            break
        if a == "s":
            continue
        conn.execute("INSERT OR REPLACE INTO judge_labels VALUES (?,?,?,?,?)",
                     (run_id, r["question_id"], metric, int(a == "y"),
                      int(json.loads(r["judge_scores"])[metric])))
        conn.commit()
        done += 1
    print(f"\nlabelled {done}. Run `report` to see agreement.")


def report(conn, run_id) -> None:
    stats = calibration.report(conn, run_id)
    if not stats:
        print("no labels yet")
    for metric, s in stats.items():
        print(f"{metric:13} n={s['n']:3}  agreement={s['accuracy']}  "
              f"judge-catches-good(TPR)={s['tpr']}  judge-catches-bad(TNR)={s['tnr']}  kappa={s['kappa']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("label")
    lb.add_argument("--run", required=True)
    lb.add_argument("--metric", choices=METRICS, required=True)
    lb.add_argument("--n", type=int, default=30)
    lb.add_argument("--seed", type=int, default=0)
    sub.add_parser("report").add_argument("--run")
    args = ap.parse_args()
    conn = db.connect(config.load().db_path)
    label(conn, args.run, args.metric, args.n, args.seed) if args.cmd == "label" else report(conn, args.run)
