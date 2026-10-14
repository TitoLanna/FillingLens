"""CI gate: run a stratified subset, diff against the baseline on the SAME questions, fail on regression.

  python3 scripts/ci_eval.py --subset 30 --pipeline hybrid --generate --judge --max-drop 0.03 --output diff.md

The baseline run is the newest one marked --baseline. Its stored per-question results are
re-aggregated over only the questions in this subset, so the comparison is like for like.
Exit codes: 0 ok, 2 = a gated quality metric dropped by more than --max-drop, 1 = the run itself failed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filinglens import config, db, embeddings  # noqa: E402
from filinglens.evals import report  # noqa: E402
from filinglens.evals.runner import rows_from_run, run_eval, summarize  # noqa: E402
from filinglens.retrieval.service import PRESETS, Retriever  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=sorted(PRESETS), default="hybrid")
    ap.add_argument("--subset", type=int, default=30, help="questions to run (0 = the full golden set)")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--max-drop", type=float, default=0.03)
    ap.add_argument("--label", default="ci run")
    ap.add_argument("--output", help="write the markdown diff here (e.g. $GITHUB_STEP_SUMMARY)")
    args = ap.parse_args()

    settings = config.load()
    embeddings.configure(settings)
    conn = db.connect(settings.db_path)
    base = conn.execute("SELECT id, label FROM eval_runs WHERE is_baseline=1 ORDER BY started_at DESC LIMIT 1").fetchone()
    if not base:
        sys.exit("no baseline run: python3 scripts/run_evals.py --label 'v0 baseline' --baseline")

    result = run_eval(conn, Retriever(conn), settings, PRESETS[args.pipeline], args.label, False,
                      args.generate, args.judge, args.subset or None)
    ids = {r["question_id"] for r in conn.execute("SELECT question_id FROM eval_results WHERE run_id=?", (result["id"],))}
    base_summary = summarize(rows_from_run(conn, base["id"], ids))
    rows = report.compare(base_summary, result["summary_metrics"])
    md = report.to_markdown(rows, f"Eval: {args.pipeline} vs baseline “{base['label']}”",
                            f"{len(ids)} questions (stratified subset), run `{result['id']}`.")
    print(md)
    if args.output:
        Path(args.output).write_text(md + "\n")
    bad = report.regressions(rows, args.max_drop)
    if bad:
        print("\nREGRESSION: " + ", ".join(f"{r['label']} {r['delta']:+.3f}" for r in bad), file=sys.stderr)
        sys.exit(2)
