"""Run the golden set: python3 scripts/run_evals.py --label "v0 baseline" --pipeline baseline --baseline"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filinglens import config, db, embeddings  # noqa: E402
from filinglens.evals.runner import run_eval  # noqa: E402
from filinglens.retrieval.service import PRESETS, Retriever  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--pipeline", choices=sorted(PRESETS), default="baseline")
    ap.add_argument("--baseline", action="store_true", help="mark this run as the baseline")
    ap.add_argument("--generate", action="store_true", help="also answer every question (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--judge", action="store_true", help="also score answers with the LLM judge (implies --generate)")
    ap.add_argument("--subset", type=int, help="run a stratified subset of N questions")
    args = ap.parse_args()
    settings = config.load()
    embeddings.configure(settings)
    conn = db.connect(settings.db_path)
    out = run_eval(conn, Retriever(conn), settings, PRESETS[args.pipeline], args.label, args.baseline,
                   args.generate, args.judge, args.subset)
    print(json.dumps(out, indent=2))
