"""Keep the baseline in git so CI compares against the same numbers every time.

  python3 scripts/baseline.py export --run run_abc123 --out data/baseline.json
  python3 scripts/baseline.py import data/baseline.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filinglens import config, db  # noqa: E402
from filinglens.evals import baseline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--run", required=True)
    ex.add_argument("--out", required=True)
    sub.add_parser("import").add_argument("path")
    args = ap.parse_args()
    conn = db.connect(config.load().db_path)
    if args.cmd == "export":
        print(baseline.export_run(conn, args.run, args.out), "results written to", args.out)
    else:
        print("imported baseline run", baseline.import_run(conn, args.path))
