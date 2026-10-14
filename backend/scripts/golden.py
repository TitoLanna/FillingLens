"""Golden dataset tool.

  python3 scripts/golden.py generate --factual 20 --table 15 --multi-hop 15 --comparison 15 --unanswerable 15
  python3 scripts/golden.py review          # accept / edit / reject each candidate
  python3 scripts/golden.py stats
  python3 scripts/golden.py export data/golden.jsonl   # verified questions only (commit this file)
  python3 scripts/golden.py import data/golden.jsonl
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filinglens import config, db, embeddings  # noqa: E402
from filinglens.evals import golden  # noqa: E402
from filinglens.retrieval.service import PRESETS, Retriever  # noqa: E402

HELP = ("  a accept   r reject   s skip   x quit\n"
        "  eq edit question   ea edit expected answer   ec edit source chunk ids\n"
        "  f  find: what does the corpus return for this question? (use it to check 'unanswerable')")


def show_chunk(conn, cid: str, width: int = 100) -> None:
    r = conn.execute("SELECT c.section, c.text, d.company, d.fiscal_year FROM chunks c "
                     "JOIN documents d ON d.id=c.document_id WHERE c.id=?", (cid,)).fetchone()
    if not r:
        print(f"  [{cid}] (missing chunk)")
        return
    print(f"  [{cid}] {r['company']} FY{r['fiscal_year']} - {r['section']}")
    print(textwrap.indent(textwrap.fill(r["text"][:900], width), "      "))


def review(conn) -> None:
    retriever = Retriever(conn)
    todo = conn.execute("SELECT * FROM golden_questions WHERE status='candidate' ORDER BY type, id").fetchall()
    print(f"{len(todo)} candidates to review.\n{HELP}\n")
    for i, q in enumerate(todo, 1):
        while True:
            q = conn.execute("SELECT * FROM golden_questions WHERE id=?", (q["id"],)).fetchone()
            print(f"\n[{i}/{len(todo)}] {q['type']}  ({q['id']})")
            print(f"Q: {q['question']}\nA: {q['expected_answer']}")
            ids = json.loads(q["relevant_chunk_ids"])
            print("Sources:" if ids else "Sources: none (unanswerable)")
            for cid in ids:
                show_chunk(conn, cid)
            cmd = input("> ").strip().lower()
            if cmd == "a":
                golden.set_status(conn, q["id"], "verified")
            elif cmd == "r":
                golden.set_status(conn, q["id"], "rejected")
            elif cmd == "s":
                pass
            elif cmd == "x":
                return
            elif cmd == "eq":
                golden.update_fields(conn, q["id"], question=input("new question: ").strip() or q["question"])
                continue
            elif cmd == "ea":
                golden.update_fields(conn, q["id"], expected_answer=input("new answer: ").strip())
                continue
            elif cmd == "ec":
                raw = input("chunk ids, comma separated (empty for none): ")
                try:
                    golden.update_fields(conn, q["id"], relevant_chunk_ids=[c.strip() for c in raw.split(",") if c.strip()])
                except ValueError as exc:
                    print(exc)
                continue
            elif cmd == "f":
                cands, _ = retriever.retrieve(q["question"], None, PRESETS["hybrid"])
                for c in cands[:3]:
                    show_chunk(conn, c["chunk_id"])
                continue
            else:
                print(HELP)
                continue
            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    for t in golden.TYPES:
        g.add_argument(f"--{t}", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    sub.add_parser("review")
    sub.add_parser("stats")
    for name in ("export", "import"):
        sub.add_parser(name).add_argument("path")
    args = ap.parse_args()

    settings = config.load()
    embeddings.configure(settings)
    conn = db.connect(settings.db_path)
    if args.cmd == "generate":
        counts = {t: getattr(args, t.replace("-", "_")) for t in golden.TYPES}
        counts = {t: n for t, n in counts.items() if n}
        if not counts:
            sys.exit("give at least one count, e.g. --factual 20")
        print(json.dumps(golden.generate_candidates(conn, settings, counts, args.seed), indent=2))
        print("Next: python3 scripts/golden.py review")
    elif args.cmd == "review":
        review(conn)
    elif args.cmd == "stats":
        print(json.dumps(golden.stats(conn), indent=2))
    elif args.cmd == "export":
        print(golden.export_verified(conn, args.path), "verified questions written")
    elif args.cmd == "import":
        print(golden.import_verified(conn, args.path), "questions imported as verified")
