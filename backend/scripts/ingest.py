"""Ingest 10-Ks from EDGAR.

  python3 scripts/ingest.py --tickers AAPL MSFT --years 2023 2024
  python3 scripts/ingest.py --manifest data/corpus.json     # {"tickers": [...], "years": [...], "ciks": {"XOM": [34088]}}
  python3 scripts/ingest.py --manifest data/corpus.json --download-only   # just save raw filings to RAW_DIR

Downloaded filings are cached in RAW_DIR, so re-running (e.g. after a parser change) does not hit the SEC again.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filinglens import config, db, embeddings  # noqa: E402
from filinglens.ingestion import edgar  # noqa: E402
from filinglens.ingestion.worker import ingest_url  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--years", nargs="+", type=int)
    ap.add_argument("--download-only", action="store_true", help="save raw filings only; no parsing, embedding or database")
    ap.add_argument("--manifest", help="JSON file with tickers and years (the corpus definition CI uses)")
    args = ap.parse_args()
    extra_ciks: dict = {}
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        args.tickers, args.years = manifest["tickers"], manifest["years"]
        extra_ciks = manifest.get("ciks", {})  # {ticker: [predecessor/successor CIKs]}
    if not args.tickers or not args.years:
        sys.exit("give --manifest, or both --tickers and --years")
    settings = config.load()
    if "example.com" in settings.sec_user_agent:
        sys.exit("Set SEC_USER_AGENT (in backend/.env) to a contact the SEC can reach, e.g. 'FilingLens Your Name you@yourdomain.com'.")
    if args.download_only:
        total = fresh = 0
        for ticker in args.tickers:
            try:
                docs = edgar.find_10k_filings(ticker, args.years, settings.sec_user_agent, extra_ciks.get(ticker, []))
            except Exception as exc:
                print(f"{ticker}: discovery failed: {exc}")
                continue
            for missing in sorted(set(args.years) - {d["fiscal_year"] for d in docs}):
                print(f"{ticker} {missing}: no 10-K found")
            for doc in docs:
                try:
                    html, cached = edgar.fetch_filing(doc["url"], settings.sec_user_agent, settings.raw_dir)
                except Exception as exc:
                    print(f"{doc['ticker']} {doc['fiscal_year']}: FAILED {exc}")
                    continue
                total += 1
                fresh += not cached
                print(f"{doc['ticker']} {doc['fiscal_year']}: {'cached' if cached else 'downloaded'} ({len(html) / 1e6:.1f} MB)")
        print(f"{total} filings in {settings.raw_dir} ({fresh} newly downloaded)")
        sys.exit(0)
    embeddings.configure(settings)
    conn = db.connect(settings.db_path)
    print("embedding model:", embeddings.model_name())
    for ticker in args.tickers:
        try:
            docs = edgar.find_10k_filings(ticker, args.years, settings.sec_user_agent, extra_ciks.get(ticker, []))
        except Exception as exc:
            print(f"{ticker}: discovery failed: {exc}")
            continue
        for missing in sorted(set(args.years) - {d["fiscal_year"] for d in docs}):
            print(f"{ticker} {missing}: no 10-K found")
        for doc in docs:
            try:
                print(f"{doc['ticker']} {doc['fiscal_year']}: {ingest_url(conn, doc, settings.sec_user_agent, settings.raw_dir)}")
            except Exception as exc:
                print(f"{doc['ticker']} {doc['fiscal_year']}: FAILED {exc}")
    n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print("total chunks:", n)
