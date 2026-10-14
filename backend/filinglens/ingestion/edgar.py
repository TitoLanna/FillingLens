"""Find 10-K filings on EDGAR for a ticker and fiscal years. Polite: descriptive UA, <10 req/s."""
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from .. import http

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
_MIN_GAP = 0.15
_last = 0.0
_tickers: dict[str, dict] | None = None


def _throttle() -> None:
    """Keep every SEC request at least _MIN_GAP apart (SEC fair access: max 10 requests/second)."""
    global _last
    wait = _MIN_GAP - (time.monotonic() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.monotonic()


def _get(url: str, user_agent: str):
    _throttle()
    return http.get_json(url, {"User-Agent": user_agent})


def raw_path(url: str, raw_dir: Path) -> Path:
    """Cache location for a filing, derived from its URL (cik / accession / document), so it
    does not depend on how we label the fiscal year and survives parser or labeling changes."""
    parts = urlparse(url).path.split("/data/", 1)[-1].strip("/").split("/")
    name = "_".join(re.sub(r"[^A-Za-z0-9._-]", "_", p) for p in parts if p)
    if not name or name.startswith("."):
        raise ValueError(f"cannot derive a cache name from {url!r}")
    return Path(raw_dir) / name


def fetch_filing(url: str, user_agent: str, raw_dir: Path | None = None) -> tuple[str, bool]:
    """Return (html, came_from_cache). With `raw_dir`, downloads are saved and reused."""
    path = raw_path(url, raw_dir) if raw_dir else None
    if path and path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8", errors="replace"), True
    _throttle()
    html = http.get_text(url, {"User-Agent": user_agent})
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")  # atomic: no half-written cache files
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html)
        os.replace(tmp, path)
    return html, False


def lookup_ticker(ticker: str, user_agent: str) -> dict:
    global _tickers
    if _tickers is None:
        _tickers = {v["ticker"].upper(): v for v in _get(TICKERS_URL, user_agent).values()}
    hit = _tickers.get(ticker.upper())
    if not hit:
        raise ValueError(f"unknown ticker: {ticker}")
    return hit


def _scan(block: dict, cik: int, wanted: set, found: dict, name: str) -> None:
    """Pick the first (newest) original 10-K per wanted fiscal year out of one submissions block."""
    for form, report_date, acc, doc in zip(block["form"], block["reportDate"],
                                           block["accessionNumber"], block["primaryDocument"]):
        if form != "10-K" or not report_date:
            continue
        year = int(report_date[:4])
        if year in wanted and year not in found:
            found[year] = {"ticker": "", "company": name, "fiscal_year": year,
                           "url": ARCHIVE_URL.format(cik=cik, acc=acc.replace("-", ""), doc=doc)}


def find_10k_filings(ticker: str, years: list[int], user_agent: str, extra_ciks: list[int] = ()) -> list[dict]:
    """Return one doc dict {ticker, company, fiscal_year, url} per requested year found.

    fiscal_year is the year of the period end date (reportDate). Searches the recent-filings feed,
    then the older history pages EDGAR splits off for heavy filers (e.g. banks with thousands of
    prospectuses). `extra_ciks` adds predecessor or successor registrants, for companies whose
    filings moved to a different CIK after a reorganization.
    """
    info = lookup_ticker(ticker, user_agent)
    ciks = [int(info["cik_str"])] + [int(c) for c in extra_ciks]
    wanted, found = set(years), {}
    name = info["title"]
    for i, cik in enumerate(ciks):
        sub = _get(SUBMISSIONS_URL.format(cik=cik), user_agent)
        if i == 0:
            name = sub.get("name") or name
        _scan(sub["filings"]["recent"], cik, wanted, found, name)
        for page in sub["filings"].get("files", []):
            if wanted <= set(found):
                break
            _scan(_get(f"https://data.sec.gov/submissions/{page['name']}", user_agent), cik, wanted, found, name)
        if wanted <= set(found):
            break
    for doc in found.values():
        doc["ticker"] = ticker.upper()
    return [found[y] for y in sorted(found)]
