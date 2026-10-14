"""Find 10-K filings on EDGAR for a ticker and fiscal years. Polite: descriptive UA, <10 req/s."""
import time

from .. import http

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
_MIN_GAP = 0.15
_last = 0.0
_tickers: dict[str, dict] | None = None


def _get(url: str, user_agent: str):
    global _last
    wait = _MIN_GAP - (time.monotonic() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.monotonic()
    return http.get_json(url, {"User-Agent": user_agent})


def lookup_ticker(ticker: str, user_agent: str) -> dict:
    global _tickers
    if _tickers is None:
        _tickers = {v["ticker"].upper(): v for v in _get(TICKERS_URL, user_agent).values()}
    hit = _tickers.get(ticker.upper())
    if not hit:
        raise ValueError(f"unknown ticker: {ticker}")
    return hit


def find_10k_filings(ticker: str, years: list[int], user_agent: str) -> list[dict]:
    """Return one doc dict {ticker, company, fiscal_year, url} per requested year found.

    fiscal_year is the year of the period end date (reportDate). Only the ~1000 most
    recent filings in the submissions feed are searched, which covers recent years.
    """
    info = lookup_ticker(ticker, user_agent)
    cik = int(info["cik_str"])
    sub = _get(SUBMISSIONS_URL.format(cik=cik), user_agent)
    recent = sub["filings"]["recent"]
    wanted, found = set(years), {}
    for form, report_date, acc, doc in zip(recent["form"], recent["reportDate"],
                                           recent["accessionNumber"], recent["primaryDocument"]):
        if form != "10-K" or not report_date:
            continue
        year = int(report_date[:4])
        if year in wanted and year not in found:  # feed is newest first; keep the original filing
            found[year] = {"ticker": ticker.upper(), "company": sub.get("name") or info["title"],
                           "fiscal_year": year,
                           "url": ARCHIVE_URL.format(cik=cik, acc=acc.replace("-", ""), doc=doc)}
    return [found[y] for y in sorted(found)]
