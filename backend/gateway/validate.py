"""Request validation. The gateway forwards a rebuilt body, never the raw client JSON."""
import re

TICKER = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")
MAX_QUESTION = 1000


class ValidationError(ValueError):
    pass


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def clean_query(body) -> dict:
    """Validate a /v1/query body and return only the fields public callers may set.

    `pipeline` is deliberately dropped: choosing retrieval configs is an operator action.
    """
    if not isinstance(body, dict):
        raise ValidationError("body must be a JSON object")
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValidationError("question is required")
    question = question.strip()
    if len(question) > MAX_QUESTION:
        raise ValidationError(f"question is too long (max {MAX_QUESTION} characters)")
    out: dict = {"question": question}

    filters = body.get("filters")
    if filters is not None:
        if not isinstance(filters, dict):
            raise ValidationError("filters must be an object")
        clean: dict = {}
        tickers = filters.get("tickers")
        if tickers is not None:
            if not isinstance(tickers, list) or len(tickers) > 10 or not all(
                    isinstance(t, str) and TICKER.match(t) for t in tickers):
                raise ValidationError("filters.tickers must be a list of up to 10 ticker symbols")
            clean["tickers"] = tickers
        years = filters.get("fiscal_years")
        if years is not None:
            if not isinstance(years, list) or len(years) > 10 or not all(
                    _is_int(y) and 1990 <= y <= 2100 for y in years):
                raise ValidationError("filters.fiscal_years must be a list of up to 10 years")
            clean["fiscal_years"] = years
        out["filters"] = clean

    top_k = body.get("top_k")
    if top_k is not None:
        if not _is_int(top_k) or not 1 <= top_k <= 20:
            raise ValidationError("top_k must be an integer from 1 to 20")
        out["top_k"] = top_k
    return out
