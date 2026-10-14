"""Structure-aware chunking: prose in overlapping word windows; tables kept whole or split by row.

A table chunk is self-contained: it starts with the caption (the short lines just above the table,
e.g. 'CONSOLIDATED STATEMENTS OF OPERATIONS' and the units line) and the header rows, and rows are
never cut in half. A table too long for one chunk is split between rows, repeating caption and header.
"""
import re

from ..text import chunk_words
from .parse import to_markdown

NUMERIC = re.compile(r"^[\(\$€£¥-]*\d[\d,\.]*[\)%]*$")
YEAR = re.compile(r"^(19|20)\d{2}$")      # column headers like "2024" look numeric but are labels
CAPTION_WORDS = 25      # a line longer than this is prose, not a caption
CAPTION_LINES = 2


def _words(row: list[str]) -> int:
    return sum(len(c.split()) for c in row)


def _header_len(rows: list[list[str]]) -> int:
    """Header = leading rows with no numeric cell (capped at 3), always leaving a data row."""
    n = 0
    for r in rows[:3]:
        if any(NUMERIC.match(c) and not YEAR.match(c) for c in r[1:]):
            break
        n += 1
    return min(n, len(rows) - 1) if len(rows) > 1 else 0


def _align_header(header: list[list[str]], body: list[list[str]]) -> list[list[str]]:
    """Column headers usually have no label cell, so they sit one column left of the numbers
    beneath them. Prepend an empty label cell when a header row is exactly one cell short."""
    if not body:
        return header
    widths = [len(r) for r in body if len(r) > 1]
    if not widths:
        return header
    width = max(set(widths), key=widths.count)   # the typical data-row width
    return [[""] + r if len(r) == width - 1 else r for r in header]


def chunk_table(rows: list[list[str]], caption: str, max_words: int) -> list[str]:
    h = _header_len(rows)
    header, body = rows[:h], rows[h:]
    header = _align_header(header, body)
    cap_words = len(caption.split())
    budget = max(40, max_words - cap_words - sum(_words(r) for r in header))
    groups: list[list[list[str]]] = [[]]
    used = 0
    for r in body:
        w = _words(r)
        if groups[-1] and used + w > budget:
            groups.append([])
            used = 0
        groups[-1].append(r)
        used += w
    out = []
    for g in groups:
        table = to_markdown(header + g) if header else to_markdown(g)
        out.append((caption + "\n" if caption else "") + table)
    return out


def chunk_blocks(blocks: list[tuple[str, object]], size: int = 200, overlap: int = 40,
                 table_words: int = 260) -> list[str]:
    """Chunk one section's blocks (from parse.parse_blocks) in document order."""
    chunks: list[str] = []
    prose: list[str] = []
    recent: list[str] = []          # short text lines just before the next table

    def flush_prose() -> None:
        if prose:
            chunks.extend(chunk_words(" ".join(prose), size, overlap))
            prose.clear()

    for kind, payload in blocks:
        if kind == "text":
            prose.append(payload)
            recent.append(payload)
            recent[:] = recent[-CAPTION_LINES:]
        else:
            flush_prose()
            caption = "\n".join(l for l in recent if len(l.split()) <= CAPTION_WORDS)
            chunks.extend(chunk_table(payload, caption, table_words))
            recent.clear()
    flush_prose()
    return chunks
