"""HTML 10-K -> sections of blocks. Prose stays text; tables become cleaned row lists.

Uses html.parser from the stdlib. `parse_blocks` is what the chunker consumes; `parse_sections`
is the flattened text view (tables as Markdown) kept for inspection and tests.
"""
import re
from html.parser import HTMLParser

ITEM = re.compile(r"^\s*(item\s+\d+[a-c]?\.?)(.*)$", re.I)
SKIP = {"script", "style", "head", "ix:header"}  # ix:header = hidden inline-XBRL metadata
BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"}
CURRENCY = {"$", "€", "£", "¥"}
SUFFIX = {")", "%", ")%", "%)"}


def clean_row(cells: list[str]) -> list[str]:
    """Collapse SEC's sparse table rows: drop empty cells and glue '$', ')' and '%' fragments to
    their numbers, so `['Products', '$', '294,866', '', '', ')']` becomes `['Products', '$294,866)']`."""
    out: list[str] = []
    prefix = ""
    for raw in cells:
        c = " ".join(raw.split())
        if not c:
            continue
        if c in CURRENCY:
            prefix = c
            continue
        if c in SUFFIX and out:
            out[-1] += c
            continue
        out.append(prefix + c)
        prefix = ""
    return out


def clean_table(rows: list[list[str]]) -> list[list[str]]:
    return [r for r in (clean_row(r) for r in rows) if r]


def is_prose_table(rows: list[list[str]]) -> bool:
    """SEC filings use tables for layout (signature blocks, bullet lists). Those are really text."""
    cells = [c for r in rows for c in r]
    if not cells:
        return True
    if max(len(r) for r in rows) <= 1:
        return True
    return sum(len(c.split()) for c in cells) / len(cells) > 15


def to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


class _Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items: list[tuple[str, object]] = []   # ("text", str) | ("table", list[list[str]])
        self.buf: list[str] = []
        self.skip = 0
        self.table: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def _flush(self):
        text = " ".join("".join(self.buf).split())
        if text:
            self.items.append(("text", text))
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
        elif tag == "table":
            self._flush()
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag in BLOCK:
            if self.cell is not None:
                self.cell.append(" ")  # <br>/<div> inside a cell must not glue words together
            elif self.table is None:
                self._flush()

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            if any(self.row):
                self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            rows = clean_table(self.table)
            if rows:
                if is_prose_table(rows):
                    self.items.append(("text", " ".join(" ".join(r) for r in rows)))
                else:
                    self.items.append(("table", rows))
            self.table = None
        elif tag in BLOCK:
            if self.cell is not None:
                self.cell.append(" ")
            elif self.table is None:
                self._flush()

    def handle_data(self, data):
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.table is None:
            self.buf.append(data)


def parse_blocks(html: str) -> list[tuple[str, list[tuple[str, object]]]]:
    """Return [(section_title, [(kind, payload)])]. Text before the first 'Item' heading is 'Preamble'."""
    p = _Extract()
    p.feed(html)
    p._flush()
    sections: list[tuple[str, list]] = [("Preamble", [])]
    for kind, payload in p.items:
        if kind == "text" and ITEM.match(payload) and len(payload) < 120:  # a heading, not a sentence
            sections.append((payload.strip(), []))
        else:
            sections[-1][1].append((kind, payload))
    return [(t, blocks) for t, blocks in sections if blocks]


def parse_sections(html: str) -> list[tuple[str, str]]:
    """Flattened view: [(section_title, text)] with tables rendered as Markdown."""
    return [(title, "\n".join(payload if kind == "text" else to_markdown(payload) for kind, payload in blocks))
            for title, blocks in parse_blocks(html)]
