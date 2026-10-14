"""HTML 10-K -> sections of clean text, tables as Markdown. Uses html.parser from the stdlib."""
import re
from html.parser import HTMLParser

ITEM = re.compile(r"^\s*(item\s+\d+[a-c]?\.?)(.*)$", re.I)
SKIP = {"script", "style", "head", "ix:header"}  # ix:header = hidden inline-XBRL metadata
BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"}


class _Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.buf: list[str] = []
        self.skip = 0
        self.table: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def _flush(self):
        text = " ".join("".join(self.buf).split())
        if text:
            self.lines.append(text)
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
        elif tag in BLOCK and self.table is None:
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
            self.lines.append(to_markdown(self.table))
            self.table = None
        elif tag in BLOCK and self.table is None:
            self._flush()

    def handle_data(self, data):
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.table is None:
            self.buf.append(data)


def to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def parse_sections(html: str) -> list[tuple[str, str]]:
    """Return [(section_title, text)]. Text before the first 'Item' heading is 'Preamble'."""
    p = _Extract()
    p.feed(html)
    p._flush()
    sections: list[tuple[str, list[str]]] = [("Preamble", [])]
    for line in p.lines:
        m = ITEM.match(line)
        if m and len(line) < 120:  # a heading, not a sentence that starts with "Item"
            sections.append((line.strip(), []))
        else:
            sections[-1][1].append(line)
    return [(t, "\n".join(ls)) for t, ls in sections if "".join(ls).strip()]
