"""Golden dataset: LLM-generated candidates that a human must verify before they count."""
import json
import random
import sqlite3
import uuid

from .. import llm

TYPES = ("factual", "table", "multi-hop", "comparison", "unanswerable")

GEN_SYSTEM = ("You write evaluation questions for a retrieval system over SEC 10-K filings. "
              "Questions must be self-contained: name the company and fiscal year, and never say "
              "'the passage' or 'the excerpt'. Reply with one JSON object and nothing else.")

TASKS = {
    "factual": "Write one factual question whose answer is stated directly in the excerpt.",
    "table": "Write one question whose answer is a specific number or row value from the table in the excerpt.",
    "multi-hop": "Write one question that can only be answered by combining facts from BOTH excerpts.",
    "comparison": "Write one question that compares the two companies or the two fiscal years shown, "
                  "and needs facts from BOTH excerpts.",
    "unanswerable": "Write one question that sounds like it belongs to this company's 10-K and is on a "
                    "related topic, but whose answer is NOT given in the excerpt and would not normally "
                    "be disclosed in a 10-K (e.g. a specific internal figure, a future plan, a person's "
                    "private opinion). Set expected_answer to null.",
}


def _is_table(text: str) -> bool:
    return "| --- |" in text


def load_pool(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT c.id, c.text, c.section, c.token_count, d.id AS doc_id, d.ticker, d.company, d.fiscal_year "
        "FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.token_count >= 40").fetchall()


def sample_sources(pool: list, qtype: str, rng: random.Random) -> list | None:
    """Pick the chunk(s) a question of this type should be built from, or None if impossible."""
    prose = [c for c in pool if not _is_table(c["text"]) and c["section"] != "Preamble"]
    if qtype == "table":
        tables = [c for c in pool if _is_table(c["text"])]
        return [rng.choice(tables)] if tables else None
    if not prose:
        return None
    a = rng.choice(prose)
    if qtype in ("factual", "unanswerable"):
        return [a]
    if qtype == "multi-hop":
        same_doc = [c for c in prose if c["doc_id"] == a["doc_id"] and c["id"] != a["id"]]
        other_section = [c for c in same_doc if c["section"] != a["section"]]
        pick = other_section or same_doc
        return [a, rng.choice(pick)] if pick else None
    if qtype == "comparison":
        same_section = [c for c in prose if c["section"] == a["section"] and c["doc_id"] != a["doc_id"]]
        tiers = ([c for c in same_section if c["ticker"] != a["ticker"] and c["fiscal_year"] == a["fiscal_year"]],
                 [c for c in same_section if c["ticker"] == a["ticker"]],
                 [c for c in same_section if c["ticker"] != a["ticker"]])
        for tier in tiers:
            if tier:
                return [a, rng.choice(tier)]
        return None
    raise ValueError(f"unknown question type: {qtype}")


def _user_prompt(qtype: str, sources: list) -> str:
    blocks = [f'<excerpt company="{c["company"]}" fiscal_year="{c["fiscal_year"]}" section="{c["section"]}">\n'
              f'{c["text"]}\n</excerpt>' for c in sources]
    return ("\n\n".join(blocks) + f"\n\n{TASKS[qtype]}\n"
            'Reply as JSON: {"question": "...", "expected_answer": "..."}')


def generate_candidates(conn: sqlite3.Connection, settings, counts: dict[str, int], seed: int = 0) -> dict:
    """Insert `counts[type]` candidate questions. Returns {"created": n, "failed": n, "usage": {...}}."""
    rng = random.Random(seed)
    pool = load_pool(conn)
    existing = {r["question"].strip().lower() for r in conn.execute("SELECT question FROM golden_questions")}
    created = failed = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for qtype, want in counts.items():
        made, attempts = 0, 0
        while made < want and attempts < want * 6:
            attempts += 1
            sources = sample_sources(pool, qtype, rng)
            if not sources:
                break
            try:
                text, u = llm.complete(settings, GEN_SYSTEM, _user_prompt(qtype, sources),
                                       max_tokens=400, model=settings.judge_model)
            except ValueError:
                raise  # misconfigured provider: fail loudly rather than count as a bad sample
            for k in usage:
                usage[k] += u[k]
            try:
                obj = llm.extract_json(text)
                question = str(obj["question"]).strip()
                answer = obj.get("expected_answer")
                if not question or (qtype != "unanswerable" and not str(answer or "").strip()):
                    raise ValueError("empty question or answer")
            except (ValueError, KeyError):
                failed += 1
                continue
            if question.lower() in existing:
                continue
            existing.add(question.lower())
            relevant = [] if qtype == "unanswerable" else [c["id"] for c in sources]
            conn.execute("INSERT INTO golden_questions (id, question, type, expected_answer, "
                         "relevant_chunk_ids, status) VALUES (?,?,?,?,?, 'candidate')",
                         ("g_" + uuid.uuid4().hex[:8], question, qtype,
                          None if qtype == "unanswerable" else str(answer).strip(), json.dumps(relevant)))
            conn.commit()
            made += 1
            created += 1
    return {"created": created, "failed": failed, "usage": usage}


# --- review helpers (used by scripts/golden.py) ---------------------------------------

def set_status(conn: sqlite3.Connection, qid: str, status: str) -> None:
    conn.execute("UPDATE golden_questions SET status=? WHERE id=?", (status, qid))
    conn.commit()


def update_fields(conn: sqlite3.Connection, qid: str, *, question: str | None = None,
                  expected_answer: str | None = None, relevant_chunk_ids: list[str] | None = None) -> None:
    if question is not None:
        conn.execute("UPDATE golden_questions SET question=? WHERE id=?", (question, qid))
    if expected_answer is not None:
        conn.execute("UPDATE golden_questions SET expected_answer=? WHERE id=?", (expected_answer, qid))
    if relevant_chunk_ids is not None:
        known = {r["id"] for r in conn.execute("SELECT id FROM chunks")}
        unknown = [c for c in relevant_chunk_ids if c not in known]
        if unknown:
            raise ValueError(f"unknown chunk ids: {unknown}")
        conn.execute("UPDATE golden_questions SET relevant_chunk_ids=? WHERE id=?",
                     (json.dumps(relevant_chunk_ids), qid))
    conn.commit()


def stats(conn: sqlite3.Connection) -> dict:
    out: dict = {}
    for r in conn.execute("SELECT type, status, COUNT(*) n FROM golden_questions GROUP BY type, status"):
        out.setdefault(r["type"], {})[r["status"]] = r["n"]
    return out


def export_verified(conn: sqlite3.Connection, path) -> int:
    rows = conn.execute("SELECT id, question, type, expected_answer, relevant_chunk_ids FROM golden_questions "
                        "WHERE status='verified' ORDER BY id").fetchall()
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({**dict(r), "relevant_chunk_ids": json.loads(r["relevant_chunk_ids"])}) + "\n")
    return len(rows)


def import_verified(conn: sqlite3.Connection, path) -> int:
    n = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            conn.execute("INSERT OR REPLACE INTO golden_questions (id, question, type, expected_answer, "
                         "relevant_chunk_ids, status) VALUES (?,?,?,?,?, 'verified')",
                         (r["id"], r["question"], r["type"], r.get("expected_answer"),
                          json.dumps(r["relevant_chunk_ids"])))
            n += 1
    conn.commit()
    return n
