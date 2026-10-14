"""Load fictional demo filings and golden questions: python3 scripts/seed.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filinglens import config, db, embeddings  # noqa: E402
from filinglens.ingestion.worker import ingest_document  # noqa: E402

FILINGS = [
    ({"ticker": "CTSO", "company": "Contoso Cloud", "fiscal_year": 2024, "url": "https://www.sec.gov/"},
     """<html><body>
     <p>Item 1A. Risk Factors</p>
     <p>We depend on a limited number of suppliers for GPUs and networking equipment. Three GPU
     suppliers account for most of our accelerator purchases, and a disruption in their supply chain
     could delay data center expansion. Our long-term supply agreements do not guarantee volume.</p>
     <p>Item 7. Management's Discussion</p>
     <p>Revenue grew 21 percent year over year, driven by demand for managed AI workloads.</p>
     <table><tr><th>Segment</th><th>Revenue ($M)</th></tr><tr><td>Compute</td><td>4,120</td></tr>
     <tr><td>Storage</td><td>1,310</td></tr></table>
     </body></html>"""),
    ({"ticker": "CTSO", "company": "Contoso Cloud", "fiscal_year": 2023, "url": "https://www.sec.gov/"},
     """<html><body><p>Item 7. Management's Discussion</p>
     <p>Operating margin improved to 18 percent as data center utilization increased.</p></body></html>"""),
]

# (id, question, type, expected_answer, relevant chunk ids)
GOLDEN = [
    ("q001", "What supply chain risks did Contoso Cloud report?", "factual",
     "Dependence on three GPU suppliers.", ["ctso-2024-0000"]),
    ("q002", "How did Contoso operating margin change in 2023?", "factual",
     "It improved to 18 percent.", ["ctso-2023-0000"]),
    ("q003", "Who is the CEO of Zebra Corp?", "unanswerable", None, []),
]

if __name__ == "__main__":
    settings = config.load()
    embeddings.configure(settings)
    conn = db.connect(settings.db_path)
    for meta, html in FILINGS:
        print(meta["ticker"], meta["fiscal_year"], ingest_document(conn, meta, html))
    for qid, question, qtype, answer, relevant in GOLDEN:
        conn.execute("INSERT OR REPLACE INTO golden_questions (id, question, type, expected_answer, "
                     "relevant_chunk_ids, status) VALUES (?,?,?,?,?, 'verified')",
                     (qid, question, qtype, answer, json.dumps(relevant)))
    conn.commit()
    print("chunks:", conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
