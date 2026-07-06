"""Integration smoke test for the FastAPI service layer (PBI 6, Task 6.1).

Exercises every route against the real backing data (Qdrant + data/axiom.db —
same corpus the Streamlit app uses), matching the acceptance criterion "smoke
test script passes against the docker-compose stack." Uses FastAPI's
TestClient (in-process, no server needed) rather than introducing pytest.

Usage (from repo root, with `docker compose up -d` + a populated index):
    python scripts/smoke_test_api.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from axiom import db
from api.main import app

client = TestClient(app)
failures = 0


def check(label: str, resp, expect_status: int = 200, expect_nonempty: bool = False):
    global failures
    ok = resp.status_code == expect_status
    body = None
    if ok and expect_nonempty:
        body = resp.json()
        ok = bool(body)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}  (status={resp.status_code})")
    if not ok:
        failures += 1
        print(f"       body: {resp.text[:300]}")
    if resp.status_code == expect_status and resp.content:
        return resp.json()
    return None


def main() -> None:
    global failures
    conn = db.connect()
    try:
        # 1. Count parity check (SQLite vs Qdrant)
        sqlite_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    finally:
        conn.close()
    
    try:
        from axiom.qdrant_client import AxiomQdrant
        qdrant_count = AxiomQdrant().count()
        ok = sqlite_count == qdrant_count
        print(f"[{'PASS' if ok else 'FAIL'}] index count parity (SQLite {sqlite_count} == Qdrant {qdrant_count})")
        if not ok: failures += 1
    except Exception as exc:
        print(f"[FAIL] index count parity check failed: {exc}")
        failures += 1

    conn = db.connect()
    try:
        sample_paper = conn.execute("SELECT openalex_id, publication_year, venue FROM papers LIMIT 1").fetchone()
        if sample_paper is None:
            raise SystemExit("No papers in data/axiom.db — run scripts/bootstrap_synthetic.py first.")
        paper_id = sample_paper["openalex_id"]
        paper_year = sample_paper["publication_year"]
        paper_venue = sample_paper["venue"]

        # 2. Add fake no-abstract paper for 422 testing
        fake_422_id = "test-no-abstract-123"
        db.insert_paper(conn, openalex_id=fake_422_id, title="Test No Abstract", abstract=None, publication_year=2024, venue_id=None, venue="TestVenue")
        conn.commit()
    finally:
        conn.close()

    check("GET /health", client.get("/health"), expect_nonempty=True)
    check("GET /search", client.get("/search", params={"q": "retrieval-augmented generation", "top_k": 5}),
          expect_nonempty=True)
    check("GET /papers/{id}", client.get(f"/papers/{paper_id}"), expect_nonempty=True)
    check("GET /papers/{id}/similar", client.get(f"/papers/{paper_id}/similar", params={"top_k": 5}))
    check("GET /trends/top", client.get("/trends/top", params={"n": 20}), expect_nonempty=True)
    check("GET /trends/top (insufficient spread)", client.get("/trends/top", params={"n": 20, "year_from": paper_year, "year_to": paper_year}), expect_nonempty=True)
    check("GET /graph/stats", client.get("/graph/stats"), expect_nonempty=True)
    check("GET /graph/influence", client.get("/graph/influence", params={"top_k": 10}))
    check("GET /graph/papers/{id}/neighbors", client.get(f"/graph/papers/{paper_id}/neighbors"))
    check("GET /graph/gaps", client.get("/graph/gaps"))
    check("GET /papers/{missing}/404", client.get("/papers/does-not-exist"), expect_status=404)
    check("POST /graph/gaps/{missing}/404", client.post("/graph/gaps/9999/hypothesize"), expect_status=404)

    check("POST /reading-list/{id}", client.post(f"/reading-list/{paper_id}"), expect_status=204)
    check("GET /reading-list (after add)", client.get("/reading-list"), expect_nonempty=True)
    check("POST /reading-list/{missing}/404",
          client.post("/reading-list/does-not-exist"), expect_status=404)
          
    check("POST /reading-list/{no_abstract_id}/summarize",
          client.post(f"/reading-list/{fake_422_id}/summarize"), expect_status=422)

    sum_resp = client.post(f"/reading-list/{paper_id}/summarize")
    if sum_resp.status_code == 503:
        print("[skip] POST /reading-list/{id}/summarize returned 503 (Ollama down) — skipping summarize/hypothesis checks")
    else:
        check("POST /reading-list/{id}/summarize",
              sum_resp, expect_nonempty=True)
        check("POST /reading-list/{id}/summarize (cached)",
              client.post(f"/reading-list/{paper_id}/summarize"), expect_nonempty=True)

    check("DELETE /reading-list/{id}",
          client.delete(f"/reading-list/{paper_id}"), expect_status=204)
    after = client.get("/reading-list").json()
    ok = paper_id not in {b["paper_id"] for b in after}
    print(f"[{'PASS' if ok else 'FAIL'}] reading-list no longer contains removed paper")
    if not ok:
        failures += 1

    gaps_resp = client.get("/graph/gaps")
    n_gaps = len(gaps_resp.json().get("gaps", []))
    if n_gaps:
        hyp_resp = client.post("/graph/gaps/0/hypothesize")
        if hyp_resp.status_code == 503:
            print("[skip] POST /graph/gaps/0/hypothesize returned 503 (Ollama down) — skipping hypothesis checks")
        else:
            hyp = check("POST /graph/gaps/0/hypothesize", hyp_resp, expect_nonempty=True)
            if hyp:
                ok = len(hyp["supporting_paper_ids"]) >= 2
                print(f"[{'PASS' if ok else 'FAIL'}] hypothesis has >=2 supporting_paper_ids")
                failures += 0 if ok else 1
                review_id = hyp["review_id"]
                check("GET /review-queue", client.get("/review-queue", params={"status": "pending"}),
                      expect_nonempty=True)
                
                # Test the reject path
                check("POST /review-queue/{id}/reject",
                      client.post(f"/review-queue/{review_id}/reject"), expect_status=204)
                
                # We need a new hypothesis to test approve since we just rejected it
                hyp2 = client.post("/graph/gaps/0/hypothesize").json()
                review_id_2 = hyp2["review_id"]

                check("POST /review-queue/{id}/approve",
                      client.post(f"/review-queue/{review_id_2}/approve"), expect_status=204)
                approved = client.get("/review-queue", params={"status": "approved"}).json()
                ok = any(r["id"] == review_id_2 for r in approved)
                print(f"[{'PASS' if ok else 'FAIL'}] review item shows approved after action")
                failures += 0 if ok else 1
    else:
        print("[skip] no gap candidates in this corpus — hypothesis/review-queue checks skipped")
        
    # Cleanup fake 422 paper
    conn = db.connect()
    try:
        conn.execute("DELETE FROM papers WHERE openalex_id = ?", (fake_422_id,))
        conn.commit()
    finally:
        conn.close()

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
