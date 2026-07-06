# LLMs for Sciences: Research Trends & Gaps Extraction Tool - Product Backlog

Axiom · full product backlog (P1-P4)

## Project Goal

**LLMs for Sciences: Research Trends & Gaps Extraction Tool** (Axiom) is a discovery platform for researchers and thesis seekers - not a paper search engine. It extracts two complementary signals from scholarly metadata: **research trends** (which concepts are accelerating via keyword velocity) and **research gaps** (under-explored conceptual regions validated by citation evidence and, for frontiers, an LLM hypothesis pitch grounded in real abstracts). Semantic search supports gap Step 1 geometry; the product centerpiece is trends + gaps extraction with mandatory HITL caveats.

## Critical Path Summary

Build greenfield from empty repo to full stack: **OpenAlex → SQLite → SPECTER2 → Qdrant → Neo4j → LangGraph → FastAPI → React**. Saleh owns the data spine and index scale-up to 5k papers. Yahor owns velocity, the 3-step gap pipeline (WHERE → WHY → WHAT), agent graph, API contracts, and calibration. Yakub owns the four-page discovery dashboard (Trending, Keyword Graph, Gap Evaluator, Reading List) via Streamlit early integration then React from Stitch. NetworkX bootstraps citation logic before Neo4j Docker migration. Gap Step 3 (LangGraph 5-node graph with Verifier) runs only for Fertile Frontier verdicts. End-state demo: mentor filters ACL 2024-2025, spots rising RAG velocity, explores a ≤50-node keyword graph, evaluates a synthetic hypothesis through all 3 gap steps with badges and `supporting_paper_ids`, bookmarks a frontier candidate to Reading List with cited summaries. PBI 8 closes with τ calibration, nDCG@10 eval, and a reproducible README runbook.

## Phase Overview

| Phase | Goal | Key deliverables | Lead |
|-------|------|------------------|------|
| P1 | Data foundation | OpenAlex ≥1k papers, SQLite, SPECTER2, Qdrant, Streamlit search drill-down | Saleh |
| P2 | Analytics core | Velocity engine, NetworkX + Neo4j gap Steps 1-2, LangGraph MVP, Streamlit pages 1-2 | Yahor + Saleh |
| P3 | UI + agents integration | FastAPI, React 4 pages, LangGraph Step 3 full, Neo4j production path | Yakub + Yahor |
| P4 | Eval & scale | 5k corpus (NeurIPS 2024), calibration notebook, nDCG + gap quality study, runbook | All |

## Execution Order Table

| Seq | ID | Task / milestone | Depends on | Owner | Priority |
|-----|-----|------------------|------------|-------|----------|
| 1 | T1.1 | Repo scaffold, docker-compose skeleton, contracts + mocks | - | Yahor | High |
| 2 | T1.2 | OpenAlex ingest + SQLite schema (≥1k ACL 2024-2025) | T1.1 | Saleh | High |
| 3 | T2.1 | SPECTER2 embedding pipeline | T1.2 | Saleh | High |
| 4 | T2.2 | Qdrant collection + `search()` API | T2.1 | Saleh | High |
| 5 | T1.3 | Streamlit dev shell + mock page routing | T1.1 | Yakub | High |
| 6 | T3.1 | Velocity engine + keyword storage | T1.2 | Yahor | High |
| 7 | T3.2 | LLM keyword canonicalization | T3.1 | Yahor | Medium |
| 8 | T4.1 | NetworkX citation graph builder | T1.2 | Yahor | High |
| 9 | T4.2 | Gap evaluate Steps 1-2 (Scenario A/B on NetworkX) | T2.2, T4.1 | Yahor | High |
| 10 | T4.3 | Neo4j Docker + migration from SQLite edges | T4.1 | Saleh | High |
| 11 | T4.4 | Gap Step 2 on Neo4j Cypher | T4.3, T4.2 | Saleh | High |
| 12 | T5.1 | LangGraph 5-node agent graph | T4.4, T3.1 | Yahor | High |
| 13 | T5.2 | Gap Step 3 + composite G scoring + HITL queue backend | T5.1 | Yahor | High |
| 14 | T6.1 | FastAPI service (search, trends, gap, reading list) | T2.2, T3.1, T5.2 | Yahor | High |
| 15 | T7.1 | Trending Topics page (Streamlit then React) | T3.1, T6.1 | Yakub | High |
| 16 | T7.2 | Keyword Graph page (≤50 nodes) | T3.1, T6.1 | Yakub | High |
| 17 | T7.3 | Gap Evaluator page (3 steps + badges) | T5.2, T6.1 | Yakub | High |
| 18 | T7.4 | Reading List page (bookmarks + cited summaries) | T5.1, T6.1 | Yakub | High |
| 19 | T2.3 | Corpus expansion to ≥5k (add NeurIPS 2024) | T2.2 | Saleh | High |
| 20 | T2.4 | PDF extract pipeline (Marker/PyMuPDF) for intro chunks | T2.3 | Saleh | Medium |
| 21 | T8.1 | Calibration notebook (τ, δ_D, δ_F) | T4.4, T5.2 | Yahor | High |
| 22 | T8.2 | nDCG@10 + gap quality evaluation | T2.3, T5.2 | All | High |
| 23 | T8.3 | README runbook + mentor demo package | T7.3, T8.1 | All | High |
| 24 | T1.4 | `docs/DECISIONS.md` (OD1-OD6 resolved) | T4.3 | Yahor | Medium |

---

## PBI 1 - Corpus & Data Foundation

### User Story:
As a researcher, I want a reliable scholarly corpus with citation metadata and provenance, so that all trend and gap extraction runs on trustworthy OpenAlex data.

### PBI Acceptance Criteria:
- SQLite schema stores papers, citation edges, keywords/concepts, and provenance per PROJECT_PLAN §6.1.
- Ingest reaches ≥1,000 ACL 2024-2025 papers with idempotent re-runs and exponential backoff on API errors.
- Citation edges from `referenced_works` are complete enough for graph construction (≥5k edges on 1k corpus).
- Ingest command and env vars documented in README.

### Task 1.1 - Repo Scaffold, Contracts & Dev Environment

- **Assignee:** Yahor
- **Sequence:** Y1
- **Priority:** High

### Description:
- Create project layout per backlog §3.3: `ingest/`, `db/`, `embeddings/`, `index/`, `graph/`, `trends/`, `gap/`, `agents/`, `api/`, `app/`, `contracts/`, `eval/`, `docs/`.
- Add `requirements.txt`, `.env.example`, `docker-compose.yml` skeleton (Qdrant + Neo4j placeholders), README with course project goal.
- Define typed contracts with mock stubs: `search()`, `get_top_velocity_keywords()`, `evaluate_gap()`, `generate_hypothesis()`, `summarize_paper()`.

### Acceptance Criteria:
- Clean clone: `pip install -r requirements.txt` succeeds.
- `contracts/` modules export documented signatures; mocks return deterministic JSON matching Pydantic models.
- `docker compose up` starts Qdrant and Neo4j containers (Neo4j may be empty until T4.3).
- README states: **LLMs for Sciences: Research Trends & Gaps Extraction Tool**.

### Task 1.2 - OpenAlex Ingest + SQLite Schema

- **Assignee:** Saleh
- **Sequence:** S1
- **Priority:** High

### Description:
- Implement `ingest/openalex.py` and `db/schema.sql`: relevance-based topic seeding, cursor pagination, abstract reconstruction, persistence to `papers`, `citation_edges`, `paper_provenance`, keyword/concept fields.
- Target ≥1,000 papers ACL 2024-2025 per PROJECT_PLAN OD2.

### Acceptance Criteria:
- `python -m ingest.openalex --venue ACL --years 2024,2025 --limit 1000` completes without error.
- SQLite: ≥1000 papers, ≥5000 citation edges, provenance with `fetched_at` and `text_hash`.
- Missing abstracts flagged `has_abstract=false`; re-run is idempotent.
- Ingest documented in README.

### Task 1.3 - Streamlit Dev Shell

- **Assignee:** Yakub
- **Sequence:** K1
- **Priority:** High

### Description:
- Create `app/streamlit_app.py` with sidebar for four pages (Trending, Keyword Graph, Gap Evaluator, Reading List) plus optional Search drill-down tab. Wire to contract mocks from T1.1.

### Acceptance Criteria:
- `streamlit run app/streamlit_app.py` starts; four pages render mock data matching contract shapes.
- Sidebar order: Trending → Keyword Graph → Gap Evaluator → Reading List.
- App title references course project name.

### Task 1.4 - Architecture Decisions Log

- **Assignee:** Yahor
- **Sequence:** Y2
- **Priority:** Medium

### Description:
- Write `docs/DECISIONS.md` recording resolved open decisions OD1-OD6 from PROJECT_PLAN §4.3: Streamlit→React path, corpus scope, SPECTER2, abstract-only then PDF intro, HITL mandatory, NetworkX→Neo4j migration.

### Acceptance Criteria:
- Dated entries for OD1-OD6 with decision, rationale, revisit trigger.
- OD5 documents badge sequence: Geometric Void → Citation Verified → Frontier Candidate.

---

## PBI 2 - Semantic Index & Search API

### User Story:
As a gap detection pipeline, I need a semantic index over paper abstracts, so that Step 1 can measure geometric novelty (`s_max`) and search supports optional drill-down.

### PBI Acceptance Criteria:
- SPECTER2 embeddings for all papers with abstracts; title-only fallback flagged.
- Qdrant `axiom_v1` populated with hybrid payload filters (venue, year).
- `search(query, top_k, filters)` returns ranked results in <2s p95 on 1k corpus, <2s p95 on 5k after scale-up.
- Corpus expandable to ≥5k without schema redesign.

### Task 2.1 - SPECTER2 Batch Embedding Pipeline

- **Assignee:** Saleh
- **Sequence:** S2
- **Priority:** High

### Description:
- Implement `embeddings/specter2.py`: batch-encode abstracts from SQLite using `allenai/specter2_base`, 768-dim output keyed by `openalex_id`.

### Acceptance Criteria:
- `python -m embeddings.specter2 --db data/axiom.db` embeds ≥95% of papers (remainder title-only, flagged).
- Spot-check: related RAG pair cosine >0.7; unrelated pair <0.5.
- Embeddings persist and reload without re-encoding unchanged papers (hash check).

### Task 2.2 - Qdrant Collection + `search()` API

- **Assignee:** Saleh
- **Sequence:** S3
- **Priority:** High

### Description:
- Implement `index/qdrant_client.py`: collection `axiom_v1`, cosine 768-dim, payload (`paper_id`, `year`, `venue`, `title`, `cited_by_count`, `concepts`), expose `search()`.

### Acceptance Criteria:
- One point per embedded paper; year/venue filters work.
- `search("RAG for low-resource NLP", top_k=10)` returns on-topic top results with scores.
- Matches `contracts/search.py` signature; importable without UI.

### Task 2.3 - Corpus Expansion to ≥5k Papers

- **Assignee:** Saleh
- **Sequence:** S4
- **Priority:** High

### Description:
- Extend ingest filter to add NeurIPS 2024 (one-parameter change per OD2). Re-embed and re-index Qdrant collection `axiom_v2` if schema changes.

### Acceptance Criteria:
- SQLite contains ≥5000 papers across ACL 2024-2025 + NeurIPS 2024.
- Qdrant index rebuilt; `search()` p95 <2s on 5k corpus.
- README documents corpus filter parameters.

### Task 2.4 - PDF Extract for Abstract+Intro Chunks (P3+)

- **Assignee:** Saleh
- **Sequence:** S5
- **Priority:** Medium

### Description:
- Implement `ingest/pdf_extract.py` using Marker or PyMuPDF on demand via DOI; store intro text hash in provenance; never redistribute PDFs.

### Acceptance Criteria:
- `python -m ingest.pdf_extract --paper-id <id>` extracts intro markdown for a paper with resolvable DOI.
- `has_fulltext=true` set in provenance; PDF deleted after extraction.
- Failures logged; paper remains abstract-only without blocking pipeline.

---

## PBI 3 - Research Trends Extraction

### User Story:
As a thesis-seeking researcher, I want to see which concepts are **accelerating** in my subfield, so that I understand what the field is doing without reading thousands of papers.

### PBI Acceptance Criteria:
- Velocity $v_k(t)$ computed with normalized frequency and log-ratio per PROJECT_PLAN §7.2.
- LLM keyword canonicalization merges synonyms (e.g. LoRA / Low-Rank Adaptation).
- `get_top_velocity_keywords(n, filters)` returns ≥50 keywords in <2s with confidence warning when $f_k < 5$.
- Trending data feeds Keyword Graph (≤50 nodes) without hairball overload.

### Task 3.1 - Velocity Engine

- **Assignee:** Yahor
- **Sequence:** Y3
- **Priority:** High

### Description:
- Implement `trends/velocity.py`: compute $v_k(t)$ across conference-year windows from SQLite keywords/concepts; persist to `KEYWORD` table; expose `get_top_velocity_keywords(n, venue, year_range)`.

### Acceptance Criteria:
- `get_top_velocity_keywords(50)` returns ranked keywords with velocity and frequency for ACL 2024 vs 2025.
- Low-volume terms flagged with confidence warning in return payload.
- Computation <2s on 1k corpus, <5s on 5k.

### Task 3.2 - LLM Keyword Canonicalization

- **Assignee:** Yahor
- **Sequence:** Y4
- **Priority:** Medium

### Description:
- Implement `trends/canonicalize.py`: batch 50 OpenAlex concept labels → LLM merge synonyms → persist canonical forms in `KEYWORD` table.

### Acceptance Criteria:
- Known synonym pairs (LoRA / Low-Rank Adaptation) map to single canonical entry.
- Canonicalization is idempotent; manual override table supported.
- No freestanding LLM claims - only label merging with logged inputs.

---

## PBI 4 - Citation Graph & Gap Steps 1-2

### User Story:
As a researcher testing a thesis idea, I want to know whether my hypothesis sits in a geometric void and whether citation history suggests a dead end or fertile frontier.

### PBI Acceptance Criteria:
- NetworkX digraph built from SQLite; Scenario A (Dead End) and Scenario B (Fertile Frontier) implemented.
- `evaluate_gap(h_syn)` returns Step 1 `s_max` and Step 2 verdict with evidence paper IDs.
- Neo4j Docker replaces NetworkX for Step 2 Cypher queries without changing API contract.
- Results never claim "proven gap"; `disclaimer: Unverified Candidate` always present.

### Task 4.1 - NetworkX Citation Graph Builder

- **Assignee:** Yahor
- **Sequence:** Y5
- **Priority:** High

### Description:
- Implement `graph/citation_graph.py`: load edges from SQLite, build `networkx.DiGraph`, expose neighbor queries and border-paper helpers for gap Step 2.

### Acceptance Criteria:
- Graph builds ≥1000 nodes / ≥5000 edges on 1k corpus in <60s.
- `get_neighbors(paper_id)` correct for a known hub paper.
- Rebuildable from SQLite on demand.

### Task 4.2 - Gap Steps 1-2 on NetworkX

- **Assignee:** Yahor
- **Sequence:** Y6
- **Priority:** High

### Description:
- Implement `gap/evaluate.py` Steps 1-2: Step 1 `s_max` via Qdrant; Step 2 Scenario A/B as NetworkX path queries. Return `GapResult` with verdict enum, evidence IDs, `s_max`, τ note.

### Acceptance Criteria:
- `evaluate_gap("multimodal RAG for clinical triage")` returns `s_max`, verdict, evidence papers in <10s on 1k corpus.
- Constructed test cases: Dead End pattern ≠ Frontier pattern.
- τ=0.65 used with "uncalibrated" flag in output.

### Task 4.3 - Neo4j Docker + Data Load

- **Assignee:** Saleh
- **Sequence:** S6
- **Priority:** High

### Description:
- Add Neo4j to `docker-compose.yml`; implement `graph/neo4j_loader.py` to load Paper nodes and CITES edges from SQLite per PROJECT_PLAN §6.2.

### Acceptance Criteria:
- `docker compose up neo4j` healthy; loader populates ≥1000 nodes on 1k corpus.
- Cypher `MATCH (p:Paper)-[:CITES]->(q:Paper) RETURN count(*)` returns expected edge count ±1%.
- Loader is idempotent (safe re-run).

### Task 4.4 - Gap Step 2 Neo4j Cypher

- **Assignee:** Saleh
- **Sequence:** S7
- **Priority:** High

### Description:
- Port Scenario A and Scenario B to Cypher templates from PROJECT_PLAN §7.3; wire into `evaluate_gap()` behind same API contract; deprecate NetworkX path when Neo4j available.

### Acceptance Criteria:
- Scenario A and B Cypher queries return results matching NetworkX on ≥10 hand-checked cases.
- `evaluate_gap()` auto-selects Neo4j when container healthy; falls back to NetworkX with degraded badge.
- End-to-end Step 1+2 <10s on 1k corpus.

---

## PBI 5 - LangGraph Agents & Gap Step 3

### User Story:
As a researcher with a citation-verified frontier, I want a grounded hypothesis pitch with supporting papers, so that I can evaluate a concrete thesis direction - not a hallucinated trend.

### PBI Acceptance Criteria:
- LangGraph 5-node graph: Ingest_Summarizer, Trend_Aggregator, Gap_Narrator, Hypothesis_Generator, Verifier.
- Step 3 triggers only when Step 2 verdict = fertile_frontier.
- Verifier requires ≥2 `supporting_paper_ids`; empty lists retry ≤3×; temperature ≤0.3.
- Composite gap score $G$ computed and returned for ranked frontier candidates.
- HITL review queue backend stores candidates for human approval.

### Task 5.1 - LangGraph 5-Node Agent Graph

- **Assignee:** Yahor
- **Sequence:** Y7
- **Priority:** High

### Description:
- Implement `agents/graph.py` and `agents/nodes/`: stateful LangGraph with nodes per PROJECT_PLAN §7.3; structured Pydantic outputs at each boundary; max 3 Verifier retries.

### Acceptance Criteria:
- Graph runs end-to-end for a frontier `h_syn` on demo corpus.
- `Hypothesis_Generator` returns JSON: title, claim, method_sketch, datasets, `supporting_paper_ids`.
- Verifier rejects outputs with <2 supporting IDs and retries.
- `summarize_paper()` produces 3-bullet abstract summary with `paper_id` cites.

### Task 5.2 - Gap Step 3, G Scoring & HITL Queue

- **Assignee:** Yahor
- **Sequence:** Y8
- **Priority:** High

### Description:
- Wire Step 3 into `evaluate_gap()` / `generate_hypothesis()`; implement `gap/scoring.py` for composite $G$; add SQLite `review_queue` table for HITL candidates with status (pending / approved / rejected).

### Acceptance Criteria:
- Full `evaluate_gap()` returns Steps 1-3 when verdict=frontier; Steps 1-2 only when dead_end.
- $G$ score returned with weight breakdown per PROJECT_PLAN §7.3 defaults.
- Review queue API: list pending, approve, reject - no candidate promoted without human action.
- Full 3-step pipeline <10s on 1k corpus (LLM latency excluded from hard fail; log p95).

---

## PBI 6 - FastAPI Service Layer

### User Story:
As a frontend developer, I want a stable REST API over all backend modules, so that React pages consume one contract instead of importing Python modules directly.

### PBI Acceptance Criteria:
- FastAPI app exposes search, trends, gap evaluation, reading list, and review queue endpoints.
- OpenAPI docs generated at `/docs`.
- CORS configured for local React dev; env-driven config; no secrets in repo.

### Task 6.1 - FastAPI Core Routes

- **Assignee:** Yahor
- **Sequence:** Y9
- **Priority:** High

### Description:
- Implement `api/main.py`: REST wrappers for `search`, `get_top_velocity_keywords`, `evaluate_gap`, `generate_hypothesis`, `summarize_paper`, reading list CRUD, review queue CRUD. Pydantic request/response models mirror `contracts/`.

### Acceptance Criteria:
- `uvicorn api.main:app` starts; `/docs` lists all endpoints.
- `POST /gap/evaluate` returns full 3-step result for frontier demo example from `docs/demo_examples.md`.
- `GET /trends/top?n=50` returns velocity table in <2s.
- `POST /reading-list/{paper_id}/summarize` returns 3 bullets each citing `paper_id`.
- Integration smoke test script passes against docker-compose stack.

---

## PBI 7 - Discovery Dashboard (4 Pages)

### User Story:
As a mentor or researcher, I want a polished four-page dashboard to explore trends and gaps with honest limitation badges, so that the LLMs for Sciences vision is tangible in one click-through.

### PBI Acceptance Criteria:
- All four PROJECT_PLAN §8 pages implemented in React (Streamlit used for early dev only).
- **Trending Topics**: velocity list, venue/year filters, sparklines.
- **Keyword Graph**: force-directed, ≤50 nodes, size=citation weight, color=velocity, renders <3s.
- **Gap Evaluator**: 3 steps, badges, mini citation diagram, $G$ breakdown, HITL queue UI, Unverified Candidate banner.
- **Reading List**: bookmarks, 3-bullet cited summaries.
- No 3D/WebGL; no cluster hairball as hero UX.

### Task 7.1 - Trending Topics Page

- **Assignee:** Yakub
- **Sequence:** K2
- **Priority:** High

### Description:
- Build Trending page in Streamlit first, migrate to React (Stitch export): velocity-ranked table, venue/year filters, sparklines, low-volume warnings.

### Acceptance Criteria:
- Top-50 keywords with velocity load <2s via FastAPI.
- Filters reduce results correctly.
- React page matches Stitch wireframe at functional level.

### Task 7.2 - Keyword Graph Page

- **Assignee:** Yakub
- **Sequence:** K3
- **Priority:** High

### Description:
- Force-directed graph ≤50 nodes from top velocity + co-occurrence; node size=citation weight, color=velocity gradient. Prevent hairball (cap nodes, filter by min citations).

### Acceptance Criteria:
- Renders <3s on 5k-backed API response.
- ≤50 nodes visible; no unreadable overlap at default zoom.
- Click node shows keyword detail panel with velocity trend.

### Task 7.3 - Gap Evaluator Page

- **Assignee:** Yakub
- **Sequence:** K4
- **Priority:** High

### Description:
- Full 3-step UI: `h_syn` input, Step 1 void map + `s_max`, Step 2 badge (Dead End / Fertile Frontier) + citation evidence, Step 3 hypothesis pitch + $G$ breakdown. HITL review queue panel. Non-suppressable Unverified Candidate banner.

### Acceptance Criteria:
- Demo Frontier and Dead End examples from `docs/demo_examples.md` render correct badges.
- Step 3 shows `supporting_paper_ids` as clickable paper links.
- User can approve/reject frontier candidates in review queue.
- Copy: "candidate directions, not validated gaps" visible on every evaluation.

### Task 7.4 - Reading List Page

- **Assignee:** Yakub
- **Sequence:** K5
- **Priority:** High

### Description:
- Bookmark papers from search or gap results; display 3-bullet LLM summaries via FastAPI; all bullets must show `paper_id` citation.

### Acceptance Criteria:
- Add/remove bookmarks persists across sessions (local storage or backend).
- Every summary bullet links to source paper; empty unsupported claims rejected by API surface as error.
- Page loads in <3s for 20 bookmarked papers.

---

## PBI 8 - Evaluation, Calibration & Scale

### User Story:
As a team presenting to mentors, we want measured quality metrics and a reproducible runbook, so that trends and gaps claims are backed by data - not vibes.

### PBI Acceptance Criteria:
- τ, $\delta_D$, $\delta_F$ calibrated per PROJECT_PLAN §7.3 protocol (20 established + 10 dead + 10 hot topics).
- nDCG@10 ≥0.65 on 20 hand-labeled queries.
- Gap quality mean ≥3.0 on rated candidates; Dead End / Frontier accuracy ≥70% on 20 labeled voids.
- README runbook: clone → compose up → ingest → index → demo in documented steps.
- Total embedding + LLM cost for 5k papers documented as <$5.

### Task 8.1 - Threshold Calibration Notebook

- **Assignee:** Yahor
- **Sequence:** Y10
- **Priority:** High

### Description:
- Implement `eval/calibration.ipynb`: labeled mini-set, precision/recall curves for τ, $\delta_D$, $\delta_F$; export chosen thresholds to config file consumed by `gap/evaluate.py`.

### Acceptance Criteria:
- Notebook runs end-to-end on 5k corpus with committed labeled set in `eval/labels/`.
- Selected τ documented with precision/recall at operating point.
- `evaluate_gap()` reads calibrated thresholds from config (overrides dev default 0.65).

### Task 8.2 - Retrieval & Gap Quality Evaluation

- **Assignee:** All
- **Sequence:** A1
- **Priority:** High

### Description:
- Create `eval/ndcg_queries.json` (20 queries, 2 annotators); run nDCG@10 on search results. Rate 10 gap candidates 1-5 (team + volunteers). Measure Dead End / Frontier accuracy on 20 labeled voids.

### Acceptance Criteria:
- nDCG@10 ≥0.65 reported in `eval/report.md`.
- Gap quality mean ≥3.0; classification accuracy ≥70% on labeled void set.
- Search p95 <2s and gap eval p95 <10s on 5k corpus logged.

### Task 8.3 - Runbook, Demo Package & Final Report

- **Assignee:** All
- **Sequence:** A2
- **Priority:** High

### Description:
- README runbook (clone → install → compose → ingest → embed → index → migrate neo4j → start API → start React). `docs/demo_examples.md` with 2 gap examples + trending highlights. Mentor slides: **LLMs for Sciences: Research Trends & Gaps Extraction Tool** with architecture diagram and limitations.

### Acceptance Criteria:
- Fresh machine can follow README to running demo without undocumented steps.
- `docs/demo_examples.md` reproducible on demo DB.
- Slides include screenshots of all 4 pages with HITL badges visible.
- Limitations section: corpus scope, calibrated vs uncalibrated periods, hallucination controls, English-only.

---

## Out of Scope

- 3D WebGL / Three.js visualization
- 2D topic-cluster hairball as primary UX (rejected per mentor feedback)
- Reuse or migration of prior POC codebases
- LLM model training or fine-tuning
- Production SLA, multi-language support, social features

---

## Known Risks & Mitigations

- **OpenAlex abstracts missing on ~15-20% of papers.** Title-only embed with flag; exclude from gap scoring; show "limited text" in UI.
- **Neo4j is new ops for the team.** Start NetworkX; migrate when Scenario A/B are correct in Python; keep fallback path in `evaluate_gap()`.
- **LangGraph Step 3 can sprawl.** Hard cap 5 nodes, 3 Verifier retries, Pydantic at every boundary; no extra agent nodes without DECISIONS entry.
- **τ wrong until PBI 8 lands.** UI shows "calibrated" only after T8.1; until then label thresholds as dev defaults.
- **Hallucinated hypothesis pitches.** Verifier enforces `supporting_paper_ids`; empty → retry; HITL queue blocks auto-promotion.
- **React/Stitch migration friction.** Streamlit hosts working demo until each React page replaces it; API (PBI 6) is the stable seam.
- **5k embed + index runtime.** Batch embeddings with hash skip; versioned Qdrant collections (`axiom_v1` → `axiom_v2`); document GPU path in README.
- **Saleh and Yakub blocked on interface changes.** Yahor owns `contracts/`; signature changes discussed in standup before merge, not mid-sprint via silent PR.

---

## Cross-Task Dependency Hints

- T1.1 (scaffold + contracts) blocks all other tasks - mocks unblock parallel UI and ingest work.
- T1.2 (ingest) blocks T2.1, T3.1, T4.1 - no embeddings, trends, or graph without papers.
- T2.1 → T2.2 (embed before Qdrant) → T2.3 (scale after base index works).
- T3.1 (velocity) blocks T7.1 and T7.2; T3.2 (canonicalization) improves T7.1 quality but is not blocking.
- T4.1 (NetworkX) → T4.2 (gap Steps 1-2) before T4.3 (Neo4j load) → T4.4 (Cypher Step 2).
- T4.4 + T3.1 block T5.1 (LangGraph needs gap verdict + trend context).
- T5.1 blocks T5.2 and T7.4 (summaries); T5.2 blocks T7.3 (full gap UI).
- T6.1 (FastAPI) should land before React pages T7.1-T7.4; Streamlit can call Python modules directly until then.
- T2.3 (5k corpus) should complete before T8.1/T8.2 eval; T8.1 calibration before final threshold claims in T8.3.
- T7.3 (Gap Evaluator) and T8.2 (eval) should precede T8.3 (demo package) so screenshots and metrics are real.
