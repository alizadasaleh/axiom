# Architecture Decisions

A running log of committed team decisions. **Any change to `db/schema.sql`
(the shared data contract) MUST be recorded here before merge.**

---

## OD6 — Graph store is NetworkX-first

NetworkX is the graph backend for P1/P2. **Do not introduce Neo4j now.** The
SQLite `citation_edges` table is the durable source of truth for edges (built
from OpenAlex `referenced_works[]`); the graph track loads it into NetworkX in
memory. Revisit a dedicated graph DB only if NetworkX becomes a bottleneck.

## Vector collection name = `axiom_v1`

The Qdrant collection is **versioned**: `axiom_v1`. Bump the suffix whenever the
embedding model or vector dimensionality changes so old and new points never mix.
One point per paper (abstract-level). Payload: `paper_id, title, year, venue,
cited_by_count, concepts[]`.

## Embedding model = `allenai/specter2_base` (768-dim)

SPECTER2 base, 768 dimensions. CPU fallback is allowed with a warning; we never
silently switch to a different model. P1 uses the base encoder with CLS pooling
over `title [SEP] abstract`. (P2 may add the SPECTER2 proximity adapter.)

## Data layer = OpenAlex → SQLite

OpenAlex is the source of truth. Abstracts arrive as an inverted index and must
be reconstructed to plain text before insertion into `papers.abstract`.

## `search()` signature (vector-store contract)

`AxiomQdrant.search(query_vector, top_k, venues=None, year_range=None)` returns
`list[SearchHit]`. Takes a **raw 768-dim vector** (the caller encodes), keeping
the vector store decoupled from the encoder. Facets AND together; `venues`
matches OR within the facet. Filters are applied natively by Qdrant. This
signature is frozen; new capability is added as new methods (e.g.
`search_hybrid`), never by changing it.

## Query-side embedding path = expansion (adapter blocked by pins)

Terse queries embed poorly when treated as documents. The preferred fix is the
SPECTER2 **ad-hoc query adapter**, but the `adapters` library pins
`transformers~=4.57`, conflicting with our committed `transformers==4.40.2`. We
do **not** bump pins, so the active path is **query expansion**: `encode_query`
wraps the query in `config.QUERY_EXPANSION_TEMPLATE` so it lands nearer the
document region. `Specter2Encoder` attempts the adapter import and logs the
active path (`[embed] query path: expansion`). Document-side encoding is
unchanged, so existing corpus points stay valid. Revisit the adapter if/when a
transformers pin bump is approved.

## Hybrid retrieval = dense + sparse, RRF-fused in Python (anticipated by D4/OD4)

Each point now stores **named vectors**: a dense SPECTER2 vector (`dense`) plus a
sparse BM25-style vector (`sparse`, pure-Python hashing-trick encoder in
`axiom/sparse.py`). `search_hybrid` runs both arms and fuses with Reciprocal
Rank Fusion. The pinned `qdrant-client==1.9.0` supports sparse vectors but **not**
the Query-API fusion primitives (`Prefetch`/`FusionQuery`, client ≥1.10), so
fusion is done in Python — no pin bump. The frozen `search` (pure dense) is
retained; the Streamlit page prefers `search_hybrid` when the collection has a
sparse vector. This is the documented win for exact terms/acronyms: query
`LoRA` now returns the *LoRA* paper at rank 1 (absent from dense top-5);
`RAG` returns the RAG paper at rank 1.

## OD7 — OpenAlex corpus is built by snowball / 2-hop expansion

Naively fetching N papers on a topic yields a graph that is almost all dangling
external edges (each paper's `referenced_works[]` mostly points outside the
fetched set), so the in-corpus citation graph is empty and every graph metric is
meaningless. Ingestion (`axiom/ingest.py`) therefore **snowballs**: seed on a
topic via the `title_and_abstract.search` filter (NOT `cited_by_count:desc`,
which returns globally-famous off-topic papers like SciPy), then expand by
pulling the papers the seeds cite *and* the papers that cite the seeds, ranking
cited candidates by co-citation frequency, until ~`CORPUS_TARGET` papers. This
densifies internal edges. OpenAlex needs no API key — only a `mailto` for the
polite pool. Abstracts are reconstructed from the inverted index before insert.
A first real run (`retrieval-augmented generation`, target 500) produced 500
papers / 1240 in-corpus edges (avg 2.5/paper), connected enough for PageRank.

## OD8 — Citation-graph influence = PageRank on the corpus-only subgraph

The graph (`axiom/graph.py`) loads `citation_edges` into a `networkx.DiGraph`
(edge `src->dst` = "src cites dst"), keeping dangling externals as nodes flagged
`in_corpus=False`. **Influence ranking runs PageRank on the corpus-only
subgraph** so metadata-less external sinks can't absorb rank; `local_in_degree`
(citations from within the corpus) is shown alongside as a transparent companion.
PageRank is a small **pure-Python power iteration** — we do NOT add `scipy` just
for `nx.pagerank` (keeps deps pinned/minimal; numpy/networkx only). This is the
NetworkX-first realisation of OD6; no Neo4j.

## OD9 — Research gaps = semantically-close, weakly-citing community pairs

The product's key value is surfacing **research gaps**, not just influence.
`axiom/gaps.py` detects them by combining the two signals we already have:
**structure** (citation graph) and **meaning** (SPECTER2 embeddings). Pipeline:
(1) Louvain community detection on the undirected corpus citation graph →
sub-topics; (2) label each community by its most *distinctive* OpenAlex concepts
(TF-IDF across communities, dropping near-universal terms like "Computer
science"); (3) an embedding centroid per community; (4) score community **pairs**
— `gap_score = centroid_cosine / (1 + inter_community_citations)`. High score =
related in meaning yet disconnected by citation = a bridge nobody has built.
NetworkX + numpy only (no scipy/torch). PageRank/influence is retained as a
companion view. Known limitation: on a sparse single-topic corpus, centroid
cosines cluster tightly (~0.95) and most cluster pairs have 0 inter-citations, so
discrimination is limited — denser/multi-topic corpora sharpen the signal.

## OD10 — Velocity computed on-demand from concepts+papers; no persisted KEYWORD table

PBI 3 / Task 3.1 asks to "persist to a `KEYWORD` table" and compute `v_k(t)`
"per PROJECT_PLAN §7.2" — **no `PROJECT_PLAN.md` file exists anywhere in this
repo** (checked 2026-07-04), so neither that table nor that formula is defined
in-tree. `axiom/velocity.py` follows the on-demand pattern OD9 already
established for community detection instead of adding a new persisted derived
table: (1) split the (optionally venue/year-filtered) corpus's available
publication years into two contiguous windows, PRIOR / RECENT, split at the
year midpoint — generalizes the backlog's "ACL 2024 vs 2025" example to a
corpus that isn't venue-year-cadenced (OD7 topic-snowball); (2) per OpenAlex
concept (`level >= 1`, dropping overly broad concepts like "Computer science",
same rationale OD9 used for community labels), compute its normalized
frequency (share of papers) in each window; (3) `velocity =
log2((recent_share + eps) / (prior_share + eps))`, epsilon-smoothed so a
brand-new or vanished concept doesn't blow up to ±inf. `recent_count < 5` is
flagged `low_confidence` (matches the backlog's `f_k < 5` acceptance
criterion) — included, not dropped. `get_top_velocity_keywords(n, venue,
year_range)` matches the backlog's function name/signature. Wired into
Streamlit as a third "📈 Trending" tab, calling `axiom.velocity` directly
(bypasses FastAPI/PBI6, same as the Search and Citation-graph tabs).

Task 3.2 (LLM keyword canonicalization — merging synonyms like LoRA / Low-Rank
Adaptation) is **deferred**: it requires introducing an LLM API call, which is
a cost/infra decision this project hasn't made yet (same reasoning PBI 5 was
deferred for).

## OD11 — Separate eval corpus (real ACL/EMNLP/COLING/NAACL) for nDCG@10, not merged into axiom_v1

A teammate's `master` branch (merged into `main`, not this branch) carries
`FULL_DATASET.jsonl`: 23,002 real papers — ACL (8,784) / EMNLP (7,124) / COLING
(3,677) / NAACL (3,417), years 2020–2025, 100% abstract coverage, plus
`gs_citation` (a Google Scholar citation *count*, not `referenced_works`) and
no concept/keyword tags. Pulled into `data/FULL_DATASET.jsonl` (gitignored,
same as `data/axiom.db`) for **Task 8.2's nDCG@10 retrieval eval only**.

Kept in a **separate** SQLite db (`data/eval.db`) + Qdrant collection
(`axiom_eval_v1`), never merged into `data/axiom.db` / `axiom_v1`: without
citation edges it can't feed OD9 gap detection or PageRank, and without
concepts it can't feed the OD10 velocity engine — it only has what retrieval
eval needs (title, abstract, venue, year, citation count).
`scripts/ingest_eval_corpus.py` stratified-samples ~1,500 papers evenly across
(venue, year) so no single slice dominates, reusing `db.insert_paper()` /
`reindex_qdrant()` unchanged (the latter gained an optional `collection=`
param to target `axiom_eval_v1` instead of `axiom_v1`).

`eval/ndcg_queries.json` (15 queries, 86 judgments) is a **single-pass,
AI-drafted** label set — built by reading real sampled titles (not guessed
blind) and grading relevance 0–3, but **not** the backlog's "2 annotators"
acceptance criterion. `scripts/eval_ndcg.py` computes real nDCG@10 against
live search results: **hybrid mean nDCG@10 = 0.453**, dense-only = 0.345
(below the backlog's ≥0.65 target). Two queries scored 0.0 not because
retrieval failed — inspection showed genuinely relevant hits the judgment set
simply didn't cover (built via keyword-pattern search over the sample, not an
exhaustive pass). Treat these numbers as a working demo signal, not a
validated quality claim, until a second human annotator reviews
`eval/ndcg_queries.json`.

## OD12 — FastAPI scoped to what's built; hypothesis/reading-list/review-queue endpoints deferred

Task 6.1 originally specified REST wrappers for `search`, `get_top_velocity_keywords`,
`evaluate_gap`, `generate_hypothesis`, `summarize_paper`, reading-list CRUD, and
review-queue CRUD. Only the first two categories — plus paper lookup and the
OD9 citation graph/gaps — actually exist in this codebase, so `api/main.py`
wraps exactly those: `/health`, `/search`, `/papers/{id}`,
`/papers/{id}/similar`, `/trends/top`, `/graph/stats`, `/graph/influence`,
`/graph/papers/{id}/neighbors`, `/graph/gaps`. `/gap/evaluate` (3-step
hypothesis), `/reading-list/*` summaries, and a review-queue endpoint are
**not built** — they depend on the LLM hypothesis pipeline (PBI 5, deferred —
needs an API-key/cost decision) and reading-list bookmark storage (PBI 7 /
T7.4, not built), neither of which exists to wrap.

Pydantic response models mirror the `axiom/*` dataclasses (`SearchHit`,
`PaperRank`, `KeywordVelocity`/`VelocityAnalysis`, `Community`/`GapCandidate`/
`GapAnalysis`, `Neighbors`) via `from_attributes=True` — this is the one place
Pydantic enters the codebase; the `axiom/*` modules keep their existing
dataclass convention (no pydantic internally, per the established repo
style) and the API layer is a thin JSON-serializable boundary on top.
Lazy singletons (`get_encoder`/`get_store`/`get_graph`/`get_gap_analysis`) use
`functools.lru_cache` — FastAPI's equivalent of `app/streamlit_app.py`'s
`st.cache_resource`/`st.cache_data` pattern, same idea, different framework
primitive. New pinned deps: `fastapi==0.111.0`, `uvicorn==0.29.0` (same
mid-2024 pin era as the rest of `requirements.txt`).

Verified for real (not just imported): `uvicorn api.main:app` starts, `/docs`
+ `/openapi.json` list all 9 routes, and `scripts/smoke_test_api.py` (a
FastAPI `TestClient` script, not a new pytest suite — matches the existing
`scripts/eval_*.py` convention) passes all 10 checks against the live
Qdrant + `data/axiom.db` backing data.

## OD13 — Reading-list bookmarks only; no LLM summaries

Task 7.4 ("Reading List page") bundles two separable things: bookmark
persistence and 3-bullet LLM-generated summaries. Only the first is buildable
without an LLM — `summarize_paper()` belongs to the deferred PBI 5 pipeline.
Split them: **schema change** — new `reading_list` table (`db/schema.sql`),
`paper_id` primary key + `added_at`, no `user_id` (single-user demo, no auth
system exists). `axiom/db.py` gained `add_bookmark`/`remove_bookmark`/
`list_bookmarks` (same "all SQL in `db.py`" convention as every other table).
Wired into both the Streamlit app (a fourth "📚 Reading list" tab, plus an
add/remove button on every Search result card — calling `axiom/db.py`
directly, same as the other tabs) and the FastAPI layer (OD12) as
`GET/POST/DELETE /reading-list[/{paper_id}]`. Verified live: the Streamlit app
runs all four tabs with zero exceptions (`AppTest`), and
`scripts/smoke_test_api.py` exercises add → list → duplicate-add-on-missing-
paper (404) → remove → confirms removal, all passing against the live stack.

Explicitly not built (at the time): cited summary bullets, since that needed
`summarize_paper()`. Resolved by OD14, below.

## OD14 — Local LLM via Ollama unblocks T3.2 and reading-list summaries

PBI 5, T3.2, and the reading-list summary half of T7.4 were all deferred for
the same reason: "needs an LLM API-key/cost decision." That decision is now:
**use a local Ollama model, not a hosted API** — no key, no per-call cost,
no data leaves the machine. `axiom/llm.py` is a thin `httpx` wrapper around
Ollama's REST API (`http://localhost:11434/api/chat`), default model
`qwen2.5:7b`, `format: "json"` for structured output with a defensive
brace-extraction fallback (small local models occasionally wrap valid JSON in
prose). No new pinned dependency — httpx is already in `requirements.txt`;
the `ollama` PyPI client is skipped on purpose, matching this repo's
"minimal deps" discipline.

Two things built on top of it, both verified against a live local Ollama
instance (`qwen2.5:7b`, models also available: `llama3.1:latest`,
`qwen2.5-coder:14b`):

- **T3.2 (`axiom/canonicalize.py`)**: batches distinct `concepts` labels
  (50/batch), asks the model to group true synonyms only (explicit
  instruction: never merge merely-related-but-distinct concepts), persists to
  a new `concept_canonical` table (`source='auto'` vs `'manual'` — manual
  rows are never overwritten by a re-run). `axiom/velocity.py`'s
  `compute_velocity()` now maps every concept through `db.canonical_map()`
  before accumulating counts, so a synonym pair merges into one trend line.
  Verified live on the demo corpus: correctly proposed **zero** merges,
  because this hand-authored 30-paper corpus has no true duplicate concept
  labels to begin with (confirmed by a separate ad-hoc test: "LoRA" +
  "Low-Rank Adaptation" *do* merge correctly when both are present in a
  batch) — a true negative, not a broken feature.
- **T7.4 summaries (`axiom/summarize.py`)**: `summarize_paper()` produces
  exactly 3 bullets grounded only in the given abstract text (system prompt
  forbids outside knowledge/invented numbers), cached in a new
  `paper_summaries` table (`paper_id`, `bullets_json`, `model`,
  `created_at`) so the LLM call runs once per paper, not on every page
  view. Wired into both the Streamlit Reading-list tab ("🧠 Summarize"
  button) and `POST /reading-list/{paper_id}/summarize` (FastAPI, `force=true`
  to bypass the cache). Verified live: a real summary of the RAG seed paper's
  abstract produced 3 bullets that paraphrase, not invent, its actual claims.

PBI 5's LangGraph hypothesis pipeline can now use the same `axiom/llm.py`
client — but it is **not built in this pass**: the backlog's own OD9 note
already flags that the hypothesis-evaluation flow needs a design pass first
(the current gap detector is corpus-level community-pairs, not OD5's original
per-hypothesis `s_max`/Scenario-A-B input), and that redesign is independent
of which LLM backend is used. Resolving the LLM question does not by itself
resolve that design gap.

## OD15 — React migration (PBI 7) deprioritized; Streamlit is the final UI

The original plan treated Streamlit as an early-dev shell only, to be replaced
by a polished React frontend (from Stitch wireframes) for the mentor-facing
demo, with FastAPI (OD12) built specifically as the seam React would consume.
**Decision: skip the migration.** Streamlit already delivers all four working
tabs (Search, Trending, Citation graph/gaps, Reading list) end-to-end, and the
Stitch wireframes the original plan was meant to implement were never part of
this repo — building React now would mean designing the UI blind rather than
following a spec, for a polish gain that isn't worth that risk given the
project's remaining priorities. FastAPI (OD12) stays built and verified (it
has value beyond React — a stable REST seam for any future consumer, and
already used by `scripts/smoke_test_api.py`), but nothing currently consumes
it as a frontend, and that's fine. Revisit only if a concrete need for a
non-Streamlit UI shows up.

## OD16 — PBI 5 scoped to narrating existing OD9 gap candidates, not per-hypothesis evaluation

The original Step 3 assumed a per-hypothesis pipeline: user types a free-text
`h_syn`, the system scores geometric novelty (`s_max`) and a citation-scenario
verdict (Dead End / Fertile Frontier), then — only for Fertile Frontier — a
5-node LangGraph (Ingest_Summarizer, Trend_Aggregator, Gap_Narrator,
Hypothesis_Generator, Verifier) pitches a grounded hypothesis. OD9 replaced
Steps 1-2 with a corpus-level community-pair detector that takes no
per-hypothesis input, so that pipeline has nothing to plug into — the
mismatch flagged back in OD9 and repeated in OD14.

**Decision:** skip the free-text hypothesis/novelty-scoring problem entirely
(it needs new, uncalibrated threshold logic — the same T8.1 gap that's
already unresolved) and scope Step 3 to what OD9 already provides: given one
of its ranked gap candidates, generate a grounded pitch citing real papers
from both sides. `axiom/hypothesis.py`'s `generate_hypothesis()` is a single
`axiom/llm.py` call (OD14) — no LangGraph dependency added, consistent with
this project's "minimal deps" discipline (same reasoning as OD10 skipping a
`KEYWORD` table). It folds in three of the original five nodes' intent
without separate agent objects: paper grounding replaces Ingest_Summarizer
(abstracts are already short enough to pass directly), the OD10 velocity
engine's top risers are passed as prompt context in place of a
Trend_Aggregator node, and Gap_Narrator + Hypothesis_Generator collapse into
one prompt since OD9 already supplies the "why this is a gap" framing
(`_render_gap_detail` in the UI).

**The Verifier is rule-based, not an LLM self-report** — a real difference
from the original design, and a stronger guarantee: it checks
`supporting_paper_ids` actually belong to the two communities (not just that
the model claims they do) and requires >=2 ids with at least one from each
side, retrying up to 3× at temperature 0.3 per the backlog's acceptance
criteria. Verified live: a real run against the top-ranked demo-corpus gap
(PEFT/LoRA cluster vs. low-resource-NLP cluster) produced a coherent,
genuinely grounded pitch on the first attempt, correctly citing one paper
from each side (`S0005` the LoRA paper, `S0010` a back-translation paper).

**No composite G-score.** The backlog wants "$G$ with a weight breakdown per
PROJECT_PLAN §7.3 defaults" — that file doesn't exist (established in OD10),
so there are no defaults to implement; inventing arbitrary weights would
manufacture false calibration-sounding precision. OD9's `gap_score` already
serves as the ranking signal upstream of hypothesis generation.

**HITL review queue** (Task 5.2's other half) is a real, minimal addition:
new `review_queue` table (`db/schema.sql`) — every generated pitch lands as
`pending`; only an explicit approve/reject action (Streamlit "🗂️ Review
queue" tab or `POST /review-queue/{id}/{approve,reject}`) changes status.
Nothing is ever auto-promoted, matching the backlog's non-negotiable
requirement. A non-suppressable "⚠️ Unverified Candidate" banner accompanies
every pitch in both the UI and the dataclass itself
(`HypothesisPitch.disclaimer`).

Verified live end-to-end: `scripts/smoke_test_api.py` (21 checks) generates a
real pitch against the live corpus, confirms >=2 supporting ids, and confirms
an approve action actually changes status; the Streamlit app renders all 5
tabs with zero exceptions (`AppTest`).

---

## OD17 — Composite gap G-score + τ calibration scaffold (supersedes OD16's "No composite G-score")

OD16 declined to build a composite G-score: with no `PROJECT_PLAN.md` §7.3 to
supply weights (established in OD10), inventing them would "manufacture false
calibration-sounding precision," so OD9's raw `gap_score = similarity /
(1 + inter_citations)` stayed the only ranking signal.

**What changed:** the missing half was never the formula — it was a way to make
the numbers empirical instead of arbitrary. OD17 adds that: a calibration
scaffold (`scripts/export_gap_labels.py` → human labels → `scripts/
calibrate_gap_thresholds.py` → `eval/calibration.json`) that turns a labeled
mini-set into a chosen threshold τ (and, optionally, weights) via a
precision/recall sweep. With that consumer in place, a composite score stops
being false precision and becomes a *calibratable* one.

**The G-score.** `axiom/gaps.GapCandidate` now carries `g_score`, its four
normalized `components`, and `meets_threshold`. All four signals already exist
on the OD9 candidates — no new data, no new deps:

```
G = w_sim·S + w_disc·D + w_vel·V + w_auth·A
  S = clamp(centroid cosine, 0, 1)                     relatedness
  D = 1 / (1 + inter_citations)                        disconnection (OD9's factor, now additive)
  V = 1 / (1 + 2^(−v̄))                                 trend heat: logistic of mean concept velocity (OD10); 0.5 when the corpus spans <2 years
  A = min-max over the run of mean log1p(cited_by)     authority/substance of the two clusters
defaults (UNCALIBRATED): w_sim=0.4, w_disc=0.3, w_vel=0.2, w_auth=0.1; τ=0.65
```

Authority uses each paper's global `cited_by_count` (already on the graph nodes)
rather than PageRank — zero extra passes; PageRank is the noted alternative if a
future calibration wants intra-corpus centrality instead. Candidates now rank by
`g_score`; with the defaults S+D dominate, so ordering stays close to OD9's.

**Consumption.** `axiom/config.load_calibration()` reads `eval/calibration.json`
when present (merged over the defaults, ignoring a malformed file) and
`gaps.analyze` calls it each run — so dropping in a calibrated file overrides
τ/weights with no code change, closing Task 8.1's acceptance criterion
("`evaluate_gap()` reads calibrated thresholds from config, overrides dev
default 0.65").

**Honest caveat / still deferred.** The weights and τ ship UNCALIBRATED and are
labeled so in code, the API (`components`/`meets_threshold` are transparent), the
Streamlit gap detail, and this doc. Producing the human labels
(`eval/labels/gap_labels.csv`) is out of scope — it needs real annotator input,
the same PBI 8 gap OD11/OD16 already flag. A "notebook" (`calibration.ipynb`)
was specified; this repo keeps eval tooling as scripts that emit markdown
reports (`scripts/eval_ndcg.py`), so the scaffold follows that discipline
instead — pure-stdlib, importable, unit-tested (`tests/test_calibrate.py`).

---

## Schema change log

| Date       | Change                                              | By |
|------------|-----------------------------------------------------|----|
| 2026-06-04 | Initial P1 schema: papers, concepts, citation_edges, paper_provenance + read-path indexes | UI track |
| 2026-07-04 | Added `reading_list` (paper_id, added_at) — bookmarks only, OD13 | UI track |
| 2026-07-04 | Added `concept_canonical` (concept, canonical, source) and `paper_summaries` (paper_id, bullets_json, model, created_at) — OD14 | UI track |
| 2026-07-04 | Added `review_queue` (id, gap_a_label, gap_b_label, title, claim, method_sketch, datasets_json, supporting_ids_json, status, created_at) — OD16 | UI track |
