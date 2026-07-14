"""Axiom P1 search page.

Free-text query -> SPECTER2 encode -> Qdrant search (hybrid dense+sparse by
default, or dense-only via the toggle), with a venue/year filter bar. Results
render as expandable cards (title/DOI, concepts, abstract) and each card can
pivot to "similar papers". Functionality only; no styling polish.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

# Make the repo root importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from axiom import config, db, gaps, graph, hypothesis, llm, summarize, velocity
from axiom.embed import Specter2Encoder
from axiom.qdrant_client import AxiomQdrant, SearchHit

st.set_page_config(
    page_title="Axiom · Thesis Discovery",
    page_icon="🔭",
    layout="wide",
)


# --- Visual polish -----------------------------------------------------------
# Streamlit's defaults are functional but plain; the theme in
# .streamlit/config.toml sets the palette, and this stylesheet handles the
# things config can't reach — the hero, pill tabs, card-like expanders/metrics,
# chips, and spacing. Injected once per run.
_CSS = """
<style>
:root {
  --ax-accent: #5b8cff;
  --ax-accent-soft: rgba(91,140,255,0.14);
  --ax-panel: #151b2b;
  --ax-panel-2: #1b2334;
  --ax-border: rgba(146,164,205,0.18);
  --ax-border-strong: rgba(146,164,205,0.32);
  --ax-muted: #93a0b8;
}

/* Tighter, centered content column. */
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }

/* --- Hero ----------------------------------------------------------------- */
.ax-hero {
  display: flex; align-items: center; gap: 16px;
  padding: 22px 26px; margin-bottom: 6px;
  border: 1px solid var(--ax-border);
  border-radius: 16px;
  background:
    radial-gradient(120% 140% at 0% 0%, rgba(91,140,255,0.16) 0%, rgba(91,140,255,0) 55%),
    linear-gradient(180deg, #131a29 0%, #0e1420 100%);
}
.ax-hero .ax-mark {
  font-size: 34px; line-height: 1;
  width: 58px; height: 58px; flex: none;
  display: grid; place-items: center;
  border-radius: 14px;
  background: var(--ax-accent-soft);
  border: 1px solid var(--ax-border-strong);
}
.ax-hero h1 {
  font-size: 1.7rem; font-weight: 750; margin: 0;
  letter-spacing: -0.02em; color: #f2f5fb;
}
.ax-hero p { margin: 3px 0 0; color: var(--ax-muted); font-size: 0.92rem; }

/* Meta pills under the hero. */
.ax-meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 6px; }
.ax-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 11px; border-radius: 999px;
  font-size: 0.78rem; font-weight: 500;
  color: var(--ax-muted);
  background: var(--ax-panel); border: 1px solid var(--ax-border);
}
.ax-pill code { background: none; color: #cdd6e4; padding: 0; font-size: 0.78rem; }
.ax-pill.ok  { color: #7ee2a8; border-color: rgba(81,207,102,0.35); background: rgba(81,207,102,0.08); }
.ax-pill.off { color: #ffb4a0; border-color: rgba(255,135,135,0.35); background: rgba(255,135,135,0.08); }
.ax-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

/* --- Tabs: pill row ------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px; border-bottom: none; margin-bottom: 8px;
  background: var(--ax-panel); padding: 6px; border-radius: 12px;
  border: 1px solid var(--ax-border);
}
.stTabs [data-baseweb="tab"] {
  height: auto; padding: 8px 16px; border-radius: 8px;
  color: var(--ax-muted); font-weight: 550; font-size: 0.9rem;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(146,164,205,0.08); color: #d6deec; }
.stTabs [aria-selected="true"] {
  background: var(--ax-accent) !important; color: #0b0f19 !important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* --- Expanders as cards --------------------------------------------------- */
[data-testid="stExpander"] {
  border: 1px solid var(--ax-border) !important;
  border-radius: 12px !important;
  background: linear-gradient(180deg, var(--ax-panel) 0%, #121828 100%);
  margin-bottom: 10px; overflow: hidden;
}
[data-testid="stExpander"] summary { padding: 12px 16px; font-weight: 550; }
[data-testid="stExpander"] summary:hover { color: var(--ax-accent); }
[data-testid="stExpander"]:hover { border-color: var(--ax-border-strong) !important; }

/* --- Metrics as tiles ----------------------------------------------------- */
[data-testid="stMetric"] {
  background: var(--ax-panel); border: 1px solid var(--ax-border);
  border-radius: 12px; padding: 14px 16px;
}
[data-testid="stMetricLabel"] { color: var(--ax-muted); }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.01em; }

/* --- Bordered containers -------------------------------------------------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 14px; border-color: var(--ax-border) !important;
}

/* --- Buttons -------------------------------------------------------------- */
.stButton > button {
  border-radius: 9px; border: 1px solid var(--ax-border-strong);
  font-weight: 550; transition: transform .05s ease, border-color .15s ease, background .15s ease;
}
.stButton > button:hover { border-color: var(--ax-accent); color: #eef2fb; }
.stButton > button:active { transform: translateY(1px); }

/* --- Inputs --------------------------------------------------------------- */
[data-testid="stTextInput"] input, [data-baseweb="select"] > div {
  border-radius: 9px;
}
.stTextInput input:focus { border-color: var(--ax-accent) !important; }

/* Section subheadings breathe a little more. */
h5 { margin-top: 0.4rem; color: #dfe5f0; }

/* Chip row for result metadata. */
.ax-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 10px; }
.ax-chip {
  font-size: 0.76rem; color: var(--ax-muted);
  padding: 2px 9px; border-radius: 6px;
  background: var(--ax-panel-2); border: 1px solid var(--ax-border);
}
.ax-chip.score { color: #bcd0ff; border-color: rgba(91,140,255,0.4); background: var(--ax-accent-soft); }
.ax-chip.grow { color: #7ee2a8; border-color: rgba(81,207,102,0.35); background: rgba(81,207,102,0.08); }
.ax-chip.fade { color: #ffb4a0; border-color: rgba(255,135,135,0.35); background: rgba(255,135,135,0.08); }
.ax-chip.warn { color: #ffd97a; border-color: rgba(255,212,59,0.35); background: rgba(255,212,59,0.08); }

/* Empty / offline states. */
.ax-empty { text-align: center; padding: 20px 12px 8px; }
.ax-empty-icon {
  font-size: 40px; line-height: 1; opacity: 0.9;
  display: inline-grid; place-items: center;
  width: 72px; height: 72px; border-radius: 18px;
  background: var(--ax-accent-soft); border: 1px solid var(--ax-border);
}
.ax-empty-title { font-weight: 650; font-size: 1.08rem; color: #eef2fb; margin-top: 12px; }
.ax-empty-sub {
  color: var(--ax-muted); font-size: 0.9rem; line-height: 1.55;
  margin: 8px auto 4px; max-width: 540px;
}
.ax-empty-sub code {
  background: rgba(146,164,205,0.16); color: #d6deec;
  padding: 1px 6px; border-radius: 5px; font-size: 0.85em;
}

/* Recolor Streamlit's top decoration bar to the brand accent (default is a
   multi-color gradient that clashes with the navy theme). */
[data-testid="stDecoration"] {
  background: linear-gradient(90deg, var(--ax-accent) 0%, #22b8cf 100%);
}

/* Hypothesis-pitch draft card. */
.ax-draft-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: 999px; font-size: 0.74rem; font-weight: 600;
  letter-spacing: 0.02em; text-transform: uppercase;
  color: #ffd97a; background: rgba(255,212,59,0.1); border: 1px solid rgba(255,212,59,0.35);
}
.ax-draft-title { font-size: 1.15rem; font-weight: 700; color: #f2f5fb; margin: 8px 0 2px; letter-spacing: -0.01em; }

/* Ranked rows (velocity keywords, etc.). */
.ax-list { display: flex; flex-direction: column; gap: 6px; }
.ax-listrow {
  display: flex; align-items: center; gap: 10px; padding: 8px 14px;
  border: 1px solid var(--ax-border); border-radius: 10px;
  background: var(--ax-panel);
}
.ax-listrow:hover { border-color: var(--ax-border-strong); background: var(--ax-panel-2); }
.ax-listrow .ax-rank {
  color: var(--ax-muted); font-size: 0.82rem; font-variant-numeric: tabular-nums;
  width: 2em; text-align: right; flex: none;
}
.ax-listrow .ax-name {
  font-weight: 550; color: #e6e9f0; flex: 1;
  min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.ax-listrow .ax-chip { flex: none; margin: 0; }

/* Narrow screens: let ranked rows wrap chips under a full-width name instead
   of overflowing horizontally. */
@media (max-width: 640px) {
  .ax-listrow { flex-wrap: wrap; }
  .ax-listrow .ax-name { flex: 1 1 100%; white-space: normal; }
  .ax-listrow .ax-rank { width: auto; }
}

/* Slimmer scrollbar. */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #2a3550; border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: #364365; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="ax-hero">'
    '<div class="ax-mark">🔭</div>'
    '<div>'
    '<h1>Axiom</h1>'
    '<p>Research trends &amp; gaps discovery over the ACL Anthology corpus</p>'
    '</div></div>',
    unsafe_allow_html=True,
)


# --- Cached singletons -------------------------------------------------------
@st.cache_resource(show_spinner="Loading SPECTER2 encoder…")
def get_encoder() -> Specter2Encoder:
    return Specter2Encoder()


@st.cache_resource(show_spinner=False)
def get_store() -> AxiomQdrant:
    return AxiomQdrant()


@st.cache_resource(show_spinner="Loading citation graph…")
def get_graph() -> nx.DiGraph:
    """In-memory citation graph from SQLite (decision OD6, NetworkX-first)."""
    return graph.load_graph()


@st.cache_data(show_spinner=False)
def get_influence_ranking(top_k: int = 50) -> list:
    """PageRank-ranked influential papers (cached; slice for the UI)."""
    return graph.influence(get_graph(), top_k=top_k)


@st.cache_data(show_spinner="Detecting communities & research gaps…")
def get_gap_analysis() -> gaps.GapAnalysis:
    """Communities + candidate research gaps (cached). Needs Qdrant vectors."""
    conn = db.connect()
    try:
        return gaps.analyze(get_graph(), conn, get_store().fetch_dense_vectors())
    finally:
        conn.close()


@st.cache_data(show_spinner="Computing keyword velocity…")
def get_velocity_analysis(venue: str | None, year_range: tuple[int, int] | None) -> velocity.VelocityAnalysis:
    """Concept velocity: normalized-frequency log2-ratio, recent vs prior window (OD10)."""
    conn = db.connect()
    try:
        return velocity.get_top_velocity_keywords(conn, n=50, venue=venue, year_range=year_range)
    finally:
        conn.close()


# Distinct, dark-background-friendly palette for community coloring.
_COMMUNITY_PALETTE = [
    "#4dabf7", "#ff922b", "#51cf66", "#cc5de8", "#ffd43b", "#ff6b6b",
    "#22b8cf", "#a9e34b", "#f783ac", "#748ffc", "#ffa94d", "#63e6be",
    "#e599f7", "#94d82d", "#ff8787", "#3bc9db",
]


def _community_color(cid: int) -> str:
    return _COMMUNITY_PALETTE[cid % len(_COMMUNITY_PALETTE)]


@st.cache_data(show_spinner=False)
def get_filter_options() -> tuple[list[str], tuple[int, int] | None]:
    """Venue list + year bounds from SQLite (drives the filter bar)."""
    conn = db.connect()
    try:
        venues = db.distinct_venues(conn)
        bounds = db.year_bounds(conn)
        return venues, bounds
    except Exception:
        return [], None
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def get_meta() -> dict[str, dict]:
    """Per-paper abstract/DOI/title from SQLite, keyed by paper_id.

    Qdrant payloads are intentionally lean; the cards enrich hits from SQLite,
    the durable source of truth.
    """
    conn = db.connect()
    try:
        rows = conn.execute("SELECT openalex_id, title, abstract, doi FROM papers").fetchall()
        return {
            r["openalex_id"]: {"title": r["title"], "abstract": r["abstract"], "doi": r["doi"]}
            for r in rows
        }
    except Exception:
        return {}
    finally:
        conn.close()

def render_year_filter(
    bounds: tuple[int, int] | None,
    key_prefix: str,
    min_year: int | None = None,
) -> tuple[int, int] | None:
    """Year range selector. `min_year` clamps the earliest selectable year —
    e.g. the Trending tab pins the floor to config.VELOCITY_MIN_YEAR so velocity
    never spans the noisy pre-2020 references pulled in by snowball ingestion."""
    if not bounds:
        return None
    lo, hi = bounds
    floor = lo if min_year is None else min(max(min_year, lo), hi)
    years = list(range(floor, hi + 1))
    ycol1, ycol2 = st.columns(2)
    with ycol1:
        y_from = st.selectbox("From year", years, index=0, key=f"{key_prefix}_from")
    with ycol2:
        y_to = st.selectbox("To year", years, index=len(years) - 1, key=f"{key_prefix}_to")
    return (min(y_from, y_to), max(y_from, y_to))


# --- Guard rails: Qdrant only gates the vector-backed tabs -------------------
# Search + Citation graph need Qdrant vectors; Trending / Reading list / Review
# queue are SQLite-only. A missing or empty index must NOT stop the whole app
# (a top-level st.stop() here previously blacked out every tab, including
# Trending). Record the status and let the two vector tabs surface it themselves.
store = get_store()
try:
    point_count = store.count()
    qdrant_ok = point_count > 0
    qdrant_msg = "" if qdrant_ok else (
        "The vector index is empty. Run the embed/index step "
        "(`python scripts/ingest_openalex.py` without `--no-index`, or "
        "`python scripts/bootstrap_synthetic.py`) to enable Search and the "
        "citation graph. Trending, Reading list and Review queue work without it."
    )
except Exception:
    point_count = 0
    qdrant_ok = False
    qdrant_msg = (
        "⚠️ Cannot reach Qdrant. Start it with `docker compose up -d`, then run "
        "the embed/index step. Search and the citation graph stay disabled until "
        "then — Trending, Reading list and Review queue work without it."
    )

meta = get_meta()
st.session_state.setdefault("similar_to", None)

# Corpus / model / index status as a compact pill row under the hero.
if qdrant_ok:
    _index_pill = f'<span class="ax-pill ok"><span class="dot"></span>index · {point_count:,} vectors</span>'
else:
    _index_pill = '<span class="ax-pill off"><span class="dot"></span>index offline</span>'
st.markdown(
    '<div class="ax-meta">'
    f'<span class="ax-pill">📚 ACL · EMNLP · COLING · NAACL · 2020–2025</span>'
    f'<span class="ax-pill">collection <code>{config.COLLECTION_NAME}</code></span>'
    f'<span class="ax-pill">model <code>{config.MODEL_ID}</code></span>'
    f'{_index_pill}'
    '</div>',
    unsafe_allow_html=True,
)


# --- Reading list (OD13): bookmarks only, no LLM summaries (PBI 5 not built) --
def get_bookmarked_ids() -> set[str]:
    """Not cached — must reflect adds/removes immediately on the next rerun."""
    conn = db.connect()
    try:
        return {r["paper_id"] for r in db.list_bookmarks(conn)}
    finally:
        conn.close()


def toggle_bookmark(paper_id: str, *, add: bool) -> None:
    conn = db.connect()
    try:
        if add:
            db.add_bookmark(conn, paper_id)
        else:
            db.remove_bookmark(conn, paper_id)
    finally:
        conn.close()


# --- Result rendering --------------------------------------------------------
def _chips(*chips: tuple[str, str]) -> None:
    """Render a horizontal chip row. Each chip is (text, css_suffix) where the
    suffix is "" for a plain chip or a modifier like "score"/"grow"/"fade"."""
    inner = "".join(f'<span class="ax-chip {cls}">{txt}</span>' for txt, cls in chips)
    st.markdown(f'<div class="ax-chips">{inner}</div>', unsafe_allow_html=True)


def _empty_state(icon: str, title: str, body_md: str | None = None) -> None:
    """Card-styled empty/offline placeholder: centered icon tile + title, with
    an optional body. The body supports a tiny markdown subset (`code`,
    **strong**, *em*) so setup commands stay formatted inside the centered card."""
    body_html = ""
    if body_md:
        # Drop a leading warning glyph — the card's own icon already signals state.
        safe = html.escape(body_md.lstrip().removeprefix("⚠️").lstrip())
        safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
        safe = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", safe)
        safe = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", safe)
        body_html = f'<div class="ax-empty-sub">{safe}</div>'
    with st.container(border=True):
        st.markdown(
            f'<div class="ax-empty"><div class="ax-empty-icon">{icon}</div>'
            f'<div class="ax-empty-title">{html.escape(title)}</div>'
            f'{body_html}</div>',
            unsafe_allow_html=True,
        )


def render_hits(hits: list[SearchHit], *, score_label: str = "score") -> None:
    """Render hits as expandable cards with concepts, abstract, and a pivot."""
    if not hits:
        _empty_state("🔍", "No papers match these filters",
                     "Try loosening the venue or year filters, or broaden the query.")
        return
    meta = get_meta()
    bookmarked = get_bookmarked_ids()
    st.caption(f"{len(hits)} result{'s' if len(hits) != 1 else ''}")
    for h in hits:
        m = meta.get(h.paper_id, {})
        # Title leads the card; dense metadata moves inside as chips so the
        # expander header stays scannable.
        with st.expander(h.title):
            _chips(
                (f"{h.score:.3f} {score_label}", "score"),
                (f"📅 {h.year}", ""),
                (f"🏛 {html.escape(str(h.venue))}", ""),
                (f"❝ {h.cited_by_count:,} cites", ""),
            )
            if m.get("doi"):
                st.markdown(f"[↗ Open paper (DOI)](https://doi.org/{m['doi']})")
            if h.concepts:
                st.markdown(" ".join(f"`{c}`" for c in h.concepts))
            st.write(m.get("abstract") or "_No abstract available._")
            bcol, scol = st.columns([1, 1])
            with bcol:
                if h.paper_id in bookmarked:
                    if st.button("📚 Remove bookmark", key=f"unbm_{h.paper_id}"):
                        toggle_bookmark(h.paper_id, add=False)
                        st.rerun()
                else:
                    if st.button("📚 Add to reading list", key=f"bm_{h.paper_id}"):
                        toggle_bookmark(h.paper_id, add=True)
                        st.rerun()
            with scol:
                # Pivot: explore semantic neighbors of this paper.
                if st.button("🔎 Similar papers", key=f"sim_{h.paper_id}"):
                    st.session_state["similar_to"] = h.paper_id
                    st.rerun()


# --- Search tab --------------------------------------------------------------
def render_search() -> None:
    """Semantic/hybrid search with the venue/year filter bar."""
    if not qdrant_ok:
        _empty_state("🗂️", "Search needs the vector index", qdrant_msg)
        return
    venues, bounds = get_filter_options()
    with st.container():
        fcol1, fcol2, fcol3 = st.columns([2, 2, 1])
        with fcol1:
            # Empty selection = all venues (no silent filtering).
            selected_venues = st.multiselect(
                "Venue", options=venues, default=[],
                help="Leave empty to search all venues.",
            )
        with fcol2:
            year_range = render_year_filter(bounds, "search")
        with fcol3:
            # Default to DEFAULT_TOP_K; cap at corpus size (no magic number).
            top_k = st.number_input(
                "Top-K", min_value=1, max_value=point_count,
                value=min(config.DEFAULT_TOP_K, point_count), step=1,
            )

    # Hybrid is only offered if the index has sparse vectors.
    if store.supports_hybrid():
        mode = st.radio(
            "Retrieval", ["Hybrid (dense + sparse)", "Dense only"],
            index=0, horizontal=True,
            help="Hybrid adds keyword/acronym matching (e.g. 'LoRA') on top of dense.",
        )
        use_hybrid = mode.startswith("Hybrid")
    else:
        use_hybrid = False
        st.caption("Retrieval: dense only (index has no sparse vectors — re-run the bootstrap for hybrid).")

    common = dict(
        top_k=int(top_k),
        venues=selected_venues or None,
        year_range=tuple(year_range) if year_range else None,
    )

    # Similar-papers pivot vs normal query view.
    similar_to = st.session_state.get("similar_to")
    if similar_to:
        title = meta.get(similar_to, {}).get("title", similar_to)
        st.subheader(f"Papers similar to: {title}")
        if st.button("← Back to search"):
            st.session_state["similar_to"] = None
            st.rerun()
        render_hits(store.similar_papers(similar_to, **common), score_label="cosine")
    else:
        query = st.text_input(
            "Search query",
            placeholder="e.g. reducing hallucination in retrieval-augmented generation",
        )
        if query:
            qvec = get_encoder().encode_query(query)
            if use_hybrid:
                hits = store.search_hybrid(query_vector=qvec, query_text=query, **common)
            else:
                hits = store.search(query_vector=qvec, **common)
            render_hits(hits, score_label="RRF rank" if use_hybrid else "cosine")
        else:
            _empty_state("💬", "Search the corpus",
                         "Enter a query above — e.g. *reducing hallucination in "
                         "retrieval-augmented generation*.")


# --- Citation-graph tab ------------------------------------------------------
_GRAPH3D_TEMPLATE = """
<div id="graph3d" style="width:100%;height:HEIGHTpx;background:#0b0f19;border-radius:10px;"></div>
<script src="https://unpkg.com/3d-force-graph@1.73.4/dist/3d-force-graph.min.js" integrity="sha384-GNPicn8pBA2/PGSyPTpxIlPurgLUYcNYJ2zskIq782dE9+gp5E32WSyuxZqA7J+u" crossorigin="anonymous"></script>
<script>
(function () {
  var el = document.getElementById('graph3d');
  function fail(msg) {
    el.innerHTML = '<p style="color:#ff8787;font-family:sans-serif;padding:1rem">' + msg + '</p>';
  }
  if (typeof ForceGraph3D === 'undefined') {
    fail('3D graph library could not load — check the browser has internet access.');
    return;
  }
  try {
    var data = __DATA__;
    el.style.position = 'relative';
    // Two background themes for the scene; toggled via the on-screen button.
    var THEMES = {
      dark:  { bg: '#0b0f19', link: 'rgba(205,222,248,0.85)', arrow: 'rgba(220,230,250,0.8)', particle: '#ffd43b' },
      light: { bg: '#f4f6fb', link: 'rgba(60,72,95,0.7)',     arrow: 'rgba(45,55,75,0.75)',   particle: '#e8590c' }
    };
    var theme = THEMES.dark;
    var Graph = ForceGraph3D()(el)
      .width(el.clientWidth || 800)
      .height(HEIGHT)
      .showNavInfo(false)
      .backgroundColor(theme.bg)
      .graphData(data)
      .nodeLabel('name')
      .nodeVal('val')
      .nodeColor('color')
      .nodeOpacity(0.95)
      .nodeResolution(16)
      .linkColor(function () { return theme.link; })
      .linkWidth(1.2)
      .linkDirectionalArrowLength(3.6)
      .linkDirectionalArrowRelPos(1)
      .linkDirectionalArrowColor(function () { return theme.arrow; })
      .linkDirectionalParticles(3)
      .linkDirectionalParticleWidth(2)
      .linkDirectionalParticleColor(function () { return theme.particle; })
      .linkDirectionalParticleSpeed(0.006)
      .onNodeClick(function (node) {
        var dist = 70;
        var ratio = 1 + dist / Math.hypot(node.x || 1, node.y || 1, node.z || 1);
        Graph.cameraPosition(
          {x: (node.x || 0) * ratio, y: (node.y || 0) * ratio, z: (node.z || 0) * ratio},
          node, 800);
      });

    // Explicit starting camera so the scene is never a blank black frame.
    Graph.cameraPosition({ z: 320 });

    // Disable the (TrackballControls) wheel-zoom so the wheel scrolls the PAGE
    // instead of the canvas; zoom is via the on-screen buttons. (Guarded.)
    try {
      var ctrls = Graph.controls();
      if (ctrls) { ctrls.noZoom = true; ctrls.enableZoom = false; }
    } catch (e) {}
    // Cap long-range repulsion so disconnected clusters don't drift far apart.
    try { if (Graph.d3Force('charge')) Graph.d3Force('charge').strength(-45).distanceMax(220); } catch (e) {}
    // Auto-fit once the layout settles (guarded; runs after physics).
    setTimeout(function () { try { Graph.zoomToFit(600, 30); } catch (e) {} }, 1800);

    function dolly(f) {
      var c = Graph.cameraPosition();
      Graph.cameraPosition({x: c.x * f, y: c.y * f, z: c.z * f}, undefined, 250);
    }
    function mkBtn(txt, fn) {
      var b = document.createElement('button');
      b.textContent = txt;
      b.style.cssText = 'width:32px;height:32px;border:1px solid #93a4bd;border-radius:6px;' +
        'background:#5b6b85;color:#ffffff;font-size:17px;font-weight:600;line-height:1;' +
        'cursor:pointer;box-shadow:0 1px 3px rgba(0,0,0,.4);';
      b.onmouseenter = function () { b.style.background = '#74859f'; };
      b.onmouseleave = function () { b.style.background = '#5b6b85'; };
      b.onclick = fn;
      return b;
    }
    // Re-apply colour accessors so the theme change takes effect immediately.
    function applyTheme(t) {
      theme = t;
      el.style.background = t.bg;
      Graph.backgroundColor(t.bg)
        .linkColor(function () { return t.link; })
        .linkDirectionalArrowColor(function () { return t.arrow; })
        .linkDirectionalParticleColor(function () { return t.particle; });
    }
    var bar = document.createElement('div');
    bar.style.cssText = 'position:absolute;top:10px;right:12px;display:flex;gap:6px;z-index:5;';
    bar.appendChild(mkBtn('+', function () { dolly(0.8); }));
    bar.appendChild(mkBtn('–', function () { dolly(1.25); }));
    bar.appendChild(mkBtn('▣', function () { try { Graph.zoomToFit(400, 30); } catch (e) {} }));
    var themeBtn = mkBtn('☀', function () {
      var light = theme === THEMES.dark;
      applyTheme(light ? THEMES.light : THEMES.dark);
      themeBtn.textContent = light ? '🌙' : '☀';
      themeBtn.title = light ? 'Switch to dark background' : 'Switch to light background';
    });
    themeBtn.title = 'Switch to light background';
    bar.appendChild(themeBtn);
    el.appendChild(bar);

    window.addEventListener('resize', function () { Graph.width(el.clientWidth || 800); });
  } catch (err) {
    fail('3D graph error: ' + (err && err.message ? err.message : err));
  }
})();
</script>
"""


def _graph_html(sub: nx.DiGraph, node_colors: dict[str, str] | None = None,
                node_cids: dict[str, int] | None = None, height: int = 500) -> str:
    """
    Renders sub to an interactive 3d-force-graph HTML snippet.
    node_colors dict maps openalex_id -> hex color.
    """
    indeg = dict(sub.in_degree())
    max_in = max(indeg.values(), default=0) or 1
    nodes = []
    for nid, d in sub.nodes(data=True):
        title = d.get("title") or nid
        if node_colors and nid in node_colors:
            color = node_colors[nid]
        else:
            color = "#4dabf7"
            
        cluster_str = f"[Cluster: {node_cids[nid]}] " if node_cids and nid in node_cids else ""
        
        nodes.append({
            "id": nid,
            "name": (f"{cluster_str}{title} ({d.get('year', '?')}) — "
                     f"{d.get('cited_by_count', 0)} cites · "
                     f"{indeg.get(nid, 0)} citing within view"),
            "val": 2 + 9 * (indeg.get(nid, 0) / max_in),
            "color": color,
        })
    links = [{"source": u, "target": v} for u, v in sub.edges()]
    payload = json.dumps({"nodes": nodes, "links": links})
    payload = payload.replace("</", "<\\/")
    return _GRAPH3D_TEMPLATE.replace("HEIGHT", str(height)).replace("__DATA__", payload)


def _short(comm, k: int = 2) -> str:
    """First k distinctive concepts of a community, for compact labels."""
    return " · ".join(comm.labels[:k]) if comm.labels else f"cluster {comm.cid}"


def _gap_map_svg(analysis, sel_idx: int | None = None, height: int = 460,
                  dark: bool = True, show_indices: list[int] | None = None) -> str:
    """Render a subset of candidate gaps as one 'gap map'.

    Sub-topics that take part in a drawn gap sit on a ring (colour = their
    cluster, dot size = paper count); each candidate gap is an arc bowing
    across the interior. Arc thickness ∝ G-score and colour marks whether it
    clears the calibrated threshold (green) or not (amber). The selected gap
    is lit gold and drawn on top. Pure inline SVG — no JS, no external assets,
    prints on any bg.

    `show_indices` restricts drawing to those positions in `analysis.gaps`
    (default: all) — with every candidate drawn at once the arcs converge
    into an unreadable hairball, so the caller typically passes only the
    top few by rank. Global indices are kept (not renumbered) so tooltips and
    the gold "selected" highlight stay consistent with the ranked list above.
    """
    import math

    bg = "#0b0f19" if dark else "#f8f9fb"
    label_fill = "#cdd6e4" if dark else "#3a4150"
    legend_fill = "#9ca3af" if dark else "#6b7280"

    all_gaps = analysis.gaps
    show_indices = list(range(len(all_gaps))) if show_indices is None else show_indices
    indexed = [(i, all_gaps[i]) for i in show_indices]
    if not indexed:
        return ""

    # Communities that participate in a drawn gap, kept in gap-rank order so
    # the busiest / strongest sub-topics land first on the ring.
    order: list[int] = []
    for _, gp in indexed:
        for comm in (gp.a, gp.b):
            if comm.cid not in order:
                order.append(comm.cid)
    comm_by_cid = {gp.a.cid: gp.a for _, gp in indexed}
    comm_by_cid.update({gp.b.cid: gp.b for _, gp in indexed})

    W, H = 820, height
    cx, cy = W / 2.0, H / 2.0
    R = min(W, H) / 2.0 - 96          # ring radius (leave room for labels)
    n = len(order)
    sizes = [comm_by_cid[c].size for c in order]
    smax = max(sizes) or 1

    # Position each community on the ring; start at the top, go clockwise.
    pos: dict[int, tuple[float, float, float]] = {}
    for k, cid in enumerate(order):
        ang = -math.pi / 2 + 2 * math.pi * k / n
        pos[cid] = (cx + R * math.cos(ang), cy + R * math.sin(ang), ang)

    g_scores = [gp.g_score for _, gp in indexed]
    g_lo, g_hi = min(g_scores), max(g_scores)
    g_span = (g_hi - g_lo) or 1.0

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    parts = [
        f'<svg viewBox="0 0 {W} {H}" width="100%" '
        f'style="max-width:{W}px;height:auto;background:{bg};border-radius:10px;'
        f'font-family:-apple-system,Segoe UI,Roboto,sans-serif;" '
        f'role="img" aria-label="Candidate research-gap map">'
    ]

    # --- arcs (gaps): weakest first so strong/selected ones sit on top ---------
    drawn = sorted(range(len(indexed)),
                   key=lambda k: (indexed[k][0] == sel_idx, indexed[k][1].g_score))
    for k in drawn:
        i, gp = indexed[k]
        ax, ay, _ = pos[gp.a.cid]
        bx, by, _ = pos[gp.b.cid]
        # Control point pulled toward centre so arcs bow inward (chord look).
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        ctrlx = cx + (mx - cx) * 0.35
        ctrly = cy + (my - cy) * 0.35
        norm = (gp.g_score - g_lo) / g_span
        selected = (i == sel_idx)
        if selected:
            stroke, width, opacity = "#ffd43b", 3.5 + 5 * norm, 1.0
        elif gp.meets_threshold:
            stroke, width, opacity = "#51cf66", 1.5 + 4.5 * norm, 0.75
        else:
            stroke, width, opacity = "#ffa94d", 1.2 + 3.5 * norm, 0.5
        tip = (f"#{i} {_short(gp.a)} ⟷ {_short(gp.b)} — "
               f"sim {gp.semantic_similarity:.2f}, {gp.inter_citations} cites, "
               f"G {gp.g_score:.2f}")
        glow = ('filter="drop-shadow(0 0 6px rgba(255,212,59,0.9))" '
                if selected else "")
        parts.append(
            f'<path d="M {ax:.1f} {ay:.1f} Q {ctrlx:.1f} {ctrly:.1f} '
            f'{bx:.1f} {by:.1f}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width:.1f}" stroke-linecap="round" '
            f'stroke-opacity="{opacity}" {glow}>'
            f'<title>{esc(tip)}</title></path>'
        )

    # --- community nodes + labels --------------------------------------------
    for cid in order:
        x, y, ang = pos[cid]
        comm = comm_by_cid[cid]
        r = 5 + 9 * (comm.size / smax)
        color = _community_color(cid)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" '
            f'stroke="{bg}" stroke-width="1.5">'
            f'<title>{esc(comm.label)} · {comm.size} papers</title></circle>'
        )
        # Label just outside the ring — one concept per line so full names show
        # (no truncation), block anchored away from the centre.
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        lines = comm.labels[:2] or [f"cluster {cid}"]
        lines = [ln if k == 0 else f"· {ln}" for k, ln in enumerate(lines)]
        if abs(cos_a) < 0.30:            # top / bottom of the ring
            anchor = "middle"
            lx = x
            ly = y + (r + 12) * (1 if sin_a > 0 else -1)
        else:                            # left / right side
            anchor = "start" if cos_a > 0 else "end"
            lx = x + (r + 8) * (1 if cos_a > 0 else -1)
            ly = y
        # Vertically centre the multi-line block on ly: first line lifted to the
        # top of the block (+0.32em baseline nudge), each subsequent line +1em.
        top = -(len(lines) - 1) / 2.0
        spans = "".join(
            f'<tspan x="{lx:.1f}" dy="{(top + 0.32) if k == 0 else 1:.2f}em">{esc(ln)}</tspan>'
            for k, ln in enumerate(lines)
        )
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'fill="{label_fill}" font-size="11">{spans}</text>'
        )

    # --- legend ---------------------------------------------------------------
    parts.append(
        f'<g font-size="11" fill="{legend_fill}">'
        f'<rect x="14" y="{H-30}" width="18" height="3" rx="1.5" fill="#51cf66"/>'
        f'<text x="38" y="{H-25}">meets threshold</text>'
        f'<rect x="150" y="{H-30}" width="18" height="3" rx="1.5" fill="#ffa94d"/>'
        f'<text x="174" y="{H-25}">below threshold</text>'
        f'<rect x="286" y="{H-30}" width="18" height="3" rx="1.5" fill="#ffd43b"/>'
        f'<text x="310" y="{H-25}">selected</text>'
        f'<text x="{W-14}" y="{H-25}" text-anchor="end">'
        f'thickness = G-score · dot size = papers</text>'
        f'</g></svg>'
    )
    return "".join(parts)


def _gap_bar_chart(analysis, sel_idx: int | None = None) -> None:
    """Ranked-list alternative to the ring diagram: one horizontal bar per
    candidate gap, sorted by G-score. Unambiguous to read (no crossing lines)
    at the cost of not showing which communities repeat across gaps."""
    rows = [
        {
            "label": f"#{i} {_short(gp.a)} ⟷ {_short(gp.b)}",
            "g_score": gp.g_score,
            "status": "selected" if i == sel_idx
                      else ("meets threshold" if gp.meets_threshold else "below threshold"),
            "a_label": gp.a.label,
            "b_label": gp.b.label,
            "similarity": gp.semantic_similarity,
            "inter_citations": gp.inter_citations,
        }
        for i, gp in enumerate(analysis.gaps)
    ]
    df = pd.DataFrame(rows)
    color_scale = alt.Scale(
        domain=["selected", "meets threshold", "below threshold"],
        range=["#ffd43b", "#51cf66", "#ffa94d"],
    )
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=340)),
        x=alt.X("g_score:Q", title="G-score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("status:N", scale=color_scale, legend=alt.Legend(title=None)),
        tooltip=[
            alt.Tooltip("a_label:N", title="Side A"),
            alt.Tooltip("b_label:N", title="Side B"),
            alt.Tooltip("g_score:Q", title="G-score", format=".2f"),
            alt.Tooltip("similarity:Q", title="semantic similarity", format=".2f"),
            alt.Tooltip("inter_citations:Q", title="inter-community citations"),
        ],
    ).properties(height=alt.Step(28))
    st.altair_chart(chart, use_container_width=True)


def _gap_matrix_chart(analysis, sel_idx: int | None = None) -> None:
    """Matrix alternative: sub-topics on both axes, cell colour = G-score.

    No crossing lines to trace — scan for the brightest cell. Only cells that
    are an actual candidate pair (in analysis.gaps) are filled; everything
    else is blank. Symmetric (both (a, b) and (b, a) filled) so the grid reads
    the same whichever axis you scan first.
    """
    labels_by_cid: dict[int, str] = {}
    for gp in analysis.gaps:
        labels_by_cid[gp.a.cid] = _short(gp.a)
        labels_by_cid[gp.b.cid] = _short(gp.b)
    # Order communities by their best (max) G-score across all their gaps, so
    # the most gap-heavy sub-topics cluster near the top-left of the grid.
    best_score: dict[int, float] = {}
    for gp in analysis.gaps:
        best_score[gp.a.cid] = max(best_score.get(gp.a.cid, 0.0), gp.g_score)
        best_score[gp.b.cid] = max(best_score.get(gp.b.cid, 0.0), gp.g_score)
    order = sorted(labels_by_cid, key=lambda c: best_score[c], reverse=True)
    order_labels = [labels_by_cid[c] for c in order]

    rows = []
    for i, gp in enumerate(analysis.gaps):
        status = ("selected" if i == sel_idx
                  else ("meets threshold" if gp.meets_threshold else "below threshold"))
        common = dict(
            g_score=gp.g_score, status=status,
            similarity=gp.semantic_similarity, inter_citations=gp.inter_citations,
            idx=i,
        )
        rows.append({"row": labels_by_cid[gp.a.cid], "col": labels_by_cid[gp.b.cid], **common})
        rows.append({"row": labels_by_cid[gp.b.cid], "col": labels_by_cid[gp.a.cid], **common})
    df = pd.DataFrame(rows)

    base = alt.Chart(df).encode(
        x=alt.X("col:N", sort=order_labels, title=None, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("row:N", sort=order_labels, title=None),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=1).encode(
        color=alt.Color("g_score:Q", title="G-score", scale=alt.Scale(scheme="greens")),
        tooltip=[
            alt.Tooltip("row:N", title="Side A"),
            alt.Tooltip("col:N", title="Side B"),
            alt.Tooltip("g_score:Q", title="G-score", format=".2f"),
            alt.Tooltip("similarity:Q", title="semantic similarity", format=".2f"),
            alt.Tooltip("inter_citations:Q", title="inter-community citations"),
        ],
    )
    selected = base.transform_filter(alt.datum.status == "selected").mark_rect(
        fill=None, stroke="#e6b800", strokeWidth=3,
    )
    chart = (cells + selected).properties(
        width=alt.Step(60), height=alt.Step(40),
    )
    st.altair_chart(chart, use_container_width=False)


def _render_gap_detail(g, gap) -> None:
    """Explain a candidate gap and list the top papers on each side."""
    st.markdown(
        f"**Why this is a candidate gap:** these two sub-topics are close in "
        f"meaning (centroid cosine **{gap.semantic_similarity:.2f}**) but only "
        f"**{gap.inter_citations}** citation(s) connect **{gap.a.size}** and "
        f"**{gap.b.size}** papers — a bridge largely unbuilt."
    )
    if getattr(gap, "components", None):
        c = gap.components
        flag = " ✅ meets threshold" if gap.meets_threshold else ""
        st.caption(
            f"**G-score {gap.g_score:.2f}**{flag} — similarity {c['similarity']:.2f} · "
            f"disconnection {c['disconnection']:.2f} · velocity {c['velocity']:.2f} · "
            f"authority {c['authority']:.2f}. ⚠️ Uncalibrated defaults (OD17); weights "
            f"and threshold need human labels — see scripts/calibrate_gap_thresholds.py."
        )
    cola, colb = st.columns(2)
    for col, comm in ((cola, gap.a), (colb, gap.b)):
        with col:
            st.markdown(f"**{comm.label}** · {comm.size} papers")
            tops = sorted(comm.members,
                          key=lambda p: g.nodes[p].get("cited_by_count", 0),
                          reverse=True)[:5]
            for p in tops:
                d = g.nodes[p]
                st.markdown(f"- {d.get('title')}  _({d.get('year')})_")


def render_graph_view() -> None:
    """Research gaps + citation structure (communities, gaps, influence)."""
    if not qdrant_ok:
        st.subheader("Research gaps & citation structure")
        _empty_state("🕸️", "The citation graph needs the vector index", qdrant_msg)
        return
    st.subheader("Research gaps & citation structure")
    st.caption(
        "Sub-topics (communities) are detected in the citation graph; a candidate "
        "**research gap** is a pair of communities close in meaning yet barely "
        "citing each other — related literatures that haven't connected."
    )

    g = get_graph()
    gs = graph.stats(g)
    if gs["edges_in_corpus"] == 0:
        _empty_state("🔗", "No in-corpus citations yet",
                     "The graph has papers but no edges between them — ingest a "
                     "connected corpus to reveal communities and gaps.")
        return

    analysis = get_gap_analysis()
    node_colors = {pid: _community_color(cid) for pid, cid in analysis.node2c.items()}

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Papers", gs["papers"])
    mcol2.metric("In-corpus citations", gs["edges_in_corpus"])
    mcol3.metric("Sub-topics", len(analysis.communities))
    mcol4.metric("Candidate gaps", len(analysis.gaps))

    st.markdown("##### Interactive 3D graph")

    with st.container(border=True):
        ccol1, ccol2, ccol3 = st.columns([1.4, 2, 1])
        with ccol1:
            view = st.radio(
                "View", ["Research gaps", "Top influential"],
                help="Research gaps (default) highlights related-but-disconnected "
                     "clusters — the openings. 'Top influential' is for orientation: "
                     "the field's central papers. Click any node to inspect it.",
            )
        sel_gap = None
        sel_gap_idx = None
        with ccol2:
            if view == "Research gaps":
                if analysis.gaps:
                    opts = [
                        f"#{i} {_short(gp.a)} ⟷ {_short(gp.b)}  "
                        f"(sim {gp.semantic_similarity:.2f} · {gp.inter_citations} cites)"
                        for i, gp in enumerate(analysis.gaps)
                    ]
                    sel = st.selectbox("Candidate gap", range(len(opts)),
                                       format_func=lambda i: opts[i])
                    sel_gap_idx = sel
                    sel_gap = analysis.gaps[sel_gap_idx]
                    members = set(sel_gap.a.members) | set(sel_gap.b.members)
                    sub = g.subgraph(members).copy()
                else:
                    sub = g.subgraph([]).copy()
            else:  # Top influential (orientation)
                v_n = st.slider("Papers to draw", 10, 50, 25, step=5)
                sub = graph.top_influential_subgraph(g, n=v_n)
        with ccol3:
            # Escape hatch: the WebGL canvas captures the mouse wheel (to zoom),
            # which blocks page scrolling while hovering it. Toggle off to scroll.
            show_3d = st.toggle("Show 3D graph", value=True,
                                help="Turn off to scroll the page freely past this section.")

    # Legend split:
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;justify-content:space-between;'
        'align-items:center;gap:16px;font-size:0.86rem;margin:2px 0 8px;">'
        # left group — what you're looking at
        '<div style="display:flex;flex-wrap:wrap;gap:18px;align-items:center;">'
        f'<span><b>{sub.number_of_nodes()}</b> papers · '
        f'<b>{sub.number_of_edges()}</b> citations in view</span>'
        '<span>🎨 colour = sub-topic cluster</span>'
        '<span style="display:inline-flex;align-items:center;gap:8px;">'
        '<span style="width:7px;height:7px;background:#868e96;border-radius:50%;display:inline-block;"></span>fewer'
        '<span style="width:17px;height:17px;background:#868e96;border-radius:50%;display:inline-block;"></span>'
        'more citations</span>'
        '</div>'
        # right group
        '<div style="color:#9ca3af;white-space:nowrap;">'
        'drag rotate · +/–/▣ zoom · click node to fly · wheel scrolls page'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if show_3d:
        components.html(
            _graph_html(sub, node_colors=node_colors, node_cids=analysis.node2c, height=520),
            height=540, scrolling=False)

    # Gap detail + ranked list (gaps view only).
    if view == "Research gaps" and sel_gap is not None:
        _render_gap_detail(g, sel_gap)

        st.info(
            "⚠️ **Unverified Candidate** — a hypothesis pitch below is an "
            "LLM-generated narrative over this gap candidate, not a validated "
            "research direction. Nothing is promoted without an explicit "
            "approve in the 🗂️ Review queue tab."
        )
        if st.button("💡 Generate hypothesis pitch", key=f"hyp_{sel_gap_idx}"):
            v = get_velocity_analysis(None, None)
            trend_context = [k.concept for k in v.keywords if k.velocity > 0][:5]
            with st.spinner("Generating + verifying pitch locally…"):
                try:
                    pitch = hypothesis.generate_hypothesis(
                        sel_gap, g, trend_context=trend_context
                    )
                    conn = db.connect()
                    try:
                        db.add_to_review_queue(
                            conn, gap_a_label=sel_gap.a.label, gap_b_label=sel_gap.b.label,
                            title=pitch.title, claim=pitch.claim,
                            method_sketch=pitch.method_sketch, datasets=pitch.datasets,
                            supporting_paper_ids=pitch.supporting_paper_ids,
                        )
                    finally:
                        conn.close()
                    st.session_state["last_pitch"] = (sel_gap_idx, pitch)
                except llm.OllamaError as exc:
                    st.error(f"Ollama error: {exc}")
                except hypothesis.VerificationError as exc:
                    st.error(f"Verifier rejected every attempt: {exc}")

        _stored = st.session_state.get("last_pitch")
        if _stored is not None and _stored[0] == sel_gap_idx:
            last_pitch = _stored[1]
            with st.container(border=True):
                st.markdown(
                    '<span class="ax-draft-badge">✦ Draft hypothesis</span>'
                    f'<div class="ax-draft-title">{html.escape(last_pitch.title)}</div>',
                    unsafe_allow_html=True,
                )
                st.write(last_pitch.claim)
                st.markdown(f"**Method sketch:** {last_pitch.method_sketch}")
                if last_pitch.datasets:
                    st.markdown("**Datasets:** " + ", ".join(f"`{d}`" for d in last_pitch.datasets))
                st.markdown(
                    "**Supporting papers:** " +
                    ", ".join(f"`{pid}`" for pid in last_pitch.supporting_paper_ids)
                )
                st.caption(f"⚠️ {last_pitch.disclaimer} Sent to the Review queue as pending.")

        st.markdown(f"##### Gap map — {len(analysis.gaps)} candidate gaps")
        gm_mode_col, gm_opt_col = st.columns([1, 2])
        with gm_mode_col:
            gap_map_mode = st.radio(
                "Display as", ["Ring diagram", "Ranked list", "Matrix"],
                horizontal=True, key="gap_map_mode",
                help="Ring diagram: sub-topics on a ring, arcs bridge related-but-"
                     "disconnected pairs (arcs can cross). Ranked list: same gaps "
                     "as plain bars, no crossing lines. Matrix: sub-topics on both "
                     "axes, cell brightness = G-score — no lines to trace at all.",
            )
        if gap_map_mode == "Ring diagram":
            with gm_opt_col:
                topk_col, light_col = st.columns([2, 1])
                with topk_col:
                    max_drawable = len(analysis.gaps)
                    top_k_show = st.slider(
                        "Gaps to draw", min(3, max_drawable), max_drawable,
                        min(6, max_drawable), key="gap_map_topk",
                        help="Drawing all candidate gaps at once tangles the arcs "
                             "into a hairball — showing just the top few by "
                             "G-score keeps it readable.",
                    ) if max_drawable > 3 else max_drawable
                with light_col:
                    gap_map_light = st.toggle("Light background", value=False,
                                               key="gap_map_light")
            st.caption(
                "Each sub-topic on the ring; the drawn candidate gaps are arcs "
                "bridging two of them, strongest G-score first. The gap picked "
                "above is lit gold. Hover any arc or dot for details."
            )
            show_indices = list(range(top_k_show))
            components.html(
                f'<div style="width:100%">'
                f'{_gap_map_svg(analysis, sel_gap_idx, dark=not gap_map_light, show_indices=show_indices)}'
                f'</div>',
                height=480, scrolling=False,
            )
        elif gap_map_mode == "Ranked list":
            st.caption(
                "All candidate gaps, ranked by G-score. Gold = the gap picked "
                "above, green = meets the calibrated threshold, amber = below it."
            )
            _gap_bar_chart(analysis, sel_gap_idx)
        else:
            st.caption(
                "Sub-topics on both axes; a filled cell is a candidate gap, "
                "darker = higher G-score. Gold outline = the gap picked above. "
                "Blank cells weren't ranked as candidates."
            )
            _gap_matrix_chart(analysis, sel_gap_idx)

        with st.expander(f"All {len(analysis.gaps)} candidate gaps (ranked, as text)"):
            for i, gp in enumerate(analysis.gaps, 1):
                st.markdown(
                    f"{i}. **{_short(gp.a)}** ⟷ **{_short(gp.b)}** — "
                    f"semantic sim `{gp.semantic_similarity:.2f}`, "
                    f"`{gp.inter_citations}` citations between them"
                )

    st.divider()
    st.markdown("##### Influence ranking")
    n = st.slider("How many to rank", min_value=5, max_value=50, value=20, step=5)
    ranking = get_influence_ranking(50)[:n]

    for i, r in enumerate(ranking, 1):
        with st.expander(f"{i}.  {r.title}"):
            _chips(
                (f"PR {r.pagerank:.4f}", "score"),
                (f"📅 {r.year}", ""),
                (f"{r.local_in_degree:,} local cites", ""),
                (f"{r.cited_by_count:,} global cites", ""),
            )
            m = meta.get(r.paper_id, {})
            if m.get("doi"):
                st.markdown(f"[↗ Open paper (DOI)](https://doi.org/{m['doi']})")
            nb = graph.neighbors(g, r.paper_id)
            ext = sum(1 for x in nb.references if not x["in_corpus"])
            st.markdown(
                f"**Cited by {len(nb.cited_by)} papers in the corpus** · "
                f"cites {len(nb.references)} works ({ext} outside the corpus)"
            )
            if nb.cited_by:
                st.markdown("**Top corpus papers citing this:**")
                for c in nb.cited_by[:8]:
                    st.markdown(f"- {c['title']}  _({c['year']} · {c['cited_by_count']} cites)_")


# --- Trending tab -------------------------------------------------------------
def _velocity_bar_chart(items: list, color: str) -> None:
    """One horizontal bar per concept, input order preserved (items arrive sorted).

    Bars diverge from a shared centered zero baseline. Each bar is labelled at its
    tip with its share change as a percentage (2**velocity - 1), so risers read as
    +N% and faders as -N%. The raw velocity stays in the tooltip only.
    """
    # log2 share-ratio -> percent change in corpus share.
    pct = [(2 ** k.velocity - 1) * 100.0 for k in items]
    df = pd.DataFrame(
        {"concept": [k.concept for k in items],
         "velocity": [k.velocity for k in items],
         "pct": pct,
         "pct_label": [f"{p:+.0f}%" for p in pct]}
    )
    # Symmetric domain so the zero baseline sits in the visual center and the
    # magnitude of a rise reads the same as an equal fade.
    bound = max((abs(v) for v in df["velocity"]), default=1.0) * 1.15 or 1.0
    xscale = alt.Scale(domain=[-bound, bound])
    # All items in one chart share a sign (risers positive, faders negative), so
    # the labels sit just beyond every bar tip: past the right end for risers,
    # past the left end for faders.
    rising = (df["velocity"] >= 0).all()
    align, dx = ("left", 12) if rising else ("right", -12)

    base = alt.Chart(df).encode(
        y=alt.Y("concept:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=220)),  # sort=None => keep input order
    )
    bars = base.mark_bar(color=color).encode(
        x=alt.X("velocity:Q",
                title="share change (velocity = log2 share ratio)",
                scale=xscale),
        tooltip=["concept",
                 alt.Tooltip("pct:Q", title="share change", format="+.1f"),
                 alt.Tooltip("velocity:Q", title="velocity", format="+.2f")],
    )
    pct_labels = base.mark_text(
        align=align, baseline="middle", dx=dx, fontWeight="bold",
    ).encode(x=alt.X("velocity:Q", scale=xscale), text="pct_label:N")

    chart = (bars + pct_labels).properties(height=alt.Step(30))
    st.altair_chart(chart, use_container_width=True)


def render_trending() -> None:
    """Concepts ranked by velocity: normalized-frequency log2-ratio, recent vs prior window."""
    st.subheader("Trending concepts")
    st.caption(
        "Concepts ranked by velocity: how much their share of the corpus has "
        "changed between the older and newer half of the available years. "
        "Positive = rising, negative = fading. Counts below "
        f"{config.VELOCITY_MIN_FREQ} recent papers are flagged low-confidence."
    )
    st.info(
        "ℹ️ **Heuristic signal** — velocity ranks concepts by corpus frequency "
        "change, not by verified research impact. Treat as directional, not definitive."
    )

    venues, bounds = get_filter_options()
    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        venue_choice = st.selectbox("Venue", ["All venues"] + venues, index=0,
                                    key="trend_venue")
        venue = None if venue_choice == "All venues" else venue_choice
    with fcol2:
        # Pin the earliest selectable year to the modern floor so velocity
        # windows never span the noisy pre-2020 references (config note).
        year_range = render_year_filter(bounds, "trend", min_year=config.VELOCITY_MIN_YEAR)

    analysis = get_velocity_analysis(venue, year_range)
    if not analysis.keywords:
        _empty_state("📉", "Not enough dated papers",
                     "This venue/year filter doesn't have enough dated papers to "
                     "compute velocity. Widen the filter to see trends.")
        return
    if analysis.insufficient_year_spread:
        st.warning("Selected range spans a single year — velocity needs at least two years to compare.")

    st.caption(
        f"Prior window {analysis.prior_window[0]}–{analysis.prior_window[1]} "
        f"({analysis.total_prior} papers) → Recent window "
        f"{analysis.recent_window[0]}–{analysis.recent_window[1]} "
        f"({analysis.total_recent} papers)"
    )

    # Charts show only meaningful movers. A concept must have >= MIN papers in
    # BOTH windows: without a real base in the prior window, a 0->N concept's
    # share-change % is dominated by the epsilon smoothing constant (e.g. +13000%)
    # and those degenerate values crowd out genuine risers. Requiring presence in
    # both windows yields real growth/decline ratios. The full ranked list below
    # still shows everything, including brand-new/vanished concepts.
    min_ct = config.VELOCITY_MIN_CHART_COUNT
    rising = [k for k in analysis.keywords
              if k.velocity > 0 and k.prior_count >= min_ct and k.recent_count >= min_ct][:15]
    fading = [k for k in reversed(analysis.keywords)
              if k.velocity < 0 and k.prior_count >= min_ct and k.recent_count >= min_ct][:10]

    if rising or fading:
        st.caption(
            f"Charts show concepts with ≥{min_ct} papers in *both* windows "
            "(real base → meaningful % change); brand-new or vanished concepts "
            "are excluded here but still listed below."
        )
    else:
        st.info(
            f"No concept reaches ≥{min_ct} papers in a window for this filter — "
            "the corpus is too sparse to chart a trend. See the ranked list below."
        )
    if rising:
        st.markdown("##### Top risers")
        _velocity_bar_chart(rising, color="#4c78a8")
    if fading:
        st.markdown("##### Top faders")
        _velocity_bar_chart(fading, color="#d62728")

    st.markdown("##### Ranked keywords")
    rows_html = []
    for i, k in enumerate(analysis.keywords, 1):
        vel_cls = "grow" if k.velocity >= 0 else "fade"
        warn = '<span class="ax-chip warn">⚠ low volume</span>' if k.low_confidence else ""
        rows_html.append(
            '<div class="ax-listrow">'
            f'<span class="ax-rank">{i}</span>'
            f'<span class="ax-name">{html.escape(k.concept)}</span>'
            f'{warn}'
            f'<span class="ax-chip">{k.prior_count} → {k.recent_count} papers</span>'
            f'<span class="ax-chip {vel_cls}">{k.velocity:+.2f}</span>'
            '</div>'
        )
    st.markdown(f'<div class="ax-list">{"".join(rows_html)}</div>', unsafe_allow_html=True)


# --- Reading list tab ---------------------------------------------------------
def render_reading_list() -> None:
    """Bookmarked papers, with 3-bullet local-LLM summaries (OD14)."""
    st.subheader("Reading list")
    st.caption(
        f"Papers bookmarked from Search. Summaries are generated locally via "
        f"Ollama (`{config.OLLAMA_MODEL}`, OD14) — grounded only in the "
        f"paper's own abstract, cached after the first request."
    )
    conn = db.connect()
    try:
        rows = db.list_bookmarks(conn)
    finally:
        conn.close()

    if not rows:
        _empty_state("📚", "Your reading list is empty",
                     "Bookmark papers from a **Search** result card and they'll "
                     "collect here for local summarization.")
        return

    st.caption(f"{len(rows)} bookmarked paper{'s' if len(rows) != 1 else ''}")
    for r in rows:
        with st.expander(r["title"]):
            _chips(
                (f"📅 {r['publication_year']}", ""),
                (f"🏛 {html.escape(str(r['venue']))}", ""),
                (f"❝ {r['cited_by_count']:,} cites", ""),
            )
            if r["doi"]:
                st.markdown(f"[↗ Open paper (DOI)](https://doi.org/{r['doi']})")
            st.write(r["abstract"] or "_No abstract available._")

            conn3 = db.connect()
            try:
                bullets = db.get_summary(conn3, r["paper_id"])
            finally:
                conn3.close()

            bcol, scol = st.columns([1, 1])
            with bcol:
                if st.button("📚 Remove bookmark", key=f"rl_unbm_{r['paper_id']}"):
                    toggle_bookmark(r["paper_id"], add=False)
                    st.rerun()

            if bullets:
                st.markdown("**LLM-generated Summary:**")
                for b in bullets:
                    st.markdown(f"- {b}  _(cites `{r['paper_id']}`)_")
            else:
                with scol:
                    if st.button("🧠 Summarize", key=f"rl_sum_{r['paper_id']}"):
                        if not r["abstract"]:
                            st.error("No abstract available to summarize.")
                        else:
                            with st.spinner("Summarizing locally…"):
                                try:
                                    result = summarize.summarize_paper(
                                        r["paper_id"], r["title"], r["abstract"]
                                    )
                                    conn2 = db.connect()
                                    try:
                                        db.save_summary(conn2, r["paper_id"],
                                                         result.bullets, config.OLLAMA_MODEL)
                                    finally:
                                        conn2.close()
                                    st.rerun()
                                except llm.OllamaError as exc:
                                    st.error(f"Ollama error: {exc}")


# --- Review queue tab (OD16) ---------------------------------------------------
def render_review_queue() -> None:
    """HITL queue for hypothesis pitches (Task 5.2, OD16). Nothing auto-promotes."""
    st.subheader("Review queue")
    st.caption(
        "Hypothesis pitches generated from the Research-gaps view. Every item "
        "starts **pending** — approve or reject explicitly; nothing is "
        "promoted automatically."
    )
    status_filter = st.radio("Filter", ["pending", "approved", "rejected", "all"],
                             horizontal=True)
    conn = db.connect()
    try:
        rows = db.list_review_queue(conn, status=None if status_filter == "all" else status_filter)
    finally:
        conn.close()

    if not rows:
        _empty_state("🗂️", f"No {status_filter} pitches",
                     "Generate a hypothesis pitch from the **Citation graph** → "
                     "Research-gaps view and it will queue here for review.")
        return

    for r in rows:
        try:
            datasets = json.loads(r["datasets_json"] or "[]")
            supporting = json.loads(r["supporting_ids_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            datasets, supporting = [], []

        badge = {"pending": "🟡", "approved": "🟢", "rejected": "🔴"}[r["status"]]
        with st.expander(f"{badge} {r['title']}  ({r['status']})"):
            st.caption(f"{r['gap_a_label']} ⟷ {r['gap_b_label']}")
            st.write(r["claim"])
            st.markdown(f"**Method sketch:** {r['method_sketch']}")
            if datasets:
                st.markdown("**Datasets:** " + ", ".join(f"`{d}`" for d in datasets))
            st.markdown("**Supporting papers:** " + ", ".join(f"`{p}`" for p in supporting))
            st.warning("⚠️ Unverified Candidate — not a validated research gap.")
            if r["status"] == "pending":
                acol, rcol = st.columns(2)
                with acol:
                    if st.button("✅ Approve", key=f"approve_{r['id']}"):
                        conn2 = db.connect()
                        try:
                            db.set_review_status(conn2, r["id"], "approved")
                        finally:
                            conn2.close()
                        st.rerun()
                with rcol:
                    if st.button("❌ Reject", key=f"reject_{r['id']}"):
                        conn2 = db.connect()
                        try:
                            db.set_review_status(conn2, r["id"], "rejected")
                        finally:
                            conn2.close()
                        st.rerun()


# --- Tab dispatch ------------------------------------------------------------
tab_graph, tab_trending, tab_search, tab_reading, tab_review = st.tabs(
    ["🕸️ Citation graph", "📈 Trending", "🔍 Search", "📚 Reading list", "🗂️ Review queue"]
)
with tab_graph:
    render_graph_view()
with tab_trending:
    render_trending()
with tab_search:
    render_search()
with tab_reading:
    render_reading_list()
with tab_review:
    render_review_queue()
