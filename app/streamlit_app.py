"""Axiom P1 search page.

Free-text query -> SPECTER2 encode -> Qdrant search (hybrid dense+sparse by
default, or dense-only via the toggle), with a venue/year filter bar. Results
render as expandable cards (title/DOI, concepts, abstract) and each card can
pivot to "similar papers". Functionality only; no styling polish.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repo root importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Make the app/ directory importable so sibling modules (vector_tools) resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from axiom import config, db, gaps, graph, hypothesis, llm, summarize, velocity
from axiom.embed import Specter2Encoder
from axiom.qdrant_client import AxiomQdrant, SearchHit
from vector_tools import render_proximity_analyzer, render_vector_gap_discovery

st.set_page_config(page_title="Axiom — Thesis Discovery", layout="wide")
st.title("Axiom — Research Trends & Gaps Discovery")
st.caption(
    f"Collection `{config.COLLECTION_NAME}` · model `{config.MODEL_ID}` · "
    "ACL Anthology corpus (ACL/EMNLP/COLING/NAACL, 2020–2025)"
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
def render_hits(hits: list[SearchHit], *, score_label: str = "score") -> None:
    """Render hits as expandable cards with concepts, abstract, and a pivot."""
    if not hits:
        st.info("No papers match the current filters. Loosen the venue/year filters.")
        return
    meta = get_meta()
    bookmarked = get_bookmarked_ids()
    st.write(f"**{len(hits)} results**")
    for h in hits:
        m = meta.get(h.paper_id, {})
        label = f"{h.score:.4f} ({score_label}) · {h.title}  ({h.year} · {h.venue} · {h.cited_by_count} cites)"
        with st.expander(label):
            if m.get("doi"):
                st.markdown(f"[Open paper (DOI)](https://doi.org/{m['doi']})")
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
        st.warning(qdrant_msg)
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
            st.info("Enter a query above to search the corpus.")


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
        st.warning(qdrant_msg)
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
        st.info("No in-corpus citation edges yet — ingest a connected corpus first.")
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
            st.markdown(f"##### {last_pitch.title}")
            st.write(last_pitch.claim)
            st.markdown(f"**Method sketch:** {last_pitch.method_sketch}")
            if last_pitch.datasets:
                st.markdown("**Datasets:** " + ", ".join(f"`{d}`" for d in last_pitch.datasets))
            st.markdown(
                "**Supporting papers:** " +
                ", ".join(f"`{pid}`" for pid in last_pitch.supporting_paper_ids)
            )
            st.caption(f"⚠️ {last_pitch.disclaimer} Sent to the Review queue as pending.")

        with st.expander(f"All {len(analysis.gaps)} candidate gaps (ranked)"):
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
        label = (f"{i}. {r.title}  ·  PR={r.pagerank:.4f}  ·  "
                 f"{r.local_in_degree} local / {r.cited_by_count} global cites  ·  {r.year}")
        with st.expander(label):
            m = meta.get(r.paper_id, {})
            if m.get("doi"):
                st.markdown(f"[Open paper (DOI)](https://doi.org/{m['doi']})")
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
        st.info("Not enough dated papers to compute velocity for this filter.")
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
    for i, k in enumerate(analysis.keywords, 1):
        flag = " ⚠️ low-volume" if k.low_confidence else ""
        st.markdown(
            f"{i}. **{k.concept}** — velocity `{k.velocity:+.2f}`{flag}  "
            f"({k.prior_count} → {k.recent_count} papers)"
        )


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
        st.info("No bookmarks yet — add one from a Search result card.")
        return

    st.write(f"**{len(rows)} bookmarked papers**")
    for r in rows:
        label = f"{r['title']}  ({r['publication_year']} · {r['venue']} · {r['cited_by_count']} cites)"
        with st.expander(label):
            if r["doi"]:
                st.markdown(f"[Open paper (DOI)](https://doi.org/{r['doi']})")
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
        st.info(f"No {status_filter} items.")
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


# --- Vector tools tab --------------------------------------------------------
def render_vector_tools() -> None:
    """Embedding-geometry analysis: similarity + KMeans gap discovery (PR #3)."""
    if not qdrant_ok:
        st.warning(qdrant_msg)
        return

    st.info(
        "**Two gap views are available in Axiom:**  \n"
        "- 🕸️ **Research gaps** (Citation graph tab) — OD9: Louvain community "
        "detection on the citation graph + semantic centroids + weak-citation "
        "scoring. Uses citation structure to validate candidate gaps.  \n"
        "- 🔗 **Vector Gap Discovery** (this tab) — KMeans topic clustering + "
        "bridge-paper scan on raw embedding geometry only. No citation data used. "
        "Complementary signal, not a replacement."
    )

    st.subheader("🔗 Paper Similarity Analyzer")
    st.caption(
        "Pick any two papers to measure how close they are in meaning, visualise "
        "each paper's research neighbourhood, and see what (if anything) currently "
        "bridges the space between them."
    )
    render_proximity_analyzer(store)

    st.divider()

    st.subheader("🕳️ Vector Gap Discovery")
    st.caption(
        "Groups every paper into topics (KMeans) and finds pairs of related topics "
        "with no paper bridging them — the places new research could go."
    )
    render_vector_gap_discovery(store)


# --- Tab dispatch ------------------------------------------------------------
tab_graph, tab_trending, tab_search, tab_reading, tab_review, tab_vector = st.tabs(
    ["🕸️ Citation graph", "📈 Trending", "🔍 Search",
     "📚 Reading list", "🗂️ Review queue", "🔗 Vector tools"]
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
with tab_vector:
    render_vector_tools()
