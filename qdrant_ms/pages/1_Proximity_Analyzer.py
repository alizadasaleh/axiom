import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from qdrant_utils import (
    concepts_list,
    cosine_similarity,
    extract_vector,
    fetch_paper_by_title,
    get_all_paper_titles,
    init_connections,
    nearest_to_vector,
)

# -----------------------------------------------------------------------------
# CONFIGURATION & INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AXIOM — Paper Similarity", layout="wide",
                   initial_sidebar_state="expanded")

try:
    client = init_connections()
except Exception as e:
    st.error(f"Could not connect to Qdrant on localhost:6333. Is Docker running? Error: {e}")
    st.stop()


def interpret(sim):
    """Plain-language reading of a cosine similarity score."""
    if sim >= 0.82:
        return "Near-duplicate territory", "These two papers sit in the same tight thematic cluster — they likely cite each other or cover the same problem.", "🟢"
    if sim >= 0.65:
        return "Closely related", "Same broad field, different angles. Strong candidates for a literature-review pairing.", "🟢"
    if sim >= 0.45:
        return "Loosely related", "Adjacent areas that share some vocabulary but tackle different problems. This is where interesting bridges can live.", "🟡"
    return "Distant", "Little semantic overlap. If a bridge between them were plausible, the space between is largely unwritten — a potential gap.", "🔴"


# -----------------------------------------------------------------------------
# HEADER + SELECTION
# -----------------------------------------------------------------------------
st.title("🔗 Paper Similarity Analyzer")
st.caption(
    "Pick any two papers to measure how close they are in meaning — and see the "
    "research neighbourhood around each, plus what (if anything) currently bridges them."
)

paper_options = get_all_paper_titles(client)
if not paper_options:
    st.warning("⚠️ No records found in the Qdrant collection. Run the ingestion pipeline first.")
    st.stop()

st.sidebar.header("🎯 Choose two papers")
st.sidebar.caption("Type to filter the titles.")
selected_title_a = st.sidebar.selectbox("Paper A", paper_options, index=0)
default_index_b = min(1, len(paper_options) - 1)
selected_title_b = st.sidebar.selectbox("Paper B", paper_options, index=default_index_b)
neighbours_k = st.sidebar.slider("Neighbours to map per paper", 3, 15, 8)

if selected_title_a == selected_title_b:
    st.info("Pick two *different* papers to compare.")
    st.stop()

paper_a = fetch_paper_by_title(client, selected_title_a)
paper_b = fetch_paper_by_title(client, selected_title_b)
if not (paper_a and paper_b):
    st.error("Could not load one of the selected papers from Qdrant.")
    st.stop()

raw_a, raw_b = extract_vector(paper_a), extract_vector(paper_b)
if raw_a is None or raw_b is None:
    st.error("One of the selected papers has no stored dense vector. Re-run ingestion, then reload.")
    st.stop()
vec_a = np.asarray(raw_a, dtype=np.float32)
vec_b = np.asarray(raw_b, dtype=np.float32)
similarity = cosine_similarity(vec_a, vec_b)

# -----------------------------------------------------------------------------
# 1. HEADLINE SCORE + VERDICT
# -----------------------------------------------------------------------------
label, blurb, emoji = interpret(similarity)
c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Cosine similarity", f"{similarity:.3f}", help="1.0 = identical meaning, 0 = unrelated.")
    st.progress(max(0.0, min(1.0, similarity)))
with c2:
    st.markdown(f"### {emoji} {label}")
    st.write(blurb)

# Shared concepts
con_a = concepts_list(paper_a.payload)
con_b = concepts_list(paper_b.payload)
shared = sorted(set(c.lower() for c in con_a) & set(c.lower() for c in con_b))
if shared:
    st.markdown("**Shared concepts:** " + " ".join(f"`{c}`" for c in shared))
else:
    st.caption("No concepts in common — any relationship is semantic rather than tag-based.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 2. NEIGHBOURHOOD MAP  (this is what makes the score legible)
# -----------------------------------------------------------------------------
st.markdown("### 🗺️ Where they sit in the research landscape")
st.caption(
    "Each paper's nearest neighbours are projected into 2-D alongside the two "
    "anchors. Overlapping clouds = related fields; a clear gap between the clouds "
    "is the unwritten space between them."
)

neigh_a = nearest_to_vector(client, vec_a, limit=neighbours_k, exclude_titles=[selected_title_a])
neigh_b = nearest_to_vector(client, vec_b, limit=neighbours_k, exclude_titles=[selected_title_b])

points, meta = [], []
points.append(vec_a); meta.append(("Paper A", selected_title_a, "anchor_a"))
points.append(vec_b); meta.append(("Paper B", selected_title_b, "anchor_b"))
for h in neigh_a:
    hv = extract_vector(h)
    if hv is not None:
        points.append(np.asarray(hv, dtype=np.float32))
        meta.append(("A-neighbour", h.payload.get("title", "Untitled"), "neigh_a"))
for h in neigh_b:
    hv = extract_vector(h)
    if hv is not None:
        points.append(np.asarray(hv, dtype=np.float32))
        meta.append(("B-neighbour", h.payload.get("title", "Untitled"), "neigh_b"))

if len(points) >= 3:
    coords = PCA(n_components=2).fit_transform(np.vstack(points))
    style = {
        "anchor_a": dict(color="#00cc96", size=18, symbol="star"),
        "anchor_b": dict(color="#ab63fa", size=18, symbol="star"),
        "neigh_a": dict(color="#00cc96", size=9, symbol="circle"),
        "neigh_b": dict(color="#ab63fa", size=9, symbol="circle"),
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[coords[0, 0], coords[1, 0]], y=[coords[0, 1], coords[1, 1]],
        mode="lines", line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
        hoverinfo="none", showlegend=False,
    ))
    for group, legend in [("neigh_a", "Paper A neighbourhood"), ("neigh_b", "Paper B neighbourhood"),
                          ("anchor_a", "Paper A"), ("anchor_b", "Paper B")]:
        idx = [i for i, m in enumerate(meta) if m[2] == group]
        if not idx:
            continue
        fig.add_trace(go.Scatter(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers", name=legend,
            marker=style[group], text=[meta[i][1] for i in idx],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig.update_layout(
        template="plotly_dark", height=430, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough neighbours in the collection to draw a landscape map.")

# -----------------------------------------------------------------------------
# 3. BRIDGE PAPERS  (the direct link to gap discovery)
# -----------------------------------------------------------------------------
st.markdown("### 🌉 What currently sits between them?")
midpoint = (vec_a + vec_b) / 2.0
bridges = nearest_to_vector(
    client, midpoint, limit=5, exclude_titles=[selected_title_a, selected_title_b], with_vectors=False
)
if bridges:
    top_bridge_sim = bridges[0].score
    if top_bridge_sim >= 0.72:
        st.success(
            f"The midpoint between these two topics is **well populated** — existing "
            f"work already connects them (closest bridge similarity {top_bridge_sim:.2f})."
        )
    else:
        st.warning(
            f"The conceptual midpoint is **sparsely populated** (closest paper only "
            f"{top_bridge_sim:.2f} similar). The space between these two papers is a candidate gap — "
            f"see the **Gap Discovery** page for the corpus-wide view."
        )
    for h in bridges:
        st.markdown(
            f"- **{h.payload.get('title','Untitled')}**  \n"
            f"  <sub>midpoint similarity `{h.score:.3f}` · "
            f"{h.payload.get('venue','N/A')} {h.payload.get('year','')}</sub>",
            unsafe_allow_html=True,
        )
else:
    st.info("No bridge papers found near the midpoint.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. SIDE-BY-SIDE METADATA
# -----------------------------------------------------------------------------
st.markdown("### 📑 The two papers")
left, right = st.columns(2)
for col, paper, dot in [(left, paper_a, "🟢"), (right, paper_b, "🟣")]:
    p = paper.payload
    with col:
        st.markdown(f"#### {dot} {p.get('title')}")
        st.markdown(f"**Venue:** `{p.get('venue', 'N/A')}` &nbsp; **Year:** `{p.get('year', 'N/A')}`",
                    unsafe_allow_html=True)
        st.markdown(f"**Citations:** `{p.get('cited_by_count', 0):,}`")
        cons = concepts_list(p)
        if cons:
            st.markdown("**Concepts:** " + ", ".join(f"*{c}*" for c in cons[:12]))
