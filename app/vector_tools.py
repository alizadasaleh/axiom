"""Vector analysis tools: Paper Similarity Analyzer and Vector Gap Discovery.

Adapted from qdrant_ms/pages/ (PR #3) for integration into the main Axiom UI.
These tools read stored Qdrant vectors directly — no embedding model needed —
using the same axiom_v1 collection and QdrantClient as the main platform.

Gap algorithm note
------------------
The "Research gaps" tab (Citation graph) uses OD9: Louvain community detection
on the citation graph + semantic centroids + weak-citation scoring. The Vector
Gap Discovery section here uses a different method: KMeans topic clustering +
bridge-paper scan on raw vector geometry (no citation structure). Both views
are retained because they surface complementary signals.
"""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from qdrant_client.models import FieldCondition, Filter, MatchValue, NamedVector
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from axiom import config
from axiom.qdrant_client import AxiomQdrant


# ---------------------------------------------------------------------------
# Internal helpers (adapted from qdrant_ms/qdrant_utils.py)
# ---------------------------------------------------------------------------

def _extract_vector(record) -> list[float] | None:
    """Extract the dense embedding as a plain float list from a Qdrant point."""
    if record is None:
        return None
    vec = getattr(record, "vector", None)
    if isinstance(vec, dict):
        dense = vec.get(config.DENSE_VECTOR_NAME)
        if dense is None:
            dense = next(
                (v for v in vec.values() if isinstance(v, (list, tuple, np.ndarray))),
                None,
            )
        return list(dense) if dense is not None else None
    if isinstance(vec, (list, tuple, np.ndarray)):
        return list(vec)
    return None


def _concepts_list(payload) -> list[str]:
    """Normalise the `concepts` payload field to a clean list of strings."""
    c = payload.get("concepts", []) if payload else []
    if isinstance(c, list):
        return [str(x).strip() for x in c if str(x).strip()]
    if isinstance(c, str):
        return [x.strip() for x in c.split(",") if x.strip()]
    return []


def _cosine_similarity(v1, v2) -> float:
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom else 0.0


@st.cache_data(ttl=300)
def _get_all_paper_titles(_raw_client) -> list[str]:
    """Scroll payloads to pull all titles for the paper selectboxes.

    `_raw_client` has a leading underscore so st.cache_data skips hashing it.
    Cache key is effectively empty — result is reused for all calls within TTL.
    """
    try:
        records, _ = _raw_client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=5000,
            with_payload=["title"],
            with_vectors=False,
        )
        return sorted(
            {r.payload["title"] for r in records if r.payload and r.payload.get("title")}
        )
    except Exception as exc:
        st.error(f"Error fetching titles: {exc}")
        return []


@st.cache_data(ttl=60)
def _fetch_paper_by_title(_raw_client, title: str):
    """Retrieve the full record (dense vector + payload) for an exact title."""
    records, _ = _raw_client.scroll(
        collection_name=config.COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="title", match=MatchValue(value=title))]
        ),
        limit=1,
        with_vectors=True,
        with_payload=True,
    )
    return records[0] if records else None


def _nearest_to_vector(raw_client, vector, limit: int = 5,
                        exclude_titles=None, with_vectors: bool = True):
    """Dense nearest neighbours of an arbitrary point in vector space."""
    exclude_titles = set(exclude_titles or [])
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    hits = raw_client.search(
        collection_name=config.COLLECTION_NAME,
        query_vector=NamedVector(name=config.DENSE_VECTOR_NAME, vector=vector),
        limit=limit + len(exclude_titles),
        with_payload=True,
        with_vectors=with_vectors,
    )
    out = []
    for h in hits:
        if h.payload and h.payload.get("title") in exclude_titles:
            continue
        out.append(h)
        if len(out) >= limit:
            break
    return out


@st.cache_data(ttl=600, show_spinner="Loading corpus vectors from Qdrant…")
def _load_corpus(_raw_client, max_points: int = 20000) -> dict:
    """Scroll the whole collection once; return dense vectors + payloads."""
    ids, vectors, titles, payloads = [], [], [], []
    offset = None
    while len(ids) < max_points:
        records, offset = _raw_client.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=1000,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if not records:
            break
        for r in records:
            vec = _extract_vector(r)
            if vec is None:
                continue
            ids.append(r.id)
            vectors.append(vec)
            payload = r.payload or {}
            titles.append(payload.get("title", "Untitled"))
            payloads.append(payload)
        if offset is None:
            break
    return {
        "ids": ids,
        "vectors": np.asarray(vectors, dtype=np.float32),
        "titles": titles,
        "payloads": payloads,
    }


@st.cache_data(ttl=600, show_spinner="Clustering corpus into topics…")
def _cluster_corpus(_raw_client, k: int) -> dict:
    """Partition every paper into `k` topics; precompute everything the gap scan needs."""
    corpus = _load_corpus(_raw_client)
    vectors = corpus["vectors"]
    payloads = corpus["payloads"]
    titles = corpus["titles"]
    if len(vectors) < k:
        k = max(2, len(vectors))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(vectors)

    vn = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    cn = km.cluster_centers_ / (
        np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True) + 1e-9
    )

    coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
    pca_full = PCA(n_components=2, random_state=42).fit(vectors)

    topics = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            topics.append({"id": c, "size": 0, "label": f"Topic {c}", "keywords": [], "rep_title": ""})
            continue
        rep = members[int(np.argmax(vn[members] @ cn[c]))]
        con_counter: Counter = Counter()
        for i in members:
            for con in _concepts_list(payloads[i]):
                con_counter[con] += 1
        top_con = [w for w, _ in con_counter.most_common(4)]
        topics.append({
            "id": c,
            "size": int(len(members)),
            "label": ", ".join(top_con) if top_con else titles[rep][:50],
            "concepts": top_con,
            "rep_title": titles[rep],
        })

    cc = cn @ cn.T
    pair_sims = [cc[a, b] for a, b in itertools.combinations(range(k), 2)]

    return {
        "labels": labels,
        "vn": vn,
        "cn": cn,
        "cc": cc,
        "coords": coords,
        "pca_full": pca_full,
        "topics": topics,
        "titles": titles,
        "payloads": payloads,
        "sim_pct": {
            "p40": float(np.percentile(pair_sims, 40)),
            "p88": float(np.percentile(pair_sims, 88)),
        },
    }


@st.cache_data(ttl=600, show_spinner="Scanning topic pairs for structural gaps…")
def _find_topic_gaps(
    _raw_client, k: int, bridge_min: float, bridge_max: float, top_n: int
) -> list[dict]:
    """Find topic pairs that are semantically related but lack bridging papers."""
    d = _cluster_corpus(_raw_client, k)
    vn, cn, cc, topics = d["vn"], d["cn"], d["cc"], d["topics"]
    S = vn @ cn.T  # (N papers) × (k topics) cosine

    gap_list = []
    for a, b in itertools.combinations(range(len(cn)), 2):
        if topics[a]["size"] == 0 or topics[b]["size"] == 0:
            continue
        cs = float(cc[a, b])
        if not (bridge_min <= cs <= bridge_max):
            continue
        joint = np.minimum(S[:, a], S[:, b])
        best = int(np.argmax(joint))
        bridge_strength = float(joint[best])
        denom = max(1e-6, 1.0 - cs)
        bridge_ratio = (bridge_strength - cs) / denom  # 0 = only edges, 1 = fully bridged

        midpoint = (cn[a] + cn[b]) / 2.0
        mid2d = d["pca_full"].transform(midpoint.reshape(1, -1))[0]

        gap_list.append({
            "topic_a": topics[a],
            "topic_b": topics[b],
            "centroid_sim": cs,
            "bridge_ratio": bridge_ratio,
            "coverage": float(np.clip(bridge_ratio, 0, 1)),
            "gap_score": float(np.clip(1 - bridge_ratio, 0, 1)),
            "best_bridge": d["payloads"][best],
            "midpoint_2d": mid2d.tolist(),
        })

    gap_list.sort(key=lambda g: g["bridge_ratio"])  # emptiest first
    return gap_list[:top_n]


# ---------------------------------------------------------------------------
# Paper Similarity Analyzer
# ---------------------------------------------------------------------------

def _interpret(sim: float) -> tuple[str, str, str]:
    """Plain-language reading of a cosine similarity score."""
    if sim >= 0.82:
        return (
            "Near-duplicate territory",
            "These two papers sit in the same tight thematic cluster — they likely "
            "cite each other or cover the same problem.",
            "🟢",
        )
    if sim >= 0.65:
        return (
            "Closely related",
            "Same broad field, different angles. Strong candidates for a "
            "literature-review pairing.",
            "🟢",
        )
    if sim >= 0.45:
        return (
            "Loosely related",
            "Adjacent areas that share some vocabulary but tackle different problems. "
            "This is where interesting bridges can live.",
            "🟡",
        )
    return (
        "Distant",
        "Little semantic overlap. If a bridge between them were plausible, the space "
        "between is largely unwritten — a potential gap.",
        "🔴",
    )


def render_proximity_analyzer(store: AxiomQdrant) -> None:
    """Pairwise paper similarity: cosine score, neighbourhood map, bridge papers.

    Adapted from qdrant_ms/pages/1_Proximity_Analyzer.py (PR #3).
    """
    raw_client = store.client

    paper_options = _get_all_paper_titles(raw_client)
    if not paper_options:
        st.warning(
            "No records found in the Qdrant collection. "
            "Run the ingestion pipeline (`python scripts/ingest_openalex.py` or "
            "`python scripts/bootstrap_synthetic.py`) first."
        )
        return

    col_a, col_b, col_k = st.columns([4, 4, 2])
    with col_a:
        selected_title_a = st.selectbox(
            "Paper A", paper_options, index=0, key="vt_paper_a"
        )
    with col_b:
        default_b = min(1, len(paper_options) - 1)
        selected_title_b = st.selectbox(
            "Paper B", paper_options, index=default_b, key="vt_paper_b"
        )
    with col_k:
        neighbours_k = st.slider(
            "Neighbours per paper", 3, 15, 8, key="vt_neighbours_k"
        )

    if selected_title_a == selected_title_b:
        st.info("Pick two *different* papers to compare.")
        return

    paper_a = _fetch_paper_by_title(raw_client, selected_title_a)
    paper_b = _fetch_paper_by_title(raw_client, selected_title_b)
    if not (paper_a and paper_b):
        st.error("Could not load one of the selected papers from Qdrant.")
        return

    raw_a = _extract_vector(paper_a)
    raw_b = _extract_vector(paper_b)
    if raw_a is None or raw_b is None:
        st.error(
            "One of the selected papers has no stored dense vector. "
            "Re-run ingestion, then reload."
        )
        return

    vec_a = np.asarray(raw_a, dtype=np.float32)
    vec_b = np.asarray(raw_b, dtype=np.float32)
    similarity = _cosine_similarity(vec_a, vec_b)

    # 1. Headline score + verdict
    label, blurb, emoji = _interpret(similarity)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Cosine similarity", f"{similarity:.3f}",
                  help="1.0 = identical meaning, 0 = unrelated.")
        st.progress(max(0.0, min(1.0, similarity)))
    with c2:
        st.markdown(f"### {emoji} {label}")
        st.write(blurb)

    con_a = _concepts_list(paper_a.payload)
    con_b = _concepts_list(paper_b.payload)
    shared = sorted(set(c.lower() for c in con_a) & set(c.lower() for c in con_b))
    if shared:
        st.markdown("**Shared concepts:** " + " ".join(f"`{c}`" for c in shared))
    else:
        st.caption(
            "No concepts in common — any relationship is semantic rather than tag-based."
        )

    st.markdown("---")

    # 2. Neighbourhood map
    st.markdown("### 🗺️ Where they sit in the research landscape")
    st.caption(
        "Each paper's nearest neighbours are projected into 2-D alongside the two "
        "anchors. Overlapping clouds = related fields; a clear gap between the clouds "
        "is the unwritten space between them."
    )

    neigh_a = _nearest_to_vector(
        raw_client, vec_a, limit=neighbours_k, exclude_titles=[selected_title_a]
    )
    neigh_b = _nearest_to_vector(
        raw_client, vec_b, limit=neighbours_k, exclude_titles=[selected_title_b]
    )

    points: list = []
    pt_meta: list = []
    points.append(vec_a)
    pt_meta.append(("Paper A", selected_title_a, "anchor_a"))
    points.append(vec_b)
    pt_meta.append(("Paper B", selected_title_b, "anchor_b"))
    for h in neigh_a:
        hv = _extract_vector(h)
        if hv is not None:
            points.append(np.asarray(hv, dtype=np.float32))
            pt_meta.append(("A-neighbour", h.payload.get("title", "Untitled"), "neigh_a"))
    for h in neigh_b:
        hv = _extract_vector(h)
        if hv is not None:
            points.append(np.asarray(hv, dtype=np.float32))
            pt_meta.append(("B-neighbour", h.payload.get("title", "Untitled"), "neigh_b"))

    if len(points) >= 3:
        coords = PCA(n_components=2).fit_transform(np.vstack(points))
        style = {
            "anchor_a": dict(color="#00cc96", size=18, symbol="star"),
            "anchor_b": dict(color="#ab63fa", size=18, symbol="star"),
            "neigh_a":  dict(color="#00cc96", size=9,  symbol="circle"),
            "neigh_b":  dict(color="#ab63fa", size=9,  symbol="circle"),
        }
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[coords[0, 0], coords[1, 0]], y=[coords[0, 1], coords[1, 1]],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
            hoverinfo="none", showlegend=False,
        ))
        for group, legend_name in [
            ("neigh_a", "Paper A neighbourhood"),
            ("neigh_b", "Paper B neighbourhood"),
            ("anchor_a", "Paper A"),
            ("anchor_b", "Paper B"),
        ]:
            idx = [i for i, m in enumerate(pt_meta) if m[2] == group]
            if not idx:
                continue
            fig.add_trace(go.Scatter(
                x=coords[idx, 0], y=coords[idx, 1],
                mode="markers", name=legend_name,
                marker=style[group],
                text=[pt_meta[i][1] for i in idx],
                hovertemplate="%{text}<extra></extra>",
            ))
        fig.update_layout(
            template="plotly_dark", height=430,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough neighbours in the collection to draw a landscape map.")

    # 3. Bridge papers
    st.markdown("### 🌉 What currently sits between them?")
    midpoint = (vec_a + vec_b) / 2.0
    bridges = _nearest_to_vector(
        raw_client, midpoint, limit=5,
        exclude_titles=[selected_title_a, selected_title_b],
        with_vectors=False,
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
                f"{top_bridge_sim:.2f} similar). The space between these two papers is a "
                f"candidate gap — see **Vector Gap Discovery** below for the corpus-wide view."
            )
        for h in bridges:
            st.markdown(
                f"- **{h.payload.get('title', 'Untitled')}**  \n"
                f"  <sub>midpoint similarity `{h.score:.3f}` · "
                f"{h.payload.get('venue', 'N/A')} {h.payload.get('year', '')}</sub>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No bridge papers found near the midpoint.")

    st.markdown("---")

    # 4. Side-by-side metadata
    st.markdown("### 📑 The two papers")
    left, right = st.columns(2)
    for col, paper, dot in [(left, paper_a, "🟢"), (right, paper_b, "🟣")]:
        p = paper.payload
        with col:
            st.markdown(f"#### {dot} {p.get('title')}")
            st.markdown(
                f"**Venue:** `{p.get('venue', 'N/A')}` &nbsp; "
                f"**Year:** `{p.get('year', 'N/A')}`",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Citations:** `{p.get('cited_by_count', 0):,}`")
            cons = _concepts_list(p)
            if cons:
                st.markdown("**Concepts:** " + ", ".join(f"*{c}*" for c in cons[:12]))


# ---------------------------------------------------------------------------
# Vector Gap Discovery
# ---------------------------------------------------------------------------

def render_vector_gap_discovery(store: AxiomQdrant) -> None:
    """Corpus-wide gap detection: KMeans topic clustering + bridgeable-pair scan.

    Adapted from qdrant_ms/pages/2_Gap_Discovery_Feed.py (PR #3).
    Groups every paper into topics, then finds pairs of related topics with no
    paper bridging them. Algorithm: pure vector geometry (no citation structure).
    """
    raw_client = store.client

    try:
        n_points = raw_client.count(config.COLLECTION_NAME, exact=True).count
    except Exception:
        n_points = 0

    k_max = max(3, min(40, n_points // 2))
    k_default = max(3, min(18, n_points // 6)) or 6

    # Controls — row 1: topics + rescan
    topics_col, rescan_col = st.columns([5, 1])
    with topics_col:
        if k_max <= 3:
            k = max(2, k_max)
            st.caption(f"Small corpus ({n_points} papers) — using {k} topics.")
        else:
            k = st.slider(
                "Number of topics", 3, k_max, min(k_default, k_max),
                key="vt_gap_k",
                help="How finely to partition the corpus. Fewer topics = broader themes.",
            )
    with rescan_col:
        st.write("")  # vertical alignment spacer
        if st.button("🔄 Rescan", key="vt_rescan",
                     help="Clear the topic-clustering cache and re-run from scratch"):
            _cluster_corpus.clear()
            _find_topic_gaps.clear()
            _load_corpus.clear()
            st.rerun()

    # Load cluster data (cached; expensive first run)
    try:
        data = _cluster_corpus(raw_client, k)
    except Exception as exc:
        st.error(f"Could not load / cluster the corpus: {exc}")
        return

    if len(data["titles"]) < 4:
        st.warning(
            "Not enough papers in the collection to find gaps. Run ingestion first."
        )
        return

    # Controls — row 2: band + top-n (defaults from actual corpus distribution)
    pct = data["sim_pct"]
    band_col, n_col = st.columns([4, 2])
    with band_col:
        band = st.slider(
            "Bridgeable similarity band", 0.0, 1.0,
            (round(pct["p40"], 2), round(pct["p88"], 2)), 0.01,
            key="vt_gap_band",
            help=(
                "Only score topic pairs whose centroids fall in this similarity band. "
                "Too low = unrelated fields (implausible bridge); too high = same topic. "
                "Defaults track this corpus's own p40/p88 distribution."
            ),
        )
    with n_col:
        top_n = st.slider("Gaps to show", 3, 15, 6, key="vt_gap_top_n")

    gaps = _find_topic_gaps(raw_client, k, band[0], band[1], top_n)

    o1, o2, o3 = st.columns(3)
    o1.metric("Papers analysed", f"{len(data['titles']):,}")
    o2.metric("Topics", f"{len(data['topics'])}")
    o3.metric("Gaps found", f"{len(gaps)}")

    if not gaps:
        st.info(
            "No related-but-unbridged topic pairs in the current band. Widen the "
            "*bridgeable similarity band* or change the *number of topics*."
        )
        return

    # Landscape map with the top gap highlighted
    st.markdown("### 🗺️ Research landscape")
    top = gaps[0]
    coords, labels = data["coords"], data["labels"]
    ta, tb = top["topic_a"]["id"], top["topic_b"]["id"]

    fig = go.Figure()
    other = np.where((labels != ta) & (labels != tb))[0]
    fig.add_trace(go.Scatter(
        x=coords[other, 0], y=coords[other, 1], mode="markers",
        marker=dict(size=4, color="rgba(150,150,150,0.35)"),
        name="Other topics", hoverinfo="skip",
    ))
    for tid, color, name in [
        (ta, "#00cc96", top["topic_a"]["label"]),
        (tb, "#ab63fa", top["topic_b"]["label"]),
    ]:
        idx = np.where(labels == tid)[0]
        fig.add_trace(go.Scatter(
            x=coords[idx, 0], y=coords[idx, 1], mode="markers",
            marker=dict(size=7, color=color), name=f"Topic: {name[:40]}",
            text=[data["titles"][i] for i in idx],
            hovertemplate="%{text}<extra></extra>",
        ))
    mx, my = top["midpoint_2d"]
    fig.add_trace(go.Scatter(
        x=[mx], y=[my], mode="markers+text",
        marker=dict(size=20, color="#ff4d4d", symbol="x", line=dict(width=2, color="white")),
        text=["GAP"], textposition="top center", name="Top gap",
    ))
    fig.update_layout(
        template="plotly_dark", height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The two related topic clouds (green / purple) with ✖ marking the empty space "
        "between them. Grey = every other topic."
    )

    st.markdown("---")

    # Ranked gap feed
    st.markdown("### 🔥 Ranked gaps")
    for i, g in enumerate(gaps):
        a, b = g["topic_a"], g["topic_b"]
        st.markdown(f"#### Gap #{i + 1} — emptiness `{g['gap_score']:.2f}`")

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Topic relatedness", f"{g['centroid_sim']:.2f}",
            help="How related the two topics are. Higher = a bridge is more plausible.",
        )
        m2.metric(
            "Bridge coverage", f"{g['coverage'] * 100:.0f}%",
            help="How far the closest existing paper gets toward truly connecting the two "
                 "topics. Low % = the space is largely unwritten.",
        )
        m3.metric("Opportunity", "HIGH" if g["gap_score"] > 0.85 else "MODERATE")

        st.markdown(
            f"**These two related topics have almost nothing written between them:**\n"
            f"- 🟢 **Topic A** — *{a['label']}*  \n"
            f"  <sub>{a['size']} papers · e.g. \"{a['rep_title'][:80]}\"</sub>\n"
            f"- 🟣 **Topic B** — *{b['label']}*  \n"
            f"  <sub>{b['size']} papers · e.g. \"{b['rep_title'][:80]}\"</sub>",
            unsafe_allow_html=True,
        )
        st.warning(
            f"Closest existing work to bridging them: "
            f"**\"{g['best_bridge'].get('title', 'Untitled')}\"** "
            f"— but it only reaches {g['coverage'] * 100:.0f}% coverage, "
            f"so the bridge is effectively unbuilt."
        )
        st.markdown("---")
