import itertools
from collections import Counter

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from qdrant_utils import COLLECTION_NAME, concepts_list, init_connections, load_corpus

# -----------------------------------------------------------------------------
# CONFIGURATION & INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AXIOM — Gap Discovery", layout="wide")

try:
    client = init_connections()
except Exception as e:
    st.error(f"Could not connect to local Qdrant instance: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# STEP 1 — CLUSTER THE WHOLE CORPUS INTO TOPICS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Clustering the corpus into topics…")
def cluster_corpus(_client, k):
    """Partition every paper into `k` topics and precompute everything the gap
    scan needs: L2-normalised vectors, unit centroids, a shared 2-D PCA
    projection, per-topic summaries, and the distribution of centroid-to-centroid
    similarities (so the UI can pick a sensible bridgeable band)."""
    corpus = load_corpus(_client)
    vectors = corpus["vectors"]
    payloads = corpus["payloads"]
    titles = corpus["titles"]
    if len(vectors) < k:
        k = max(2, len(vectors))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(vectors)

    # unit-normalise so plain dot products == cosine similarity
    vn = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    cn = km.cluster_centers_ / (np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True) + 1e-9)

    coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
    pca_full = PCA(n_components=2, random_state=42).fit(vectors)  # to place midpoints

    topics = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            topics.append({"id": c, "size": 0, "label": f"Topic {c}", "keywords": [], "rep_title": ""})
            continue
        rep = members[int(np.argmax(vn[members] @ cn[c]))]  # closest member to centroid
        con_counter = Counter()
        for i in members:
            for con in concepts_list(payloads[i]):
                con_counter[con] += 1
        top_con = [w for w, _ in con_counter.most_common(4)]
        topics.append({
            "id": c,
            "size": int(len(members)),
            "label": ", ".join(top_con) if top_con else titles[rep][:50],
            "concepts": top_con,
            "rep_title": titles[rep],
        })

    cc = cn @ cn.T  # centroid-to-centroid cosine
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


# -----------------------------------------------------------------------------
# STEP 2 — FIND TOPIC PAIRS WITH NO PAPER BRIDGING THEM
# -----------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner="Scanning topic pairs for structural gaps…")
def find_topic_gaps(_client, k, bridge_min, bridge_max, top_n):
    """A real bridge between topics A and B is a paper close to *both* topics at
    once — i.e. high `min(sim to A, sim to B)`. If the best such paper barely
    beats the trivial 'sits on one topic's edge' baseline (= centroid similarity),
    the two topics are effectively unconnected: a gap."""
    d = cluster_corpus(_client, k)
    vn, cn, cc, topics = d["vn"], d["cn"], d["cc"], d["topics"]
    S = vn @ cn.T  # (N papers) x (k topics) cosine

    gaps = []
    for a, b in itertools.combinations(range(len(cn)), 2):
        if topics[a]["size"] == 0 or topics[b]["size"] == 0:
            continue
        cs = float(cc[a, b])
        if not (bridge_min <= cs <= bridge_max):  # only plausibly-bridgeable pairs
            continue

        joint = np.minimum(S[:, a], S[:, b])       # each paper's closeness to BOTH topics
        best = int(np.argmax(joint))
        bridge_strength = float(joint[best])
        denom = max(1e-6, 1.0 - cs)
        bridge_ratio = (bridge_strength - cs) / denom  # 0 = only edges, 1 = fully bridged

        midpoint = (cn[a] + cn[b]) / 2.0
        mid2d = d["pca_full"].transform(midpoint.reshape(1, -1))[0]

        gaps.append({
            "topic_a": topics[a],
            "topic_b": topics[b],
            "centroid_sim": cs,
            "bridge_ratio": bridge_ratio,
            "coverage": float(np.clip(bridge_ratio, 0, 1)),
            "gap_score": float(np.clip(1 - bridge_ratio, 0, 1)),
            "best_bridge": d["payloads"][best],
            "midpoint_2d": mid2d.tolist(),
        })

    gaps.sort(key=lambda g: g["bridge_ratio"])  # emptiest (least-bridged) first
    return gaps[:top_n]


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🕳️ Gap Discovery — the whole corpus")
st.caption(
    "AXIOM groups every paper into topics, then looks for pairs of **related** "
    "topics with no paper bridging them. Those are the places new research could go."
)

try:
    n_points = client.count(COLLECTION_NAME, exact=True).count
except Exception:
    n_points = 0
k_max = max(3, min(40, n_points // 2))
k_default = max(3, min(18, n_points // 6)) or 6

with st.sidebar:
    st.header("⚙️ Scan settings")
    if k_max <= 3:
        k = max(2, k_max)
        st.caption(f"Small corpus ({n_points} papers) — using {k} topics.")
    else:
        k = st.slider("Number of topics", 3, k_max, min(k_default, k_max),
                      help="How finely to partition the corpus. Fewer topics = broader themes.")
    if st.button("🔄 Rescan", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    data = cluster_corpus(client, k)
except Exception as e:
    st.error(f"Could not load / cluster the corpus: {e}")
    st.stop()

if len(data["titles"]) < 4:
    st.warning("Not enough papers in the collection to find gaps. Run ingestion first.")
    st.stop()

# Band + count sliders, defaulted to the corpus's own similarity distribution.
pct = data["sim_pct"]
with st.sidebar:
    band = st.slider(
        "Bridgeable similarity band", 0.0, 1.0,
        (round(pct["p40"], 2), round(pct["p88"], 2)), 0.01,
        help="Only score topic pairs whose centroids are this similar — too low = "
             "unrelated fields (implausible bridge), too high = the same topic. "
             "Defaults track this corpus's own distribution.",
    )
    top_n = st.slider("Gaps to show", 3, 15, 6)

gaps = find_topic_gaps(client, k, band[0], band[1], top_n)

o1, o2, o3 = st.columns(3)
o1.metric("Papers analysed", f"{len(data['titles']):,}")
o2.metric("Topics", f"{len(data['topics'])}")
o3.metric("Gaps found", f"{len(gaps)}")

if not gaps:
    st.info(
        "No related-but-unbridged topic pairs in the current band. Widen the "
        "*bridgeable similarity band* or change the *number of topics* in the sidebar."
    )
    st.stop()

# ---- Landscape map with the top gap highlighted ----------------------------
st.markdown("### 🗺️ Research landscape")
top = gaps[0]
coords, labels = data["coords"], data["labels"]
ta, tb = top["topic_a"]["id"], top["topic_b"]["id"]

fig = go.Figure()
other = np.where((labels != ta) & (labels != tb))[0]
fig.add_trace(go.Scatter(
    x=coords[other, 0], y=coords[other, 1], mode="markers",
    marker=dict(size=4, color="rgba(150,150,150,0.35)"), name="Other topics", hoverinfo="skip",
))
for tid, color, name in [(ta, "#00cc96", top["topic_a"]["label"]),
                         (tb, "#ab63fa", top["topic_b"]["label"])]:
    idx = np.where(labels == tid)[0]
    fig.add_trace(go.Scatter(
        x=coords[idx, 0], y=coords[idx, 1], mode="markers",
        marker=dict(size=7, color=color), name=f"Topic: {name[:40]}",
        text=[data["titles"][i] for i in idx], hovertemplate="%{text}<extra></extra>",
    ))
mx, my = top["midpoint_2d"]
fig.add_trace(go.Scatter(
    x=[mx], y=[my], mode="markers+text",
    marker=dict(size=20, color="#ff4d4d", symbol="x", line=dict(width=2, color="white")),
    text=["GAP"], textposition="top center", name="Top gap",
))
fig.update_layout(
    template="plotly_dark", height=460, margin=dict(l=10, r=10, t=10, b=10),
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

# ---- Ranked gap feed -------------------------------------------------------
st.markdown("### 🔥 Ranked gaps")
for i, g in enumerate(gaps):
    a, b = g["topic_a"], g["topic_b"]
    st.markdown(f"#### Gap #{i + 1} — emptiness `{g['gap_score']:.2f}`")

    m1, m2, m3 = st.columns(3)
    m1.metric("Topic relatedness", f"{g['centroid_sim']:.2f}",
              help="How related the two topics are. Higher = a bridge is more plausible.")
    m2.metric("Bridge coverage", f"{g['coverage'] * 100:.0f}%",
              help="How far the closest existing paper gets toward truly connecting "
                   "the two topics. Low = the space is unwritten.")
    m3.metric("Opportunity", "HIGH" if g["gap_score"] > 0.85 else "MODERATE")

    st.markdown(
        f"**These two related topics have almost nothing written between them:**\n"
        f"- 🟢 **Topic A** — *{a['label']}*  \n"
        f"  <sub>{a['size']} papers · e.g. “{a['rep_title'][:80]}”</sub>\n"
        f"- 🟣 **Topic B** — *{b['label']}*  \n"
        f"  <sub>{b['size']} papers · e.g. “{b['rep_title'][:80]}”</sub>",
        unsafe_allow_html=True,
    )
    st.warning(
        f"Closest existing work to bridging them: **“{g['best_bridge'].get('title', 'Untitled')}”** "
        f"— but it only reaches {g['coverage'] * 100:.0f}% coverage, so the bridge is effectively unbuilt."
    )
    st.markdown("---")
