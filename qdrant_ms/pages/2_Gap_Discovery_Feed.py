import streamlit as st
import numpy as np
import random
from qdrant_client import QdrantClient
from qdrant import init_connections, COLLECTION_NAME

# -----------------------------------------------------------------------------
# CONFIGURATION & INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AXIOM — Gap Discovery Feed", layout="wide")

try:
    client, _ = init_connections()
except Exception as e:
    st.error(f"Could not connect to local Qdrant instance: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# ALGORITHMIC GAP GENERATION DISPATCHER
# -----------------------------------------------------------------------------
def discover_void_candidates(sample_limit=1000, target_voids=5):
    """
    Scans the database index by cross-evaluating random point intersections,
    calculating their midpoints, and filtering for low density neighborhoods.
    """
    try:
        # Fetch an arbitrary batch pool of items containing vectors and payloads
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=sample_limit,
            with_vectors=True,
            with_payload=True
        )
    except Exception:
        return []

    if not records or len(records) < 2:
        return []

    discovered_gaps = []
    checked_pairs = set()

    # Combinatorial sampling loop
    for _ in range(sample_limit * 2):
        if len(discovered_gaps) >= target_voids:
            break

        idx_a, idx_b = random.sample(range(len(records)), 2)
        pair_key = tuple(sorted([idx_a, idx_b]))
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)

        paper_a = records[idx_a]
        paper_b = records[idx_b]

        # Ensure we are not comparing a paper with itself or items with missing metadata
        if paper_a.payload.get("title") == paper_b.payload.get("title"):
            continue

        vec_a = np.array(paper_a.vector)
        vec_b = np.array(paper_b.vector)
        
        # 1. Compute Geometric Midpoint
        midpoint_vector = ((vec_a + vec_b) / 2).tolist()

        # 2. Query Qdrant for the absolute nearest neighbor to this exact coordinate
       # 2. Query Qdrant for the absolute nearest neighbor to this exact coordinate
        try:
            closest_neighbors = client.query_points(
                collection_name=COLLECTION_NAME,
                query=midpoint_vector,  # In v1.x client, query takes the raw float list directly
                limit=1,
                with_payload=True
            ).points
        except Exception as e:
            st.error(f"Error querying midpoint vector coordinates: {e}")
            continue

        if closest_neighbors:
            nearest_neighbor = closest_neighbors[0]
            # query_points returns proximity as a 'score' attribute (Cosine Similarity)
            s_max = nearest_neighbor.score  

            # 3. Check against AXIOM Void Threshold (s_max < 0.68 means empty space)
            if s_max < 0.73:
                gap_score = 1.0 - s_max  # Higher score means a larger, cleaner void
                
                discovered_gaps.append({
                    "gap_score": gap_score,
                    "s_max": s_max,
                    "paper_a": paper_a.payload,
                    "paper_b": paper_b.payload,
                    "closest_bridge": nearest_neighbor.payload,
                })


    # Sort candidates so the largest gaps (highest gap score) appear first
    return sorted(discovered_gaps, key=lambda x: x["gap_score"], reverse=True)

# -----------------------------------------------------------------------------
# UI RENDERING
# -----------------------------------------------------------------------------
st.title("🕳️ AXIOM // Cross-Venue Gap Discovery Feed")
st.caption("Automated global scan highlighting unpopulated geometric voids across domain boundaries.")
st.markdown("---")

# Controls top segment
col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl1:
    st.markdown("This feed dynamically pairs distinct literature clusters, calculates their conceptual midpoint, and flags intersections where **no written research paper actively sits**.")
with col_ctrl2:
    refresh_feed = st.button("🔄 Rescan Global Collection", type="primary", use_container_width=True)

if "gap_data" not in st.session_state or refresh_feed:
    with st.spinner("Analyzing high-dimensional vector spaces for voids..."):
        st.session_state.gap_data = discover_void_candidates()

gaps = st.session_state.gap_data

if not gaps:
    st.info("💡 The scanner evaluated the sample index but didn't identify any clear voids below the threshold. Try triggering a 'Rescan' to evaluate a fresh cross-section of data.")
else:
    # Render feed stream
    for idx, gap in enumerate(gaps):
        # Unique visual containers for each feed card
        with st.container():
            st.markdown(f"### 🔥 GAP CANDIDATE #{idx+1} — Void Scale: `{gap['gap_score']:.4f}`")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Max Neighborhood Proximity ($s_{max}$)", f"{gap['s_max']:.4f}")
            m_col2.metric("Inferred Opportunity Index", "HIGH" if gap['gap_score'] > 0.35 else "MODERATE")
            m_col3.markdown(f"**Structural Gap Scope:** \nThis empty space resides at the nexus of venue paths: `{gap['paper_a'].get('inferred_venue','N/A').upper()}` and `{gap['paper_b'].get('inferred_venue','N/A').upper()}`.")

            # Display the two separate worlds that create this gap
            st.markdown("#### 🔬 Unconnected Domain Anchors:")
            card_col1, card_col2 = st.columns(2)
            
            with card_col1:
                st.markdown(f"**⚓ Domain Cluster A:** {gap['paper_a'].get('title')}")
                st.caption(f"Venue: {gap['paper_a'].get('inferred_venue','N/A').upper()} | Keywords: {', '.join(gap['paper_a'].get('keywords', []))[:80]}...")
                with st.expander("Show Anchor A Summary"):
                    st.write(gap['paper_a'].get('abstract', "Abstract not indexed."))

            with card_col2:
                st.markdown(f"**⚓ Domain Cluster B:** {gap['paper_b'].get('title')}")
                st.caption(f"Venue: {gap['paper_b'].get('inferred_venue','N/A').upper()} | Keywords: {', '.join(gap['paper_b'].get('keywords', []))[:80]}...")
                with st.expander("Show Anchor B Summary"):
                    st.write(gap['paper_b'].get('abstract', "Abstract not indexed."))

            # Display the closest attempt to bridge the gap
            st.markdown("#### 🪵 Closest Existing Literature Boundary:")
            st.warning(f"👉 **\"{gap['closest_bridge'].get('title')}\"** — Sits furthest away from the midpoint coordinate, leaving the core concept unbuilt.")
            
            st.markdown("---")