import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from qdrant import init_connections, get_all_paper_titles, fetch_paper_by_title

# -----------------------------------------------------------------------------
# CONFIGURATION & INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AXIOM — Searchable Dropdown Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    client, model = init_connections()
except Exception as e:
    st.error(f"Could not connect to Qdrant on localhost:6333. Is Docker running? Error: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# -----------------------------------------------------------------------------
# UI LAYOUT & BRANDING
# -----------------------------------------------------------------------------
st.title("📟 AXIOM // Searchable Dropdown Proximity Engine")
st.caption("Click or type inside the dropdown containers to dynamically search and pair your literature index.")
st.markdown("---")

# Load index properties
paper_options = get_all_paper_titles(client)

if not paper_options:
    st.warning("⚠️ No records found in the Qdrant collection. Please run your ingestion pipeline script first!")
    st.stop()

# Sidebar: Interactive Search-as-you-type Selectboxes
st.sidebar.header("🎯 Document Targets")
st.sidebar.markdown("Type a keyword inside the fields to filter titles instantly.")

selected_title_a = st.sidebar.selectbox("Search / Select Paper A", paper_options, index=0)
default_index_b = min(1, len(paper_options) - 1)
selected_title_b = st.sidebar.selectbox("Search / Select Paper B", paper_options, index=default_index_b)

# -----------------------------------------------------------------------------
# DATA VISUALIZATION & PROCESS
# -----------------------------------------------------------------------------
paper_a = fetch_paper_by_title(client, selected_title_a)
paper_b = fetch_paper_by_title(client, selected_title_b)

if paper_a and paper_b:
    vec_a = paper_a.vector
    vec_b = paper_b.vector
    
    similarity = cosine_similarity(vec_a, vec_b)
    distance = 1.0 - similarity
    
    # Simple Layout Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Cosine Similarity", value=f"{similarity:.4f}")
    with col2:
        st.metric(label="Cosine Distance", value=f"{distance:.4f}")

    st.markdown("### 🗺️ Vector Space Orientation")
    
    # Dimensional reduction loop
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform([vec_a, vec_b])
    coords_a = reduced_vectors[0]
    coords_b = reduced_vectors[1]
    
    fig = go.Figure()
    
    # Connecting vector path
    fig.add_trace(go.Scatter(
        x=[coords_a[0], coords_b[0]],
        y=[coords_a[1], coords_b[1]],
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.25)', width=2),
        hoverinfo='none'
    ))
    
    # Node Paper A
    fig.add_trace(go.Scatter(
        x=[coords_a[0]], y=[coords_a[1]],
        mode='markers+text',
        marker=dict(size=14, color='#00cc96', line=dict(width=1, color='white')),
        text=["Paper A"], textposition="top center",
        hovertemplate=f"<b>Paper A</b><br>{paper_a.payload.get('title')}<br><extra></extra>"
    ))
    
    # Node Paper B
    fig.add_trace(go.Scatter(
        x=[coords_b[0]], y=[coords_b[1]],
        mode='markers+text',
        marker=dict(size=14, color='#ab63fa', line=dict(width=1, color='white')),
        text=["Paper B"], textposition="top center",
        hovertemplate=f"<b>Paper B</b><br>{paper_b.payload.get('title')}<br><extra></extra>"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        height=350,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Clean Metadata Displays
    st.markdown("### 📑 Literature Breakdown")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🟢 Paper A Context")
        st.info(f"**{paper_a.payload.get('title')}**")
        st.markdown(f"**Venue:** `{paper_a.payload.get('inferred_venue', 'N/A').upper()}`")
        st.markdown(f"**Keywords:** *{', '.join(paper_a.payload.get('keywords', []))}*")
        with st.expander("Read Abstract"):
            st.write(paper_a.payload.get("abstract", "No summary data found."))
            
    with col_right:
        st.markdown("#### 🟣 Paper B Context")
        st.success(f"**{paper_b.payload.get('title')}**")
        st.markdown(f"**Venue:** `{paper_b.payload.get('inferred_venue', 'N/A').upper()}`")
        st.markdown(f"**Keywords:** *{', '.join(paper_b.payload.get('keywords', []))}*")
        with st.expander("Read Abstract"):
            st.write(paper_b.payload.get("abstract", "No summary data found."))