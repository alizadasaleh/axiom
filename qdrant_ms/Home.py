import streamlit as st

st.set_page_config(
    page_title="AXIOM - Academic Paper Analysis Toolkit",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 AXIOM — Vector Analysis of the Research Corpus")

st.markdown(
    """
Every paper in the corpus is stored in **Qdrant** as a vector — a point in a
high-dimensional space where *closeness = similarity in meaning*. Two tools turn
that geometry into something useful:
"""
)

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
### 🔗 Paper Similarity Analyzer
*How close are two specific papers?*

- A cosine-similarity score with a plain-language verdict.
- A **neighbourhood map** showing each paper's nearest work, so the score has
  context instead of being a bare number.
- **Bridge papers** — what currently sits between the two topics (and a flag when
  that space is empty).

→ Open **Paper Similarity** in the sidebar.
"""
    )
with c2:
    st.markdown(
        """
### 🕳️ Gap Discovery
*Where is research missing across the whole corpus?*

- Groups every paper into **topics**, then finds pairs of topics with
  **nothing written between them**.
- A landscape map pinpointing the empty midpoint.
- A ranked feed explaining each gap: *Topic A ↔ Topic B, no bridge yet.*

→ Open **Gap Discovery** in the sidebar.
"""
    )

st.markdown("---")
st.caption(
    "Requires the Qdrant container (`docker compose up -d qdrant`) with the "
    "`axiom_v1` collection populated (same collection as the main platform). "
    "These tools are also available as the **🔗 Vector tools** tab in the main UI "
    "(`streamlit run app/streamlit_app.py`)."
)
