import streamlit as st

st.set_page_config(
    page_title="AXIOM - Academic Paper Analysis Toolkit",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 AXIOM: Academic Paper Analysis Toolkit")

st.markdown("""
Welcome to AXIOM, a suite of tools for exploring and analyzing a vector database of academic papers.

This application demonstrates the power of vector search for discovering relationships and gaps in research literature.

### Available Tools:

*   **Proximity Analyzer**: Select two papers and visualize their conceptual distance in the vector space.
*   **Gap Discovery Feed**: Automatically scans the paper collection to find potential "voids" or unexplored research areas between different topics.

Please select a tool from the sidebar to begin.
""")