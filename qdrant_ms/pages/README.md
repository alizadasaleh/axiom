# AXIOM: Academic Paper Analysis Toolkit

Welcome to AXIOM, a suite of tools for exploring and analyzing a vector database of academic papers. This application demonstrates the power of vector search for discovering relationships and gaps in research literature.

This project is built with Streamlit and uses Qdrant as its vector database.

## Features

*   **Proximity Analyzer**: Select two papers and visualize their conceptual distance in the vector space.
*   **Gap Discovery Feed**: Automatically scans the paper collection to find potential "voids" or unexplored research areas between different topics.

## Setup

1.  **Prerequisites**:
    *   Python 3.8+
    *   Docker

2.  **Installation**:
    *   Clone this repository:
        ```bash
        git clone https://github.com/your-username/axiom-academic-analyzer.git
        cd axiom-academic-analyzer
        ```
    *   Install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Running the Application**:
    *   Start the Qdrant database using Docker.
    *   (Add instructions for your data ingestion pipeline here).
    *   Run the Streamlit app:
        ```bash
        streamlit run Home.py
        ```