import numpy as np
import pandas as pd
import plotly.express as px
from qdrant_client import QdrantClient
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE

# 1. Connect to your local Qdrant container
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "academic_papers"

print("Fetching vectors and payloads from Qdrant...")
# Pull down the papers (adjust limit based on your local dataset size)
response = client.scroll(
    collection_name=COLLECTION_NAME,
    limit=1000,
    with_vectors=True,
    with_payload=True
)

points = response[0]

if not points:
    raise ValueError(f"No data found in collection '{COLLECTION_NAME}'. Did you run ingestion?")

# 2. Extract vectors and metadata fields
all_vectors = []
all_titles = []
all_areas = []
all_keywords = []

for pt in points:
    all_vectors.append(pt.vector)
    all_titles.append(pt.payload.get("title", "Untitled"))
    all_areas.append(pt.payload.get("primary_area", "Unknown"))
    
    kw = pt.payload.get("keywords", [])
    all_keywords.append(", ".join(kw) if isinstance(kw, list) else str(kw))

X = np.array(all_vectors)

# 3. Clustering: Find the dense mathematical "islands"
print("Running density-based clustering (HDBSCAN)...")
# min_cluster_size determines the smallest group of papers that can form a topic planet
clusterer = HDBSCAN(min_cluster_size=4, metric="cosine")
cluster_labels = clusterer.fit_predict(X)

# 4. Dimensionality Reduction: Flatten 384 dimensions to 2D for human eyes
print("Projecting high-dimensional space into 2D (t-SNE)...")
tsne = TSNE(n_components=2, metric="cosine", random_state=42, perplexity=15)
X_2d = tsne.fit_transform(X)

# 5. Build a structured DataFrame for Plotly
df = pd.DataFrame({
    "x": X_2d[:, 0],
    "y": X_2d[:, 1],
    "Title": all_titles,
    "Area": all_areas,
    "Keywords": all_keywords,
    "Cluster": [f"Cluster {label}" if label != -1 else "Dark Matter (Gap Space)" for label in cluster_labels]
})

# 6. Generate the Interactive Knowledge Nebula Map
print("Generating your interactive Nebula Map...")
fig = px.scatter(
    df, 
    x="x", 
    y="y", 
    color="Cluster",
    hover_data={"Title": True, "Keywords": True, "Area": True, "x": False, "y": False},
    title="Research Landscape Nebula: Topic Clusters & Voids",
    labels={"Cluster": "Identified Domains"},
    template="plotly_dark"  # Dark mode fits the nebula theme perfectly
)

# Customize marker style
fig.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.5, color="white")))

# Save and automatically display in your browser
fig.write_html("nebula_map.html")
fig.show()