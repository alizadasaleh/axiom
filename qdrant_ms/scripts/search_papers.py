from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Type any research concept you want to test
query_text = "Optimizing LLM inference on hardware constrained edge devices"
query_vector = model.encode(query_text).tolist()

results = client.query_points(
    collection_name="academic_papers",
    query=query_vector,          # 'query_vector' was renamed to 'query'
    limit=3,
    with_payload=True
)

print(f"\n--- Top Results for: '{query_text}' ---")
# The new API returns objects inside a .points attribute
for hit in results.points:       
    print(f"\nScore: {hit.score:.4f} | Title: {hit.payload.get('title')}")
