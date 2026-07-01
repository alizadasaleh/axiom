import argparse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np

def calculate_cosine_similarity(vec1, vec2):
    """Computes the dot product of normalized vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def find_paper_by_title(client, collection_name, title_query):
    """Searches the collection payloads to find an exact or close title match."""
    # Using a scroll filter to scan for the title match in payload strings
    result, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {
                    "key": "title",
                    "match": {
                        "text": title_query
                    }
                }
            ]
        },
        limit=1,
        with_vectors=True,
        with_payload=True
    )
    if result:
        return result[0]
    return None

def main():
    parser = argparse.ArgumentParser(description="Compare semantic similarity between papers in Qdrant.")
    parser.add_argument("--title1", type=str, required=True, help="Title or substring of the first paper")
    parser.add_argument("--title2", type=str, required=True, help="Title or substring of the second paper")
    parser.add_argument("--collection", type=str, default="academic_papers", help="Qdrant collection name")
    args = parser.parse_args()

    # 1. Connect to local Qdrant container
    client = QdrantClient(host="localhost", port=6333)
    
    print(f"🔍 Searching for Paper 1: '{args.title1}'...")
    paper1 = find_paper_by_title(client, args.collection, args.title1)
    
    print(f"🔍 Searching for Paper 2: '{args.title2}'...")
    paper2 = find_paper_by_title(client, args.collection, args.title2)

    # Error handling if papers aren't matched
    if not paper1:
        print(f"❌ Could not find an ingested paper matching title: '{args.title1}'")
        return
    if not paper2:
        print(f"❌ Could not find an ingested paper matching title: '{args.title2}'")
        return

    # 2. Extract payload and vector representations
    p1_meta = paper1.payload
    p2_meta = paper2.payload
    
    v1 = paper1.vector
    v2 = paper2.vector

    # 3. Compute Similarity Metrics
    similarity_score = calculate_cosine_similarity(v1, v2)

    # 4. Format and Print Report
    print("\n" + "="*80)
    print("📋 AXIOM SEMANTIC SIMILARITY REPORT")
    print("="*80)
    
    print(f"\n📄 PAPER 1:")
    print(f"   • Title:   {p1_meta.get('title')}")
    print(f"   • Venue:   {p1_meta.get('inferred_venue', 'N/A').upper()} ({p1_meta.get('year', 'N/A')})")
    print(f"   • Keyword: {', '.join(p1_meta.get('keywords', []))[:100]}...")
    
    print(f"\n📄 PAPER 2:")
    print(f"   • Title:   {p2_meta.get('title')}")
    print(f"   • Venue:   {p2_meta.get('inferred_venue', 'N/A').upper()} ({p2_meta.get('year', 'N/A')})")
    print(f"   • Keyword: {', '.join(p2_meta.get('keywords', []))[:100]}...")
    
    print("\n" + "-"*80)
    print(f"📊 Cosine Similarity Metric: {similarity_score:.4f}")
    print("-"*80)

    # Geometric evaluation criteria aligned with your project specification
    if similarity_score >= 0.82:
        print("💡 Interpretation: Very high semantic overlap. These papers live in the same close thematic cluster.")
    elif similarity_score >= 0.65:
        print("💡 Interpretation: Moderate similarity. Part of related fields but addressing different aspects.")
    else:
        print("🕳️ Interpretation: Low similarity. The distance matches your project's 'Geometric Void' profile. Potential space for a bridge hypothesis!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()