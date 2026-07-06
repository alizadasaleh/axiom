import os
import json
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Recursively ingest all paperlists JSON configurations into Qdrant.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./paperlists",
        help="Path to the top-level paperlist folder containing subdirectories"
    )
    args = parser.parse_args()

    if not os.path.exists(args.root_dir):
        raise FileNotFoundError(f"Root folder not found at: {args.root_dir}")

    # 1. Initialize local Qdrant Client (pointing to your Docker instance)
    client = QdrantClient(host="localhost", port=6333)

    COLLECTION_NAME = "academic_papers"
    VECTOR_SIZE = 384  # Dimension size for 'all-MiniLM-L6-v2'

    # 2. Re-create or verify the collection exists
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        print(
            f"Target collection '{COLLECTION_NAME}' detected. Appending records...")

    # 3. Load the embedding model locally
    print("Loading text embedding model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4. Gather all JSON files in the directory tree recursively
    json_files = []
    for root, _, files in os.walk(args.root_dir):
        for file in files:
            if file.lower().endswith(".json") and file != "croissant.json":
                json_files.append(os.path.join(root, file))

    print(f"Found {len(json_files)} configuration payloads to ingest.")

    # Global point counter ensures vectors do not overwrite each other across files
    global_point_id = 0
    BATCH_SIZE = 64

    # 5. Process each JSON target sequentially
    for file_path in json_files:
        # Extract metadata context based on folder patterns (e.g. /iclr/ or /neurips/)
        path_segments = file_path.split(os.sep)
        detected_venue = path_segments[-2] if len(
            path_segments) > 2 else "Unknown"

        print(
            f"\nProcessing target: {os.path.basename(file_path)} [Venue: {detected_venue.upper()}]")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                papers_data = json.load(f)
        except Exception as e:
            print(f"🛑 Skipping {file_path} due to file read error: {e}")
            continue

        # Convert dictionary formats to iterable lists safely
        if isinstance(papers_data, dict):
            papers_list = list(papers_data.values())
        elif isinstance(papers_data, list):
            papers_list = papers_data
        else:
            print(
                f"⚠️ Unknown JSON root structure inside {file_path}. Skipping.")
            continue

        points = []

        # Stream embeddings inside the individual file
        for paper in tqdm(papers_list, desc="Embedding generation"):
            # Extract basic strings
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            keywords = paper.get("keywords", [])

            # Format messy keyword structures cleanly
            if isinstance(keywords, list):
                keywords_str = ", ".join([str(k) for k in keywords])
            else:
                keywords_str = str(keywords)

            # Create the structural document layout for the model
            text_to_embed = f"Title: {title}\nKeywords: {keywords_str}\nAbstract: {abstract}"

            # Vector calculation loop
            try:
                vector = model.encode(text_to_embed).tolist()
            except Exception as e:
                # Fallback capture if parsing breaks down on malformed abstracts
                continue

            # Assemble comprehensive payload tags
            payload = {
                "title": title,
                "abstract": abstract,
                "keywords": keywords,
                "primary_area": paper.get("primary_area", "N/A"),
                "authors": paper.get("authors", []),
                "url": paper.get("url", ""),
                "pdf_url": paper.get("pdf_url", ""),
                "inferred_venue": detected_venue.lower(),
                "source_file": os.path.basename(file_path)
            }

            # Append structural vector configuration
            point = PointStruct(
                id=global_point_id,
                vector=vector,
                payload=payload
            )
            points.append(point)
            global_point_id += 1

            # Flush batch to Qdrant if buffer size is reached
            if len(points) >= BATCH_SIZE:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                points = []

        # Flush any trailing leftovers within this file context
        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(
        f"\n✨ Ingestion cycle complete! Total vectors written: {global_point_id}")


if __name__ == "__main__":
    main()
