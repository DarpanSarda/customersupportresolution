"""
Amazon FAQ Ingestion Script

Loads Amazon FAQ data into the Qdrant vector store for FAQ lookup.
Run with: python scripts/ingest_amazon_faqs.py
"""

import sys
import os
import json
import asyncio

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


async def load_config():
    """Load embedding configuration."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "embeddings.json")

    if not os.path.exists(config_path):
        # Fallback configuration
        return {
            "model_name": "all-MiniLM-L6-v2",
            "vector_size": 384,
            "device": "cpu"
        }

    with open(config_path, "r") as f:
        return json.load(f)


async def ingest_amazon_faqs():
    """Ingest Amazon FAQs into Qdrant vector store."""

    print("=" * 60)
    print("Amazon FAQ Ingestion Script")
    print("=" * 60)

    # Load FAQ data
    faq_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "amazon_faqs.json")

    if not os.path.exists(faq_path):
        print(f"ERROR: FAQ file not found at {faq_path}")
        print("Please ensure data/amazon_faqs.json exists.")
        return

    with open(faq_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    tenant_id = faq_data.get("tenant_id", "amazon")
    faqs = faq_data.get("faqs", [])

    print(f"\nLoaded {len(faqs)} FAQs for tenant: {tenant_id}")

    # Load configuration
    config = await load_config()
    model_name = config.get("model_name", "all-MiniLM-L6-v2")
    vector_size = config.get("vector_size", 384)

    print(f"\nEmbedding model: {model_name}")
    print(f"Vector size: {vector_size}")

    # Initialize embedding model
    print("\nLoading embedding model...")
    embedder = SentenceTransformer(model_name)
    print("Embedding model loaded successfully.")

    # Initialize Qdrant client
    print("\nConnecting to Qdrant...")
    qdrant = QdrantClient(url="http://localhost:6333")

    collection_name = f"faq_{tenant_id}"

    # Check if collection exists, delete and recreate
    collections = qdrant.get_collections().collections
    collection_names = [c.name for c in collections]

    if collection_name in collection_names:
        print(f"\nCollection '{collection_name}' already exists.")
        response = input("Do you want to delete and recreate it? (y/N): ").strip().lower()
        if response == 'y':
            qdrant.delete_collection(collection_name)
            print(f"Deleted collection '{collection_name}'")
        else:
            print("Aborting ingestion.")
            return

    # Create collection
    print(f"\nCreating collection '{collection_name}'...")
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Collection '{collection_name}' created successfully.")

    # Embed and insert FAQs
    print(f"\nEmbedding and inserting {len(faqs)} FAQs...")
    points = []

    for idx, faq in enumerate(faqs):
        question = faq.get("question", "")
        answer = faq.get("answer", "")
        category = faq.get("category", "")

        # Combine question and answer for better semantic search
        text_to_embed = f"{question} {answer}"

        # Generate embedding
        embedding = embedder.encode(text_to_embed).tolist()

        # Create point
        point = PointStruct(
            id=idx + 1,
            vector=embedding,
            payload={
                "question": question,
                "answer": answer,
                "category": category,
                "keywords": faq.get("keywords", []),
                "priority": faq.get("priority", "medium")
            }
        )
        points.append(point)

        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(faqs)} FAQs...")

    # Batch insert
    print(f"\nInserting {len(points)} points into Qdrant...")
    qdrant.upsert(
        collection_name=collection_name,
        points=points
    )

    print(f"\n{'=' * 60}")
    print(f"SUCCESS: Ingested {len(faqs)} Amazon FAQs!")
    print(f"{'=' * 60}")
    print(f"\nCollection: {collection_name}")
    print(f"Total FAQs: {len(faqs)}")
    print(f"Categories: {set(faq.get('category') for faq in faqs)}")
    print(f"\nYou can now use FAQTool to search these FAQs.")


def main():
    """Main entry point."""
    try:
        asyncio.run(ingest_amazon_faqs())
    except KeyboardInterrupt:
        print("\n\nIngestion cancelled by user.")
    except Exception as e:
        print(f"\n\nERROR: Ingestion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
