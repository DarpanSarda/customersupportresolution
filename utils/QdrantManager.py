"""
QdrantManager - Generalized Qdrant vector database manager.

Handles collection management and vector operations with tenant isolation.
"""

import os
import uuid
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from dotenv import load_dotenv

load_dotenv()


class QdrantManager:
    """
    Generalized Qdrant manager for vector operations.

    Features:
    - Collection management (create, delete, recreate, check exists)
    - Document operations (add, search, delete)
    - Tenant isolation via filters and collection naming
    - Configurable vector size and location

    Collection Naming Convention:
    - knowledge_base_{tenant_id} - Each tenant gets their own collection

    Attributes:
        client: QdrantClient instance
        collection_name: Name of the collection (includes tenant_id)
        vector_size: Dimension of vectors

    Example:
        >>> manager = QdrantManager(tenant_id="amazon")
        >>> await manager.add_documents(documents, embeddings)
        >>> results = await manager.search(query_embedding, top_k=5)
    """

    # Default collection prefix
    COLLECTION_PREFIX = "knowledge_base_"

    def __init__(
        self,
        tenant_id: str = "default",
        location: Optional[str] = None,
        vector_size: int = 768,
        recreate_collection: bool = False
    ):
        """
        Initialize QdrantManager.

        Args:
            tenant_id: Tenant identifier (used in collection name)
            location: Qdrant server URL (defaults to env QDRANT_URL)
            vector_size: Vector dimension (must match embedding model)
            recreate_collection: If True, delete and recreate collection
        """
        if location is None:
            location = os.getenv("QDRANT_URL", "http://localhost:6333")

        self.tenant_id = tenant_id
        self.collection_name = f"{self.COLLECTION_PREFIX}{tenant_id}"
        self.vector_size = vector_size

        # Initialize Qdrant client
        self.client = QdrantClient(location=location, check_compatibility=False)

        # Ensure collection exists
        if recreate_collection:
            self.delete_collection()
        self._ensure_collection_exists()

        print(f"✅ QdrantManager initialized: collection={self.collection_name} (dim={vector_size})")

    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created Qdrant collection: {self.collection_name}")

    def collection_exists(self) -> bool:
        """
        Check if the tenant's collection exists.

        Returns:
            True if collection exists, False otherwise
        """
        return self.client.collection_exists(self.collection_name)

    def delete_collection(self):
        """Delete the tenant's collection."""
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)
            print(f"🗑️  Deleted Qdrant collection: {self.collection_name}")

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Add documents to the collection.

        Args:
            documents: List of document dictionaries
                Each document should have:
                - id: Unique identifier (str) - stored in payload, not used as point ID
                - text: Document content (str)
                - metadata: Optional metadata dict (source, title, etc.)
            embeddings: List of embedding vectors (one per document)

        Returns:
            Dict with operation status

        Example:
            >>> documents = [
            ...     {"id": "doc1", "text": "Refund policy...", "metadata": {"source": "policy.pdf"}},
            ...     {"id": "doc2", "text": "Shipping info...", "metadata": {"source": "shipping.pdf"}}
            ... ]
            >>> embeddings = await embedding_manager.embed_batch([d["text"] for d in documents])
            >>> result = await manager.add_documents(documents, embeddings)
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        points = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),  # Generate proper UUID
                vector=embedding,
                payload={
                    "text": doc.get("text", ""),
                    "tenant_id": self.tenant_id,
                    "doc_id": doc.get("id", f"{self.tenant_id}_{idx}"),  # Store original ID in payload
                    **doc.get("metadata", {})
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return {
            "status": "success",
            "added_count": len(documents),
            "collection": self.collection_name
        }

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional payload filters (e.g., {"source": "policy.pdf"})

        Returns:
            List of search results with score and payload

        Example:
            >>> query_emb = await embedding_manager.embed_query("refund policy")
            >>> results = await manager.search(query_emb, top_k=3)
            >>> for result in results:
            ...     print(f"{result['score']:.3f}: {result['payload']['text'][:50]}...")
        """
        # Build filter if provided
        query_filter = None
        if filters:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                for key, value in filters.items()
            ]
            query_filter = Filter(must=conditions)

        # Search (compatible with Qdrant client 1.17.0+)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k
        ).points

        # Format results (query_points returns points with nested score)
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,  # query_points puts score at the point level
                "payload": result.payload
            })

        return formatted_results

    async def delete(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Delete documents by IDs.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Dict with operation status
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=document_ids
        )

        return {
            "status": "success",
            "deleted_count": len(document_ids),
            "collection": self.collection_name
        }

    async def delete_by_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete documents by payload filter.

        Args:
            filters: Filter conditions (e.g., {"source": "policy.pdf"})

        Returns:
            Dict with operation status
        """
        conditions = [
            FieldCondition(
                key=key,
                match=MatchValue(value=value)
            )
            for key, value in filters.items()
        ]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=conditions)
        )

        return {
            "status": "success",
            "filter": filters,
            "collection": self.collection_name
        }

    async def count_documents(self) -> int:
        """
        Count documents in the collection.

        Returns:
            Number of documents in the collection
        """
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get collection information.

        Returns:
            Dict with collection details
        """
        if not self.collection_exists():
            return {
                "exists": False,
                "collection": self.collection_name
            }

        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "collection": self.collection_name,
            "vectors_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance)
        }

    async def delete_tenant_knowledge(self) -> Dict[str, Any]:
        """
        Delete all knowledge for the current tenant.

        This deletes the entire collection for the tenant.

        Returns:
            Dict with operation status
        """
        self.delete_collection()

        return {
            "status": "success",
            "tenant_id": self.tenant_id,
            "message": f"All knowledge deleted for tenant: {self.tenant_id}"
        }
