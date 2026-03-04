"""Abstract vector store interface with Qdrant implementation.

Supports multi-tenant collections and cascade retrieval.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result from vector store."""
    content: str
    doc_id: str
    metadata: Dict[str, Any]
    score: float
    payload: Optional[Dict[str, Any]] = None


class VectorStore(ABC):
    """Abstract vector store interface.

    Allows switching between Qdrant, ChromaDB, Pinecone, etc.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        collection_name: str = "default"
    ) -> None:
        """Add documents with embeddings to vector store.

        Args:
            documents: List of document text content
            ids: List of unique document IDs
            embeddings: List of embedding vectors (pre-computed)
            metadata: List of metadata dicts
            collection_name: Collection to add to
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        collection_name: str = "default",
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents by embedding.

        Args:
            query_embedding: Query embedding vector
            collection_name: Collection to search in
            top_k: Number of results to return
            score_threshold: Minimum score threshold (optional)
            filter_metadata: Metadata filter (optional)

        Returns:
            List[SearchResult]: Search results
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.

        Args:
            collection_name: Collection to delete
        """
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.

        Args:
            collection_name: Collection name to check

        Returns:
            bool: True if collection exists
        """
        pass

    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics.

        Args:
            collection_name: Collection name

        Returns:
            Dict with stats (count, dimension, etc.)
        """
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation.

    Supports:
    - Multi-tenant collections
    - Metadata filtering
    - Score thresholding
    - Cascade retrieval (multiple embedding dimensions)
    """

    def __init__(
        self,
        url: str = "http://103.180.31.44:8082",
        api_key: Optional[str] = None,
        timeout: int = 60,
        prefer_grpc: bool = False
    ):
        """Initialize Qdrant vector store.

        Args:
            url: Qdrant server URL
            api_key: Optional API key for cloud Qdrant
            timeout: Request timeout in seconds
            prefer_grpc: Whether to use gRPC (faster for large operations)
        """
        self._url = url
        self._api_key = api_key
        self._timeout = timeout
        self._prefer_grpc = prefer_grpc
        self._client = None
        self._collections: Dict[str, Any] = {}

    def _get_client(self):
        """Get or create Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError(
                    "qdrant-client is required. "
                    "Install with: pip install qdrant-client"
                )

            self._client = QdrantClient(
                url=self._url,
                api_key=self._api_key,
                timeout=self._timeout,
                prefer_grpc=self._prefer_grpc,
                check_compatibility=False
            )

            logger.info(f"Connected to Qdrant at {self._url}")

        return self._client

    @property
    def client(self):
        """Get Qdrant client (lazy loading)."""
        return self._get_client()

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        collection_name: str = "default"
    ) -> None:
        """Add documents with embeddings to Qdrant.

        Args:
            documents: List of document text content
            ids: List of unique document IDs
            embeddings: List of embedding vectors (pre-computed)
            metadata: List of metadata dicts
            collection_name: Collection to add to
        """
        if len(documents) != len(embeddings) or len(documents) != len(ids) or len(documents) != len(metadata):
            raise ValueError("documents, ids, embeddings, and metadata must have same length")

        if not embeddings:
            return

        # Get embedding dimension from first embedding
        vector_size = len(embeddings[0])

        # Ensure collection exists
        self._ensure_collection(collection_name, vector_size)

        # Prepare points for Qdrant
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=doc_id,
                vector=embedding,
                payload={
                    "content": content,
                    **meta
                }
            )
            for doc_id, content, embedding, meta in zip(ids, documents, embeddings, metadata)
        ]

        # Upload in batches (Qdrant recommends max 100-1000 per batch)
        batch_size = 500
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )
            logger.debug(f"Uploaded batch {i // batch_size + 1} to {collection_name}")

        logger.info(f"Added {len(points)} documents to collection '{collection_name}'")

    def search(
        self,
        query_embedding: List[float],
        collection_name: str = "default",
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents by embedding.

        Args:
            query_embedding: Query embedding vector
            collection_name: Collection to search in
            top_k: Number of results to return
            score_threshold: Minimum score threshold (optional)
            filter_metadata: Metadata filter (optional)

        Returns:
            List[SearchResult]: Search results
        """
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist")
            return []

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter if specified
        search_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                for key, value in filter_metadata.items()
            ]
            search_filter = Filter(must=conditions)

        # Search - use query_points for newer qdrant-client versions
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=search_filter,
            score_threshold=score_threshold
        ).points

        # Convert to SearchResult format
        return [
            SearchResult(
                content=result.payload.get("content", ""),
                doc_id=str(result.id),
                metadata={k: v for k, v in result.payload.items() if k != "content"},
                score=result.score if hasattr(result, 'score') else 0.0,
                payload=result.payload
            )
            for result in results
        ]

    def _ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Ensure collection exists, create if not.

        Args:
            collection_name: Collection name
            vector_size: Vector dimension
        """
        if self.collection_exists(collection_name):
            return

        from qdrant_client.models import Distance, VectorParams, CreateCollection

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE  # BGE models use cosine similarity
            )
        )

        logger.info(f"Created collection '{collection_name}' with vector_size={vector_size}")

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.

        Args:
            collection_name: Collection name to check

        Returns:
            bool: True if collection exists
        """
        try:
            collections = self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection.

        Args:
            collection_name: Collection to delete
        """
        if self.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics.

        Args:
            collection_name: Collection name

        Returns:
            Dict with stats (count, dimension, etc.)
        """
        if not self.collection_exists(collection_name):
            return {"exists": False}

        info = self.client.get_collection(collection_name)

        return {
            "exists": True,
            "name": collection_name,
            "count": info.points_count,
            "vector_size": info.config.params.vectors.size if info.config.params.vectors else None
        }


class CascadeVectorStore:
    """Vector store wrapper for cascade retrieval.

    Manages multiple collections for different embedding dimensions:
    - Stage 1: 384 dims (bge-small)
    - Stage 2: 768 dims (bge-base)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        collection_prefix: str = "cs_"
    ):
        """Initialize cascade vector store.

        Args:
            vector_store: Base vector store instance
            collection_prefix: Prefix for collection names
        """
        self.vector_store = vector_store
        self.collection_prefix = collection_prefix

    def get_collection_name(
        self,
        tenant_id: str,
        stage: str = "stage_1"
    ) -> str:
        """Get collection name for tenant and stage.

        Args:
            tenant_id: Tenant identifier
            stage: Cascade stage ("stage_1" or "stage_2")

        Returns:
            str: Collection name
        """
        # Format: {prefix}{tenant}_{stage}
        # Example: cs_test_stage_1, cs_amazon_stage_2
        return f"{self.collection_prefix}{tenant_id}_{stage}"

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        tenant_id: str,
        stage: str = "stage_1"
    ) -> None:
        """Add documents to tenant-specific stage collection.

        Args:
            documents: List of document text
            ids: List of unique IDs
            embeddings: List of embedding vectors
            metadata: List of metadata dicts
            tenant_id: Tenant identifier
            stage: Cascade stage
        """
        collection_name = self.get_collection_name(tenant_id, stage)
        self.vector_store.add_documents(
            documents=documents,
            ids=ids,
            embeddings=embeddings,
            metadata=metadata,
            collection_name=collection_name
        )

    def search(
        self,
        query_embedding: List[float],
        tenant_id: str,
        stage: str = "stage_1",
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search in tenant-specific stage collection.

        Args:
            query_embedding: Query embedding vector
            tenant_id: Tenant identifier
            stage: Cascade stage
            top_k: Number of results
            score_threshold: Minimum score
            filter_metadata: Metadata filter

        Returns:
            List[SearchResult]: Search results
        """
        collection_name = self.get_collection_name(tenant_id, stage)
        return self.vector_store.search(
            query_embedding=query_embedding,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_metadata=filter_metadata
        )

    def delete_tenant(self, tenant_id: str) -> None:
        """Delete all collections for a tenant.

        Args:
            tenant_id: Tenant identifier
        """
        for stage in ["stage_1", "stage_2"]:
            collection_name = self.get_collection_name(tenant_id, stage)
            self.vector_store.delete_collection(collection_name)
