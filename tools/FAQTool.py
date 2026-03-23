"""
FAQTool - Tool for FAQ lookups from vector store.

Provides FAQ search functionality using Qdrant vector database.
Frequently asked questions are stored with embeddings for semantic search.
"""

import os
from typing import Dict, Any, Optional, List
from core.BaseTools import BaseTool
from models.tool import ToolResult, ToolConfig
from utils.QdrantManager import QdrantManager
from utils.EmbeddingManager import EmbeddingManager
from dotenv import load_dotenv

load_dotenv()


class FAQTool(BaseTool):
    """
    FAQ lookup tool using vector search.

    Searches for frequently asked questions in the FAQ collection
    using semantic similarity with embeddings.

    Collection naming: faq_{tenant_id}

    Example:
        >>> tool = FAQTool()
        >>> result = await tool.execute({
        ...     "query": "What is the refund policy?",
        ...     "tenant_id": "amazon",
        ...     "top_k": 3
        ... })
    """

    def __init__(self, config: Optional[ToolConfig] = None):
        """
        Initialize FAQTool.

        Args:
            config: Tool configuration (optional)
        """
        super().__init__(config)

        # Initialize embeddings manager
        self._embeddings: Optional[EmbeddingManager] = None

        # Qdrant managers per tenant (cached)
        self._qdrant_managers: Dict[str, QdrantManager] = {}

    @property
    def embeddings(self) -> EmbeddingManager:
        """Lazy load embeddings manager."""
        if self._embeddings is None:
            self._embeddings = EmbeddingManager()
        return self._embeddings

    def _get_qdrant(self, tenant_id: str) -> QdrantManager:
        """
        Get or create Qdrant manager for tenant's FAQ collection.

        Args:
            tenant_id: Tenant identifier

        Returns:
            QdrantManager instance
        """
        if tenant_id not in self._qdrant_managers:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            vector_size = self.embeddings.dimension

            manager = QdrantManager(
                tenant_id=tenant_id,
                vector_size=vector_size,
                url=qdrant_url
            )

            # Override collection name for FAQ
            manager.collection_name = f"faq_{tenant_id}"

            self._qdrant_managers[tenant_id] = manager

        return self._qdrant_managers[tenant_id]

    async def execute(self, payload: Dict[str, Any]) -> ToolResult:
        """
        Execute FAQ search.

        Args:
            payload: Input data with keys:
                - query (str): Search query
                - tenant_id (str): Tenant identifier
                - top_k (int, optional): Number of results (default: 3)
                - threshold (float, optional): Similarity threshold (default: 0.0)

        Returns:
            ToolResult with FAQ matches
        """
        import time
        start_time = time.time()

        # Validate required fields
        is_valid, error = self.validate_payload(payload, ["query", "tenant_id"])
        if not is_valid:
            return ToolResult.failed(error)

        query = payload["query"]
        tenant_id = payload["tenant_id"]
        top_k = payload.get("top_k", 3)
        threshold = payload.get("threshold", 0.0)

        try:
            # Get Qdrant manager for tenant
            qdrant = self._get_qdrant(tenant_id)

            # Check if collection exists
            collection_info = qdrant.get_collection_info()
            if not collection_info or not collection_info.get("exists", False):
                # Collection doesn't exist
                return ToolResult.success(
                    data={
                        "results": [],
                        "total_found": 0,
                        "message": "No FAQ collection found for this tenant. Please ingest FAQs first."
                    },
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )

            # Generate embedding for query
            query_embedding = await self.embeddings.embed_query(query)

            # Search in Qdrant
            results = await qdrant.search(
                query_embedding=query_embedding,
                top_k=top_k * 2  # Get more to filter by threshold
            )

            # Filter by threshold and format results
            faq_results = []
            for result in results:
                score = result.get("score", 0.0)
                if score >= threshold:
                    payload = result.get("payload", {})
                    faq_results.append({
                        "question": payload.get("text", ""),
                        "answer": payload.get("answer", ""),
                        "score": score,
                        "source": payload.get("source", "")
                    })

            # Limit to top_k
            faq_results = faq_results[:top_k]

            execution_time_ms = int((time.time() - start_time) * 1000)

            return ToolResult.success(
                data={
                    "results": faq_results,
                    "total_found": len(faq_results),
                    "query": query,
                    "tenant_id": tenant_id,
                    "threshold": threshold,
                    "top_k": top_k
                },
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            return ToolResult.failed(
                error=f"FAQ search failed: {str(e)}",
                error_code="faq_search_error"
            )

    async def exact_match_search(
        self,
        query: str,
        tenant_id: str,
        faq_dict: Optional[Dict[str, str]] = None
    ) -> ToolResult:
        """
        Perform exact/normalized match search for FAQs.

        This is a fast lookup for FAQ exact matches using text normalization.
        Useful for quick FAQ hits without vector search.

        Args:
            query: User query
            tenant_id: Tenant identifier
            faq_dict: Optional pre-loaded FAQ dict {normalized_question: answer}

        Returns:
            ToolResult with exact match if found
        """
        import re
        from typing import Dict

        # If no FAQ dict provided, try to get from config
        if faq_dict is None:
            faq_dict = self.config.url if self.config else {}  # Use config.url as faq_dict storage

        # Normalize query for matching
        normalized_query = self._normalize_text(query)

        # Check for exact match
        if normalized_query in faq_dict:
            return ToolResult.success(
                data={
                    "results": [{
                        "question": query,
                        "answer": faq_dict[normalized_query],
                        "score": 1.0,  # Exact match = 100% confidence
                        "match_type": "exact"
                    }],
                    "total_found": 1,
                    "match_type": "exact"
                }
            )

        # No exact match found
        return ToolResult.success(
            data={
                "results": [],
                "total_found": 0,
                "match_type": "no_exact_match"
            }
        )

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for exact matching.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def get_info(self) -> Dict[str, Any]:
        """Get FAQTool information."""
        info = super().get_info()
        info.update({
            "description": "FAQ lookup tool using vector search",
            "collection_pattern": "faq_{tenant_id}",
            "search_modes": ["vector", "exact_match"]
        })
        return info
