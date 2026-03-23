"""
RerankerManager - Generalized reranker for result refinement.

Supports Cohere API and BGE local rerankers for improving search results.
"""

import os
from typing import List, Dict, Optional, Literal
from cohere import Client as CohereClient
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()


class RerankerManager:
    """
    Generalized reranker for refining search results.

    Features:
    - Cohere API reranker (recommended, requires API key)
    - BGE local reranker (free, runs locally)
    - Top-k result selection
    - Score-based result filtering

    Attributes:
        reranker_type: Type of reranker ("cohere" or "bge")
        model: Reranker model instance
        top_k: Number of top results to return

    Example:
        >>> manager = RerankerManager()
        >>> results = await manager.rerank(
        ...     query="refund policy",
        ...     documents=[{"text": "Refunds are processed..."}, ...],
        ...     top_k=3
        ... )
    """

    # Available reranker types
    RERANKER_COHERE = "cohere"
    RERANKER_BGE = "bge"

    # Cohere models
    COHERE_MODELS = {
        "rerank-english-v3.0": "rerank-english-v3.0",  # Best for English
        "rerank-multilingual-v3.0": "rerank-multilingual-v3.0"  # Multilingual
    }

    # BGE reranker models
    BGE_MODELS = {
        "bge-reranker-base": "BAAI/bge-reranker-base",
        "bge-reranker-large": "BAAI/bge-reranker-v2-m3"
    }

    def __init__(
        self,
        reranker_type: Literal["cohere", "bge"] = "cohere",
        model_name: Optional[str] = None,
        top_k: int = 5,
        api_key: Optional[str] = None
    ):
        """
        Initialize RerankerManager.

        Args:
            reranker_type: Type of reranker ("cohere" or "bge")
            model_name: Model name (defaults to best model for each type)
            top_k: Default number of top results to return
            api_key: Cohere API key (for cohere type, defaults to env COHERE_API_KEY)
        """
        self.reranker_type = reranker_type
        self.top_k = top_k

        if reranker_type == self.RERANKER_COHERE:
            # Initialize Cohere client
            if api_key is None:
                api_key = os.getenv("COHERE_API_KEY")

            if not api_key:
                raise ValueError(
                    "Cohere API key required. Set COHERE_API_KEY in .env or pass api_key parameter. "
                    "Alternatively, use reranker_type='bge' for local reranking."
                )

            self.model = CohereClient(api_key=api_key)
            self.model_name = model_name or self.COHERE_MODELS["rerank-english-v3.0"]
            print(f"✅ RerankerManager initialized: Cohere ({self.model_name})")

        elif reranker_type == self.RERANKER_BGE:
            # Initialize BGE reranker
            model_name = model_name or self.BGE_MODELS["bge-reranker-large"]
            self.model = CrossEncoder(model_name)
            self.model_name = model_name
            print(f"✅ RerankerManager initialized: BGE ({self.model_name})")

        else:
            raise ValueError(f"Invalid reranker_type: {reranker_type}. Use 'cohere' or 'bge'")

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, str]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Rerank documents based on query relevance.

        Args:
            query: Search query
            documents: List of document dicts with "text" field
                [{"text": "document content", "id": "doc1", ...}, ...]
            top_k: Number of top results to return (defaults to init top_k)

        Returns:
            List of reranked documents with relevance scores

        Example:
            >>> query = "refund policy"
            >>> documents = [
            ...     {"id": "doc1", "text": "Our refund policy allows..."},
            ...     {"id": "doc2", "text": "Shipping takes 3-5 days..."},
            ...     {"id": "doc3", "text": "Returns are accepted within..."}
            ... ]
            >>> results = await manager.rerank(query, documents, top_k=2)
            >>> # Returns doc1 and doc3 reranked by relevance
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not documents:
            return []

        if top_k is None:
            top_k = self.top_k

        # Extract texts and preserve metadata
        texts = [doc.get("text", "") for doc in documents]

        if self.reranker_type == self.RERANKER_COHERE:
            return await self._rerank_cohere(query, texts, documents, top_k)
        else:
            return await self._rerank_bge(query, texts, documents, top_k)

    async def _rerank_cohere(
        self,
        query: str,
        texts: List[str],
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Rerank using Cohere API."""
        try:
            response = self.model.rerank(
                model=self.model_name,
                query=query,
                documents=texts,
                top_n=top_k,
                return_documents=False
            )

            # Build results
            results = []
            for result in response.results:
                original_doc = documents[result.index]
                results.append({
                    **original_doc,
                    "rerank_score": result.relevance_score,
                    "original_index": result.index
                })

            return results

        except Exception as e:
            print(f"⚠️  Cohere reranking failed: {str(e)}")
            # Fallback: return original documents
            return documents[:top_k]

    async def _rerank_bge(
        self,
        query: str,
        texts: List[str],
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Rerank using BGE local model."""
        try:
            # Create query-document pairs
            pairs = [[query, text] for text in texts]

            # Get scores
            scores = self.model.predict(pairs)

            # Sort by score
            indexed_scores = [(i, score) for i, score in enumerate(scores)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # Build results
            results = []
            for i, (original_idx, score) in enumerate(indexed_scores[:top_k]):
                original_doc = documents[original_idx]
                results.append({
                    **original_doc,
                    "rerank_score": float(score),
                    "original_index": original_idx
                })

            return results

        except Exception as e:
            print(f"⚠️  BGE reranking failed: {str(e)}")
            # Fallback: return original documents
            return documents[:top_k]

    def get_model_info(self) -> Dict[str, str]:
        """Get reranker model information."""
        return {
            "type": self.reranker_type,
            "model": self.model_name,
            "top_k": self.top_k
        }
