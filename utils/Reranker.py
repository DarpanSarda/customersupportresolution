"""Reranker wrapper for cross-encoder models.

Supports BGE reranker and FlashRank for final result refinement.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Single reranked result."""
    content: str
    doc_id: str
    metadata: Dict[str, Any]
    score: float  # Re-reranked score
    original_rank: int  # Original position before reranking


class Reranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """Rerank documents based on query relevance.

        Args:
            query: User query
            documents: List of dicts with 'content', 'doc_id', 'metadata', 'score'
            top_k: Number of top results to return

        Returns:
            List[RerankResult]: Reranked results
        """
        pass


class BGEReranker(Reranker):
    """BGE cross-encoder reranker.

    Model: BAAI/bge-reranker-large or bge-reranker-base
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        """Initialize BGE reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ("cpu" or "cuda")
            cache_dir: Optional cache directory
        """
        self.model_name = model_name
        self.device = device
        self._cache_dir = cache_dir
        self._model = None

    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )

            logger.info(f"Loading reranker model: {self.model_name} on {self.device}")
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                cache_folder=self._cache_dir
            )

    @property
    def model(self):
        """Get loaded model (lazy loading)."""
        if self._model is None:
            self._load_model()
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """Rerank documents using cross-encoder.

        Args:
            query: User query
            documents: List of dicts with 'content', 'doc_id', 'metadata', 'score'
            top_k: Number of top results to return

        Returns:
            List[RerankResult]: Reranked results
        """
        if not documents:
            return []

        # Prepare pairs for cross-encoder
        pairs = [[query, doc.get("content", "")] for doc in documents]

        # Get scores from cross-encoder
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Create reranked results
        reranked = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            reranked.append(RerankResult(
                content=doc.get("content", ""),
                doc_id=doc.get("doc_id", ""),
                metadata=doc.get("metadata", {}),
                score=float(score),
                original_rank=i
            ))

        # Sort by score and return top_k
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]


class FlashRankReranker(Reranker):
    """FlashRank reranker (fast Rust-based reranker).

    Uses the FlashRank Python wrapper for blazing fast reranking.
    Model: ms-marco-MiniLM-L-6-v2 (default)
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2"):
        """Initialize FlashRank reranker.

        Args:
            model_name: Model name (default: ms-marco-MiniLM-L-6-v2)
        """
        self.model_name = model_name
        self._ranker = None

    def _load_ranker(self):
        """Lazy load ranker on first use."""
        if self._ranker is None:
            try:
                from flashrank import Reranker
            except ImportError:
                raise ImportError(
                    "flashrank is required. "
                    "Install with: pip install flashrank"
                )

            logger.info(f"Loading FlashRank reranker: {self.model_name}")
            self._ranker = Reranker(model_name=self.model_name)

    @property
    def ranker(self):
        """Get loaded ranker (lazy loading)."""
        if self._ranker is None:
            self._load_ranker()
        return self._ranker

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[RerankResult]:
        """Rerank documents using FlashRank.

        Args:
            query: User query
            documents: List of dicts with 'content', 'doc_id', 'metadata', 'score'
            top_k: Number of top results to return

        Returns:
            List[RerankResult]: Reranked results
        """
        if not documents:
            return []

        # Prepare documents for FlashRank format
        flashrank_docs = [
            {"id": doc.get("doc_id", ""), "text": doc.get("content", "")}
            for doc in documents
        ]

        # Rerank
        reranked_docs = self.ranker.rank(
            query=query,
            documents=flashrank_docs,
            top_n=top_k
        )

        # Convert to RerankResult format
        results = []
        for result in reranked_docs:
            # Find original doc for metadata
            original_doc = next(
                (d for d in documents if d.get("doc_id") == result.get("id")),
                {}
            )

            results.append(RerankResult(
                content=result.get("text", ""),
                doc_id=result.get("id", ""),
                metadata=original_doc.get("metadata", {}),
                score=result.get("score", 0.0),
                original_rank=-1  # FlashRank doesn't preserve original rank
            ))

        return results


def create_reranker(
    provider: str = "huggingface",
    model_name: Optional[str] = None,
    device: str = "cpu"
) -> Reranker:
    """Factory function to create reranker.

    Args:
        provider: Reranker provider ("huggingface", "flashrank")
        model_name: Model name (optional, uses default if not specified)
        device: Device to run on

    Returns:
        Reranker: Initialized reranker instance
    """
    if provider == "huggingface":
        model = model_name or "BAAI/bge-reranker-large"
        return BGEReranker(model_name=model, device=device)

    elif provider == "flashrank":
        return FlashRankReranker(model_name=model_name or "ms-marco-MiniLM-L-6-v2")

    else:
        raise ValueError(f"Unknown reranker provider: {provider}")
