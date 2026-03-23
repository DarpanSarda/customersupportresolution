"""
EmbeddingManager - Generalized embeddings manager for text vectorization.

Supports BGE (BAAI General Embedding) models with instruction-based query enhancement.
"""

import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class EmbeddingManager:
    """
    Generalized embeddings manager for text vectorization.

    Features:
    - BGE model support (base, large, m3)
    - Instruction-based query enhancement for better retrieval
    - Batch processing
    - Multi-language support (via BGE-m3)

    Attributes:
        model: SentenceTransformer model instance
        dimension: Vector dimension size
        device: Device to run model on ("cpu" or "cuda")
        query_instruction: Instruction prefix for query embeddings

    Example:
        >>> manager = EmbeddingManager()
        >>> embedding = await manager.embed_text("Hello world")
        >>> embeddings = await manager.embed_batch(["Hello", "World"])
        >>> query_emb = await manager.embed_query("Search query")
    """

    # BGE model configurations
    MODELS = {
        "bge-base": "BAAI/bge-base-en-v1.5",
        "bge-large": "BAAI/bge-large-en-v1.5",
        "bge-m3": "BAAI/bge-m3"
    }

    # Query instruction for BGE models (enhances retrieval quality)
    QUERY_INSTRUCTION = "Represent this query for retrieving relevant documents: "

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize EmbeddingManager.

        Args:
            model_name: BGE model name (bge-base, bge-large, bge-m3) or full HuggingFace path
            device: Device to run model on ("cpu" or "cuda", defaults to env DEVICE)
            dimension: Expected dimension (validated against model output)
        """
        # Load from environment if not specified
        if model_name is None:
            env_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            # Map short names to full model names
            model_name = self.MODELS.get(env_model, env_model)

        if device is None:
            device = os.getenv("DEVICE", "cpu")

        if dimension is None:
            dimension = int(os.getenv("DIMENSION", "768"))

        self.model_name = model_name
        self.device = device
        self._expected_dimension = dimension

        # Initialize model
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Validate dimension
        if self.dimension != self._expected_dimension:
            print(
                f"⚠️  Warning: Model dimension ({self.dimension}) differs from "
                f"expected dimension ({self._expected_dimension})"
            )

        print(f"✅ EmbeddingManager initialized: {model_name} (dim={self.dimension}, device={device})")

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector

        Example:
            >>> embedding = await manager.embed_text("Customer wants a refund")
            >>> len(embedding)
            768
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).

        More efficient than calling embed_text multiple times.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors

        Example:
            >>> texts = ["Refund policy", "Shipping info", "Contact support"]
            >>> embeddings = await manager.embed_batch(texts)
            >>> len(embeddings)
            3
        """
        if not texts:
            return []

        # Filter empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []

        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
        return embeddings.tolist()

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query with instruction prefix.

        BGE models perform better with query-specific instructions.
        Use this for search queries, not document indexing.

        Args:
            query: Search query text

        Returns:
            List of floats representing the query embedding

        Example:
            >>> query = "How do I request a refund?"
            >>> query_emb = await manager.embed_query(query)
            >>> len(query_emb)
            768
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Prepend instruction for better retrieval
        query_with_instruction = self.QUERY_INSTRUCTION + query
        embedding = self.model.encode(query_with_instruction, convert_to_numpy=True)
        return embedding.tolist()

    def get_dimension(self) -> int:
        """Get the embedding vector dimension."""
        return self.dimension

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device
        }
