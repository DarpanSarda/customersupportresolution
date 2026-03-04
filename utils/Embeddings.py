"""HuggingFace embedding model wrapper for BGE models.

Supports cascade retrieval with multiple BGE models:
- bge-small-en-v1.5: 384 dims, fast (Stage 1)
- bge-base-en-v1.5: 768 dims, quality (Stage 2)
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract embedding model interface."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass


class HuggingFaceEmbeddings(EmbeddingModel):
    """HuggingFace sentence-transformers embeddings.

    Supports BGE models optimized for retrieval:
    - BAAI/bge-small-en-v1.5: 384 dims
    - BAAI/bge-base-en-v1.5: 768 dims
    - BAAI/bge-large-en-v1.5: 1024 dims
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None
    ):
        """Initialize HuggingFace embedding model.

        Args:
            model_name: HuggingFace model name (default: bge-small)
            device: Device to run on ("cpu" or "cuda")
            normalize_embeddings: Whether to normalize embeddings (important for cosine similarity)
            cache_dir: Optional cache directory for models
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = None
        self._cache_dir = cache_dir

    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )

            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(
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

    def embed(self, text: str) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Input text to embed

        Returns:
            List[float]: Embedding vector
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []

        # BGE models expect instruction format for queries
        # Format: "Represent this sentence for searching relevant passages:"
        encoded = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False
        )

        return encoded.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension for this model."""
        # Known dimensions for BGE models
        dimensions = {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-m3": 1024,
        }

        if self.model_name in dimensions:
            return dimensions[self.model_name]

        # Fallback: load model and check
        self._load_model()
        return self.model.get_sentence_embedding_dimension()

    def encode_query(self, query: str) -> List[float]:
        """Encode query with instruction prefix (BGE specific).

        BGE models perform better with instruction prefix for queries:
        "Represent this sentence for searching relevant passages:"

        Args:
            query: User query

        Returns:
            List[float]: Query embedding
        """
        instruction = "Represent this sentence for searching relevant passages: "
        query_with_instruction = instruction + query
        return self.embed(query_with_instruction)


class CascadeEmbeddings:
    """Manages multiple embedding models for cascade retrieval.

    Usage:
        cascade = CascadeEmbeddings({
            "stage_1": HuggingFaceEmbeddings("BAAI/bge-small-en-v1.5"),
            "stage_2": HuggingFaceEmbeddings("BAAI/bge-base-en-v1.5"),
        })

        stage_1_embedding = cascade.embed("query", stage="stage_1")
    """

    def __init__(
        self,
        models: Dict[str, EmbeddingModel],
        default_stage: str = "stage_1"
    ):
        """Initialize cascade embeddings.

        Args:
            models: Dict mapping stage names to embedding models
            default_stage: Default stage to use if not specified
        """
        self.models = models
        self.default_stage = default_stage

    def embed(self, text: str, stage: Optional[str] = None) -> List[float]:
        """Generate embedding using specified stage model.

        Args:
            text: Input text
            stage: Which stage model to use (default: default_stage)

        Returns:
            List[float]: Embedding vector
        """
        stage = stage or self.default_stage
        if stage not in self.models:
            raise ValueError(f"Stage '{stage}' not found. Available: {list(self.models.keys())}")

        return self.models[stage].embed(text)

    def embed_batch(
        self,
        texts: List[str],
        stage: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            stage: Which stage model to use

        Returns:
            List[List[float]]: Embedding vectors
        """
        stage = stage or self.default_stage
        if stage not in self.models:
            raise ValueError(f"Stage '{stage}' not found. Available: {list(self.models.keys())}")

        return self.models[stage].embed_batch(texts)

    def get_model(self, stage: str) -> EmbeddingModel:
        """Get embedding model for specific stage.

        Args:
            stage: Stage name

        Returns:
            EmbeddingModel: The embedding model
        """
        if stage not in self.models:
            raise ValueError(f"Stage '{stage}' not found. Available: {list(self.models.keys())}")

        return self.models[stage]

    @classmethod
    def from_config(cls, config: Dict[str, Any], device: str = "cpu") -> "CascadeEmbeddings":
        """Create CascadeEmbeddings from configuration.

        Args:
            config: RAG embeddings config from Config.py
            device: Device to run models on

        Returns:
            CascadeEmbeddings: Initialized cascade embeddings
        """
        models = {}

        # Stage 1: Fast, broad search
        stage_1_config = config.get("stage_1", {})
        models["stage_1"] = HuggingFaceEmbeddings(
            model_name=stage_1_config.get("model", "BAAI/bge-small-en-v1.5"),
            device=device,
            normalize_embeddings=config.get("normalize_embeddings", True)
        )

        # Stage 2: Quality refinement
        stage_2_config = config.get("stage_2", {})
        if stage_2_config:
            models["stage_2"] = HuggingFaceEmbeddings(
                model_name=stage_2_config.get("model", "BAAI/bge-base-en-v1.5"),
                device=device,
                normalize_embeddings=config.get("normalize_embeddings", True)
            )

        return cls(models, default_stage="stage_1")
