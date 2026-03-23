# RAG Agent Implementation Plan

## Overview

The **RAG (Retrieval-Augmented Generation) Agent** is responsible for searching and retrieving relevant knowledge from the system's knowledge base to support customer support responses. It acts as the knowledge retrieval layer in the multi-agent architecture.

---

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Vector DB** | **Qdrant** | High-performance, filtering, hybrid search support |
| **Embeddings** | **BGE (BAAI General Embedding)** | Open-source, excellent retrieval quality, multilingual support |
| **Reranker** | **Cohere Rerank / BGE Reranker** | Cross-encoder for result refinement, significantly improves accuracy |
| **Chunking** | **LangChain TextSplitter** | Robust strategies: RecursiveCharacter, Semantic, etc. |

### Two-Stage RAG Pipeline

```
User Query
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: FAQ Exact Match Check                         │
│ ├── Normalize query (lowercase, trim, remove punctuation)│
│ ├── Compare with FAQ questions (100% exact match)       │
│ └── Match found? → Return FAQ answer immediately ✅     │
│                                                         │
│ No match? → Proceed to Stage 2                          │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ STAGE 2: Knowledge Base Search                         │
│ ├── Query Embedding (BGE)                               │
│ ├── Vector Search (Qdrant) → Top-K (e.g., top 50)       │
│ ├── Reranker (Cohere/BGE) → Re-rank to Top-N (e.g., 5)  │
│ └── Return relevant passages with citations ✅          │
└─────────────────────────────────────────────────────────┘
```

### FAQ Matching Strategy

| Strategy | Description |
|----------|-------------|
| **Normalization** | Lowercase, trim, remove punctuation before comparison |
| **Exact Match** | 100% string match after normalization |
| **No Fuzzy Matching** | Only return FAQ if perfect match exists |
| **Early Exit** | Return immediately on FAQ match, skip KB search |

This approach ensures:
- **Fast responses** for common questions (FAQs)
- **Accurate answers** from verified FAQ content
- **Efficient** - no vector search needed for FAQ matches
- **Fallback** to knowledge base for complex queries

---

## Architecture Context

### Where RAG Agent Fits

```
User Message
    ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 1: Understanding Agents                          │
│  ├── IntentAgent ✅ (Completed)                         │
│  └── SentimentAgent ✅ (Completed)                      │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 2: Knowledge & Decision Agents                   │
│  ├── RAGRetrievalAgent 🔄 (Current Task)                │
│  ├── VannaAgent (Separate - for DB queries)              │
│  ├── PolicyEvaluationAgent (Pending)                    │
│  └── ContextBuilderAgent (Pending)                      │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 3: Response & Action Agents                      │
│  ├── ResolutionGeneratorAgent (Pending)                 │
│  ├── EscalationDecisionAgent (Pending)                  │
│  └── TicketActionAgent (Pending)                        │
└─────────────────────────────────────────────────────────┘
```

### Data Flow with RAG Agent

```
User Message
    ↓
IntentAgent → Intent classification
SentimentAgent → Sentiment analysis
    ↓
RAGRetrievalAgent → Two-stage retrieval
    │
    ├─ STAGE 1: FAQ Exact Match
    │   ├── Normalize query
    │   ├── 100% exact match check
    │   └─ Found? → Return FAQ answer immediately
    │
    └─ STAGE 2: Knowledge Base (if no FAQ match)
        ├── Query Embedding (BGE)
        ├── Vector Search (Qdrant) → Top-K
        ├── Reranker (Cohere/BGE) → Top-N
        └─ Return relevant passages with citations
    ↓
Returns: relevant_passages[], citations[], source_type (faq/knowledge), confidence_score
```

---

## RAG Agent Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Two-Stage Pipeline** | FAQ first (100% match), then KB search | High |
| **FAQ Exact Match** | 100% string matching after normalization | High |
| **Early Exit** | Return immediately on FAQ match | High |
| **Vector Search (Qdrant)** | Semantic similarity search using BGE embeddings | High |
| **Reranking** | Cohere/BGE reranker for result refinement | High |
| **LangChain Chunking** | Robust text splitting strategies | High |
| **Citation Support** | Return source references | Medium |
| **Tenant Isolation** | Tenant-specific knowledge bases | High |
| **Confidence Scoring** | Return retrieval confidence | High |
| **Source Type Tracking** | Indicate if result from FAQ or KB | High |
| **API-based Training** | Knowledge ingestion via REST API | High |

---

## Implementation Phases

### Phase 1: Utils Managers (Core Infrastructure)

#### 1.1 QdrantManager

**File:** `utils/QdrantManager.py`

```python
"""
QdrantManager - Generalized Qdrant vector store manager.
Provides methods for collection management, document operations, and search.
"""

from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, CreateCollection
import os


class QdrantManager:
    """
    Generalized Qdrant manager for vector operations.

    Features:
    - Collection management (create, delete, recreate, check exists)
    - Document operations (add, search, delete)
    - Tenant isolation via filters
    - Configurable vector size and location
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        location: Optional[str] = None,
        vector_size: int = 768,  # BGE-base dimension
        recreate_collection: bool = False
    ):
        """
        Initialize Qdrant manager.

        Args:
            collection_name: Name of the collection
            location: Qdrant storage location (default: ./data/qdrant)
            vector_size: Vector dimension size
            recreate_collection: Whether to recreate collection on init
        """
        self.collection_name = collection_name
        self.location = location or os.getenv("QDRANT_LOCATION", "./data/qdrant")
        self.vector_size = vector_size

        # Initialize Qdrant client
        self.client = QdrantClient(path=self.location)

        # Ensure collection exists
        self._ensure_collection(recreate_collection)

    def _ensure_collection(self, recreate: bool = False):
        """Ensure collection exists, recreate if requested"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if recreate and self.collection_name in collection_names:
            self.delete_collection()

        if self.collection_name not in collection_names or recreate:
            self.create_collection()

    def create_collection(self):
        """Create a new collection"""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Created Qdrant collection: {self.collection_name}")

    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection_name)
        print(f"🗑️ Deleted Qdrant collection: {self.collection_name}")

    def collection_exists(self) -> bool:
        """Check if collection exists"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        return self.collection_name in collection_names

    def get_collection_info(self) -> Dict:
        """Get collection information"""
        if not self.collection_exists():
            return {"exists": False}

        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status
        }

    async def add_documents(
        self,
        documents: List[Dict],
        embeddings: List[List[float]]
    ) -> Dict:
        """
        Add documents with embeddings to Qdrant.

        Args:
            documents: List of document dicts with metadata
            embeddings: List of embedding vectors

        Returns:
            Result dict with status and count
        """
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=doc.get("id", f"doc_{i}"),
                vector=embedding,
                payload={k: v for k, v in doc.items() if k != "id"}
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return {
            "success": True,
            "added_count": len(points),
            "collection": self.collection_name
        }

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search Qdrant for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional payload filters (e.g., {"tenant_id": "default"})
            score_threshold: Minimum similarity score

        Returns:
            List of search results with metadata
        """
        # Build filter if provided
        query_filter = None
        if filters:
            conditions = [
                FieldCondition(
                    key=k,
                    match=MatchValue(value=v)
                )
                for k, v in filters.items()
            ]
            query_filter = Filter(must=conditions)

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )

        return [
            {
                "id": str(result.id),
                "text": result.payload.get("text", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                "score": result.score
            }
            for result in results
        ]

    async def delete(self, document_ids: List[str]) -> Dict:
        """
        Delete documents by ID.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Result dict with status
        """
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=document_ids)
        )

        return {
            "success": True,
            "deleted_count": len(document_ids)
        }

    async def delete_by_filter(self, filters: Dict) -> Dict:
        """
        Delete documents matching filter.

        Args:
            filters: Filter dict (e.g., {"tenant_id": "default"})

        Returns:
            Result dict with status
        """
        from qdrant_client.models import Filter

        conditions = [
            FieldCondition(
                key=k,
                match=MatchValue(value=v)
            )
            for k, v in filters.items()
        ]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(must=conditions)
        )

        return {
            "success": True,
            "filter": filters
        }

    async def count_documents(self, filters: Optional[Dict] = None) -> int:
        """
        Count documents in collection.

        Args:
            filters: Optional filters to count by

        Returns:
            Document count
        """
        if not filters:
            info = self.client.get_collection(self.collection_name)
            return info.points_count

        # Count with filter
        from qdrant_client.models import Filter
        conditions = [
            FieldCondition(
                key=k,
                match=MatchValue(value=v)
            )
            for k, v in filters.items()
        ]

        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(must=conditions)
        )

        return result.count
```

#### 1.2 EmbeddingManager

**File:** `utils/EmbeddingManager.py`

```python
"""
EmbeddingManager - Generalized embeddings service.
Supports BGE embeddings with instruction-based query enhancement.
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import os


class EmbeddingManager:
    """
    Generalized embeddings manager for text vectorization.

    Features:
    - BGE model support (base, large, m3)
    - Instruction-based query enhancement
    - Batch processing
    - Multi-language support (via BGE-m3)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize embedding manager.

        Args:
            model_name: BGE model to use
                - "BAAI/bge-base-en-v1.5" - English, 768 dim (default)
                - "BAAI/bge-large-en-v1.5" - English, 1024 dim
                - "BAAI/bge-m3" - Multilingual, 1024 dim
            device: Device to use ("cpu" or "cuda", default: from env or "cpu")
        """
        self.model_name = model_name or os.getenv(
            "BGE_MODEL",
            "BAAI/bge-base-en-v1.5"
        )
        self.device = device or os.getenv("BGE_DEVICE", "cpu")

        # Load model
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Embedding model loaded (dimension: {self.dimension})")

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for search query with instruction prefix.

        BGE models perform better with instruction-based queries.

        Args:
            query: Search query

        Returns:
            Embedding vector as list of floats
        """
        # Instruction prefix for BGE models (improves retrieval)
        instruction = "Represent this sentence for searching relevant passages: "
        query_with_instruction = instruction + query
        return await self.embed_text(query_with_instruction)

    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.dimension

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name
```

#### 1.3 RerankerManager

**File:** `utils/RerankerManager.py`

```python
"""
RerankerManager - Generalized reranker service.
Supports Cohere Rerank API and BGE cross-encoder reranker.
"""

from typing import List, Dict, Optional
import os


class RerankerManager:
    """
    Generalized reranker manager for result refinement.

    Supports:
    - Cohere Rerank API (cloud-based, high quality)
    - BGE Reranker (local, free)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize reranker manager.

        Args:
            provider: Reranker provider ("cohere" or "bge")
            model: Model name (provider-specific)
            api_key: API key for Cohere (if using Cohere)
        """
        self.provider = provider or os.getenv("RERANKER_PROVIDER", "bge")
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self._reranker = None

        # Initialize reranker
        if self.provider == "cohere":
            self._init_cohere()
        elif self.provider == "bge":
            self._init_bge()
        else:
            print(f"Warning: Unknown provider '{self.provider}', reranking disabled")

    def _init_cohere(self):
        """Initialize Cohere reranker"""
        if not self.api_key:
            print("Warning: COHERE_API_KEY not set, Cohere reranking disabled")
            return

        self.model = self.model or "rerank-english-v2.0"
        print(f"✅ Cohere reranker initialized (model: {self.model})")

    def _init_bge(self):
        """Initialize BGE reranker"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = self.model or "BAAI/bge-reranker-base"
            self._reranker = CrossEncoder(self.model)
            print(f"✅ BGE reranker initialized (model: {self.model})")
        except Exception as e:
            print(f"Error initializing BGE reranker: {e}")

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank documents based on query relevance.

        Args:
            query: Search query
            documents: List of document dicts with "text" field
            top_k: Number of top results to return

        Returns:
            Reranked list of documents
        """
        if not documents:
            return []

        if self.provider == "cohere":
            return await self._rerank_cohere(query, documents, top_k)
        elif self.provider == "bge" and self._reranker:
            return await self._rerank_bge(query, documents, top_k)

        # Fallback: return original documents
        print(f"Reranking not available, returning original order")
        return documents[:top_k]

    async def _rerank_cohere(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Rerank using Cohere API"""
        if not self.api_key:
            return documents[:top_k]

        try:
            import cohere
            co = cohere.Client(self.api_key)

            # Extract texts
            texts = [doc.get("text", "") for doc in documents]

            # Rerank
            results = co.rerank(
                model=self.model,
                query=query,
                documents=texts,
                top_n=top_k
            )

            # Reorder documents based on rerank results
            reranked = []
            for result in results:
                doc = documents[result.index].copy()
                doc["rerank_score"] = result.relevance_score
                reranked.append(doc)

            return reranked

        except Exception as e:
            print(f"Cohere reranking error: {e}")
            return documents[:top_k]

    async def _rerank_bge(
        self,
        query: str,
        documents: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """Rerank using BGE cross-encoder"""
        try:
            # Create query-document pairs
            pairs = [[query, doc.get("text", "")] for doc in documents]

            # Score pairs
            scores = self._reranker.predict(pairs)

            # Add scores and sort
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)

            # Sort by score
            documents.sort(key=lambda x: x["rerank_score"], reverse=True)

            return documents[:top_k]

        except Exception as e:
            print(f"BGE reranking error: {e}")
            return documents[:top_k]

    def get_provider(self) -> str:
        """Get the reranker provider"""
        return self.provider
```

#### 1.4 DocumentProcessor

**File:** `utils/DocumentProcessor.py`

```python
"""
DocumentProcessor - Generalized document processing utility.
Supports PDF, TXT, MD, HTML formats with LangChain chunking.
"""

from typing import List, Dict, Optional
from pathlib import Path
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
)


class DocumentProcessor:
    """
    Generalized document processor for knowledge base ingestion.

    Features:
    - Multiple format support (PDF, TXT, MD, HTML)
    - LangChain-based chunking
    - Configurable chunk size and overlap
    - Custom separators
    - Metadata extraction
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            separators: Custom separators for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )

    def load_document(self, file_path: str) -> Dict:
        """
        Load document from file using LangChain loaders.

        Args:
            file_path: Path to the document file

        Returns:
            Document dict with content and metadata
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        file_type = path.suffix.lower()

        # Select appropriate loader
        if file_type == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_type in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_type == ".html":
            loader = UnstructuredHTMLLoader(file_path)
        else:  # .txt and others
            loader = TextLoader(file_path)

        # Load document
        docs = loader.load()

        # Combine all pages into single content
        content = "\n\n".join([doc.page_content for doc in docs])

        return {
            "id": self._generate_id(file_path),
            "source": file_path,
            "filename": path.name,
            "type": file_type,
            "content": content
        }

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Split document into chunks using LangChain.

        Args:
            document: Document dict with content

        Returns:
            List of chunk dicts
        """
        content = document["content"]

        # Use LangChain text splitter
        chunks = self.text_splitter.split_text(content)

        # Create chunk dicts
        chunked_docs = []
        for chunk_num, chunk_text in enumerate(chunks):
            chunked_docs.append({
                "id": f"{document['id']}_chunk_{chunk_num}",
                "text": chunk_text,
                "source": document["source"],
                "filename": document["filename"],
                "chunk_index": chunk_num,
                "type": document["type"]
            })

        return chunked_docs

    def process_file(self, file_path: str, tenant_id: str = "default") -> List[Dict]:
        """
        Load and chunk a document in one step.

        Args:
            file_path: Path to the document file
            tenant_id: Tenant identifier for metadata

        Returns:
            List of chunk dicts with tenant_id
        """
        # Load document
        document = self.load_document(file_path)

        # Chunk document
        chunks = self.chunk_document(document)

        # Add tenant_id to each chunk
        for chunk in chunks:
            chunk["tenant_id"] = tenant_id

        return chunks

    def process_directory(
        self,
        directory_path: str,
        tenant_id: str = "default",
        extensions: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Process all documents in a directory.

        Args:
            directory_path: Path to the directory
            tenant_id: Tenant identifier
            extensions: File extensions to process (default: .txt, .md, .pdf, .html)

        Returns:
            List of all chunks from all documents
        """
        extensions = extensions or [".txt", ".md", ".pdf", ".html"]
        dir_path = Path(directory_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        all_chunks = []

        for file_path in dir_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    chunks = self.process_file(str(file_path), tenant_id)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        return all_chunks

    def _generate_id(self, file_path: str) -> str:
        """Generate unique document ID from file path"""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]
```

---

### Phase 2: Training Service & Routes

#### 2.1 TrainingRAGAgent Service

**File:** `services/TrainingRAGAgent.py`

```python
"""
TrainingRAGAgent - Service for knowledge base ingestion and management.
Provides methods for ingesting, managing, and querying the knowledge base.
"""

from typing import List, Dict, Optional
from pathlib import Path
from fastapi import HTTPException, UploadFile
import aiofiles
import os
import tempfile
import shutil

from utils.QdrantManager import QdrantManager
from utils.EmbeddingManager import EmbeddingManager
from utils.RerankerManager import RerankerManager
from utils.DocumentProcessor import DocumentProcessor
from utils.FileConfigLoader import FileConfigLoader


class TrainingRAGAgent:
    """
    Service for RAG knowledge base operations.

    Features:
    - Document ingestion from files/directories
    - Collection management
    - Knowledge base statistics
    - Tenant isolation
    """

    def __init__(self):
        """Initialize training service with managers"""
        self.config_loader = FileConfigLoader()
        self.rag_config = self._load_rag_config()

        # Initialize managers
        self.qdrant = QdrantManager(
            collection_name=self.rag_config["vector_store"]["collection_name"],
            location=self.rag_config["vector_store"]["location"],
            vector_size=self.rag_config["embeddings"]["dimension"]
        )

        self.embeddings = EmbeddingManager(
            model_name=self.rag_config["embeddings"]["model"]
        )

        self.reranker = RerankerManager(
            provider=self.rag_config["reranker"]["provider"],
            model=self.rag_config["reranker"]["model"]
        )

        self.document_processor = DocumentProcessor(
            chunk_size=self.rag_config["chunking"]["chunk_size"],
            chunk_overlap=self.rag_config["chunking"]["chunk_overlap"]
        )

    def _load_rag_config(self) -> Dict:
        """Load RAG configuration"""
        try:
            return self.config_loader.get_rag_config()
        except FileNotFoundError:
            return {
                "vector_store": {
                    "collection_name": "knowledge_base",
                    "location": "./data/qdrant",
                    "vector_size": 768
                },
                "embeddings": {
                    "model": "BAAI/bge-base-en-v1.5",
                    "dimension": 768
                },
                "reranker": {
                    "provider": "bge",
                    "model": "BAAI/bge-reranker-base"
                },
                "chunking": {
                    "chunk_size": 500,
                    "chunk_overlap": 50
                }
            }

    async def ingest_from_directory(
        self,
        directory_path: str,
        tenant_id: str = "default"
    ) -> Dict:
        """
        Ingest all documents from a directory.

        Args:
            directory_path: Path to directory containing documents
            tenant_id: Tenant identifier

        Returns:
            Result dict with ingestion stats
        """
        if not Path(directory_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Directory not found: {directory_path}"
            )

        # Process all documents
        chunks = self.document_processor.process_directory(
            directory_path=directory_path,
            tenant_id=tenant_id
        )

        if not chunks:
            return {
                "success": True,
                "message": "No documents found to process",
                "ingested_count": 0,
                "tenant_id": tenant_id
            }

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.embeddings.embed_batch(texts)

        # Add to Qdrant
        result = await self.qdrant.add_documents(chunks, embeddings)

        return {
            "success": True,
            "message": f"Successfully ingested {result['added_count']} chunks",
            "ingested_count": result["added_count"],
            "tenant_id": tenant_id,
            "collection": result["collection"]
        }

    async def ingest_from_file(
        self,
        file_path: str,
        tenant_id: str = "default"
    ) -> Dict:
        """
        Ingest a single document file.

        Args:
            file_path: Path to the document file
            tenant_id: Tenant identifier

        Returns:
            Result dict with ingestion stats
        """
        if not Path(file_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}"
            )

        # Process document
        chunks = self.document_processor.process_file(file_path, tenant_id)

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.embeddings.embed_batch(texts)

        # Add to Qdrant
        result = await self.qdrant.add_documents(chunks, embeddings)

        return {
            "success": True,
            "message": f"Successfully ingested {result['added_count']} chunks",
            "ingested_count": result["added_count"],
            "file": file_path,
            "tenant_id": tenant_id
        }

    async def ingest_from_upload(
        self,
        file: UploadFile,
        tenant_id: str = "default"
    ) -> Dict:
        """
        Ingest a document from uploaded file.

        Args:
            file: Uploaded file object
            tenant_id: Tenant identifier

        Returns:
            Result dict with ingestion stats
        """
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            # Write uploaded content
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Process and ingest
            result = await self.ingest_from_file(tmp_path, tenant_id)
            result["file"] = file.filename
            return result
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)

    async def delete_tenant_knowledge(self, tenant_id: str) -> Dict:
        """
        Delete all knowledge for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Result dict with deletion stats
        """
        result = await self.qdrant.delete_by_filter({"tenant_id": tenant_id})

        return {
            "success": True,
            "message": f"Deleted all knowledge for tenant '{tenant_id}'",
            "tenant_id": tenant_id,
            "filter": result["filter"]
        }

    async def get_knowledge_stats(self, tenant_id: Optional[str] = None) -> Dict:
        """
        Get knowledge base statistics.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            Statistics dict
        """
        collection_info = self.qdrant.get_collection_info()

        stats = {
            "collection": {
                "name": collection_info.get("name"),
                "exists": collection_info.get("exists", False),
                "vectors_count": collection_info.get("vectors_count", 0),
                "points_count": collection_info.get("points_count", 0),
                "status": collection_info.get("status")
            },
            "embeddings": {
                "model": self.embeddings.get_model_name(),
                "dimension": self.embeddings.get_dimension()
            },
            "reranker": {
                "provider": self.reranker.get_provider()
            }
        }

        # Add tenant-specific count if tenant_id provided
        if tenant_id:
            tenant_count = await self.qdrant.count_documents({"tenant_id": tenant_id})
            stats["tenant_count"] = tenant_count

        return stats

    async def recreate_collection(self) -> Dict:
        """
        Recreate the entire collection (WARNING: Deletes all data).

        Returns:
            Result dict
        """
        self.qdrant.delete_collection()
        self.qdrant.create_collection()

        return {
            "success": True,
            "message": "Collection recreated successfully",
            "collection": self.qdrant.collection_name
        }
```

#### 2.2 Training Routes

**File:** `routes/training.py`

```python
"""
Training routes for RAG knowledge base management.
Provides REST API endpoints for knowledge ingestion and management.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from pydantic import BaseModel

from services.TrainingRAGAgent import TrainingRAGAgent

router = APIRouter(prefix="/training", tags=["training"])


class IngestDirectoryRequest(BaseModel):
    """Request for directory ingestion"""
    directory_path: str
    tenant_id: str = "default"


class DeleteKnowledgeRequest(BaseModel):
    """Request for knowledge deletion"""
    tenant_id: str


@router.post("/ingest/directory")
async def ingest_directory(request: IngestDirectoryRequest):
    """
    Ingest all documents from a directory into the knowledge base.

    Supported formats: .txt, .md, .pdf, .html
    """
    training_service = TrainingRAGAgent()

    result = await training_service.ingest_from_directory(
        directory_path=request.directory_path,
        tenant_id=request.tenant_id
    )

    return result


@router.post("/ingest/file")
async def ingest_file(
    file_path: str = Form(...),
    tenant_id: str = Form("default")
):
    """
    Ingest a single document file into the knowledge base.

    Supported formats: .txt, .md, .pdf, .html
    """
    training_service = TrainingRAGAgent()

    result = await training_service.ingest_from_file(
        file_path=file_path,
        tenant_id=tenant_id
    )

    return result


@router.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    tenant_id: str = Form("default")
):
    """
    Ingest an uploaded document file into the knowledge base.

    Supported formats: .txt, .md, .pdf, .html
    """
    training_service = TrainingRAGAgent()

    result = await training_service.ingest_from_upload(
        file=file,
        tenant_id=tenant_id
    )

    return result


@router.delete("/knowledge")
async def delete_knowledge(request: DeleteKnowledgeRequest):
    """
    Delete all knowledge for a tenant.
    """
    training_service = TrainingRAGAgent()

    result = await training_service.delete_tenant_knowledge(
        tenant_id=request.tenant_id
    )

    return result


@router.get("/stats")
async def get_stats(tenant_id: Optional[str] = None):
    """
    Get knowledge base statistics.

    Args:
        tenant_id: Optional tenant filter for stats
    """
    training_service = TrainingRAGAgent()

    stats = await training_service.get_knowledge_stats(tenant_id=tenant_id)

    return stats


@router.post("/collection/recreate")
async def recreate_collection():
    """
    Recreate the entire knowledge base collection.

    WARNING: This will delete ALL existing data!
    """
    training_service = TrainingRAGAgent()

    result = await training_service.recreate_collection()

    return result
```

---

### Phase 3: RAG Agent Implementation

#### 3.1 RAGRetrievalAgent

**File:** `agents/RAGRetrievalAgent.py`

```python
"""
RAGRetrievalAgent - Two-stage knowledge retrieval.

Stage 1: FAQ exact match (100% match required)
  - If match found, return FAQ answer immediately

Stage 2: Knowledge base search (if no FAQ match)
  - Vector search with BGE embeddings
  - Reranking for result refinement
"""

from typing import Dict, List, Optional, Any
import string
from agents.BaseAgent import BaseAgent, ResponsePatch
from utils.QdrantManager import QdrantManager
from utils.EmbeddingManager import EmbeddingManager
from utils.RerankerManager import RerankerManager
from utils.FileConfigLoader import FileConfigLoader


class RAGRetrievalAgent(BaseAgent):
    """
    Two-stage RAG retrieval agent.

    Stage 1: FAQ exact match (100% match required)
    Stage 2: Knowledge base search with vector + rerank

    Input:
    - query: Search query
    - intent: User's intent (for filtering)
    - tenant_id: Tenant identifier
    - top_k: Number of results to return

    Output:
    - relevant_passages: List of retrieved passages
    - citations: Source references
    - confidence_score: Average relevance score
    - source_type: "faq" or "knowledge_base"
    """

    def __init__(
        self,
        llm_client: Any,
        system_prompt: str,
        qdrant_manager: Optional[QdrantManager] = None,
        embeddings_manager: Optional[EmbeddingManager] = None,
        reranker_manager: Optional[RerankerManager] = None,
        config_loader: Optional[FileConfigLoader] = None
    ):
        super().__init__(llm_client, system_prompt)

        self.qdrant = qdrant_manager
        self.embeddings = embeddings_manager or EmbeddingManager()
        self.reranker = reranker_manager or RerankerManager()
        self.config_loader = config_loader or FileConfigLoader()

        # Load RAG configuration
        self.config = self._load_rag_config()

    def _load_rag_config(self) -> Dict:
        """Load RAG configuration from config/rag.json"""
        try:
            return self.config_loader.get_rag_config()
        except FileNotFoundError:
            return self._get_default_rag_config()

    def _get_default_rag_config(self) -> Dict:
        """Default RAG configuration"""
        return {
            "default_top_k": 5,
            "vector_search_top_k": 50,  # Fetch more for reranking
            "min_relevance_score": 0.6,
            "enable_faq_exact_match": True,
            "enable_reranking": True,
        }

    async def process(self, input_data: Dict) -> ResponsePatch:
        """
        Process retrieval request with two-stage pipeline.

        Args:
            input_data: Dictionary containing:
                - query: Search query (required)
                - intent: User's intent (optional)
                - tenant_id: Tenant identifier (optional, default: "default")
                - top_k: Number of results (optional)

        Returns:
            ResponsePatch with retrieval results
        """
        try:
            # Extract inputs
            query = input_data.get("query")
            tenant_id = input_data.get("tenant_id", "default")
            top_k = input_data.get("top_k", self.config["default_top_k"])

            if not query:
                return ResponsePatch(
                    patch_type="rag_retrieval",
                    data={"error": "Query is required"},
                    confidence=0.0
                )

            # ============ STAGE 1: FAQ Exact Match ============
            faq_result = await self._faq_exact_match(query, tenant_id)

            if faq_result:
                # FAQ match found - return immediately
                return ResponsePatch(
                    patch_type="rag_retrieval",
                    data={
                        "relevant_passages": [faq_result["answer"]],
                        "citations": [{
                            "source": "faq",
                            "question": faq_result["question"],
                            "category": faq_result.get("category", "general")
                        }],
                        "confidence_score": 1.0,  # 100% match = full confidence
                        "source_type": "faq",
                        "retrieval_metadata": {
                            "query": query,
                            "method": "faq_exact_match",
                            "total_results": 1,
                            "tenant_id": tenant_id
                        }
                    },
                    confidence=1.0
                )

            # ============ STAGE 2: Knowledge Base Search ============
            # No FAQ match - proceed to vector search + rerank
            kb_results = await self._knowledge_base_search(
                query=query,
                tenant_id=tenant_id,
                top_k=top_k
            )

            # Build response
            confidence_score = self._calculate_confidence(kb_results)

            return ResponsePatch(
                patch_type="rag_retrieval",
                data={
                    "relevant_passages": [r["text"] for r in kb_results],
                    "citations": [self._build_citation(r) for r in kb_results],
                    "confidence_score": confidence_score,
                    "source_type": "knowledge_base",
                    "retrieval_metadata": {
                        "query": query,
                        "method": "vector_search_rerank",
                        "total_results": len(kb_results),
                        "tenant_id": tenant_id
                    }
                },
                confidence=confidence_score
            )

        except Exception as e:
            return ResponsePatch(
                patch_type="rag_retrieval",
                data={"error": str(e)},
                confidence=0.0
            )

    async def _faq_exact_match(self, query: str, tenant_id: str) -> Optional[Dict]:
        """
        Check for 100% FAQ match.

        Args:
            query: User query
            tenant_id: Tenant identifier

        Returns:
            FAQ dict if match found, None otherwise
        """
        try:
            faq_config = self.config_loader.get_faq_config(tenant_id)
            faqs = faq_config.get("faqs", [])
        except FileNotFoundError:
            return None

        # Normalize query for comparison
        normalized_query = self._normalize_text(query)

        # Check each FAQ for exact match
        for faq in faqs:
            question = faq.get("question", "")
            normalized_question = self._normalize_text(question)

            if normalized_query == normalized_question:
                # 100% match found
                return faq

        return None

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for exact matching.

        - Convert to lowercase
        - Remove punctuation
        - Strip whitespace
        - Remove extra spaces
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Strip and normalize whitespace
        text = " ".join(text.split())

        return text

    async def _knowledge_base_search(
        self,
        query: str,
        tenant_id: str,
        top_k: int
    ) -> List[Dict]:
        """
        Perform knowledge base search with vector + rerank.

        Args:
            query: Search query
            tenant_id: Tenant identifier
            top_k: Final number of results to return

        Returns:
            List of retrieved and reranked passages
        """
        if not self.qdrant:
            return []

        # Generate query embedding
        query_embedding = await self.embeddings.embed_query(query)

        # Stage 2a: Vector search (fetch more for reranking)
        vector_top_k = self.config.get("vector_search_top_k", 50)
        initial_results = await self.qdrant.search(
            query_embedding=query_embedding,
            top_k=vector_top_k,
            filters={"tenant_id": tenant_id}
        )

        if not initial_results:
            return []

        # Stage 2b: Rerank results
        if self.config.get("enable_reranking", True):
            reranked_results = await self.reranker.rerank(
                query=query,
                documents=initial_results,
                top_k=top_k
            )
            return reranked_results

        # Fallback: Return top-k from vector search
        return initial_results[:top_k]

    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate overall confidence from results"""
        if not results:
            return 0.0

        # Use rerank_score if available, otherwise use score
        scores = [
            r.get("rerank_score", r.get("score", 0))
            for r in results
        ]
        return sum(scores) / len(scores)

    def _build_citation(self, result: Dict) -> Dict:
        """Build citation from result"""
        metadata = result.get("metadata", {})

        return {
            "source": metadata.get("source", "unknown"),
            "type": metadata.get("type", "text"),
            "chunk": metadata.get("chunk_index", 0),
            "score": result.get("score", 0)
        }
```

---

### Phase 4: Configuration Files

#### 4.1 RAG Configuration

**File:** `config/rag.json`

```json
{
  "default_top_k": 5,
  "vector_search_top_k": 50,
  "min_relevance_score": 0.6,
  "enable_faq_exact_match": true,
  "enable_reranking": true,
  "vector_store": {
    "provider": "qdrant",
    "collection_name": "knowledge_base",
    "location": "./data/qdrant",
    "vector_size": 768
  },
  "embeddings": {
    "model": "BAAI/bge-base-en-v1.5",
    "dimension": 768
  },
  "reranker": {
    "provider": "bge",
    "model": "BAAI/bge-reranker-base"
  },
  "chunking": {
    "chunk_size": 500,
    "chunk_overlap": 50
  }
}
```

#### 4.2 FAQ Configuration

**File:** `config/faqs.json`

```json
{
  "default": {
    "faqs": [
      {
        "id": "faq_001",
        "question": "What are your business hours?",
        "answer": "Our customer support is available 24/7. For billing inquiries, please contact us Monday-Friday 9 AM - 6 PM EST.",
        "category": "general"
      },
      {
        "id": "faq_002",
        "question": "How do I track my order?",
        "answer": "You can track your order by logging into your account and visiting the Orders section. You'll also receive email updates at each stage of delivery.",
        "category": "orders"
      },
      {
        "id": "faq_003",
        "question": "What is your return policy?",
        "answer": "We offer a 30-day return policy for most items. Products must be unused and in original packaging. To initiate a return, go to Orders > Return Request.",
        "category": "returns"
      },
      {
        "id": "faq_004",
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 5-7 business days. Express shipping (2-3 days) and overnight shipping options are available at checkout.",
        "category": "shipping"
      },
      {
        "id": "faq_005",
        "question": "How do I cancel my order?",
        "answer": "Orders can be cancelled within 1 hour of placement. After that, please contact customer support. For orders already shipped, our return policy applies.",
        "category": "orders"
      }
    ]
  },
  "amazon": {
    "faqs": [
      {
        "id": "amazon_faq_001",
        "question": "Amazon Prime shipping times",
        "answer": "Prime members get free 2-day shipping on eligible items. Same-day and next-day options available in select areas.",
        "category": "shipping"
      }
    ]
  }
}
```

#### 4.3 RAG Agent Prompt

**File:** `prompts/RAGAgent/v1.txt`

```
You are a RAG (Retrieval-Augmented Generation) knowledge retrieval assistant.

Your role is to help find the most relevant information from the knowledge base to answer customer questions.

## Two-Stage Search Process

### Stage 1: FAQ Exact Match
- Check for 100% exact match with FAQ questions
- Normalize text (lowercase, remove punctuation)
- If match found, return FAQ answer immediately

### Stage 2: Knowledge Base Search
- If no FAQ match, search knowledge base
- Use vector similarity with BGE embeddings
- Rerank results for better accuracy
- Return top relevant passages

## Response Format

```json
{
  "relevant_passages": ["Passage text 1", "Passage text 2"],
  "citations": [
    {
      "source": "faq",
      "question": "Original question",
      "category": "orders"
    }
  ],
  "confidence_score": 0.85,
  "source_type": "faq"
}
```

## Guidelines

- Always provide source citations
- Return only results above relevance threshold (0.6)
- Limit results to top_k most relevant
- Handle tenant-specific knowledge isolation
- Fallback gracefully if no relevant results found
```

---

### Phase 5: Schemas

#### 5.1 RAG Schemas

**File:** `schemas/rag.py`

```python
"""
RAG-related schemas for retrieval results.
"""

from pydantic import BaseModel
from typing import Optional, List, Dict


class Citation(BaseModel):
    """Source citation"""
    source: str
    type: str
    chunk: int
    score: float


class RAGResult(BaseModel):
    """Result of RAG retrieval"""
    relevant_passages: List[str]
    citations: List[Citation]
    confidence_score: float
    source_type: str  # "faq" or "knowledge_base"
    retrieval_metadata: Dict


class RAGRequest(BaseModel):
    """Request for RAG retrieval"""
    query: str
    intent: Optional[str] = None
    tenant_id: str = "default"
    top_k: Optional[int] = 5


class TrainingStatsResponse(BaseModel):
    """Knowledge base statistics response"""
    collection: Dict
    embeddings: Dict
    reranker: Dict
    tenant_count: Optional[int] = None
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `utils/QdrantManager.py` | Generalized Qdrant vector store manager |
| `utils/EmbeddingManager.py` | BGE embeddings manager |
| `utils/RerankerManager.py` | Cohere/BGE reranker manager |
| `utils/DocumentProcessor.py` | LangChain document processor |
| `services/TrainingRAGAgent.py` | Knowledge ingestion service |
| `routes/training.py` | Training API endpoints |
| `agents/RAGRetrievalAgent.py` | RAG retrieval agent with two-stage pipeline |
| `config/rag.json` | RAG configuration |
| `config/faqs.json` | FAQ knowledge base |
| `prompts/RAGAgent/v1.txt` | RAG agent system prompt |
| `schemas/rag.py` | RAG result schemas |

---

## Files to Modify

| File | Changes |
|------|---------|
| `services/ChatService.py` | Add `get_rag_agent()` and update `process_chat_request()` |
| `services/ConfigService.py` | Add `get_rag_config()` and `get_faq_config()` |
| `utils/FileConfigLoader.py` | Add `get_rag_config()` and `get_faq_config()` |
| `main.py` | Register training router |
| `pyproject.toml` | Add dependencies |

---

## Dependencies to Add

**File:** `pyproject.toml`

```toml
[project]
dependencies = [
    # ... existing dependencies ...

    # Vector Store
    "qdrant-client>=1.7.0",

    # Embeddings
    "sentence-transformers>=2.2.0",

    # Reranker
    "cohere>=5.0.0",  # Optional, for Cohere reranker

    # Document Processing
    "langchain>=0.1.0",
    "langchain-community>=0.0.20",
    "pypdf>=3.0.0",
    "unstructured>=0.11.0",

    # File Upload
    "python-multipart>=0.0.6",
    "aiofiles>=23.0.0",
]
```

---

## Environment Variables

```bash
# .env

# Vector Store
QDRANT_LOCATION=./data/qdrant
QDRANT_COLLECTION_NAME=knowledge_base

# Embeddings
BGE_MODEL=BAAI/bge-base-en-v1.5
BGE_DEVICE=cpu  # or "cuda" for GPU

# Reranker
RERANKER_PROVIDER=bge  # or "cohere"
COHERE_API_KEY=your-cohere-api-key  # Only if using Cohere

# RAG Configuration
RAG_DEFAULT_TOP_K=5
RAG_VECTOR_SEARCH_TOP_K=50
RAG_MIN_RELEVANCE_SCORE=0.6
```

---

## Verification Checklist

| Component | Status | File |
|-----------|--------|------|
| QdrantManager | ⏳ | `utils/QdrantManager.py` |
| EmbeddingManager | ⏳ | `utils/EmbeddingManager.py` |
| RerankerManager | ⏳ | `utils/RerankerManager.py` |
| DocumentProcessor | ⏳ | `utils/DocumentProcessor.py` |
| TrainingRAGAgent service | ⏳ | `services/TrainingRAGAgent.py` |
| Training routes | ⏳ | `routes/training.py` |
| RAGRetrievalAgent | ⏳ | `agents/RAGRetrievalAgent.py` |
| RAG configuration | ⏳ | `config/rag.json` |
| FAQ configuration | ⏳ | `config/faqs.json` |
| RAG prompt template | ⏳ | `prompts/RAGAgent/v1.txt` |
| RAG schemas | ⏳ | `schemas/rag.py` |
| ChatService integration | ⏳ | `services/ChatService.py` |
| ConfigService updates | ⏳ | `services/ConfigService.py` |
| FileConfigLoader updates | ⏳ | `utils/FileConfigLoader.py` |
| main.py router registration | ⏳ | `main.py` |
| Dependencies added | ⏳ | `pyproject.toml` |

---

## API Endpoints (Training)

### Ingest from Directory
```bash
POST /training/ingest/directory
{
  "directory_path": "/path/to/docs",
  "tenant_id": "default"
}
```

### Ingest from File
```bash
POST /training/ingest/file
Content-Type: multipart/form-data

file_path: /path/to/file.pdf
tenant_id: "default"
```

### Ingest from Upload
```bash
POST /training/ingest/upload
Content-Type: multipart/form-data

file: <uploaded file>
tenant_id: "default"
```

### Get Knowledge Stats
```bash
GET /training/stats?tenant_id=default
```

### Delete Tenant Knowledge
```bash
DELETE /training/knowledge
{
  "tenant_id": "default"
}
```

### Recreate Collection
```bash
POST /training/collection/recreate
```

---

## Next Steps

Once RAG Agent is complete:

1. ✅ **IntentAgent** - Completed
2. ✅ **SentimentAgent** - Completed
3. 🔄 **RAGRetrievalAgent** - Current Task
4. ⏳ **VannaAgent** - Separate agent for database queries
5. ⏳ **PolicyEvaluationAgent** - Pending
6. ⏳ **ContextBuilderAgent** - Pending
7. ⏳ **ResolutionGeneratorAgent** - Pending
8. ⏳ **EscalationDecisionAgent** - Pending
9. ⏳ **TicketActionAgent** - Pending

---

**Status**: Plan Restructured ✅
**Last Updated**: 2026-03-17
