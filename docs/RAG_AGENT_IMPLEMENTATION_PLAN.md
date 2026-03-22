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

---

## Implementation Phases

### Phase 1: Core RAG Infrastructure

#### 1.1 Vector Store Integration

**File:** `llms/vector_store.py`

```python
"""
Vector store for semantic search using embeddings.
Supports multiple backends: ChromaDB, FAISS, Pinecone
"""

from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
import numpy as np

class VectorStore(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    async def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """Add documents with embeddings to the store"""

    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5, filters: Dict = None) -> List[Dict]:
        """Search for similar documents"""

    @abstractmethod
    async def delete(self, document_ids: List[str]):
        """Delete documents by ID"""

class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store"""

    def __init__(self, collection_name: str = "knowledge_base", persist_directory: str = "./data/chroma"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    async def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """Add documents to ChromaDB"""
        ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
        texts = [doc.get("text", "") for doc in documents]
        metadatas = [{k: v for k, v in doc.items() if k != "text"} for doc in documents]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    async def search(self, query_embedding: List[float], top_k: int = 5, filters: Dict = None) -> List[Dict]:
        """Search ChromaDB"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )

        return [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert distance to similarity
            }
            for i in range(len(results["ids"][0]))
        ]

    async def delete(self, document_ids: List[str]):
        """Delete documents"""
        self.collection.delete(ids=document_ids)
```

#### 1.2 Embeddings Service

**File:** `llms/embeddings.py`

```python
"""
Embeddings service for generating text embeddings.
Supports OpenAI, SentenceTransformers, etc.
"""

from typing import List, Union
from llms.BaseLLM import LLMClient

class EmbeddingsService:
    """Service for generating text embeddings"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        response = await self.llm_client.generate_embedding(text)
        return response

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        # Batch embedding for efficiency
        return await self.llm_client.generate_embeddings(texts)

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        return await self.embed_text(query)
```

#### 1.3 Document Processor

**File:** `llms/document_processor.py`

```python
"""
Document processing for knowledge base ingestion.
Supports PDF, TXT, MD, HTML formats.
"""

from typing import List, Dict, Optional
import os
from pathlib import Path
import hashlib

class DocumentProcessor:
    """Process and prepare documents for knowledge base"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: str) -> Dict:
        """Load document from file"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        file_type = path.suffix.lower()
        content = ""

        if file_type == ".pdf":
            content = self._load_pdf(file_path)
        elif file_type == ".txt":
            content = self._load_txt(file_path)
        elif file_type in [".md", ".markdown"]:
            content = self._load_txt(file_path)
        elif file_type == ".html":
            content = self._load_html(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return {
            "id": self._generate_id(file_path),
            "source": file_path,
            "type": file_type,
            "content": content
        }

    def chunk_document(self, document: Dict) -> List[Dict]:
        """Split document into chunks for embedding"""
        content = document["content"]
        chunks = []

        # Simple chunking by character count
        start = 0
        chunk_num = 0

        while start < len(content):
            end = start + self.chunk_size
            chunk_text = content[start:end]

            chunks.append({
                "id": f"{document['id']}_chunk_{chunk_num}",
                "text": chunk_text,
                "source": document["source"],
                "chunk_index": chunk_num,
                "type": document["type"]
            })

            start = end - self.chunk_overlap
            chunk_num += 1

        return chunks

    def _load_pdf(self, file_path: str) -> str:
        """Load PDF content"""
        # Implementation depends on PDF library
        # For now, return placeholder
        return f"PDF content from {file_path}"

    def _load_txt(self, file_path: str) -> str:
        """Load text file content"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_html(self, file_path: str) -> str:
        """Load HTML content"""
        # Strip HTML tags
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text()

    def _generate_id(self, file_path: str) -> str:
        """Generate unique document ID"""
        return hashlib.md5(file_path.encode()).hexdigest()[:12]
```

### Phase 2: RAG Agent Implementation

#### 2.1 RAGRetrievalAgent

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
from llms.vector_store import QdrantVectorStore
from llms.embeddings import BGEEmbeddings
from llms.reranker import CohereReranker
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
        vector_store: Optional[QdrantVectorStore] = None,
        embeddings_service: Optional[BGEEmbeddings] = None,
        reranker: Optional[CohereReranker] = None,
        config_loader: Optional[FileConfigLoader] = None
    ):
        super().__init__(llm_client, system_prompt)

        self.vector_store = vector_store
        self.embeddings = embeddings_service or BGEEmbeddings()
        self.reranker = reranker or CohereReranker()
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
            "enable_vector_search": True,
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
        # Generate query embedding
        query_embedding = await self.embeddings.embed_query(query)

        # Stage 2a: Vector search (fetch more for reranking)
        vector_top_k = self.config.get("vector_search_top_k", 50)
        initial_results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=vector_top_k,
            filters={"tenant_id": tenant_id}
        )

        if not initial_results:
            return []

        # Stage 2b: Rerank results
        if self.config.get("enable_reranking", True) and self.reranker:
            reranked_results = await self.reranker.rerank(
                query=query,
                documents=initial_results,
                top_k=top_k
            )
            return reranked_results

        # Fallback: Return top-k from vector search
        return initial_results[:top_k]

    async def _vector_search(self, query: str, tenant_id: str, top_k: int) -> List[Dict]:
        """Vector-based semantic search"""
        # Generate query embedding
        query_embedding = await self.embeddings.embed_query(query)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters={"tenant_id": tenant_id}
        )

        # Filter by relevance threshold
        min_score = self.config.get("min_relevance_score", 0.6)
        return [r for r in results if r.get("score", 0) >= min_score]

    async def _faq_search(self, query: str, tenant_id: str, top_k: int) -> List[Dict]:
        """FAQ-based exact/pattern search"""
        # Load FAQ for tenant
        try:
            faq_config = self.config_loader.get_faq_config(tenant_id)
            faqs = faq_config.get("faqs", [])
        except FileNotFoundError:
            faqs = []

        # Simple keyword matching
        query_lower = query.lower()
        results = []

        for faq in faqs:
            question = faq.get("question", "").lower()
            answer = faq.get("answer", "")

            # Calculate simple relevance
            relevance = self._calculate_keyword_relevance(query_lower, question)

            if relevance > 0.3:
                results.append({
                    "text": answer,
                    "metadata": {
                        "source": "faq",
                        "question": faq.get("question"),
                        "category": faq.get("category", "general")
                    },
                    "score": relevance
                })

        # Sort by relevance and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def _hybrid_search(self, query: str, tenant_id: str, top_k: int) -> List[Dict]:
        """Combine vector and FAQ search"""
        # Get results from both methods
        vector_results = await self._vector_search(query, tenant_id, top_k)
        faq_results = await self._faq(query, tenant_id, top_k)

        # Combine and deduplicate
        combined = {}

        for r in vector_results:
            combined[r.get("id", "")] = r

        for r in faq_results:
            id_key = r.get("id", f"faq_{hash(r['text'])}")
            if id_key in combined:
                # Average the scores
                combined[id_key]["score"] = (combined[id_key]["score"] + r["score"]) / 2
            else:
                combined[id_key] = r

        # Sort by score
        results = list(combined.values())
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def _calculate_keyword_relevance(self, query: str, text: str) -> float:
        """Calculate keyword relevance score"""
        query_words = set(query.split())
        text_words = set(text.split())

        if not query_words:
            return 0.0

        intersection = query_words & text_words
        return len(intersection) / len(query_words)

    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate overall confidence from results"""
        if not results:
            return 0.0

        scores = [r.get("score", 0) for r in results]
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

    async def add_knowledge(
        self,
        content: str,
        tenant_id: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Add new knowledge to the vector store.

        Args:
            content: Text content to add
            tenant_id: Tenant identifier
            metadata: Optional metadata

        Returns:
            Result with document ID
        """
        doc_id = f"{tenant_id}_{hash(content) % 1000000:06d}"

        # Generate embedding
        embedding = await self.embeddings.embed_text(content)

        # Add to vector store
        await self.vector_store.add_documents(
            documents=[{
                "id": doc_id,
                "text": content,
                "tenant_id": tenant_id,
                **(metadata or {})
            }],
            embeddings=[embedding]
        )

        return {"document_id": doc_id, "status": "added"}
```

### Phase 3: Configuration Files

#### 3.1 RAG Configuration

**File:** `config/rag.json`

```json
{
  "default_top_k": 5,
  "min_relevance_score": 0.6,
  "enable_vector_search": true,
  "enable_faq_search": true,
  "enable_hybrid_search": false,
  "retrieval_method": "vector",
  "chunk_size": 500,
  "chunk_overlap": 50,
  "vector_store": {
    "provider": "chroma",
    "collection_name": "knowledge_base",
    "persist_directory": "./data/chroma"
  },
  "embeddings": {
    "model": "text-embedding-ada-002",
    "dimension": 1536
  }
}
```

#### 3.2 FAQ Configuration

**File:** `config/faqs.json`

```json
{
  "default": {
    "faqs": [
      {
        "id": "faq_001",
        "question": "What are your business hours?",
        "answer": "Our customer support is available 24/7. For billing inquiries, please contact us Monday-Friday 9 AM - 6 PM EST.",
        "category": "general",
        "keywords": ["hours", "support", "contact", "time"]
      },
      {
        "id": "faq_002",
        "question": "How do I track my order?",
        "answer": "You can track your order by logging into your account and visiting the Orders section. You'll also receive email updates at each stage of delivery.",
        "category": "orders",
        "keywords": ["track", "order", "shipping", "delivery", "status"]
      },
      {
        "id": "faq_003",
        "question": "What is your return policy?",
        "answer": "We offer a 30-day return policy for most items. Products must be unused and in original packaging. To initiate a return, go to Orders > Return Request.",
        "category": "returns",
        "keywords": ["return", "refund", "policy", "exchange"]
      },
      {
        "id": "faq_004",
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 5-7 business days. Express shipping (2-3 days) and overnight shipping options are available at checkout.",
        "category": "shipping",
        "keywords": ["shipping", "delivery", "days", "how long"]
      },
      {
        "id": "faq_005",
        "question": "How do I cancel my order?",
        "answer": "Orders can be cancelled within 1 hour of placement. After that, please contact customer support. For orders already shipped, our return policy applies.",
        "category": "orders",
        "keywords": ["cancel", "order", "stop"]
      }
    ]
  },
  "amazon": {
    "faqs": [
      {
        "id": "amazon_faq_001",
        "question": "Amazon Prime shipping times",
        "answer": "Prime members get free 2-day shipping on eligible items. Same-day and next-day options available in select areas.",
        "category": "shipping",
        "keywords": ["prime", "shipping", "delivery"]
      }
    ]
  }
}
```

#### 3.3 RAG Agent Prompt

**File:** `prompts/RAGAgent/v1.txt`

```
You are a RAG (Retrieval-Augmented Generation) knowledge retrieval assistant.

Your role is to help find the most relevant information from the knowledge base to answer customer questions.

## Search Strategy

1. **Understand the Query**: Identify key concepts and intent
2. **Vector Search**: Use semantic similarity for conceptual matches
3. **FAQ Lookup**: Check for exact FAQ matches
4. **Combine Results**: Merge and rank by relevance

## Relevance Scoring

- Score 0.9-1.0: Exact match, highly relevant
- Score 0.7-0.9: Strong semantic match
- Score 0.5-0.7: Partially relevant
- Score < 0.5: Not relevant (filter out)

## Response Format

Return results as a JSON object:

```json
{
  "relevant_passages": [
    "Passage text 1",
    "Passage text 2"
  ],
  "citations": [
    {
      "source": "faq",
      "question": "Original question",
      "score": 0.95
    }
  ],
  "confidence_score": 0.85,
  "retrieval_method": "vector"
}
```

## Guidelines

- Always provide source citations
- Return only results above relevance threshold (0.6)
- Limit results to top_k most relevant
- Handle tenant-specific knowledge isolation
- Fallback gracefully if no relevant results found
```

### Phase 4: ChatService Integration

#### 4.1 Update ChatService

**File:** `services/ChatService.py` (MODIFY)

Add RAG Agent support to ChatService:

```python
from agents.RAGRetrievalAgent import RAGRetrievalAgent
from llms.vector_store import ChromaVectorStore
from llms.embeddings import EmbeddingsService

class ChatService:
    # ... existing code ...

    async def get_rag_agent(
        self,
        llm_client,
        tenant_id: str
    ) -> RAGRetrievalAgent:
        """
        Create and initialize RAGRetrievalAgent.

        Args:
            llm_client: LLM client instance
            tenant_id: Tenant identifier

        Returns:
            Initialized RAGRetrievalAgent

        Raises:
            HTTPException: If prompt not configured
        """
        # Load prompt for RAGAgent
        try:
            prompt = await self.config_service.get_prompt(
                agent_name="RAGAgent",
                version="v1",
                tenant_id=tenant_id
            )
        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail=f"RAGAgent prompt not configured: {str(e)}"
            )

        # Initialize vector store and embeddings
        vector_store = ChromaVectorStore(
            collection_name=f"kb_{tenant_id}"
        )
        embeddings_service = EmbeddingsService(llm_client)

        # Initialize RAGAgent
        return RAGRetrievalAgent(
            llm_client=llm_client,
            system_prompt=prompt,
            vector_store=vector_store,
            embeddings_service=embeddings_service,
            config_loader=self.config_service.config_loader
        )

    async def process_chat_request(
        self,
        request: ChatRequest
    ) -> ChatResponse:
        """
        Process a chat request with Intent, Sentiment, and RAG agents.
        """
        # ... existing IntentAgent and SentimentAgent code ...

        # Create RAGAgent
        rag_agent = await self.get_rag_agent(
            llm_client=llm_client,
            tenant_id=request.tenant_id
        )

        # Run RAGAgent
        rag_result = await rag_agent.process({
            "query": request.message,
            "intent": detected_intent,
            "tenant_id": request.tenant_id,
            "top_k": 5
        })

        print(f"RAGAgent result: {rag_result}")

        # Check for RAG errors
        if rag_result.data and "error" in rag_result.data:
            # Don't fail the request, just log
            print(f"RAG retrieval error: {rag_result.data['error']}")
            rag_data = {}
        else:
            rag_data = rag_result.data or {}

        # Build enhanced response
        response_parts = [
            f"Detected intent: {detected_intent} (confidence: {intent_confidence:.2f})",
            f"Sentiment: {sentiment} (urgency: {urgency_score:.2f})"
        ]

        # Add RAG results if available
        if rag_data.get("relevant_passages"):
            passages_count = len(rag_data["relevant_passages"])
            rag_confidence = rag_data.get("confidence_score", 0)
            response_parts.append(f"Knowledge: {passages_count} relevant passages found (confidence: {rag_confidence:.2f})")

        if toxicity_flag:
            response_parts.append("⚠️ Toxic language detected - consider escalation")

        if urgency_score >= 0.7:
            response_parts.append("⚠️ High urgency - requires prompt attention")

        response_text = " | ".join(response_parts)

        return ChatResponse(
            response=response_text,
            tenant_id=request.tenant_id,
            chatbot_id=request.chatbot_id,
            session_id=request.session_id
        )
```

### Phase 5: Knowledge Base Setup

#### 5.1 Knowledge Ingestion Script

**File:** `scripts/ingest_knowledge.py`

```python
"""
Script to ingest documents into the knowledge base.
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(".")

from llms.embeddings import EmbeddingsService
from llms.vector_store import ChromaVectorStore
from llms.document_processor import DocumentProcessor
from llms.BaseLLM import LLMConfig, LLMFactory
from utils.DBManager import get_db_manager
from utils.encryption import decrypt

async def ingest_documents(
    documents_dir: str,
    tenant_id: str = "default",
    collection_name: str = "knowledge_base"
):
    """Ingest documents from directory into vector store"""

    # Initialize components
    vector_store = ChromaVectorStore(collection_name=collection_name)

    # Create LLM client for embeddings
    # You might want to use a separate embeddings API
    config = LLMConfig(
        provider="openai",
        api_key="your-api-key",  # Load from config or env
        model="text-embedding-ada-002"
    )
    llm_client = LLMFactory.create("openai", config)
    embeddings_service = EmbeddingsService(llm_client)

    document_processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=50
    )

    # Process all documents
    documents_path = Path(documents_dir)
    all_chunks = []

    for file_path in documents_path.glob("**/*"):
        if file_path.is_file() and file_path.suffix in [".txt", ".md", ".pdf"]:
            print(f"Processing: {file_path}")

            try:
                # Load document
                document = document_processor.load_document(str(file_path))

                # Chunk document
                chunks = document_processor.chunk_document(document)

                # Add tenant_id to each chunk
                for chunk in chunks:
                    chunk["tenant_id"] = tenant_id

                all_chunks.extend(chunks)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Total chunks to ingest: {len(all_chunks)}")

    # Generate embeddings
    print("Generating embeddings...")
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = await embeddings_service.embed_batch(texts)

    # Add to vector store
    print("Adding to vector store...")
    await vector_store.add_documents(all_chunks, embeddings)

    print(f"✅ Ingested {len(all_chunks)} chunks for tenant '{tenant_id}'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into knowledge base")
    parser.add_argument("--dir", required=True, help="Directory containing documents")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument("--collection", default="knowledge_base", help="Collection name")

    args = parser.parse_args()

    asyncio.run(ingest_documents(args.dir, args.tenant, args.collection))
```

### Phase 6: Schema Updates

#### 6.1 RAG Schema

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
    retrieval_metadata: Dict


class RAGRequest(BaseModel):
    """Request for RAG retrieval"""
    query: str
    intent: Optional[str] = None
    tenant_id: str = "default"
    top_k: Optional[int] = 5
```

### Phase 7: ConfigService Updates

#### 7.1 Update ConfigService

**File:** `services/ConfigService.py` (MODIFY)

Add RAG and FAQ config loading:

```python
class ConfigService:
    # ... existing code ...

    async def get_rag_config(self, tenant_id: str = "default") -> Dict:
        """Get RAG configuration"""
        try:
            return self.config_loader.get_rag_config()
        except FileNotFoundError:
            return self._get_default_rag_config()

    async def get_faq_config(self, tenant_id: str = "default") -> Dict:
        """Get FAQ configuration"""
        try:
            return self.config_loader.get_faq_config(tenant_id)
        except FileNotFoundError:
            return {"faqs": []}

    def _get_default_rag_config(self) -> Dict:
        """Default RAG configuration"""
        return {
            "default_top_k": 5,
            "min_relevance_score": 0.6,
            "enable_vector_search": True,
            "enable_faq_search": True,
            "retrieval_method": "vector"
        }
```

### Phase 8: FileConfigLoader Updates

#### 8.1 Update FileConfigLoader

**File:** `utils/FileConfigLoader.py` (MODIFY)

Add RAG and FAQ config loading methods:

```python
class FileConfigLoader:
    # ... existing code ...

    def get_rag_config(self, tenant_id: str = "default") -> Dict:
        """Load RAG configuration"""
        config_path = self._get_config_path("rag.json")
        return self._load_json_config(config_path)

    def get_faq_config(self, tenant_id: str = "default") -> Dict:
        """Load FAQ configuration for tenant"""
        config_path = self._get_config_path("faqs.json")
        config = self._load_json_config(config_path)

        # Return tenant-specific FAQs or default
        return config.get(tenant_id, config.get("default", {"faqs": []}))
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `agents/RAGRetrievalAgent.py` | RAG retrieval agent implementation |
| `llms/vector_store.py` | Vector store abstraction |
| `llms/embeddings.py` | Embeddings service |
| `llms/document_processor.py` | Document processing and chunking |
| `config/rag.json` | RAG configuration |
| `config/faqs.json` | FAQ knowledge base |
| `prompts/RAGAgent/v1.txt` | RAG agent prompt |
| `schemas/rag.py` | RAG result schemas |
| `scripts/ingest_knowledge.py` | Knowledge ingestion script |
| `data/knowledge/default/` | Default tenant knowledge directory |

---

## Files to Modify

| File | Changes |
|------|---------|
| `services/ChatService.py` | Add `get_rag_agent()` and update `process_chat_request()` |
| `services/ConfigService.py` | Add `get_rag_config()` and `get_faq_config()` |
| `utils/FileConfigLoader.py` | Add `get_rag_config()` and `get_faq_config()` |
| `main.py` | Add knowledge ingestion endpoint (optional) |

---

## Implementation Steps

### Step 1: Infrastructure Layer
1. Create `llms/vector_store.py` with ChromaDB integration
2. Create `llms/embeddings.py` for embedding generation
3. Create `llms/document_processor.py` for document processing

### Step 2: Configuration Layer
1. Create `config/rag.json` with RAG settings
2. Create `config/faqs.json` with FAQ entries
3. Create `prompts/RAGAgent/v1.txt` with system prompt
4. Create `schemas/rag.py` with result models

### Step 3: Agent Layer
1. Create `agents/RAGRetrievalAgent.py` with full implementation
2. Update `services/ConfigService.py` for RAG/FAQ config
3. Update `utils/FileConfigLoader.py` for config loading

### Step 4: Integration Layer
1. Update `services/ChatService.py` to include RAG agent
2. Test chat endpoint with RAG results

### Step 5: Knowledge Base Setup
1. Create `data/knowledge/default/` directory
2. Add sample FAQ documents
3. Create and run `scripts/ingest_knowledge.py`
4. Verify vector store has data

### Step 6: Testing
1. Test vector search with sample queries
2. Test FAQ lookup
3. Test hybrid retrieval
4. Verify tenant isolation
5. Test relevance scoring

---

## Testing Strategy

### Unit Tests

```python
# tests/test_rag_agent.py

import pytest
from agents.RAGRetrievalAgent import RAGRetrievalAgent

@pytest.mark.asyncio
async def test_vector_search():
    """Test vector-based retrieval"""
    agent = RAGRetrievalAgent(...)
    result = await agent.process({
        "query": "How do I track my order?",
        "tenant_id": "default",
        "top_k": 3
    })

    assert result.patch_type == "rag_retrieval"
    assert len(result.data["relevant_passages"]) > 0
    assert result.data["confidence_score"] > 0.6

@pytest.mark.asyncio
async def test_faq_search():
    """Test FAQ-based retrieval"""
    agent = RAGRetrievalAgent(...)
    result = await agent.process({
        "query": "What are your business hours?",
        "tenant_id": "default"
    })

    assert result.data["retrieval_metadata"]["method"] == "faq"
    assert len(result.data["citations"]) > 0

@pytest.mark.asyncio
async def test_no_results():
    """Test handling of no relevant results"""
    agent = RAGRetrievalAgent(...)
    result = await agent.process({
        "query": "quantum physics equations",
        "tenant_id": "default"
    })

    assert len(result.data["relevant_passages"]) == 0
    assert result.data["confidence_score"] == 0.0
```

### Integration Tests

```bash
# Test 1: Knowledge retrieval
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I track my order?",
    "tenant_id": "default",
    "chatbot_id": "test-chatbot"
  }'

# Expected: RAG passages about order tracking

# Test 2: FAQ match
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your business hours?",
    "tenant_id": "default",
    "chatbot_id": "test-chatbot"
  }'

# Expected: FAQ answer about business hours

# Test 3: No relevant knowledge
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about medieval history",
    "tenant_id": "default",
    "chatbot_id": "test-chatbot"
  }'

# Expected: Empty RAG results, low confidence
```

---

## Dependencies to Add

**File:** `pyproject.toml`

```toml
[project]
dependencies = [
    # ... existing dependencies ...

    # Vector Store
    "chromadb>=0.4.0",  # Vector database

    # Embeddings
    # OpenAI embeddings included in openai package

    # Document Processing
    "pypdf>=3.0.0",      # PDF parsing
    "beautifulsoup4>=4.12.0",  # HTML parsing
]
```

---

## Environment Variables

```bash
# .env

# Vector Store
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=knowledge_base

# Embeddings
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

# RAG Configuration
RAG_DEFAULT_TOP_K=5
RAG_MIN_RELEVANCE_SCORE=0.6
```

---

## Directory Structure

```
customersupportresolution/
├── agents/
│   ├── RAGRetrievalAgent.py      # NEW
│   ├── IntentAgent.py             # ✅ Existing
│   └── SentimentAgent.py          # ✅ Existing
├── config/
│   ├── rag.json                   # NEW
│   ├── faqs.json                  # NEW
│   ├── intents.json               # ✅ Existing
│   └── sentiments.json            # ✅ Existing
├── prompts/
│   └── RAGAgent/
│       └── v1.txt                 # NEW
├── llms/
│   ├── vector_store.py            # NEW
│   ├── embeddings.py              # NEW
│   └── document_processor.py      # NEW
├── schemas/
│   ├── rag.py                     # NEW
│   ├── chat.py                    # ✅ Existing
│   └── sentiment.py               # ✅ Existing
├── services/
│   ├── ChatService.py             # MODIFY
│   └── ConfigService.py           # MODIFY
├── utils/
│   └── FileConfigLoader.py        # MODIFY
├── scripts/
│   └── ingest_knowledge.py        # NEW
└── data/
    └── knowledge/
        └── default/               # NEW
            ├── faqs.txt
            ├── policies.txt
            └── products.txt
```

---

## Verification Checklist

| Component | Status | File |
|-----------|--------|------|
| Vector store implementation | ⏳ | `llms/vector_store.py` |
| Embeddings service | ⏳ | `llms/embeddings.py` |
| Document processor | ⏳ | `llms/document_processor.py` |
| RAGRetrievalAgent | ⏳ | `agents/RAGRetrievalAgent.py` |
| RAG configuration | ⏳ | `config/rag.json` |
| FAQ configuration | ⏳ | `config/faqs.json` |
| RAG prompt template | ⏳ | `prompts/RAGAgent/v1.txt` |
| RAG schemas | ⏳ | `schemas/rag.py` |
| ChatService integration | ⏳ | `services/ChatService.py` |
| ConfigService updates | ⏳ | `services/ConfigService.py` |
| FileConfigLoader updates | ⏳ | `utils/FileConfigLoader.py` |
| Knowledge ingestion script | ⏳ | `scripts/ingest_knowledge.py` |
| ChromaDB dependency | ⏳ | `pyproject.toml` |
| Knowledge base directory | ⏳ | `data/knowledge/default/` |

---

## Next Steps

Once RAG Agent is complete:

1. ✅ **IntentAgent** - Completed
2. ✅ **SentimentAgent** - Completed
3. 🔄 **RAGRetrievalAgent** - Current Task
4. ⏳ **PolicyEvaluationAgent** - Next
5. ⏳ **ContextBuilderAgent** - Pending
6. ⏳ **ResolutionGeneratorAgent** - Pending
7. ⏳ **EscalationDecisionAgent** - Pending
8. ⏳ **TicketActionAgent** - Pending

---

**Status**: Planning Complete ✅
**Last Updated**: 2026-03-17
