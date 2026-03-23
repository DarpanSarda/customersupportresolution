"""
RAGRetrievalAgent - Two-stage RAG retrieval agent.

Implements a two-stage retrieval pipeline:
1. FAQ Lookup via FAQTool (100% confidence if match)
2. Knowledge Base Search (vector + rerank)
"""

import os
from typing import Dict, Optional, List, Any
from datetime import datetime
from core.BaseAgent import BaseAgent
from schemas.response import ResponsePatch
from utils.QdrantManager import QdrantManager
from utils.EmbeddingManager import EmbeddingManager
from utils.RerankerManager import RerankerManager
from dotenv import load_dotenv

load_dotenv()


class RAGRetrievalAgent(BaseAgent):
    """
    Two-stage RAG retrieval agent.

    Stage 1: FAQ Lookup via FAQTool (Required)
    - Uses FAQTool for FAQ vector search
    - Returns immediately if high-confidence FAQ match found
    - Requires tool_registry with FAQTool permission

    Stage 2: Knowledge Base Search
    - Vector search in Qdrant
    - Reranking for better results
    - Returns top-k relevant passages

    Collection Naming:
    - knowledge_base_{tenant_id} - Knowledge base collection
    - faq_{tenant_id} - FAQ collection (managed by FAQTool)

    Error Handling:
    - If collection not found: returns "please train on the knowledge base first"

    Example:
        >>> agent = RAGRetrievalAgent(llm_client=llm, tool_registry=registry)
        >>> result = await agent.process({
        ...     "query": "What is the refund policy?",
        ...     "tenant_id": "amazon"
        ... })
    """

    def __init__(
        self,
        llm_client,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tool_registry = None,
        qdrant_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        reranker_type: Optional[str] = None,
        reranker_api_key: Optional[str] = None
    ):
        """
        Initialize RAGRetrievalAgent.

        Args:
            llm_client: LLM client for inference
            system_prompt: System prompt (optional, not used for retrieval)
            config: Additional configuration
            tool_registry: Tool registry (optional)
            qdrant_url: Qdrant server URL
            embedding_model: BGE model name
            reranker_type: Reranker type ("cohere" or "bge")
            reranker_api_key: Cohere API key (if using cohere)
        """
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry
        )

        # Configuration
        self._qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self._embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self._reranker_type = reranker_type or os.getenv("RERANKER_TYPE", "bge")
        self._reranker_api_key = reranker_api_key

        # Initialize managers (lazy loaded)
        self._embeddings: Optional[EmbeddingManager] = None
        self._reranker: Optional[RerankerManager] = None
        self._qdrant_managers: Dict[str, QdrantManager] = {}

        # RAG config
        self.default_top_k = config.get("top_k", 5) if config else 5
        self.use_reranker = config.get("use_reranker", True) if config else True

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return "You are a RAG retrieval agent. Find relevant passages from the knowledge base."

    @property
    def embeddings(self) -> EmbeddingManager:
        """Get or create EmbeddingManager."""
        if self._embeddings is None:
            self._embeddings = EmbeddingManager(model_name=self._embedding_model)
        return self._embeddings

    @property
    def reranker(self) -> Optional[RerankerManager]:
        """Get or create RerankerManager."""
        if self._reranker is None and self.use_reranker:
            try:
                self._reranker = RerankerManager(
                    reranker_type=self._reranker_type,
                    api_key=self._reranker_api_key
                )
            except Exception as e:
                print(f"⚠️  Could not initialize reranker: {str(e)}")
                self._reranker = None
                self.use_reranker = False
        return self._reranker

    def _get_qdrant_manager(self, tenant_id: str) -> QdrantManager:
        """Get or create QdrantManager for tenant."""
        if tenant_id not in self._qdrant_managers:
            self._qdrant_managers[tenant_id] = {}

        # Knowledge base manager
        if "kb" not in self._qdrant_managers[tenant_id]:
            self._qdrant_managers[tenant_id]["kb"] = QdrantManager(
                tenant_id=tenant_id,
                location=self._qdrant_url,
                vector_size=self.embeddings.dimension
            )
        return self._qdrant_managers[tenant_id]["kb"]

    async def process(self, input_data: Dict[str, Any], **kwargs) -> ResponsePatch:
        """
        Process input and retrieve relevant passages.

        Args:
            input_data: Dict containing:
                - query: Search query
                - tenant_id: Tenant identifier
                - top_k: Number of results (optional, defaults to config)
                - use_reranker: Whether to use reranking (optional)
            **kwargs: Additional parameters

        Returns:
            ResponsePatch with retrieval results
        """
        query = input_data.get("query", "")
        tenant_id = input_data.get("tenant_id", "default")
        top_k = input_data.get("top_k", self.default_top_k)
        use_reranker = input_data.get("use_reranker", self.use_reranker)

        if not query:
            return ResponsePatch(
                agent_name="RAGRetrievalAgent",
                patch_type="error",
                data={"error": "Query is required"},
                confidence=0.0,
                timestamp=datetime.utcnow().isoformat()
            )

        # ============ STAGE 1: FAQ Exact Match ============
        faq_result = await self._faq_exact_match(query, tenant_id)

        if faq_result:
            # FAQ match found - return immediately with 100% confidence
            return ResponsePatch(
                agent_name="RAGRetrievalAgent",
                patch_type="rag_retrieval",
                data={
                    "relevant_passages": [faq_result["answer"]],
                    "source_type": "faq",
                    "faq_question": faq_result["question"],
                    "confidence_score": 1.0
                },
                content=faq_result["answer"],
                confidence=1.0,
                timestamp=datetime.utcnow().isoformat()
            )

        # ============ STAGE 2: Knowledge Base Search ============
        try:
            kb_results = await self._knowledge_base_search(
                query=query,
                tenant_id=tenant_id,
                top_k=top_k,
                use_reranker=use_reranker
            )

            return ResponsePatch(
                agent_name="RAGRetrievalAgent",
                patch_type="rag_retrieval",
                data={
                    "relevant_passages": [r["text"] for r in kb_results],
                    "source_type": "knowledge_base",
                    "results": kb_results,
                    "confidence_score": kb_results[0]["score"] if kb_results else 0.0
                },
                content=self._format_results(kb_results),
                confidence=kb_results[0]["score"] if kb_results else 0.0,
                timestamp=datetime.utcnow().isoformat()
            )

        except Exception as e:
            error_msg = str(e)
            # Check if collection doesn't exist
            if "not found" in error_msg.lower() or "doesn't exist" in error_msg.lower():
                return ResponsePatch(
                    agent_name="RAGRetrievalAgent",
                    patch_type="rag_retrieval",
                    data={
                        "error": "collection_not_found",
                        "message": "please train on the knowledge base first"
                    },
                    content="please train on the knowledge base first",
                    confidence=0.0,
                    timestamp=datetime.utcnow().isoformat()
                )

            # Other errors
            return ResponsePatch(
                agent_name="RAGRetrievalAgent",
                patch_type="error",
                data={"error": error_msg},
                confidence=0.0,
                timestamp=datetime.utcnow().isoformat()
            )

    async def _faq_exact_match(self, query: str, tenant_id: str) -> Optional[Dict[str, str]]:
        """
        Check for FAQ match using FAQTool.

        Uses FAQTool for vector search with high confidence threshold.

        Args:
            query: User query
            tenant_id: Tenant identifier

        Returns:
            Dict with question and answer if match found, None otherwise
        """
        print(f"[DEBUG] _faq_exact_match called: query={query}, tenant_id={tenant_id}")
        print(f"[DEBUG] tool_registry exists: {self.tool_registry is not None}")
        print(f"[DEBUG] agent_name: {self.get_agent_name()}")

        if self.tool_registry:
            available_tools = self.tool_registry.list_tools()
            print(f"[DEBUG] Available tools: {available_tools}")
            my_tools = self.tool_registry.get_agent_tools(self.get_agent_name())
            print(f"[DEBUG] Tools for {self.get_agent_name()}: {my_tools}")

        if not self.tool_registry or not self.can_use_tool("FAQTool"):
            # No FAQTool available - skip FAQ lookup
            print(f"[DEBUG] FAQTool not available - skipping FAQ lookup")
            return None

        print(f"[DEBUG] Calling FAQTool...")
        try:
            result = await self.use_tool(
                "FAQTool",
                {
                    "query": query,
                    "tenant_id": tenant_id,
                    "top_k": 1,
                    "threshold": 0.85  # High confidence threshold for FAQ
                }
            )

            print(f"[DEBUG] FAQTool result: status={result.status}, data={result.data}")

            # Check if FAQTool found a match
            if result.status == "success" and result.data.get("total_found", 0) > 0:
                faq_match = result.data["results"][0]
                print(f"[DEBUG] FAQ match found: {faq_match['question']}")
                return {
                    "question": faq_match["question"],
                    "answer": faq_match["answer"],
                    "match_type": "vector",
                    "score": faq_match["score"]
                }
            else:
                print(f"[DEBUG] No FAQ match found")

        except Exception as e:
            print(f"⚠️  FAQTool lookup failed: {str(e)}")
            import traceback
            traceback.print_exc()

        return None

    async def _knowledge_base_search(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        use_reranker: bool
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base with optional reranking.

        Args:
            query: Search query
            tenant_id: Tenant identifier
            top_k: Number of results
            use_reranker: Whether to use reranking

        Returns:
            List of search results with text, score, and metadata
        """
        # Get Qdrant manager
        qdrant = self._get_qdrant_manager(tenant_id)

        # Check if collection exists
        if not qdrant.collection_exists():
            raise ValueError(f"Collection not found: {qdrant.collection_name}")

        # Generate query embedding
        query_embedding = await self.embeddings.embed_query(query)

        # Search in Qdrant
        search_results = await qdrant.search(
            query_embedding=query_embedding,
            top_k=top_k * 2 if use_reranker else top_k  # Fetch more for reranking
        )

        if not search_results:
            return []

        # Rerank if enabled
        if use_reranker and self.reranker:
            rerank_results = await self.reranker.rerank(
                query=query,
                documents=[
                    {"text": r["payload"]["text"], "id": r["id"], **r.get("metadata", {})}
                    for r in search_results
                ],
                top_k=top_k
            )

            # Combine original scores with rerank scores
            final_results = []
            for rerank_result in rerank_results:
                original_result = next(
                    (r for r in search_results if r["id"] == rerank_result["id"]),
                    None
                )
                if original_result:
                    final_results.append({
                        "id": rerank_result["id"],
                        "text": rerank_result["text"],
                        "score": original_result["score"],  # Original vector score
                        "rerank_score": rerank_result["rerank_score"],  # Rerank score
                        "metadata": original_result["payload"]
                    })

            return final_results

        # No reranking - return vector search results
        return [
            {
                "id": r["id"],
                "text": r["payload"]["text"],
                "score": r["score"],
                "metadata": r["payload"]
            }
            for r in search_results[:top_k]
        ]

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format results as text."""
        if not results:
            return "No relevant information found in the knowledge base."

        formatted = []
        for i, result in enumerate(results, 1):
            score = result.get("rerank_score") or result.get("score", 0)
            text = result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
            formatted.append(f"{i}. (score: {score:.3f}) {text}")

        return "\n".join(formatted)

    @classmethod
    def get_agent_info(cls) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": "RAGRetrievalAgent",
            "description": "Two-stage RAG retrieval with FAQ exact match and knowledge base search",
            "stages": ["FAQ Exact Match", "Knowledge Base Search"],
            "features": ["Vector search", "Reranking", "Tenant isolation", "Collection management"]
        }
