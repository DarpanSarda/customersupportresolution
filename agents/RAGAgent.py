"""RAG Retrieval Agent with cascade multi-stage retrieval.

Architecture:
    Query → bge-small (100 docs) → bge-base (50 docs) → reranker (15 docs) → LLM
"""

import time
from typing import Dict, Any, List, Optional
from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.patch import Patch
from models.sections import RAGSchema, RetrievedDocument
import logging

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """
    Retrieves relevant knowledge using cascade multi-stage retrieval.

    Reads:
    - understanding.intent: To filter retrieval by intent
    - understanding.input.raw_text: User query for search
    - context.tenant_id: For tenant-specific collection

    Writes:
    - knowledge.query: Original query
    - knowledge.documents: Retrieved documents (after all stages)
    - knowledge.has_relevant_content: Whether relevant docs found
    - knowledge.stage_1_count: Stage 1 retrieval count
    - knowledge.stage_2_count: Stage 2 retrieval count
    - knowledge.reranker_count: After reranker count
    - knowledge.retrieval_latency_ms: Total retrieval time
    """

    agent_name = "RAGAgent"
    allowed_section = "knowledge"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")
        self.embeddings = config.get("embeddings")  # CascadeEmbeddings
        self.vector_store = config.get("vector_store")  # CascadeVectorStore
        self.reranker = config.get("reranker")  # Optional Reranker

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """Execute cascade retrieval: bge-small → bge-base → reranker."""
        start_time = time.time()

        # -------------------------------------------------
        # 1️⃣ Check if RAG is enabled
        # -------------------------------------------------
        if not self.config_loader.is_rag_enabled():
            return self._empty_result("RAG disabled", context)

        # -------------------------------------------------
        # 2️⃣ Extract query from state
        # -------------------------------------------------
        understanding = state.get("understanding", {})
        query = understanding.get("input", {}).get("raw_text", "")

        if not query:
            return self._empty_result("No query found", context)

        # -------------------------------------------------
        # 3️⃣ Get tenant-specific collection
        # -------------------------------------------------
        context_data = state.get("context", {})
        tenant_id = context_data.get("tenant_id", context.tenant_id or "default")

        # -------------------------------------------------
        # 4️⃣ Build metadata filter based on intent
        # -------------------------------------------------
        intent_data = understanding.get("intent", {})
        intent_name = intent_data.get("name")
        filter_metadata = self._build_filter(intent_name)

        # -------------------------------------------------
        # 5️⃣ Get retrieval configuration
        # -------------------------------------------------
        stage_1_config = self.config_loader.get_stage_retrieval_config("stage_1")
        stage_2_config = self.config_loader.get_stage_retrieval_config("stage_2")
        reranker_config = self.config_loader.get_reranker_config()

        stage_1_top_k = stage_1_config.get("top_k", 100)
        stage_2_top_k = stage_2_config.get("top_k", 50)
        reranker_top_k = reranker_config.get("top_k", 15)
        min_score = stage_1_config.get("score_threshold", 0.3)

        # -------------------------------------------------
        # 6️⃣ Stage 1: Fast, broad search (bge-small)
        # -------------------------------------------------
        stage_1_results = self._stage_1_search(
            query=query,
            tenant_id=tenant_id,
            top_k=stage_1_top_k,
            min_score=min_score,
            filter_metadata=filter_metadata
        )

        if not stage_1_results:
            # No results in stage 1, return empty
            return self._empty_result("No documents found in stage 1", context, query=query)

        # -------------------------------------------------
        # 7️⃣ Stage 2: Quality refinement (bge-base)
        # -------------------------------------------------
        stage_2_results = self._stage_2_search(
            query=query,
            tenant_id=tenant_id,
            stage_1_results=stage_1_results,
            top_k=stage_2_top_k,
            min_score=min_score
        )

        if not stage_2_results:
            # Stage 2 filtered everything, use stage 1 top results
            stage_2_results = stage_1_results[:stage_2_top_k]

        # -------------------------------------------------
        # 8️⃣ Stage 3: Reranker (final selection)
        # -------------------------------------------------
        final_results = stage_2_results
        reranker_count = len(stage_2_results)

        if self.reranker and self.config_loader.is_reranker_enabled():
            final_results = self._rerank(
                query=query,
                documents=stage_2_results,
                top_k=reranker_top_k
            )
            reranker_count = len(final_results)

        # -------------------------------------------------
        # 9️⃣ Build final result
        # -------------------------------------------------
        retrieval_latency_ms = int((time.time() - start_time) * 1000)
        has_relevant = len(final_results) > 0
        confidence = sum(r.score for r in final_results) / len(final_results) if final_results else 0.0

        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,
            changes={
                "query": query,
                "documents": [self._doc_to_dict(r) for r in final_results],
                "total_retrieved": len(final_results),
                "retrieval_method": "cascade",
                "confidence": confidence,
                "has_relevant_content": has_relevant,
                "stage_1_count": len(stage_1_results),
                "stage_2_count": len(stage_2_results),
                "reranker_count": reranker_count,
                "retrieval_latency_ms": retrieval_latency_ms
            }
        )

    def _stage_1_search(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        min_score: float,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """Stage 1: Fast, broad search using bge-small."""
        try:
            # Get embedding model for stage 1
            embedding_model = self.embeddings.get_model("stage_1")

            # Encode query (with instruction for BGE)
            query_embedding = embedding_model.encode_query(query)

            # Search in stage 1 collection
            from utils.VectorStore import SearchResult
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                tenant_id=tenant_id,
                stage="stage_1",
                top_k=top_k,
                score_threshold=min_score,
                filter_metadata=filter_metadata
            )

            # Convert to RetrievedDocument
            return [
                RetrievedDocument(
                    content=r.content,
                    doc_id=r.doc_id,
                    source=r.metadata.get("source", "unknown"),
                    metadata=r.metadata,
                    score=r.score,
                    stage="stage_1"
                )
                for r in search_results
            ]

        except Exception as e:
            logger.error(f"Stage 1 search error: {e}")
            return []

    def _stage_2_search(
        self,
        query: str,
        tenant_id: str,
        stage_1_results: List[RetrievedDocument],
        top_k: int,
        min_score: float
    ) -> List[RetrievedDocument]:
        """Stage 2: Quality refinement using bge-base.

        Strategy:
        1. If stage_2 collection exists: Re-encode and re-rank top stage_1 results
        2. If stage_2 collection doesn't exist: Return top stage_1 results
        """
        # Check if stage 2 collection exists
        from utils.VectorStore import SearchResult
        stage_2_collection = self.vector_store.get_collection_name(tenant_id, "stage_2")

        if not self.vector_store.vector_store.collection_exists(stage_2_collection):
            # Stage 2 collection doesn't exist, return stage 1 results
            logger.info(f"Stage 2 collection '{stage_2_collection}' doesn't exist, using stage 1 results")
            return stage_1_results[:top_k]

        try:
            # Get embedding model for stage 2
            embedding_model = self.embeddings.get_model("stage_2")

            # Encode query
            query_embedding = embedding_model.encode_query(query)

            # Search in stage 2 collection with higher top_k
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                tenant_id=tenant_id,
                stage="stage_2",
                top_k=top_k * 2,  # Get more to filter by doc_ids
                score_threshold=min_score,
                filter_metadata=None  # No filter, use stage 1 results as filter
            )

            # Filter to only include docs that were in stage_1
            stage_1_ids = set(r.doc_id for r in stage_1_results)

            # Build a map of doc_id to stage_1 result (for metadata)
            stage_1_map = {r.doc_id: r for r in stage_1_results}

            # Convert to RetrievedDocument, keeping only those from stage_1
            stage_2_results = []
            for r in search_results:
                if r.doc_id in stage_1_ids:
                    original = stage_1_map[r.doc_id]
                    stage_2_results.append(RetrievedDocument(
                        content=r.content,
                        doc_id=r.doc_id,
                        source=original.source,
                        metadata=original.metadata,
                        score=r.score,  # Use stage 2 score
                        stage="stage_2"
                    ))

            return stage_2_results[:top_k]

        except Exception as e:
            logger.error(f"Stage 2 search error: {e}")
            return stage_1_results[:top_k]

    def _rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int
    ) -> List[RetrievedDocument]:
        """Stage 3: Rerank using cross-encoder."""
        try:
            # Convert to format expected by reranker
            doc_dicts = [
                {
                    "content": d.content,
                    "doc_id": d.doc_id,
                    "metadata": d.metadata,
                    "score": d.score
                }
                for d in documents
            ]

            # Rerank
            rerank_results = self.reranker.rerank(
                query=query,
                documents=doc_dicts,
                top_k=top_k
            )

            # Convert back to RetrievedDocument
            return [
                RetrievedDocument(
                    content=r.content,
                    doc_id=r.doc_id,
                    source=r.metadata.get("source", "unknown"),
                    metadata=r.metadata,
                    score=r.score,
                    stage="reranker"
                )
                for r in rerank_results
            ]

        except Exception as e:
            logger.error(f"Reranker error: {e}")
            return documents[:top_k]

    def _build_filter(self, intent_name: str) -> Optional[Dict[str, Any]]:
        """Build metadata filter based on intent."""
        if not intent_name:
            return None

        # Intent category mapping
        intent_category_map = {
            "FAQ_QUERY": {"category": "faq"},
            "REFUND_REQUEST": {"category": "policy"},
            "COMPLAINT": {"category": "sop"},
        }

        return intent_category_map.get(intent_name)

    def _empty_result(
        self,
        reason: str,
        context: AgentExecutionContext,
        query: str = ""
    ) -> Patch:
        """Return empty result patch."""
        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,
            changes={
                "query": query,
                "documents": [],
                "total_retrieved": 0,
                "retrieval_method": "none",
                "confidence": 0.0,
                "has_relevant_content": False,
                "stage_1_count": 0,
                "stage_2_count": 0,
                "reranker_count": 0,
                "retrieval_latency_ms": 0
            }
        )

    def _doc_to_dict(self, doc: RetrievedDocument) -> Dict[str, Any]:
        """Convert RetrievedDocument to dict for state storage."""
        return {
            "content": doc.content,
            "doc_id": doc.doc_id,
            "source": doc.source,
            "metadata": doc.metadata,
            "score": doc.score,
            "stage": doc.stage
        }
