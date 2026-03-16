"""FAQ service for managing FAQs in Qdrant vector store.

Handles:
- Single FAQ creation
- Bulk FAQ upload (CSV/XLSX)
- FAQ retrieval by tenant
- FAQ update/delete
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from utils.Embeddings import HuggingFaceEmbeddings
from utils.VectorStore import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass
class FAQItem:
    """FAQ item data structure."""
    tenant_id: str
    category: str
    question: str
    answer: str


class FAQService:
    """Service for FAQ management with Qdrant vector store."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_model: Optional[HuggingFaceEmbeddings] = None
    ):
        """Initialize FAQ service.

        Args:
            vector_store: Qdrant vector store instance
            embedding_model: Optional embedding model (created from env if not provided)
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model or self._create_embedding_model()

    def _create_embedding_model(self) -> HuggingFaceEmbeddings:
        """Create embedding model from environment variables."""
        model_name = os.getenv("BGE_BASE", "BAAI/bge-base-en-v1.5")
        logger.info(f"Creating FAQ embedding model: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)

    def _get_collection_name(self, tenant_id: str) -> str:
        """Get FAQ collection name for tenant."""
        return f"{tenant_id}_faqs"

    def add_faq(self, faq: FAQItem) -> Dict[str, Any]:
        """Add single FAQ to vector store.

        Args:
            faq: FAQ item to add

        Returns:
            Dict with created FAQ info
        """
        # Combine question and answer for embedding
        text = f"Question: {faq.question}\nAnswer: {faq.answer}"

        # Generate embedding
        embedding = self.embedding_model.embed(text)

        # Create unique ID
        faq_id = str(uuid.uuid4())

        # Metadata
        metadata = {
            "tenant_id": faq.tenant_id,
            "category": faq.category,
            "question": faq.question,
            "answer": faq.answer
        }

        # Add to vector store (collection will be created if not exists)
        collection_name = self._get_collection_name(faq.tenant_id)
        self.vector_store.add_documents(
            documents=[text],
            ids=[faq_id],
            embeddings=[embedding],
            metadata=[metadata],
            collection_name=collection_name
        )

        logger.info(f"Added FAQ for tenant '{faq.tenant_id}': {faq.question[:50]}...")

        return {
            "id": faq_id,
            "tenant_id": faq.tenant_id,
            "category": faq.category,
            "question": faq.question,
            "answer": faq.answer
        }

    def add_bulk_faqs(self, faqs: List[FAQItem]) -> Dict[str, Any]:
        """Add multiple FAQs to vector store.

        Args:
            faqs: List of FAQ items

        Returns:
            Dict with creation results
        """
        if not faqs:
            return {"success": False, "error": "No FAQs provided"}

        # Group by tenant
        tenant_faqs: Dict[str, List[FAQItem]] = {}
        for faq in faqs:
            if faq.tenant_id not in tenant_faqs:
                tenant_faqs[faq.tenant_id] = []
            tenant_faqs[faq.tenant_id].append(faq)

        results = {"success": True, "created": 0, "failed": 0, "details": []}

        # Process each tenant
        for tenant_id, tenant_faq_list in tenant_faqs.items():
            try:
                # Prepare batch data
                texts = []
                ids = []
                embeddings = []
                metadata_list = []

                for faq in tenant_faq_list:
                    text = f"Question: {faq.question}\nAnswer: {faq.answer}"
                    texts.append(text)
                    ids.append(str(uuid.uuid4()))

                    metadata = {
                        "tenant_id": faq.tenant_id,
                        "category": faq.category,
                        "question": faq.question,
                        "answer": faq.answer
                    }
                    metadata_list.append(metadata)

                # Generate embeddings
                embeddings = self.embedding_model.embed_batch(texts)

                # Add to vector store (collection will be created if not exists)
                collection_name = self._get_collection_name(tenant_id)
                self.vector_store.add_documents(
                    documents=texts,
                    ids=ids,
                    embeddings=embeddings,
                    metadata=metadata_list,
                    collection_name=collection_name
                )

                results["created"] += len(tenant_faq_list)
                results["details"].append({
                    "tenant_id": tenant_id,
                    "count": len(tenant_faq_list),
                    "status": "success"
                })

                logger.info(f"Added {len(tenant_faq_list)} FAQs for tenant '{tenant_id}'")

            except Exception as e:
                results["failed"] += len(tenant_faq_list)
                results["details"].append({
                    "tenant_id": tenant_id,
                    "count": len(tenant_faq_list),
                    "status": "failed",
                    "error": str(e)
                })
                logger.error(f"Failed to add FAQs for tenant '{tenant_id}': {e}")

        return results

    def get_all_faqs(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get all FAQs for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of FAQ items
        """
        collection_name = self._get_collection_name(tenant_id)

        if not self.vector_store.collection_exists(collection_name):
            return []

        # Get all points from collection
        from qdrant_client.models import ScrollRequest, Filter

        try:
            results = []
            offset = None
            limit = 100

            while True:
                scroll_result = self.vector_store.client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points = scroll_result[0]
                if not points:
                    break

                for point in points:
                    payload = point.payload
                    results.append({
                        "id": str(point.id),
                        "tenant_id": payload.get("tenant_id"),
                        "category": payload.get("category"),
                        "question": payload.get("question"),
                        "answer": payload.get("answer")
                    })

                offset = scroll_result[1]
                if offset is None:
                    break

            return results

        except Exception as e:
            logger.error(f"Error getting FAQs for tenant '{tenant_id}': {e}")
            return []

    def update_faq(self, faq_id: str, faq: FAQItem) -> Dict[str, Any]:
        """Update an existing FAQ.

        Args:
            faq_id: FAQ ID to update
            faq: Updated FAQ data

        Returns:
            Dict with update result
        """
        collection_name = self._get_collection_name(faq.tenant_id)

        if not self.vector_store.collection_exists(collection_name):
            return {"success": False, "error": "Collection not found"}

        # Combine question and answer for embedding
        text = f"Question: {faq.question}\nAnswer: {faq.answer}"

        # Generate new embedding
        embedding = self.embedding_model.embed(text)

        # Metadata
        metadata = {
            "tenant_id": faq.tenant_id,
            "category": faq.category,
            "question": faq.question,
            "answer": faq.answer
        }

        try:
            from qdrant_client.models import PointStruct

            # Update in Qdrant
            self.vector_store.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=faq_id,
                        vector=embedding,
                        payload={
                            "content": text,
                            **metadata
                        }
                    )
                ]
            )

            logger.info(f"Updated FAQ {faq_id} for tenant '{faq.tenant_id}'")

            return {
                "success": True,
                "id": faq_id,
                "tenant_id": faq.tenant_id,
                "category": faq.category,
                "question": faq.question,
                "answer": faq.answer
            }

        except Exception as e:
            logger.error(f"Error updating FAQ {faq_id}: {e}")
            return {"success": False, "error": str(e)}

    def delete_faq(self, tenant_id: str, faq_id: str) -> Dict[str, Any]:
        """Delete an FAQ.

        Args:
            tenant_id: Tenant identifier
            faq_id: FAQ ID to delete

        Returns:
            Dict with delete result
        """
        collection_name = self._get_collection_name(tenant_id)

        if not self.vector_store.collection_exists(collection_name):
            return {"success": False, "error": "Collection not found"}

        try:
            self.vector_store.client.delete(
                collection_name=collection_name,
                points_selector=[faq_id]
            )

            logger.info(f"Deleted FAQ {faq_id} for tenant '{tenant_id}'")

            return {"success": True, "id": faq_id}

        except Exception as e:
            logger.error(f"Error deleting FAQ {faq_id}: {e}")
            return {"success": False, "error": str(e)}

    def delete_all_faqs(self, tenant_id: str) -> Dict[str, Any]:
        """Delete all FAQs for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dict with delete result
        """
        collection_name = self._get_collection_name(tenant_id)

        if not self.vector_store.collection_exists(collection_name):
            return {"success": False, "error": "Collection not found"}

        try:
            # Get count before deletion
            stats = self.vector_store.get_collection_stats(collection_name)
            count = stats.get("count", 0)

            # Delete collection
            self.vector_store.delete_collection(collection_name)

            logger.info(f"Deleted all FAQs ({count}) for tenant '{tenant_id}'")

            return {
                "success": True,
                "tenant_id": tenant_id,
                "deleted_count": count
            }

        except Exception as e:
            logger.error(f"Error deleting FAQs for tenant '{tenant_id}': {e}")
            return {"success": False, "error": str(e)}
