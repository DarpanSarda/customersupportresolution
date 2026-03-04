"""Document ingestion service for building vector knowledge base.

Supports:
- FAQ ingestion from JSON
- Text document chunking and ingestion
- Multi-tenant isolation
- Cascade embedding (stage_1 + stage_2)
"""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from utils.VectorStore import CascadeVectorStore
from utils.Embeddings import CascadeEmbeddings
import logging

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Service for ingesting documents into vector store.

    Handles:
    - FAQ ingestion (Q/A pairs)
    - Text document chunking
    - Multi-tenant collections
    - Cascade embeddings (stage_1 + stage_2)
    """

    def __init__(
        self,
        vector_store: CascadeVectorStore,
        embeddings: CascadeEmbeddings,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """Initialize document ingester.

        Args:
            vector_store: Cascade vector store
            embeddings: Cascade embeddings
            chunk_size: Text chunk size for splitting large documents
            chunk_overlap: Overlap between chunks
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_faq(
        self,
        faq_data: List[Dict[str, str]],
        tenant_id: str = "default",
        stages: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Ingest FAQ documents into vector store.

        Args:
            faq_data: List of {"question": "...", "answer": "..."}
            tenant_id: Tenant identifier
            stages: Which stages to ingest into (default: ["stage_1", "stage_2"])

        Returns:
            Dict with counts per stage
        """
        stages = stages or ["stage_1", "stage_2"]
        counts = {}

        for stage in stages:
            # Get embedding model for this stage
            embedding_model = self.embeddings.get_model(stage)

            # Prepare documents
            documents = []
            ids = []
            metadata = []

            for i, faq in enumerate(faq_data):
                # Combine question and answer for better retrieval
                text = f"Question: {faq['question']}\nAnswer: {faq['answer']}"

                documents.append(text)
                ids.append(f"{tenant_id}_faq_{i}")
                metadata.append({
                    "tenant_id": tenant_id,
                    "source": "faq",
                    "category": "faq",
                    "question": faq["question"]
                })

            # Generate embeddings
            embeddings_list = embedding_model.embed_batch(documents)

            # Add to vector store
            self.vector_store.add_documents(
                documents=documents,
                ids=ids,
                embeddings=embeddings_list,
                metadata=metadata,
                tenant_id=tenant_id,
                stage=stage
            )

            counts[stage] = len(documents)
            logger.info(f"Ingested {len(documents)} FAQs for tenant '{tenant_id}' into {stage}")

        return counts

    def ingest_texts(
        self,
        texts: List[str],
        tenant_id: str = "default",
        source: str = "manual",
        category: str = "general",
        metadata: Optional[List[Dict[str, Any]]] = None,
        stages: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Ingest text documents into vector store.

        Args:
            texts: List of text documents
            tenant_id: Tenant identifier
            source: Source type (manual, sop, policy, etc.)
            category: Document category
            metadata: Optional metadata per document
            stages: Which stages to ingest into

        Returns:
            Dict with counts per stage
        """
        stages = stages or ["stage_1", "stage_2"]
        metadata = metadata or [{}] * len(texts)

        # Chunk documents if needed
        chunks = []
        chunk_ids = []
        chunk_metadata = []

        for doc_idx, text in enumerate(texts):
            doc_chunks = self._chunk_text(text)
            base_metadata = metadata[doc_idx]

            for chunk_idx, chunk in enumerate(doc_chunks):
                chunks.append(chunk)
                chunk_ids.append(f"{tenant_id}_{source}_{doc_idx}_chunk_{chunk_idx}")
                chunk_metadata.append({
                    "tenant_id": tenant_id,
                    "source": source,
                    "category": category,
                    "doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(doc_chunks),
                    **base_metadata
                })

        # Ingest chunks
        counts = {}
        for stage in stages:
            embedding_model = self.embeddings.get_model(stage)
            embeddings_list = embedding_model.embed_batch(chunks)

            self.vector_store.add_documents(
                documents=chunks,
                ids=chunk_ids,
                embeddings=embeddings_list,
                metadata=chunk_metadata,
                tenant_id=tenant_id,
                stage=stage
            )

            counts[stage] = len(chunks)
            logger.info(f"Ingested {len(chunks)} text chunks for tenant '{tenant_id}' into {stage}")

        return counts

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap

        return chunks

    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get statistics for a tenant's collections.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dict with stats per stage
        """
        stats = {}
        for stage in ["stage_1", "stage_2"]:
            collection_name = self.vector_store.get_collection_name(tenant_id, stage)
            stats[stage] = self.vector_store.vector_store.get_collection_stats(collection_name)

        return stats

    def delete_tenant(self, tenant_id: str) -> None:
        """Delete all collections for a tenant.

        Args:
            tenant_id: Tenant identifier
        """
        self.vector_store.delete_tenant(tenant_id)
        logger.info(f"Deleted all collections for tenant '{tenant_id}'")

    def ingest_files(
        self,
        file_paths: List[str],
        tenant_id: str = "default",
        source: str = "document",
        category: str = "general",
        metadata: Optional[List[Dict[str, Any]]] = None,
        stages: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Ingest documents from files.

        Args:
            file_paths: List of file paths to ingest
            tenant_id: Tenant identifier
            source: Source type
            category: Document category
            metadata: Optional metadata per document
            stages: Which stages to ingest into

        Returns:
            Dict with counts per stage
        """
        from utils.DocumentParser import DocumentParser

        parser = DocumentParser()
        texts = []
        file_metadata = []

        for file_path in file_paths:
            try:
                text = parser.parse_file(file_path)
                texts.append(text)

                # Extract filename for metadata
                filename = Path(file_path).name
                ext = Path(file_path).suffix.lower()

                file_meta = {
                    "filename": filename,
                    "file_type": ext[1:],  # Remove dot
                    "source_path": file_path
                }
                file_metadata.append(file_meta)

            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")

        # Merge with provided metadata
        if metadata:
            for i, meta in enumerate(metadata):
                if i < len(file_metadata):
                    file_metadata[i].update(meta)

        return self.ingest_texts(
            texts=texts,
            tenant_id=tenant_id,
            source=source,
            category=category,
            metadata=file_metadata,
            stages=stages
        )

    def ingest_directory(
        self,
        directory: str,
        tenant_id: str = "default",
        source: str = "document",
        category: str = "general",
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        stages: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """Ingest all documents from a directory.

        Args:
            directory: Directory path
            tenant_id: Tenant identifier
            source: Source type
            category: Document category
            extensions: File extensions to include
            recursive: Search recursively
            stages: Which stages to ingest into

        Returns:
            Dict with counts per stage
        """
        from utils.DocumentParser import BatchDocumentParser

        batch_parser = BatchDocumentParser()
        results = batch_parser.parse_directory(
            directory=directory,
            extensions=extensions,
            recursive=recursive
        )

        file_paths = list(results.keys())
        file_metadata = [
            {
                "filename": Path(p).name,
                "file_type": Path(p).suffix[1:],
                "source_path": p
            }
            for p in file_paths
        ]

        return self.ingest_texts(
            texts=list(results.values()),
            tenant_id=tenant_id,
            source=source,
            category=category,
            metadata=file_metadata,
            stages=stages
        )


class FAQIngester:
    """Pre-built FAQ ingester with common FAQ formats."""

    @staticmethod
    def from_dict(faq_dict: Dict[str, str]) -> List[Dict[str, str]]:
        """Convert FAQ dict to list format.

        Args:
            faq_dict: Dict mapping questions to answers

        Returns:
            List of {"question": "...", "answer": "..."}
        """
        return [
            {"question": q, "answer": a}
            for q, a in faq_dict.items()
        ]

    @staticmethod
    def from_list(faq_list: List[tuple]) -> List[Dict[str, str]]:
        """Convert FAQ list of tuples to standard format.

        Args:
            faq_list: List of (question, answer) tuples

        Returns:
            List of {"question": "...", "answer": "..."}
        """
        return [
            {"question": q, "answer": a}
            for q, a in faq_list
        ]


# ============================================================
# Ingestion Helper Functions
# ============================================================

def ingest_default_faqs(
    ingester: DocumentIngester,
    tenant_id: str = "default"
) -> Dict[str, int]:
    """Ingest default customer support FAQs.

    Args:
        ingester: DocumentIngester instance
        tenant_id: Tenant identifier

    Returns:
        Dict with counts per stage
    """
    default_faqs = [
        {
            "question": "How do I reset my password?",
            "answer": "To reset your password, click on 'Forgot Password' on the login page and follow the instructions sent to your email."
        },
        {
            "question": "What is your refund policy?",
            "answer": "Our refund policy allows refunds within 7 days of purchase. The item must be unused and in original packaging."
        },
        {
            "question": "How do I track my order?",
            "answer": "You can track your order in the 'My Orders' section of your account. You'll also receive email updates at each step."
        },
        {
            "question": "How do I cancel my order?",
            "answer": "You can cancel your order within 30 minutes of placing it from the My Orders page. After that, please contact support."
        },
        {
            "question": "What payment methods do you accept?",
            "answer": "We accept credit cards, debit cards, net banking, UPI, and digital wallets like PayPal and Apple Pay."
        },
        {
            "question": "How do I contact customer support?",
            "answer": "You can reach our support team at support@example.com or call 1-800-SUPPORT. Our live chat is available 24/7."
        },
        {
            "question": "How long does shipping take?",
            "answer": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days. Same-day delivery is available in select cities."
        },
        {
            "question": "Can I return an item?",
            "answer": "Yes, you can return items within 30 days of purchase. Go to Your Orders > Select the item > Return or replace items."
        },
        {
            "question": "How do I update my account information?",
            "answer": "Go to Account Settings > Profile. You can update your email, phone, address, and password from there."
        },
        {
            "question": "Where can I find my invoice?",
            "answer": "Your invoice is available in the My Orders section. Click on 'View Details' for any order to download the invoice."
        }
    ]

    return ingester.ingest_faq(default_faqs, tenant_id=tenant_id)
