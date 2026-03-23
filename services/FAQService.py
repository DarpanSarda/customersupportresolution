"""
FAQService - FAQ ingestion service with separate collection naming.
"""

import os
from utils.QdrantManager import QdrantManager
from utils.EmbeddingManager import EmbeddingManager
from utils.FAQProcessor import FAQProcessor
from dotenv import load_dotenv

load_dotenv()


class FAQService:
    """FAQ ingestion service with faq_{tenant_id} collection naming."""

    def __init__(self):
        self.embeddings = EmbeddingManager()
        self.faq_processor = FAQProcessor()
        self._qdrant_managers = {}

    def _get_qdrant(self, tenant_id: str) -> QdrantManager:
        """Get or create FAQ Qdrant manager for tenant."""
        if tenant_id not in self._qdrant_managers:
            # Create manager with FAQ collection naming
            self._qdrant_managers[tenant_id] = QdrantManager(
                tenant_id=tenant_id,
                vector_size=self.embeddings.dimension
            )
            # Override collection name for FAQ
            self._qdrant_managers[tenant_id].collection_name = f"faq_{tenant_id}"
            # Ensure FAQ collection exists
            self._qdrant_managers[tenant_id]._ensure_collection_exists()
        return self._qdrant_managers[tenant_id]

    async def ingest_from_directory(self, directory_path: str, tenant_id: str = "default") -> dict:
        """Ingest all FAQ files from directory."""
        chunks = await self.document_processor.process_directory(directory_path)
        if not chunks:
            return {"status": "error", "message": "No files found"}

        texts = [c["text"] for c in chunks]
        embeddings = await self.embeddings.embed_batch(texts)

        qdrant = self._get_qdrant(tenant_id)
        await qdrant.add_documents(chunks, embeddings)

        return {
            "status": "success",
            "tenant_id": tenant_id,
            "collection": qdrant.collection_name,
            "chunks_added": len(chunks)
        }

    async def ingest_from_file(self, file_path: str, tenant_id: str = "default", original_filename: str = None) -> dict:
        """Ingest single FAQ file - extracts Q&A pairs and embeds only questions."""
        qa_pairs = await self.faq_processor.process_file(file_path, original_filename)
        if not qa_pairs:
            return {"status": "error", "message": "No Q&A pairs found"}

        # Embed only the questions
        texts = [qa["text"] for qa in qa_pairs]  # This is the question
        embeddings = await self.embeddings.embed_batch(texts)

        qdrant = self._get_qdrant(tenant_id)
        await qdrant.add_documents(qa_pairs, embeddings)

        return {
            "status": "success",
            "tenant_id": tenant_id,
            "collection": qdrant.collection_name,
            "qa_pairs_added": len(qa_pairs)
        }

    def get_collection_info(self, tenant_id: str = "default") -> dict:
        """Get FAQ collection information."""
        qdrant = self._get_qdrant(tenant_id)
        return qdrant.get_collection_info()
