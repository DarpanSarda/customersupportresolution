"""
TrainingRAGAgent - Simple RAG training service.
"""

import os
from utils.QdrantManager import QdrantManager
from utils.EmbeddingManager import EmbeddingManager
from utils.DocumentProcessor import DocumentProcessor
from dotenv import load_dotenv

load_dotenv()


class TrainingRAGAgent:
    """Train RAG with PDF, MD, TXT files."""

    def __init__(self):
        self.embeddings = EmbeddingManager()
        self.document_processor = DocumentProcessor()
        self._qdrant_managers = {}

    def _get_qdrant(self, tenant_id: str) -> QdrantManager:
        if tenant_id not in self._qdrant_managers:
            self._qdrant_managers[tenant_id] = QdrantManager(
                tenant_id=tenant_id,
                vector_size=self.embeddings.dimension
            )
        return self._qdrant_managers[tenant_id]

    async def ingest_from_directory(self, directory_path: str, tenant_id: str = "default") -> dict:
        """Ingest all files from directory."""
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
        """Ingest single file."""
        chunks = await self.document_processor.process_file(file_path, original_filename)
        if not chunks:
            return {"status": "error", "message": "No content"}

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
