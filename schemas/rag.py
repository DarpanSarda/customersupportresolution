"""
RAG-related schemas for request/response models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class RerankerType(str, Enum):
    """Available reranker types."""
    cohere = "cohere"
    bge = "bge"


class IngestDirectoryRequest(BaseModel):
    """Request for ingesting documents from a directory."""
    directory_path: str = Field(..., description="Path to the directory containing documents")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    chunk_size: Optional[int] = Field(default=500, description="Maximum characters per chunk")
    chunk_overlap: Optional[int] = Field(default=50, description="Overlap between chunks")
    recursive: Optional[bool] = Field(default=True, description="Process subdirectories")


class IngestFileRequest(BaseModel):
    """Request for ingesting a single file."""
    file_path: str = Field(..., description="Path to the file to ingest")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    chunk_size: Optional[int] = Field(default=500, description="Maximum characters per chunk")
    chunk_overlap: Optional[int] = Field(default=50, description="Overlap between chunks")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class IngestTextRequest(BaseModel):
    """Request for ingesting raw text."""
    text: str = Field(..., description="Text content to ingest")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    chunk_size: Optional[int] = Field(default=500, description="Maximum characters per chunk")
    chunk_overlap: Optional[int] = Field(default=50, description="Overlap between chunks")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class IngestJsonRequest(BaseModel):
    """Request for ingesting JSON data."""
    data: List[Dict[str, Any]] = Field(..., description="List of JSON objects")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    text_field: str = Field(default="text", description="Field name containing text content")
    metadata_fields: Optional[List[str]] = Field(default=None, description="Fields to include as metadata")
    chunk_size: Optional[int] = Field(default=500, description="Maximum characters per chunk")
    chunk_overlap: Optional[int] = Field(default=50, description="Overlap between chunks")


class DeleteKnowledgeRequest(BaseModel):
    """Request for deleting knowledge base."""
    tenant_id: str = Field(..., description="Tenant identifier")


class RecreateCollectionRequest(BaseModel):
    """Request for recreating collection."""
    tenant_id: str = Field(default="default", description="Tenant identifier")


class RAGSearchRequest(BaseModel):
    """Request for RAG search."""
    query: str = Field(..., description="Search query")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    top_k: Optional[int] = Field(default=5, description="Number of results to return")
    use_reranker: Optional[bool] = Field(default=True, description="Whether to use reranking")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters for search")


class RAGSearchResult(BaseModel):
    """Single RAG search result."""
    id: str
    text: str
    score: float
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGSearchResponse(BaseModel):
    """Response for RAG search."""
    query: str
    results: List[RAGSearchResult]
    source_type: str = Field(default="knowledge_base", description="Source of results (faq or knowledge_base)")
    total_results: int
    tenant_id: str


class KnowledgeStats(BaseModel):
    """Knowledge base statistics."""
    tenant_id: str
    collection_name: str
    collection_exists: bool
    document_count: Optional[int] = None
    vector_size: Optional[int] = None


class IngestResponse(BaseModel):
    """Response for document ingestion."""
    status: str
    tenant_id: str
    collection_name: str
    chunks_added: int
    total_documents: Optional[int] = None
    message: str


class RAGConfig(BaseModel):
    """RAG configuration."""
    embedding_model: str
    embedding_dimension: int
    reranker_type: str
    reranker_model: str
    chunk_size: int
    chunk_overlap: int


class CollectionInfo(BaseModel):
    """Collection information."""
    exists: bool
    collection_name: str
    vectors_count: Optional[int] = None
    vector_size: Optional[int] = None
    distance: Optional[str] = None
