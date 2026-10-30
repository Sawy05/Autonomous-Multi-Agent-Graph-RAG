"""Pydantic v2 request and response schemas for the FastAPI Graph-RAG service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.retrieval.graph_traversal import GraphSearchResult
from src.retrieval.hybrid_search import RetrievedContext


class HealthResponse(BaseModel):
    """Service readiness and storage metrics payload."""

    status: Literal["ok"] = "ok"
    service: str
    version: str
    indexed_chunks: int = Field(default=0, ge=0)
    knowledge_triples: int = Field(default=0, ge=0)


class IndexDocumentRequest(BaseModel):
    """Payload for indexing a new technical document into Qdrant, BM25, and Neo4j."""

    doc_id: str = Field(..., min_length=1, max_length=256, description="Unique document identifier")
    text: str = Field(..., min_length=10, description="Full raw text of the document")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional provenance metadata (author, domain, version, tags)",
    )


class IndexDocumentResponse(BaseModel):
    """Summary of chunks and Neo4j knowledge triples created during ingestion."""

    status: Literal["indexed"] = "indexed"
    doc_id: str
    chunks_indexed: int = Field(..., ge=0)
    triples_extracted: int = Field(..., ge=0)
    chunk_hashes: list[str] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Payload for querying the self-correcting Graph-RAG agent."""

    query: str = Field(..., min_length=3, max_length=2000, description="User question")
    stream: bool = Field(
        default=False,
        description="When true, streams Server-Sent Events (SSE) for each LangGraph node transition",
    )
    max_retries: int | None = Field(
        default=None,
        ge=0,
        le=5,
        description="Optional override for self-reflection query rewrite retries",
    )


class QueryResponse(BaseModel):
    """Grounded response with node/chunk citations, relevance metrics, and execution trace."""

    query: str
    answer: str
    complexity: Literal["simple", "multi_hop", "analytical"] = "simple"
    extracted_entities: list[str] = Field(default_factory=list)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    retry_count: int = Field(default=0, ge=0)
    hallucination_grade: Literal["grounded", "needs_rewrite", "exhausted"] = "grounded"
    citations: list[str] = Field(default_factory=list)
    vector_contexts: list[RetrievedContext] = Field(default_factory=list)
    graph_contexts: list[GraphSearchResult] = Field(default_factory=list)
    trace_steps: list[str] = Field(default_factory=list)
