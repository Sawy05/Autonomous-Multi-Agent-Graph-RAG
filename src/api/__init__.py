"""FastAPI HTTP & SSE microservice layer for the Autonomous Graph-RAG Agent."""

from src.api.schemas import (
    HealthResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    QueryRequest,
    QueryResponse,
)
from src.api.server import app, create_app

__all__ = [
    "HealthResponse",
    "IndexDocumentRequest",
    "IndexDocumentResponse",
    "QueryRequest",
    "QueryResponse",
    "app",
    "create_app",
]
