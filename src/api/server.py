"""FastAPI server exposing /v1/query (JSON + SSE streaming) and /v1/index-document."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src import __version__
from src.agent.graph import AutonomousGraphRAGAgent
from src.api.schemas import (
    HealthResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    QueryRequest,
    QueryResponse,
)
from src.config import Settings, get_settings

_agent_instance: AutonomousGraphRAGAgent | None = None


def get_agent() -> AutonomousGraphRAGAgent:
    """Return or initialize the singleton AutonomousGraphRAGAgent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AutonomousGraphRAGAgent(settings=get_settings())
    return _agent_instance


def set_agent(agent: AutonomousGraphRAGAgent | None) -> None:
    """Override the singleton agent instance (used for testing and dependency injection)."""
    global _agent_instance
    _agent_instance = agent


def _serialize_sse_payload(data: dict[str, Any]) -> str:
    """Serialize node transition payload into SSE wire format."""

    def _default_serializer(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        return str(obj)

    encoded = json.dumps(data, default=_default_serializer)
    return f"data: {encoded}\n\n"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = settings or get_settings()
    application = FastAPI(
        title="Autonomous Graph-RAG Agent API",
        description=(
            "Production-grade, self-correcting Enterprise Graph-RAG microservice combining "
            "Qdrant dense retrieval, BM25 lexical search, Cross-Encoder reranking, "
            "Neo4j multi-hop graph traversal, and LangGraph agentic self-reflection."
        ),
        version=__version__,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check(
        agent: AutonomousGraphRAGAgent = Depends(get_agent),
    ) -> HealthResponse:
        """Return operational status and current index counts."""
        return HealthResponse(
            status="ok",
            service=cfg.app.name,
            version=__version__,
            indexed_chunks=agent.vector_indexer.count,
            knowledge_triples=agent.graph_store.triple_count,
        )

    @application.post(
        "/v1/index-document",
        response_model=IndexDocumentResponse,
        status_code=201,
        tags=["Ingestion"],
    )
    async def index_document(
        request: IndexDocumentRequest,
        agent: AutonomousGraphRAGAgent = Depends(get_agent),
    ) -> IndexDocumentResponse:
        """Chunk document, compute dense/sparse representations, and extract Neo4j triples."""
        try:
            summary = agent.ingest_document(
                doc_id=request.doc_id,
                text=request.text,
                metadata=request.metadata,
            )
            return IndexDocumentResponse(
                status="indexed",
                doc_id=str(summary["doc_id"]),
                chunks_indexed=int(summary["chunks_indexed"]),
                triples_extracted=int(summary["triples_extracted"]),
                chunk_hashes=list(summary["chunk_hashes"]),
                node_ids=list(summary["node_ids"]),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Document ingestion failed: {exc}"
            ) from exc

    @application.post(
        "/v1/query",
        response_model=QueryResponse,
        tags=["Retrieval & Reasoning"],
    )
    async def query_knowledge_base(
        request: QueryRequest,
        agent: AutonomousGraphRAGAgent = Depends(get_agent),
    ) -> Any:
        """Execute hybrid Graph-RAG query; returns SSE event stream when stream=True."""
        if request.stream:

            async def _event_generator() -> AsyncIterator[str]:
                for step_event in agent.stream_query(
                    question=request.query,
                    max_retries=request.max_retries,
                ):
                    for node_name, delta in step_event.items():
                        yield _serialize_sse_payload(
                            {"event": "node_update", "node": node_name, "data": delta}
                        )
                yield "data: [DONE]\n\n"

            return StreamingResponse(_event_generator(), media_type="text/event-stream")

        state = agent.query(question=request.query, max_retries=request.max_retries)
        response_payload = QueryResponse(
            query=request.query,
            answer=state.get("final_answer", ""),
            complexity=state.get("query_complexity", "simple"),
            extracted_entities=state.get("extracted_entities", []),
            relevance_score=float(state.get("relevance_score", 0.0)),
            retry_count=int(state.get("retry_count", 0)),
            hallucination_grade=state.get("hallucination_grade", "grounded"),
            citations=state.get("citations", []),
            vector_contexts=state.get("vector_contexts", []),
            graph_contexts=state.get("graph_contexts", []),
            trace_steps=state.get("trace_steps", []),
        )
        return JSONResponse(content=response_payload.model_dump())

    return application


app = create_app()
