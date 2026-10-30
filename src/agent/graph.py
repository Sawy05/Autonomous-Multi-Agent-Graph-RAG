"""LangGraph state machine assembly with cyclic self-reflection and query rewriting."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from src.agent.nodes import AgentNodes, LLMClient
from src.agent.state import AgentState, create_initial_state
from src.config import Settings, get_settings
from src.ingestion.chunker import SemanticChunker
from src.ingestion.graph_extractor import GraphExtractor
from src.ingestion.vector_indexer import VectorIndexer
from src.retrieval.graph_traversal import Neo4jGraphStore
from src.retrieval.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


def _route_after_grading(state: AgentState) -> str:
    """Conditional edge router: loop to rewrite_query if relevance is low and retries remain."""
    grade = state.get("hallucination_grade", "grounded")
    if grade == "needs_rewrite":
        return "rewrite_query"
    return "generate_grounded_response"


class FallbackCompiledGraph:
    """Deterministic state-graph runner matching LangGraph's .invoke() and .stream() interface."""

    def __init__(self, nodes: AgentNodes) -> None:
        self.nodes = nodes

    def invoke(self, state: AgentState) -> AgentState:
        """Execute the cyclic state graph to completion and return final AgentState."""
        current: AgentState = dict(state)  # type: ignore[assignment]
        for step_update in self.stream(current):
            for _node_name, delta in step_update.items():
                current.update(delta)  # type: ignore[typeddict-item]
        return current

    def stream(self, state: AgentState) -> Iterator[dict[str, dict[str, Any]]]:
        """Yield node-by-node state updates following the cyclic reflection graph."""
        current: AgentState = dict(state)  # type: ignore[assignment]

        delta_analyze = self.nodes.analyze_query(current)
        current.update(delta_analyze)  # type: ignore[typeddict-item]
        yield {"analyze_query": delta_analyze}

        while True:
            delta_retrieve = self.nodes.retrieve_and_aggregate(current)
            current.update(delta_retrieve)  # type: ignore[typeddict-item]
            yield {"retrieve_and_aggregate": delta_retrieve}

            delta_grade = self.nodes.grade_and_reflect(current)
            current.update(delta_grade)  # type: ignore[typeddict-item]
            yield {"grade_and_reflect": delta_grade}

            next_node = _route_after_grading(current)
            if next_node == "rewrite_query":
                delta_rewrite = self.nodes.rewrite_query(current)
                current.update(delta_rewrite)  # type: ignore[typeddict-item]
                yield {"rewrite_query": delta_rewrite}
            else:
                break

        delta_gen = self.nodes.generate_grounded_response(current)
        current.update(delta_gen)  # type: ignore[typeddict-item]
        yield {"generate_grounded_response": delta_gen}


def build_agent_graph(nodes: AgentNodes) -> Any:
    """Build and compile the LangGraph StateGraph (or FallbackCompiledGraph if LangGraph is absent)."""
    try:
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(AgentState)
        workflow.add_node("analyze_query", nodes.analyze_query)
        workflow.add_node("retrieve_and_aggregate", nodes.retrieve_and_aggregate)
        workflow.add_node("grade_and_reflect", nodes.grade_and_reflect)
        workflow.add_node("rewrite_query", nodes.rewrite_query)
        workflow.add_node("generate_grounded_response", nodes.generate_grounded_response)

        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "retrieve_and_aggregate")
        workflow.add_edge("retrieve_and_aggregate", "grade_and_reflect")
        workflow.add_conditional_edges(
            "grade_and_reflect",
            _route_after_grading,
            {
                "rewrite_query": "rewrite_query",
                "generate_grounded_response": "generate_grounded_response",
            },
        )
        workflow.add_edge("rewrite_query", "retrieve_and_aggregate")
        workflow.add_edge("generate_grounded_response", END)
        return workflow.compile()
    except Exception as exc:
        logger.info("Using built-in deterministic StateGraph engine (LangGraph import: %s)", exc)
        return FallbackCompiledGraph(nodes)


class AutonomousGraphRAGAgent:
    """High-level facade coordinating document ingestion and self-correcting Graph-RAG queries."""

    def __init__(
        self,
        settings: Settings | None = None,
        hybrid_search: HybridSearchEngine | None = None,
        graph_store: Neo4jGraphStore | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chunker = SemanticChunker(
            chunk_size=self.settings.ingestion.chunk_size,
            chunk_overlap=self.settings.ingestion.chunk_overlap,
            min_chunk_length=self.settings.ingestion.min_chunk_length,
        )
        self.llm_client = llm_client or LLMClient(self.settings.llm)
        self.graph_extractor = GraphExtractor(
            llm_client=(
                self.llm_client
                if self.settings.llm.provider != "mock" and self.settings.llm.api_key
                else None
            )
        )
        self.vector_indexer = (
            hybrid_search.vector_indexer
            if hybrid_search is not None
            else VectorIndexer(
                qdrant_config=self.settings.qdrant,
                retrieval_config=self.settings.retrieval,
            )
        )
        self.hybrid_search = hybrid_search or HybridSearchEngine(
            vector_indexer=self.vector_indexer,
            config=self.settings.retrieval,
        )
        self.graph_store = graph_store or Neo4jGraphStore(config=self.settings.neo4j)
        self.nodes = AgentNodes(
            hybrid_search=self.hybrid_search,
            graph_store=self.graph_store,
            llm_client=self.llm_client,
            agent_config=self.settings.agent,
        )
        self.graph = build_agent_graph(self.nodes)

    def ingest_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Chunk document, index dense + sparse vectors, and extract Neo4j knowledge triples."""
        chunks = self.chunker.chunk_document(doc_id=doc_id, text=text, metadata=metadata)
        indexed_chunks = self.hybrid_search.add_chunks(chunks)

        extracted_triples = []
        if self.settings.ingestion.extract_graph_triples:
            extracted_triples = self.graph_extractor.extract_from_chunks(chunks)
            self.graph_store.upsert_triples(extracted_triples)

        return {
            "doc_id": doc_id,
            "chunks_indexed": indexed_chunks,
            "triples_extracted": len(extracted_triples),
            "chunk_hashes": [c.chunk_hash for c in chunks],
            "node_ids": [t.triple_id for t in extracted_triples],
        }

    def query(self, question: str, max_retries: int | None = None) -> AgentState:
        """Run the full self-correcting Graph-RAG workflow for a user question."""
        retries = max_retries if max_retries is not None else self.settings.agent.max_retries
        initial_state = create_initial_state(query=question, max_retries=retries)
        result: AgentState = self.graph.invoke(initial_state)
        return result

    def stream_query(
        self,
        question: str,
        max_retries: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream intermediate LangGraph node transitions and final grounded synthesis."""
        retries = max_retries if max_retries is not None else self.settings.agent.max_retries
        initial_state = create_initial_state(query=question, max_retries=retries)
        for event in self.graph.stream(initial_state):
            yield dict(event)
