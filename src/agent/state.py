"""State definitions and structured schemas for the LangGraph self-reflective workflow."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from src.retrieval.graph_traversal import GraphSearchResult
from src.retrieval.hybrid_search import RetrievedContext

QueryComplexity = Literal["simple", "multi_hop", "analytical"]
HallucinationGrade = Literal["grounded", "needs_rewrite", "exhausted"]


class QueryAnalysisResult(BaseModel):
    """Structured output of the query analysis and deconstruction node."""

    complexity: QueryComplexity = Field(default="simple")
    entities: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)


class ReflectionGradeResult(BaseModel):
    """Structured output of the self-reflection and relevance grading node."""

    relevance_score: float = Field(..., ge=0.0, le=1.0)
    is_relevant: bool
    grade: HallucinationGrade
    reasoning: str = Field(default="")


class AgentState(TypedDict, total=False):
    """Mutable graph state passed across LangGraph nodes."""

    original_query: str
    active_query: str
    query_complexity: QueryComplexity
    extracted_entities: list[str]
    sub_queries: list[str]
    vector_contexts: list[RetrievedContext]
    graph_contexts: list[GraphSearchResult]
    relevance_score: float
    is_relevant: bool
    retry_count: int
    max_retries: int
    hallucination_grade: HallucinationGrade
    final_answer: str
    citations: list[str]
    trace_steps: list[str]


def create_initial_state(query: str, max_retries: int = 2) -> AgentState:
    """Create a fully initialized AgentState dictionary for a new user query."""
    return AgentState(
        original_query=query,
        active_query=query,
        query_complexity="simple",
        extracted_entities=[],
        sub_queries=[query],
        vector_contexts=[],
        graph_contexts=[],
        relevance_score=0.0,
        is_relevant=False,
        retry_count=0,
        max_retries=max_retries,
        hallucination_grade="needs_rewrite",
        final_answer="",
        citations=[],
        trace_steps=[],
    )
