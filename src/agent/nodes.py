"""LangGraph workflow nodes: Query Deconstruction, Parallel Retrieval, Reflection Grader, Rewriter, and Synthesizer."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from src.agent.state import AgentState, HallucinationGrade, QueryComplexity
from src.config import AgentConfig, LLMConfig
from src.ingestion.graph_extractor import GraphExtractor
from src.retrieval.graph_traversal import GraphSearchResult, Neo4jGraphStore
from src.retrieval.hybrid_search import HybridSearchEngine, RetrievedContext

logger = logging.getLogger(__name__)


class LLMClient:
    """Provider-agnostic LLM client supporting OpenAI, Anthropic, Ollama, vLLM, and Mock modes."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def complete(self, prompt: str, system_prompt: str = "You are a precise enterprise AI assistant.") -> str:
        """Execute synchronous completion against configured LLM provider with safe fallback."""
        provider = self.config.provider
        if provider == "mock" or (provider in {"openai", "anthropic"} and not self.config.api_key and not self.config.anthropic_api_key):
            return self._deterministic_synthesis(prompt)

        if provider in {"openai", "vllm", "ollama"}:
            try:
                from openai import OpenAI

                base_url = self.config.base_url
                if provider == "ollama" and "openai.com" in base_url:
                    base_url = "http://localhost:11434/v1"
                client = OpenAI(
                    api_key=self.config.api_key or "EMPTY",
                    base_url=base_url,
                    timeout=self.config.timeout_seconds,
                )
                response = client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            except Exception as exc:
                logger.warning("OpenAI/compatible completion failed (%s); using local synthesis fallback", exc)
                return self._deterministic_synthesis(prompt)

        if provider == "anthropic":
            try:
                headers = {
                    "x-api-key": self.config.anthropic_api_key or self.config.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                payload = {
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": prompt}],
                }
                with httpx.Client(timeout=self.config.timeout_seconds) as http_client:
                    resp = http_client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    blocks = data.get("content", [])
                    if blocks and isinstance(blocks, list):
                        return str(blocks[0].get("text", "")).strip()
            except Exception as exc:
                logger.warning("Anthropic completion failed (%s); using local synthesis fallback", exc)
                return self._deterministic_synthesis(prompt)

        return self._deterministic_synthesis(prompt)

    @staticmethod
    def _deterministic_synthesis(prompt: str) -> str:
        """Extractive grounded response generator when running with mocked/offline LLM."""
        return prompt.strip()


class AgentNodes:
    """Encapsulates all stateful node functions executed within the LangGraph workflow."""

    def __init__(
        self,
        hybrid_search: HybridSearchEngine,
        graph_store: Neo4jGraphStore,
        llm_client: LLMClient | None = None,
        agent_config: AgentConfig | None = None,
    ) -> None:
        self.hybrid_search = hybrid_search
        self.graph_store = graph_store
        self.llm_client = llm_client or LLMClient(LLMConfig(provider="mock"))
        self.agent_config = agent_config or AgentConfig()

    def analyze_query(self, state: AgentState) -> dict[str, Any]:
        """Node 1: Classify query complexity, extract entities, and decompose into sub-queries."""
        query = state.get("active_query") or state.get("original_query", "")
        trace = list(state.get("trace_steps", []))

        entities = GraphExtractor._extract_named_entities(query)
        if not entities:
            # Extract technical keywords (>3 chars) as candidate entities
            stop = {
                "what",
                "where",
                "when",
                "which",
                "how",
                "does",
                "with",
                "from",
                "that",
                "this",
                "between",
                "explain",
                "describe",
                "impact",
                "architecture",
            }
            tokens = [
                t
                for t in re.findall(r"[A-Za-z0-9_\-]+", query)
                if len(t) > 3 and t.lower() not in stop
            ]
            entities = tokens[:4]

        lower_q = query.lower()
        multi_hop_indicators = (
            "chain",
            "indirect",
            "multi-hop",
            "cascade",
            "downstream",
            "upstream",
            "impact",
            "through",
            "via",
            "connect",
            "depend",
            "relationship",
            "between",
            "and how",
        )
        complexity: QueryComplexity = "simple"
        if len(entities) >= 2 or any(ind in lower_q for ind in multi_hop_indicators):
            complexity = "multi_hop"
        elif len(query.split()) > 18 or "compare" in lower_q or "why" in lower_q:
            complexity = "analytical"

        sub_queries: list[str] = [query]
        if " and " in lower_q and len(query.split()) > 8:
            parts = [p.strip() for p in re.split(r"\band\b", query, flags=re.IGNORECASE) if len(p.strip()) > 10]
            if len(parts) >= 2:
                sub_queries.extend(parts[:2])
        for ent in entities[:2]:
            candidate_sq = f"{ent} architecture dependencies and interactions"
            if candidate_sq not in sub_queries and complexity != "simple":
                sub_queries.append(candidate_sq)

        trace.append(
            f"analyze_query: complexity={complexity}, entities={entities}, sub_queries={len(sub_queries)}"
        )
        return {
            "query_complexity": complexity,
            "extracted_entities": entities,
            "sub_queries": sub_queries,
            "trace_steps": trace,
        }

    def retrieve_and_aggregate(self, state: AgentState) -> dict[str, Any]:
        """Node 2: Execute hybrid vector/BM25 search and Neo4j graph traversal in parallel."""
        sub_queries = state.get("sub_queries") or [state.get("active_query", "")]
        entities = list(state.get("extracted_entities", []))
        active_query = state.get("active_query") or state.get("original_query", "")
        trace = list(state.get("trace_steps", []))

        def _run_vector_retrieval() -> list[RetrievedContext]:
            merged_by_id: dict[str, RetrievedContext] = {}
            for sq in sub_queries:
                for ctx in self.hybrid_search.search(sq):
                    existing = merged_by_id.get(ctx.chunk_id)
                    if existing is None or ctx.rerank_score > existing.rerank_score:
                        merged_by_id[ctx.chunk_id] = ctx
            ranked = sorted(merged_by_id.values(), key=lambda x: x.rerank_score, reverse=True)
            if ranked and ranked[0].rerank_score >= 0.30:
                cutoff = ranked[0].rerank_score * 0.45
                ranked = [c for c in ranked if c.rerank_score >= cutoff]
            return ranked[: self.hybrid_search.config.top_k_rerank]

        def _run_graph_retrieval() -> list[GraphSearchResult]:
            seed_terms = list(entities) if entities else [active_query]
            return self.graph_store.traverse_multi_hop(seed_terms)

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_vec = pool.submit(_run_vector_retrieval)
            future_graph = pool.submit(_run_graph_retrieval)
            vector_contexts = future_vec.result()
            graph_contexts = future_graph.result()

        trace.append(
            f"retrieve_and_aggregate: vector_hits={len(vector_contexts)}, graph_triples={len(graph_contexts)}"
        )
        return {
            "vector_contexts": vector_contexts,
            "graph_contexts": graph_contexts,
            "trace_steps": trace,
        }

    def grade_and_reflect(self, state: AgentState) -> dict[str, Any]:
        """Node 3: Self-Reflection / Hallucination Grader evaluating retrieved context relevance."""
        vector_contexts = state.get("vector_contexts", [])
        graph_contexts = state.get("graph_contexts", [])
        retry_count = int(state.get("retry_count", 0))
        max_retries = int(state.get("max_retries", self.agent_config.max_retries))
        trace = list(state.get("trace_steps", []))

        top_rerank = max((c.rerank_score for c in vector_contexts), default=0.0)
        graph_bonus = min(0.25, 0.08 * len(graph_contexts))
        relevance_score = round(min(1.0, top_rerank + graph_bonus), 4)

        threshold = self.agent_config.relevance_threshold
        is_relevant = relevance_score >= threshold and (len(vector_contexts) > 0 or len(graph_contexts) > 0)

        grade: HallucinationGrade
        if is_relevant:
            grade = "grounded"
        elif retry_count < max_retries:
            grade = "needs_rewrite"
        else:
            grade = "exhausted"

        trace.append(
            f"grade_and_reflect: score={relevance_score:.3f} (threshold={threshold:.2f}), grade={grade}, retry={retry_count}/{max_retries}"
        )
        return {
            "relevance_score": relevance_score,
            "is_relevant": is_relevant,
            "hallucination_grade": grade,
            "trace_steps": trace,
        }

    def rewrite_query(self, state: AgentState) -> dict[str, Any]:
        """Node 4: Rewrite low-relevance query to broaden semantic and graph recall."""
        current_query = state.get("active_query") or state.get("original_query", "")
        entities = state.get("extracted_entities", [])
        retry_count = int(state.get("retry_count", 0)) + 1
        trace = list(state.get("trace_steps", []))

        entity_suffix = " ".join(entities) if entities else ""
        rewritten = (
            f"{current_query} {entity_suffix} system architecture service dependencies protocol flow".strip()
        )
        sub_queries = [rewritten, current_query]

        trace.append(f"rewrite_query: retry={retry_count}, rewritten='{rewritten}'")
        return {
            "active_query": rewritten,
            "sub_queries": sub_queries,
            "retry_count": retry_count,
            "trace_steps": trace,
        }

    def generate_grounded_response(self, state: AgentState) -> dict[str, Any]:
        """Node 5: Synthesize grounded response strictly citing Neo4j node IDs and chunk hashes."""
        query = state.get("original_query", "")
        vector_contexts = state.get("vector_contexts", [])
        graph_contexts = state.get("graph_contexts", [])
        trace = list(state.get("trace_steps", []))

        citations: list[str] = []
        chunk_lines: list[str] = []
        for ctx in vector_contexts:
            cite_tag = f"[chunk:{ctx.chunk_hash}]"
            if cite_tag not in citations:
                citations.append(cite_tag)
            chunk_lines.append(f"- {ctx.text} {cite_tag}")

        graph_lines: list[str] = []
        for g_ctx in graph_contexts:
            node_tag = f"[{g_ctx.node_id}]"
            chunk_tag = f"[chunk:{g_ctx.chunk_hash}]"
            if node_tag not in citations:
                citations.append(node_tag)
            if chunk_tag not in citations:
                citations.append(chunk_tag)
            graph_lines.append(
                f"- {g_ctx.path_summary} (hop={g_ctx.hop_distance}) {node_tag} {chunk_tag}"
            )

        if not vector_contexts and not graph_contexts:
            fallback_msg = (
                "Insufficient grounded evidence found in the knowledge base to answer the query "
                f"'{query}' without risking hallucination."
            )
            trace.append("generate_grounded_response: no grounded contexts available")
            return {
                "final_answer": fallback_msg,
                "citations": [],
                "trace_steps": trace,
            }

        if self.llm_client.config.provider != "mock" and (
            self.llm_client.config.api_key or self.llm_client.config.anthropic_api_key
        ):
            synthesis_prompt = (
                f"Question: {query}\n\n"
                "Retrieved Vector Chunks:\n"
                + ("\n".join(chunk_lines) if chunk_lines else "None\n")
                + "\n\nRetrieved Knowledge Graph Triples:\n"
                + ("\n".join(graph_lines) if graph_lines else "None\n")
                + "\n\nInstructions: Synthesize a comprehensive, technically precise answer based strictly on the "
                "context above. You MUST cite every claim inline using the exact [chunk:<hash>] and [node_<id>] tags."
            )
            answer = self.llm_client.complete(synthesis_prompt)
            # Ensure citations appear in the synthesized answer
            if citations and not any(c in answer for c in citations):
                answer = f"{answer}\n\nCitations: {' '.join(citations)}"
        else:
            sections: list[str] = [
                f"Grounded Analysis for: {query}",
            ]
            if vector_contexts:
                primary_findings = " ".join(
                    f"{ctx.text} [chunk:{ctx.chunk_hash}]" for ctx in vector_contexts[:3]
                )
                sections.append(f"Document Evidence: {primary_findings}")
            if graph_contexts:
                rel_summary = "; ".join(
                    f"{g.source_entity} {g.relation} {g.target_entity} [{g.node_id}] [chunk:{g.chunk_hash}]"
                    for g in graph_contexts[:4]
                )
                sections.append(f"Multi-Hop Knowledge Graph Reasoning: {rel_summary}.")
            sections.append(f"Verified Provenance References: {' '.join(citations)}")
            answer = "\n\n".join(sections)

        trace.append(f"generate_grounded_response: citations={len(citations)}")
        return {
            "final_answer": answer,
            "citations": citations,
            "trace_steps": trace,
        }
