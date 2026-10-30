"""DeepEval / Ragas evaluation harness & synthetic test dataset generator."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.agent.graph import AutonomousGraphRAGAgent
from src.config import Settings

logger = logging.getLogger(__name__)

SEED_ENTERPRISE_DOCS: list[dict[str, str]] = [
    {
        "doc_id": "doc_auth_topology",
        "title": "Zero-Trust Gateway & Token Revocation Architecture",
        "text": (
            "ApolloGateway serves as the edge ingress controller for all enterprise microservices. "
            "ApolloGateway authenticates via SentinelAuth using mutual TLS (mTLS) and Ed25519 JWT validation. "
            "SentinelAuth stores data in RedisCluster to track real-time revoked token signatures with sub-millisecond TTL lookup. "
            "Furthermore, SentinelAuth depends on VaultKeyManager for automated cryptographic key rotation every 24 hours."
        ),
    },
    {
        "doc_id": "doc_data_pipeline",
        "title": "Knowledge Ingestion & Multi-Index Synchronization",
        "text": (
            "ChronosScheduler orchestrates NebulaIngestionPipeline on a continuous 5-minute cadence. "
            "NebulaIngestionPipeline connects to KafkaEventMesh to consume raw enterprise document streams. "
            "During indexing, NebulaIngestionPipeline stores data in QdrantVectorCluster for dense HNSW vector retrieval "
            "and writes to Neo4jCore for multi-hop entity-relationship traversal."
        ),
    },
    {
        "doc_id": "doc_payment_resilience",
        "title": "Distributed Ledger & Failover Guarantees",
        "text": (
            "HyperionPaymentEngine processes high-throughput institutional settlement transactions. "
            "HyperionPaymentEngine depends on CockroachDBLedger for serializable ACID consensus across three availability zones. "
            "HyperionPaymentEngine mitigates double-charge race conditions by enforcing deterministic UUIDv5 idempotency keys "
            "and authenticates via SentinelAuth before committing ledger mutations."
        ),
    },
    {
        "doc_id": "doc_observability",
        "title": "Observability Mesh & Automated Incident Routing",
        "text": (
            "PrometheusScraper connects to QdrantVectorCluster and Neo4jCore to collect p99 latency and memory saturation metrics. "
            "PrometheusScraper routes to AlertManagerService whenever tail latency exceeds 150ms. "
            "AlertManagerService connects to PagerDutyRouter to trigger automated on-call incident remediation workflows."
        ),
    },
]


class SyntheticTestCase(BaseModel):
    """Single evaluation test case with reference ground truth and expected entities."""

    case_id: str
    category: str
    question: str
    expected_answer: str
    expected_entities: list[str] = Field(default_factory=list)
    seed_doc_id: str


class CaseMetricScores(BaseModel):
    """Per-question DeepEval / Ragas metric scores."""

    case_id: str
    question: str
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevance: float = Field(..., ge=0.0, le=1.0)
    context_precision: float = Field(..., ge=0.0, le=1.0)
    has_strict_citations: bool
    latency_steps: int


class EvaluationReport(BaseModel):
    """Aggregated LLMOps benchmark evaluation report."""

    total_cases: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_precision: float
    citation_compliance_rate: float
    case_results: list[CaseMetricScores] = Field(default_factory=list)


class SyntheticDatasetGenerator:
    """Generates synthetic multi-hop and hybrid retrieval test cases from technical seed documents."""

    def __init__(self, seed_docs: list[dict[str, str]] | None = None) -> None:
        self.seed_docs = seed_docs or SEED_ENTERPRISE_DOCS

    def generate_dataset(self) -> list[SyntheticTestCase]:
        """Generate structured QA benchmark pairs from the seed technical corpus."""
        cases: list[SyntheticTestCase] = []
        for idx, doc in enumerate(self.seed_docs, start=1):
            doc_id = doc["doc_id"]
            text = doc["text"]
            entities = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b", text)
            unique_entities = list(dict.fromkeys(entities))
            if len(unique_entities) >= 2:
                src = unique_entities[0]
                tgt = unique_entities[-1]
                question = f"How does {src} interact with downstream services to reach {tgt}?"
            else:
                question = f"What is the architecture described in {doc['title']}?"

            cases.append(
                SyntheticTestCase(
                    case_id=f"syn_{idx:03d}",
                    category="multi_hop_reasoning",
                    question=question,
                    expected_answer=text,
                    expected_entities=unique_entities,
                    seed_doc_id=doc_id,
                )
            )
        return cases


class RAGMetricsEvaluator:
    """Evaluates Faithfulness, Answer Relevance, and Context Precision (with DeepEval hook)."""

    def __init__(self, use_deepeval_llm_judge: bool = False) -> None:
        self.use_deepeval_llm_judge = use_deepeval_llm_judge

    @staticmethod
    def _tokenize_content(text: str) -> set[str]:
        stop = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "of",
            "to",
            "for",
            "and",
            "or",
            "is",
            "are",
            "with",
            "by",
            "via",
            "how",
            "what",
            "where",
            "which",
            "does",
            "from",
            "as",
            "at",
        }
        return {
            tok
            for tok in re.findall(r"[a-z0-9_]+", text.lower())
            if len(tok) > 2 and tok not in stop
        }

    def score_faithfulness(self, answer: str, retrieved_contexts: list[str]) -> float:
        """Measure fraction of factual claims/tokens in answer grounded in retrieved contexts."""
        if not answer.strip() or not retrieved_contexts:
            return 0.0

        context_blob = " ".join(retrieved_contexts)
        context_tokens = self._tokenize_content(context_blob)
        # Strip structural template boilerplate before checking grounding
        cleaned_answer = re.sub(
            r"(Grounded Analysis for:.*?Document Evidence:|Multi-Hop Knowledge Graph Reasoning:|Verified Provenance References:|\bchunk:[a-f0-9]+\b|\bnode_[a-f0-9]+\b)",
            " ",
            answer,
        )
        answer_tokens = self._tokenize_content(cleaned_answer)
        if not answer_tokens:
            return 1.0

        grounded_count = sum(1 for tok in answer_tokens if tok in context_tokens)
        return round(min(1.0, grounded_count / len(answer_tokens)), 4)

    def score_answer_relevance(
        self,
        question: str,
        answer: str,
        expected_entities: list[str],
    ) -> float:
        """Measure semantic/entity alignment between question, expected entities, and generated answer."""
        if not answer.strip():
            return 0.0

        q_tokens = self._tokenize_content(question)
        a_tokens = self._tokenize_content(answer)
        q_overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens))

        if expected_entities:
            lower_ans = answer.lower()
            entity_hits = sum(1 for ent in expected_entities if ent.lower() in lower_ans)
            entity_recall = entity_hits / len(expected_entities)
            score = 0.45 * q_overlap + 0.55 * entity_recall
        else:
            score = q_overlap

        return round(min(1.0, max(0.0, score)), 4)

    def score_context_precision(
        self,
        retrieved_contexts: list[str],
        expected_answer: str,
        expected_entities: list[str],
    ) -> float:
        """Calculate Average Precision (AP@K) of retrieved chunks/triples with respect to ground truth."""
        if not retrieved_contexts:
            return 0.0

        target_tokens = self._tokenize_content(expected_answer)
        entity_lowers = [e.lower() for e in expected_entities]

        relevant_hits = 0
        precision_sum = 0.0
        for rank, ctx in enumerate(retrieved_contexts, start=1):
            ctx_lower = ctx.lower()
            ctx_tokens = self._tokenize_content(ctx)
            token_overlap = len(ctx_tokens & target_tokens) / max(1, len(target_tokens))
            has_entity = any(ent in ctx_lower for ent in entity_lowers)
            is_relevant = token_overlap >= 0.15 or has_entity
            if is_relevant:
                relevant_hits += 1
                precision_sum += relevant_hits / rank

        if relevant_hits == 0:
            return 0.0
        return round(min(1.0, precision_sum / relevant_hits), 4)


def run_evaluation_suite(
    test_cases_path: Path | None = None,
) -> EvaluationReport:
    """Seed an isolated Graph-RAG agent, execute test cases, and compute DeepEval/Ragas metrics."""
    settings = Settings()
    settings.llm.provider = "mock"
    settings.retrieval.use_lightweight_embeddings = True
    settings.qdrant.in_memory = True

    agent = AutonomousGraphRAGAgent(settings=settings)
    for doc in SEED_ENTERPRISE_DOCS:
        agent.ingest_document(
            doc_id=doc["doc_id"],
            text=doc["text"],
            metadata={"title": doc["title"]},
        )

    resolved_path = test_cases_path or Path(__file__).resolve().parent / "test_cases.json"
    if resolved_path.exists():
        raw_cases: list[dict[str, Any]] = json.loads(resolved_path.read_text(encoding="utf-8"))
        test_cases = [SyntheticTestCase.model_validate(c) for c in raw_cases]
    else:
        test_cases = SyntheticDatasetGenerator().generate_dataset()

    evaluator = RAGMetricsEvaluator()
    case_scores: list[CaseMetricScores] = []

    for tc in test_cases:
        state = agent.query(tc.question)
        answer = state.get("final_answer", "")
        citations = state.get("citations", [])
        vec_texts = [c.text for c in state.get("vector_contexts", [])]
        graph_texts = [
            f"{g.source_entity} {g.relation} {g.target_entity} {g.evidence}"
            for g in state.get("graph_contexts", [])
        ]
        all_contexts = vec_texts + graph_texts

        faithfulness = evaluator.score_faithfulness(answer, all_contexts)
        relevance = evaluator.score_answer_relevance(tc.question, answer, tc.expected_entities)
        precision = evaluator.score_context_precision(
            all_contexts,
            tc.expected_answer,
            tc.expected_entities,
        )
        has_citations = len(citations) > 0 and any(tag in answer for tag in citations)

        case_scores.append(
            CaseMetricScores(
                case_id=tc.case_id,
                question=tc.question,
                faithfulness=faithfulness,
                answer_relevance=relevance,
                context_precision=precision,
                has_strict_citations=has_citations,
                latency_steps=len(state.get("trace_steps", [])),
            )
        )

    n = max(1, len(case_scores))
    return EvaluationReport(
        total_cases=len(case_scores),
        mean_faithfulness=round(sum(c.faithfulness for c in case_scores) / n, 4),
        mean_answer_relevance=round(sum(c.answer_relevance for c in case_scores) / n, 4),
        mean_context_precision=round(sum(c.context_precision for c in case_scores) / n, 4),
        citation_compliance_rate=round(
            sum(1.0 for c in case_scores if c.has_strict_citations) / n,
            4,
        ),
        case_results=case_scores,
    )


if __name__ == "__main__":
    report = run_evaluation_suite()
    print(json.dumps(report.model_dump(), indent=2))
