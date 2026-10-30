"""Integration tests for LangGraph self-reflection workflow, FastAPI SSE endpoints, and DeepEval harness."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.agent.graph import AutonomousGraphRAGAgent
from src.api.server import create_app, set_agent
from src.config import Settings
from src.eval.run_evals import run_evaluation_suite


def _build_test_agent(relevance_threshold: float = 0.25, max_retries: int = 2) -> AutonomousGraphRAGAgent:
    settings = Settings()
    settings.llm.provider = "mock"
    settings.retrieval.use_lightweight_embeddings = True
    settings.qdrant.in_memory = True
    settings.agent.relevance_threshold = relevance_threshold
    settings.agent.max_retries = max_retries
    return AutonomousGraphRAGAgent(settings=settings)


def test_agent_workflow_grounded_multi_hop_citations() -> None:
    agent = _build_test_agent(relevance_threshold=0.20, max_retries=2)
    ingest_res = agent.ingest_document(
        doc_id="doc_security",
        text=(
            "ApolloGateway authenticates via SentinelAuth using mutual TLS. "
            "SentinelAuth stores data in RedisCluster for revoked JWT token tracking."
        ),
    )
    assert ingest_res["chunks_indexed"] >= 1
    assert ingest_res["triples_extracted"] >= 2

    state = agent.query(
        "How does ApolloGateway connect to SentinelAuth and where are tokens stored in RedisCluster?"
    )
    assert state["query_complexity"] == "multi_hop"
    assert state["hallucination_grade"] == "grounded"
    assert state["retry_count"] == 0
    assert len(state["citations"]) >= 2
    assert any(c.startswith("[chunk:") for c in state["citations"])
    assert any(c.startswith("[node_") for c in state["citations"])
    for citation in state["citations"]:
        assert citation in state["final_answer"]


def test_agent_self_reflection_query_rewrite_loop() -> None:
    # Set an unreachable relevance threshold (1.0 + epsilon effectively) by setting threshold=1.0
    # and indexing unrelated content so the grader triggers max_retries=2 rewrites.
    agent = _build_test_agent(relevance_threshold=0.99, max_retries=2)
    agent.ingest_document(
        doc_id="doc_misc",
        text="ChronosScheduler orchestrates NebulaIngestionPipeline over KafkaEventMesh.",
    )

    state = agent.query("Unrelated quantum photonic telemetry calibration", max_retries=2)
    assert state["retry_count"] == 2
    assert state["hallucination_grade"] == "exhausted"
    rewrite_steps = [s for s in state["trace_steps"] if s.startswith("rewrite_query:")]
    assert len(rewrite_steps) == 2


def test_fastapi_endpoints_and_sse_streaming() -> None:
    agent = _build_test_agent(relevance_threshold=0.20, max_retries=2)
    set_agent(agent)
    app = create_app(agent.settings)
    client = TestClient(app)

    # 1. Health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"

    # 2. Index document
    index_resp = client.post(
        "/v1/index-document",
        json={
            "doc_id": "doc_ledger",
            "text": (
                "HyperionPaymentEngine depends on CockroachDBLedger for serializable ACID consensus. "
                "HyperionPaymentEngine authenticates via SentinelAuth."
            ),
            "metadata": {"team": "payments"},
        },
    )
    assert index_resp.status_code == 201
    index_data = index_resp.json()
    assert index_data["status"] == "indexed"
    assert index_data["chunks_indexed"] >= 1
    assert index_data["triples_extracted"] >= 1

    # 3. Synchronous query
    query_resp = client.post(
        "/v1/query",
        json={
            "query": "How does HyperionPaymentEngine use CockroachDBLedger and SentinelAuth?",
            "stream": False,
        },
    )
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert "HyperionPaymentEngine" in query_data["answer"]
    assert len(query_data["citations"]) >= 1

    # 4. Streaming SSE query
    stream_resp = client.post(
        "/v1/query",
        json={
            "query": "How does HyperionPaymentEngine use CockroachDBLedger?",
            "stream": True,
        },
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    body_text = stream_resp.text
    assert "analyze_query" in body_text
    assert "retrieve_and_aggregate" in body_text
    assert "generate_grounded_response" in body_text
    assert "data: [DONE]" in body_text


def test_llmops_evaluation_harness() -> None:
    report = run_evaluation_suite()
    assert report.total_cases == 4
    assert report.mean_faithfulness >= 0.75
    assert report.mean_answer_relevance >= 0.70
    assert report.mean_context_precision >= 0.75
    assert report.citation_compliance_rate == 1.0
