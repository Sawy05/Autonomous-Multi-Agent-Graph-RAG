"""Unit tests for semantic chunking, hybrid Dense+BM25 RRF retrieval, Cross-Encoder reranking, and Neo4j traversal."""

from __future__ import annotations

from typing import Any

from src.config import Neo4jConfig, QdrantConfig, RetrievalConfig
from src.ingestion.chunker import DocumentChunk, SemanticChunker
from src.ingestion.graph_extractor import GraphExtractor
from src.ingestion.vector_indexer import EmbeddingEncoder, VectorIndexer
from src.retrieval.graph_traversal import Neo4jGraphStore
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.reranker import CrossEncoderReranker


class MockStructuredLLM:
    """Mock LLM returning deterministic JSON triples for extractor testing."""

    def complete(self, prompt: str) -> str:
        _ = prompt
        return (
            '[{"source_entity": "ApolloGateway", "source_type": "Service", '
            '"relation": "AUTHENTICATES_VIA", "target_entity": "SentinelAuth", "target_type": "Service"}]'
        )


def test_semantic_chunker_deterministic_hashes() -> None:
    chunker = SemanticChunker(chunk_size=180, chunk_overlap=30, min_chunk_length=20)
    text = (
        "ApolloGateway authenticates via SentinelAuth using mTLS certificates.\n\n"
        "SentinelAuth stores data in RedisCluster for low-latency token revocation checks.\n\n"
        "HyperionPaymentEngine depends on CockroachDBLedger for distributed consensus."
    )
    chunks_first = chunker.chunk_document(doc_id="doc_test", text=text, metadata={"env": "prod"})
    chunks_second = chunker.chunk_document(doc_id="doc_test", text=text, metadata={"env": "prod"})

    assert len(chunks_first) >= 2
    assert [c.chunk_hash for c in chunks_first] == [c.chunk_hash for c in chunks_second]
    assert all(len(c.chunk_hash) == 12 for c in chunks_first)
    assert all(c.metadata["doc_id"] == "doc_test" for c in chunks_first)


def test_graph_extractor_patterns_and_llm_mock() -> None:
    chunk = DocumentChunk(
        chunk_id="doc_1#c0000_abcdef123456",
        chunk_hash="abcdef123456",
        doc_id="doc_1",
        text=(
            "ApolloGateway authenticates via SentinelAuth. "
            "SentinelAuth stores data in RedisCluster."
        ),
        chunk_index=0,
    )

    pattern_extractor = GraphExtractor(llm_client=None)
    triples = pattern_extractor.extract_from_chunk(chunk)
    assert len(triples) >= 2
    relations = {(t.source_entity, t.relation, t.target_entity) for t in triples}
    assert ("ApolloGateway", "AUTHENTICATES_VIA", "SentinelAuth") in relations
    assert ("SentinelAuth", "STORES_IN", "RedisCluster") in relations

    llm_extractor = GraphExtractor(llm_client=MockStructuredLLM())
    llm_triples = llm_extractor.extract_from_chunk(chunk)
    assert len(llm_triples) == 1
    assert llm_triples[0].source_entity == "ApolloGateway"
    assert llm_triples[0].relation == "AUTHENTICATES_VIA"
    assert llm_triples[0].target_entity == "SentinelAuth"


def test_hybrid_search_rrf_and_reranker() -> None:
    retrieval_cfg = RetrievalConfig(
        use_lightweight_embeddings=True,
        top_k_dense=5,
        top_k_sparse=5,
        top_k_rerank=3,
        rrf_k=60,
    )
    qdrant_cfg = QdrantConfig(in_memory=True)
    encoder = EmbeddingEncoder(use_lightweight=True, dimension=128)
    indexer = VectorIndexer(qdrant_config=qdrant_cfg, retrieval_config=retrieval_cfg, encoder=encoder)
    reranker = CrossEncoderReranker(use_lightweight=True)
    engine = HybridSearchEngine(vector_indexer=indexer, reranker=reranker, config=retrieval_cfg)

    chunker = SemanticChunker(chunk_size=300, chunk_overlap=30)
    chunks = (
        chunker.chunk_document(
            "doc_auth",
            "ApolloGateway authenticates incoming API traffic via SentinelAuth using Ed25519 JWT tokens.",
        )
        + chunker.chunk_document(
            "doc_pay",
            "HyperionPaymentEngine mitigates double-charge race conditions using CockroachDBLedger.",
        )
        + chunker.chunk_document(
            "doc_obs",
            "PrometheusScraper collects telemetry metrics from QdrantVectorCluster and routes to PagerDutyRouter.",
        )
    )
    added = engine.add_chunks(chunks)
    assert added == 3

    results = engine.search("How does HyperionPaymentEngine prevent double-charge race conditions?")
    assert len(results) >= 1
    assert results[0].doc_id == "doc_pay"
    assert results[0].rrf_score > 0.0
    assert results[0].rerank_score > 0.0


def test_neo4j_multi_hop_graph_traversal() -> None:
    chunker = SemanticChunker(chunk_size=400, chunk_overlap=40)
    chunks = chunker.chunk_document(
        "doc_chain",
        "ApolloGateway authenticates via SentinelAuth. SentinelAuth stores data in RedisCluster.",
    )
    extractor = GraphExtractor()
    triples = extractor.extract_from_chunks(chunks)

    graph_store = Neo4jGraphStore(config=Neo4jConfig(max_hops=2, max_triples=10))
    inserted = graph_store.upsert_triples(triples)
    assert inserted >= 2

    traversed = graph_store.traverse_multi_hop(["ApolloGateway"], max_hops=2)
    assert len(traversed) >= 2
    entities_reached = {t.target_entity for t in traversed} | {t.source_entity for t in traversed}
    assert "ApolloGateway" in entities_reached
    assert "SentinelAuth" in entities_reached
    assert "RedisCluster" in entities_reached
    assert any(t.hop_distance == 2 for t in traversed)
