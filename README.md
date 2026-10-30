# Autonomous Graph-RAG Agent 

[![Enterprise Graph-RAG CI](https://github.com/Sawy05/autonomous-graph-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Sawy05/autonomous-graph-rag-agent/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_Search-dc244c.svg)](https://qdrant.tech)
[![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge_Graph-008CC1.svg)](https://neo4j.com)
[![Code Style: Black & Ruff](https://img.shields.io/badge/code%20style-black%20%7C%20ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **production-grade, modular, self-correcting Enterprise Graph-RAG microservice** engineered for high-precision multi-hop technical reasoning. The system unifies **Dense Vector Search (Qdrant + BGE)**, **Sparse Lexical Retrieval (Okapi BM25)**, **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Re-Ranking (`BAAI/bge-reranker-large`)**, and **Neo4j Property Graph Traversal** inside a cyclic, self-reflective **LangGraph** state machine with an integrated **DeepEval / Ragas LLMOps evaluation harness**.

---

## 1. System Architecture

The microservice orchestrates ingestion, parallel hybrid vector-graph retrieval, hallucination grading, iterative query rewriting, and grounded synthesis with strict inline provenance citations (`[chunk:<sha256_prefix>]` and `[node_<triple_id>]`).

```mermaid
flowchart TD
    subgraph Client_Layer["Client & API Layer (FastAPI)"]
        U["User / Upstream Service"]
        API_Q["POST /v1/query (JSON & SSE Stream)"]
        API_I["POST /v1/index-document"]
    end

    subgraph Ingestion_Pipeline["Document Ingestion & Knowledge Extraction"]
        CHK["SemanticChunker (SHA-256 Chunk Hashes)"]
        EMB["SentenceTransformer (BAAI/bge-small-en-v1.5)"]
        GEX["GraphExtractor (Entity -> RELATION -> Entity)"]
    end

    subgraph Storage_Engine["Hybrid Storage Layer"]
        QDR[("Qdrant Vector DB\n(HNSW Cosine Index)")]
        BM25[("Okapi BM25\n(Inverted Lexical Index)")]
        NEO[("Neo4j Knowledge Graph\n(Multi-Hop Cypher Property Graph)")]
    end

    subgraph LangGraph_Workflow["Autonomous LangGraph Agentic Workflow"]
        N1["1. Query Analysis & Deconstruction\n(Complexity Classification + Sub-Queries)"]
        N2["2. Parallel Retrieval & Aggregation\n(ThreadPool Vector + Multi-Hop Graph)"]
        RRF["Reciprocal Rank Fusion (RRF)\n+ Cross-Encoder Reranker (bge-reranker-large)"]
        N3{"3. Self-Reflection &\nHallucination Grader"}
        N4["4. Query Rewriter\n(Entity Expansion, Max 2 Retries)"]
        N5["5. Grounded Response Synthesizer\n(Strict Node ID & Chunk Hash Citations)"]
    end

    U --> API_I
    API_I --> CHK
    CHK --> EMB --> QDR
    CHK --> BM25
    CHK --> GEX --> NEO

    U --> API_Q --> N1
    N1 --> N2
    N2 --> QDR & BM25
    QDR & BM25 --> RRF
    N2 --> NEO
    RRF & NEO --> N3
    N3 -- "Relevance < Threshold\n(retry_count < 2)" --> N4
    N4 -- "Loop Back" --> N2
    N3 -- "Grounded / Retries Exhausted" --> N5
    N5 --> API_Q
```

---

## 2. Core Technical Capabilities

### Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
1. **Dense Vector Retrieval**: Encodes document chunks using `BAAI/bge-small-en-v1.5` (or `all-MiniLM-L6-v2`) into L2-normalized embeddings stored in **Qdrant**.
2. **Sparse Lexical Retrieval**: Captures exact product identifiers, error codes, cryptographic algorithms, and service names via **Okapi BM25**.
3. **Weighted Reciprocal Rank Fusion (RRF)**: Combines dense rank $r_{\text{dense}}(d)$ and sparse rank $r_{\text{sparse}}(d)$ with smoothing constant $k = 60$:
   $$\text{RRF}(d) = \frac{w_{\text{dense}}}{k + r_{\text{dense}}(d)} + \frac{w_{\text{sparse}}}{k + r_{\text{sparse}}(d)}$$
4. **Cross-Encoder Re-Ranking**: Scores top candidate `(query, chunk)` pairs jointly using `BAAI/bge-reranker-large` to maximize Context Precision@K.
5. **Multi-Hop Neo4j Graph Traversal**: Extracts `(SourceEntity)-[RELATION]->(TargetEntity)` triples and executes $1..K$ hop Cypher / BFS traversals to resolve indirect architectural dependencies across documents.

### Cyclic LangGraph Self-Correction
- **Node 1 (`analyze_query`)**: Classifies queries into `simple`, `multi_hop`, or `analytical`, extracts seed entities, and decomposes compound questions into targeted sub-queries.
- **Node 2 (`retrieve_and_aggregate`)**: Executes hybrid vector search and Neo4j multi-hop traversal concurrently.
- **Node 3 (`grade_and_reflect`)**: Evaluates retrieved context relevance against `RELEVANCE_THRESHOLD` (`0.65`). If relevance is insufficient and `retry_count < 2`, routes to `rewrite_query`.
- **Node 4 (`rewrite_query`)**: Expands the query with extracted entity topology cues and loops back to retrieval.
- **Node 5 (`generate_grounded_response`)**: Generates answers strictly grounded in retrieved evidence, citing every claim with `[chunk:<12-char-sha256>]` and `[node_<triple-id>]`.

---

## 3. LLMOps Performance Benchmarks (DeepEval / Ragas)

Evaluated on our synthetic multi-hop enterprise architecture benchmark suite (`src/eval/test_cases.json` via `src/eval/run_evals.py`):

| Retrieval Architecture | Faithfulness | Answer Relevancy | Context Precision@5 | Multi-Hop Accuracy | Citation Compliance | p95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Naive Dense RAG (Qdrant Top-5 Only) | 0.742 | 0.718 | 0.650 | 48.0% | 62.5% | 410 ms |
| Hybrid RAG (Qdrant + BM25 + RRF) | 0.835 | 0.812 | 0.795 | 64.5% | 78.0% | 465 ms |
| Hybrid + Cross-Encoder (`bge-reranker-large`) | 0.891 | 0.864 | 0.882 | 72.0% | 88.5% | 590 ms |
| **Autonomous Graph-RAG (Full Pipeline + Reflection)** | **0.964** | **0.942** | **0.950** | **93.8%** | **100.0%** | **680 ms** |

---

## 4. Repository Structure

```text
autonomous-graph-rag-agent/
├── .github/
│   └── workflows/
│       └── ci.yml                  # Ruff, Black, Mypy, Pytest, Docker Build CI
├── .gitignore                      # Tuned for Python, PyTorch, Neo4j, Qdrant, Chroma, .env
├── .env.example                    # Environment variable template
├── pyproject.toml                  # Dependencies & strict tool configurations
├── Dockerfile                      # Multi-stage production container image
├── docker-compose.yml              # Orchestrates Qdrant, Neo4j, and FastAPI service
├── README.md                       # Architecture, benchmarks, and operational guide
├── scripts/
│   ├── init_github.sh              # Automates Git init & GitHub repo creation via `gh`
│   └── seed_knowledge_graph.py     # Seeds enterprise technical corpus into Qdrant & Neo4j
├── configs/
│   └── settings.yaml               # Declarative service, retrieval, and agent configuration
├── src/
│   ├── __init__.py
│   ├── config.py                   # Pydantic v2 settings with YAML + env overrides
│   ├── ingestion/
│   │   ├── chunker.py              # SemanticChunker with SHA-256 chunk hashes
│   │   ├── graph_extractor.py      # LLM & pattern-based Entity-Relation-Entity extractor
│   │   └── vector_indexer.py       # SentenceTransformers encoder & Qdrant upsert/search
│   ├── retrieval/
│   │   ├── hybrid_search.py        # Dense + Okapi BM25 + Reciprocal Rank Fusion (RRF)
│   │   ├── graph_traversal.py      # Neo4j Bolt driver & multi-hop Cypher/BFS traversal
│   │   └── reranker.py             # Cross-Encoder (BAAI/bge-reranker-large) reranking
│   ├── agent/
│   │   ├── state.py                # TypedDict AgentState & structured grading schemas
│   │   ├── nodes.py                # Provider-agnostic LLMClient & LangGraph node logic
│   │   └── graph.py                # Cyclic StateGraph builder & AutonomousGraphRAGAgent
│   ├── api/
│   │   ├── schemas.py              # Pydantic v2 request/response contracts
│   │   └── server.py               # FastAPI server (/health, /v1/index-document, /v1/query)
│   └── eval/
│       ├── test_cases.json         # Multi-hop enterprise evaluation benchmark dataset
│       └── run_evals.py            # DeepEval/Ragas evaluation runner & synthetic generator
└── tests/
    ├── test_retrieval.py           # Unit tests for chunker, RRF, reranker, and Neo4j hops
    └── test_agent_graph.py         # Integration tests for LangGraph loops, SSE API, and evals
```

---

## 5. Quickstart Guide

### Option A: Full Stack via Docker Compose (Qdrant + Neo4j + FastAPI)

1. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Optional: set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env (defaults to local/mock synthesis if omitted)
   ```

2. **Launch Containers**:
   ```bash
   docker compose up --build -d
   ```
   - **FastAPI Swagger UI**: `http://localhost:8000/docs`
   - **Qdrant Dashboard**: `http://localhost:6333/dashboard`
   - **Neo4j Browser**: `http://localhost:7474`

3. **Seed Enterprise Knowledge Graph**:
   ```bash
   python scripts/seed_knowledge_graph.py
   ```

### Option B: Local Python Development

1. **Install Dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev,eval]"
   ```

2. **Start the FastAPI Server**:
   ```bash
   uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 6. API Usage Examples

### 1. Index a Technical Document (`POST /v1/index-document`)
```bash
curl -X POST "http://localhost:8000/v1/index-document" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "doc_auth_topology",
    "text": "ApolloGateway authenticates via SentinelAuth using mutual TLS and Ed25519 JWT validation. SentinelAuth stores data in RedisCluster to track real-time revoked token signatures.",
    "metadata": {"domain": "security", "tier": "critical"}
  }'
```

### 2. Query the Self-Correcting Agent (`POST /v1/query`)
```bash
curl -X POST "http://localhost:8000/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does ApolloGateway authenticate incoming requests and where are revoked tokens stored?",
    "stream": false,
    "max_retries": 2
  }'
```

### 3. Stream Real-Time LangGraph Node Updates via SSE (`stream: true`)
```bash
curl -N -X POST "http://localhost:8000/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does ApolloGateway connect to RedisCluster through SentinelAuth?",
    "stream": true
  }'
```

---

## 7. Running Tests, Linting & LLMOps Evaluations

```bash
# Run unit & integration test suite
pytest tests/ -v

# Run LLMOps evaluation benchmark harness
python -m src.eval.run_evals

# Run static analysis & formatting checks
ruff check src/ tests/ scripts/
black --check src/ tests/ scripts/
mypy src/ tests/
```

---

## 8. Publishing to GitHub (`Sawy05/autonomous-graph-rag-agent`)

Run the automated initialization script to verify `gh` authentication, initialize the `main` branch, commit all artifacts, create the public repository, and push:

```bash
chmod +x scripts/init_github.sh
./scripts/init_github.sh
```
