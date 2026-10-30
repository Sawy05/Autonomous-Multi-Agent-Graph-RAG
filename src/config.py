"""Centralized configuration management with YAML and environment variable overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

LLMProviderType = Literal["openai", "anthropic", "ollama", "vllm", "mock"]


class AppConfig(BaseModel):
    """Application server metadata and runtime settings."""

    name: str = Field(default="autonomous-graph-rag-agent")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)


class LLMConfig(BaseModel):
    """Provider-agnostic LLM configuration."""

    provider: LLMProviderType = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=64)
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    timeout_seconds: float = Field(default=60.0, gt=0.0)


class IngestionConfig(BaseModel):
    """Document chunking and entity extraction settings."""

    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=64, ge=0)
    min_chunk_length: int = Field(default=30, ge=1)
    extract_graph_triples: bool = Field(default=True)


class RetrievalConfig(BaseModel):
    """Hybrid dense + sparse + cross-encoder reranking parameters."""

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dim: int = Field(default=384, ge=8)
    reranker_model: str = Field(default="BAAI/bge-reranker-large")
    use_lightweight_embeddings: bool = Field(default=False)
    top_k_dense: int = Field(default=10, ge=1)
    top_k_sparse: int = Field(default=10, ge=1)
    top_k_rerank: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.4, ge=0.0, le=1.0)


class QdrantConfig(BaseModel):
    """Qdrant vector database connection settings."""

    host: str = Field(default="localhost")
    port: int = Field(default=6333, ge=1, le=65535)
    grpc_port: int = Field(default=6334, ge=1, le=65535)
    api_key: str = Field(default="")
    collection_name: str = Field(default="enterprise_knowledge_chunks")
    in_memory: bool = Field(default=False)


class Neo4jConfig(BaseModel):
    """Neo4j graph database connection and traversal settings."""

    uri: str = Field(default="bolt://localhost:7687")
    username: str = Field(default="neo4j")
    password: str = Field(default="neo4j_enterprise_pass")
    database: str = Field(default="neo4j")
    max_hops: int = Field(default=2, ge=1, le=5)
    max_triples: int = Field(default=25, ge=1)


class AgentConfig(BaseModel):
    """LangGraph self-reflection and retry loop configuration."""

    max_retries: int = Field(default=2, ge=0, le=5)
    relevance_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    enable_self_reflection: bool = Field(default=True)
    strict_citations: bool = Field(default=True)


class Settings(BaseModel):
    """Root application settings aggregating all subsystem configurations."""

    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    @classmethod
    def from_yaml_and_env(cls, yaml_path: Path | None = None) -> Settings:
        """Load settings from YAML file and override with environment variables."""
        raw_data: dict[str, Any] = {}
        resolved_path = (
            yaml_path or Path(__file__).resolve().parents[1] / "configs" / "settings.yaml"
        )
        if resolved_path.exists():
            with resolved_path.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
                if isinstance(loaded, dict):
                    raw_data = loaded

        settings = cls.model_validate(raw_data)
        settings._apply_env_overrides()
        return settings

    def _apply_env_overrides(self) -> None:
        """Override YAML defaults with explicit environment variables when present."""
        if val := os.getenv("APP_NAME"):
            self.app.name = val
        if val := os.getenv("APP_ENV"):
            self.app.environment = val
        if val := os.getenv("LOG_LEVEL"):
            self.app.log_level = val
        if val := os.getenv("API_HOST"):
            self.app.host = val
        if val := os.getenv("API_PORT"):
            self.app.port = int(val)

        if (val := os.getenv("LLM_PROVIDER")) and val in {
            "openai",
            "anthropic",
            "ollama",
            "vllm",
            "mock",
        }:
            self.llm.provider = val  # type: ignore[assignment]
        if val := os.getenv("LLM_MODEL"):
            self.llm.model = val
        if val := os.getenv("LLM_TEMPERATURE"):
            self.llm.temperature = float(val)
        if val := os.getenv("LLM_MAX_TOKENS"):
            self.llm.max_tokens = int(val)
        if val := os.getenv("LLM_BASE_URL"):
            self.llm.base_url = val
        if val := os.getenv("OPENAI_API_KEY"):
            self.llm.api_key = val
        if val := os.getenv("ANTHROPIC_API_KEY"):
            self.llm.anthropic_api_key = val

        if val := os.getenv("EMBEDDING_MODEL_NAME"):
            self.retrieval.embedding_model = val
        if val := os.getenv("EMBEDDING_DIMENSION"):
            self.retrieval.embedding_dim = int(val)
        if val := os.getenv("RERANKER_MODEL_NAME"):
            self.retrieval.reranker_model = val
        if val := os.getenv("USE_LIGHTWEIGHT_EMBEDDINGS"):
            self.retrieval.use_lightweight_embeddings = val.lower() in {"1", "true", "yes"}
        if val := os.getenv("TOP_K_DENSE"):
            self.retrieval.top_k_dense = int(val)
        if val := os.getenv("TOP_K_SPARSE"):
            self.retrieval.top_k_sparse = int(val)
        if val := os.getenv("TOP_K_RERANK"):
            self.retrieval.top_k_rerank = int(val)
        if val := os.getenv("RRF_K"):
            self.retrieval.rrf_k = int(val)

        if val := os.getenv("QDRANT_HOST"):
            self.qdrant.host = val
        if val := os.getenv("QDRANT_PORT"):
            self.qdrant.port = int(val)
        if val := os.getenv("QDRANT_GRPC_PORT"):
            self.qdrant.grpc_port = int(val)
        if val := os.getenv("QDRANT_API_KEY"):
            self.qdrant.api_key = val
        if val := os.getenv("QDRANT_COLLECTION_NAME"):
            self.qdrant.collection_name = val
        if val := os.getenv("QDRANT_IN_MEMORY"):
            self.qdrant.in_memory = val.lower() in {"1", "true", "yes"}

        if val := os.getenv("NEO4J_URI"):
            self.neo4j.uri = val
        if val := os.getenv("NEO4J_USERNAME"):
            self.neo4j.username = val
        if val := os.getenv("NEO4J_PASSWORD"):
            self.neo4j.password = val
        if val := os.getenv("NEO4J_DATABASE"):
            self.neo4j.database = val
        if val := os.getenv("GRAPH_MAX_HOPS"):
            self.neo4j.max_hops = int(val)

        if val := os.getenv("MAX_RETRIES"):
            self.agent.max_retries = int(val)
        if val := os.getenv("RELEVANCE_THRESHOLD"):
            self.agent.relevance_threshold = float(val)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings.from_yaml_and_env()
