"""Dense vector embedding and Qdrant indexing engine with deterministic lightweight fallback."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import uuid
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.config import QdrantConfig, RetrievalConfig
from src.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class IndexedVectorRecord(BaseModel):
    """Represents a stored vector point with its associated chunk payload."""

    point_id: str
    chunk: DocumentChunk
    vector: list[float] = Field(default_factory=list)


class EmbeddingEncoder:
    """Wraps sentence-transformers (BGE / MiniLM) with deterministic n-gram hashing fallback."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        use_lightweight: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.use_lightweight = use_lightweight
        self._st_model: Any | None = None

        if not self.use_lightweight:
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(self.model_name)
                dim = self._st_model.get_sentence_embedding_dimension()
                if isinstance(dim, int) and dim > 0:
                    self.dimension = dim
                logger.info(
                    "Loaded SentenceTransformer model '%s' (dim=%d)",
                    self.model_name,
                    self.dimension,
                )
            except Exception as exc:
                logger.info(
                    "Using deterministic feature-hashing embeddings (SentenceTransformer unavailable: %s)",
                    exc,
                )
                self._st_model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of texts into L2-normalized dense vectors."""
        if not texts:
            return []

        if self._st_model is not None and not self.use_lightweight:
            embeddings = self._st_model.encode(texts, normalize_embeddings=True)
            arr = np.asarray(embeddings, dtype=np.float32)
            return [row.tolist() for row in arr]

        return [self._hash_embed(text) for text in texts]

    def encode_query(self, query: str) -> list[float]:
        """Encode a single search query into an L2-normalized vector."""
        results = self.encode([query])
        return results[0] if results else [0.0] * self.dimension

    def _hash_embed(self, text: str) -> list[float]:
        """Deterministic signed feature-hashing over word unigrams, bigrams, and character trigrams."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        if not tokens:
            return vec.tolist()

        features: list[tuple[str, float]] = []
        for tok in tokens:
            features.append((f"u:{tok}", 1.5))
            if len(tok) >= 4:
                for i in range(len(tok) - 2):
                    features.append((f"c:{tok[i:i+3]}", 0.5))
        for i in range(len(tokens) - 1):
            features.append((f"b:{tokens[i]}_{tokens[i+1]}", 1.2))

        for feat, weight in features:
            digest = hashlib.md5(feat.encode("utf-8"), usedforsecurity=False).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[idx] += sign * weight

        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return [float(x) for x in vec]


class VectorIndexer:
    """Manages Qdrant vector collection indexing and dense similarity search."""

    def __init__(
        self,
        qdrant_config: QdrantConfig | None = None,
        retrieval_config: RetrievalConfig | None = None,
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        self.qdrant_config = qdrant_config or QdrantConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.encoder = encoder or EmbeddingEncoder(
            model_name=self.retrieval_config.embedding_model,
            dimension=self.retrieval_config.embedding_dim,
            use_lightweight=self.retrieval_config.use_lightweight_embeddings,
        )
        self._memory_records: dict[str, IndexedVectorRecord] = {}
        self._qdrant_client: Any | None = None
        self._init_qdrant_client()

    def _init_qdrant_client(self) -> None:
        """Initialize Qdrant client if available and reachable; otherwise use in-memory store."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels

            if self.qdrant_config.in_memory:
                client = QdrantClient(location=":memory:")
            else:
                client = QdrantClient(
                    host=self.qdrant_config.host,
                    port=self.qdrant_config.port,
                    api_key=self.qdrant_config.api_key or None,
                    timeout=2.0,
                )
            collections = client.get_collections().collections
            existing = {c.name for c in collections}
            if self.qdrant_config.collection_name not in existing:
                client.create_collection(
                    collection_name=self.qdrant_config.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.encoder.dimension,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
            self._qdrant_client = client
            logger.info("Connected to Qdrant collection '%s'", self.qdrant_config.collection_name)
        except Exception as exc:
            logger.info("Qdrant server fallback to high-speed in-memory vector store (%s)", exc)
            self._qdrant_client = None

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Compute dense embeddings and upsert chunks into Qdrant and local index."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = self.encoder.encode(texts)

        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            self._memory_records[chunk.chunk_id] = IndexedVectorRecord(
                point_id=point_id,
                chunk=chunk,
                vector=vector,
            )

        if self._qdrant_client is not None:
            try:
                from qdrant_client.http import models as qmodels

                points = [
                    qmodels.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                        vector=vec,
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "chunk_hash": chunk.chunk_hash,
                            "doc_id": chunk.doc_id,
                            "text": chunk.text,
                            "chunk_index": chunk.chunk_index,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "metadata": chunk.metadata,
                        },
                    )
                    for chunk, vec in zip(chunks, vectors, strict=True)
                ]
                self._qdrant_client.upsert(
                    collection_name=self.qdrant_config.collection_name,
                    points=points,
                )
            except Exception as exc:
                logger.warning("Qdrant upsert failed; records preserved in memory index: %s", exc)

        return len(chunks)

    def search_dense(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        """Perform dense cosine similarity search returning (DocumentChunk, score) pairs."""
        query_vec = self.encoder.encode_query(query)

        if self._qdrant_client is not None:
            try:
                hits = self._qdrant_client.search(
                    collection_name=self.qdrant_config.collection_name,
                    query_vector=query_vec,
                    limit=top_k,
                )
                results: list[tuple[DocumentChunk, float]] = []
                for hit in hits:
                    payload = hit.payload or {}
                    chunk = DocumentChunk(
                        chunk_id=str(payload.get("chunk_id", "")),
                        chunk_hash=str(payload.get("chunk_hash", "")),
                        doc_id=str(payload.get("doc_id", "")),
                        text=str(payload.get("text", "")),
                        chunk_index=int(payload.get("chunk_index", 0)),
                        start_char=int(payload.get("start_char", 0)),
                        end_char=int(payload.get("end_char", 0)),
                        metadata=dict(payload.get("metadata", {})),
                    )
                    results.append((chunk, float(hit.score)))
                if results:
                    return results
            except Exception as exc:
                logger.warning(
                    "Qdrant search failed; falling back to in-memory cosine search: %s", exc
                )

        if not self._memory_records:
            return []

        q_arr = np.asarray(query_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_arr))
        scored: list[tuple[DocumentChunk, float]] = []
        for record in self._memory_records.values():
            v_arr = np.asarray(record.vector, dtype=np.float32)
            v_norm = float(np.linalg.norm(v_arr))
            denom = q_norm * v_norm
            sim = float(np.dot(q_arr, v_arr) / denom) if denom > 1e-8 else 0.0
            # Map cosine similarity [-1, 1] to non-negative score
            score = max(0.0, sim)
            if not math.isnan(score):
                scored.append((record.chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Return all indexed chunks currently held in the store."""
        return [rec.chunk for rec in self._memory_records.values()]

    @property
    def count(self) -> int:
        """Total number of indexed chunks."""
        return len(self._memory_records)
