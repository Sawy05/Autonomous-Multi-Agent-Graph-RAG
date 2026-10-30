"""Hybrid retrieval combining Qdrant dense vectors, Okapi BM25 lexical search, RRF, and reranking."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from src.config import RetrievalConfig
from src.ingestion.chunker import DocumentChunk
from src.ingestion.vector_indexer import VectorIndexer
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class RetrievedContext(BaseModel):
    """Unified retrieval result containing provenance, fusion metrics, and reranker confidence."""

    chunk_id: str
    chunk_hash: str
    doc_id: str
    text: str
    dense_score: float = Field(default=0.0)
    sparse_score: float = Field(default=0.0)
    rrf_score: float = Field(default=0.0)
    rerank_score: float = Field(default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OkapiBM25Index:
    """Lexical BM25 scorer using rank-bm25 when installed with built-in Okapi BM25 fallback."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[DocumentChunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._external_bm25: Any | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Rebuild the BM25 inverted index over the provided chunks."""
        self._chunks = list(chunks)
        self._tokenized_corpus = [self.tokenize(c.text) for c in self._chunks]
        corpus_size = len(self._tokenized_corpus)
        if corpus_size == 0:
            self._avgdl = 0.0
            return

        try:
            from rank_bm25 import BM25Okapi

            self._external_bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        except Exception:
            self._external_bm25 = None

        self._doc_lens = [len(doc) for doc in self._tokenized_corpus]
        self._avgdl = sum(self._doc_lens) / corpus_size
        df: dict[str, int] = {}
        for doc in self._tokenized_corpus:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1
        self._doc_freqs = df
        self._idf = {
            token: math.log(1.0 + (corpus_size - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        """Score all indexed chunks with Okapi BM25 and return top_k (chunk, score) pairs."""
        if not self._chunks:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        scores: list[float] = []
        if self._external_bm25 is not None:
            raw_scores = self._external_bm25.get_scores(q_tokens)
            scores = [max(0.0, float(s)) for s in raw_scores]
        else:
            for idx, doc_tokens in enumerate(self._tokenized_corpus):
                dl = self._doc_lens[idx]
                tf_counts = Counter(doc_tokens)
                doc_score = 0.0
                for term in q_tokens:
                    if term not in tf_counts:
                        continue
                    tf = tf_counts[term]
                    idf = self._idf.get(term, 0.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(1e-6, self._avgdl)))
                    doc_score += idf * ((tf * (self.k1 + 1.0)) / denom)
                scores.append(max(0.0, doc_score))

        paired = list(zip(self._chunks, scores, strict=True))
        paired.sort(key=lambda item: item[1], reverse=True)
        return paired[:top_k]


class HybridSearchEngine:
    """Executes Dense + Sparse BM25 retrieval fused via Reciprocal Rank Fusion and Cross-Encoder reranking."""

    def __init__(
        self,
        vector_indexer: VectorIndexer,
        reranker: CrossEncoderReranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.vector_indexer = vector_indexer
        self.config = config or RetrievalConfig()
        self.reranker = reranker or CrossEncoderReranker(
            model_name=self.config.reranker_model,
            use_lightweight=self.config.use_lightweight_embeddings,
        )
        self.bm25 = OkapiBM25Index()
        self._sync_sparse_index()

    def _sync_sparse_index(self) -> None:
        chunks = self.vector_indexer.get_all_chunks()
        if chunks:
            self.bm25.build_index(chunks)

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Index new document chunks into both dense vector store and sparse BM25 index."""
        added = self.vector_indexer.index_chunks(chunks)
        self._sync_sparse_index()
        return added

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: list[tuple[DocumentChunk, float]],
        sparse_results: list[tuple[DocumentChunk, float]],
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> list[RetrievedContext]:
        """Fuse ranked dense and sparse retrieval lists using weighted Reciprocal Rank Fusion (RRF)."""
        chunk_map: dict[str, DocumentChunk] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        rrf_scores: dict[str, float] = {}

        for rank, (chunk, score) in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_scores[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (dense_weight / (rrf_k + rank))

        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            sparse_scores[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (sparse_weight / (rrf_k + rank))

        fused: list[RetrievedContext] = []
        for cid, chunk in chunk_map.items():
            fused.append(
                RetrievedContext(
                    chunk_id=cid,
                    chunk_hash=chunk.chunk_hash,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    dense_score=round(dense_scores.get(cid, 0.0), 4),
                    sparse_score=round(sparse_scores.get(cid, 0.0), 4),
                    rrf_score=round(rrf_scores.get(cid, 0.0), 6),
                    rerank_score=round(rrf_scores.get(cid, 0.0), 4),
                    metadata=chunk.metadata,
                )
            )

        fused.sort(key=lambda item: item.rrf_score, reverse=True)
        return fused

    def search(
        self,
        query: str,
        top_k: int | None = None,
        enable_rerank: bool = True,
    ) -> list[RetrievedContext]:
        """Run full hybrid search pipeline: Dense + BM25 -> RRF -> Cross-Encoder Reranking."""
        final_k = top_k or self.config.top_k_rerank
        dense_hits = self.vector_indexer.search_dense(query, top_k=self.config.top_k_dense)
        sparse_hits = self.bm25.search(query, top_k=self.config.top_k_sparse)

        fused = self.reciprocal_rank_fusion(
            dense_results=dense_hits,
            sparse_results=sparse_hits,
            rrf_k=self.config.rrf_k,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
        )

        if not fused:
            return []

        if enable_rerank:
            return self.reranker.rerank(query=query, candidates=fused, top_k=final_k)

        return fused[:final_k]
