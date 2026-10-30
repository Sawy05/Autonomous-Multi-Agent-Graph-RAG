"""Cross-Encoder re-ranker (BAAI/bge-reranker-large) with calibrated lexical-semantic fallback."""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.retrieval.hybrid_search import RetrievedContext

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-ranks candidate contexts produced by Reciprocal Rank Fusion using a Cross-Encoder."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        use_lightweight: bool = False,
    ) -> None:
        self.model_name = model_name
        self.use_lightweight = use_lightweight
        self._cross_encoder: Any | None = None

        if not self.use_lightweight:
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(self.model_name)
                logger.info("Loaded CrossEncoder model '%s'", self.model_name)
            except Exception as exc:
                logger.info(
                    "Using calibrated cross-interaction scorer (CrossEncoder model offline: %s)",
                    exc,
                )
                self._cross_encoder = None

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedContext],
        top_k: int = 5,
    ) -> list[RetrievedContext]:
        """Score (query, passage) pairs and return top_k sorted by descending rerank_score."""
        if not candidates:
            return []

        if self._cross_encoder is not None and not self.use_lightweight:
            try:
                pairs = [(query, c.text) for c in candidates]
                raw_scores = self._cross_encoder.predict(pairs)
                scored_candidates: list[RetrievedContext] = []
                for cand, raw in zip(candidates, raw_scores, strict=True):
                    prob = self._sigmoid(float(raw))
                    updated = cand.model_copy(update={"rerank_score": round(prob, 4)})
                    scored_candidates.append(updated)
                scored_candidates.sort(key=lambda item: item.rerank_score, reverse=True)
                return scored_candidates[:top_k]
            except Exception as exc:
                logger.warning("CrossEncoder prediction failed; using fallback scorer: %s", exc)

        rescored: list[RetrievedContext] = []
        for cand in candidates:
            score = self._compute_interaction_score(
                query, cand.text, cand.rrf_score, cand.dense_score
            )
            rescored.append(cand.model_copy(update={"rerank_score": round(score, 4)}))

        rescored.sort(key=lambda item: item.rerank_score, reverse=True)
        return rescored[:top_k]

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        stop_words = {
            "the",
            "a",
            "an",
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "and",
            "or",
            "how",
            "what",
            "which",
            "where",
            "when",
            "does",
            "do",
        }
        # Expand CamelCase identifiers alongside full tokens for higher recall
        expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        raw_tokens = re.findall(r"[a-z0-9_]+", f"{text.lower()} {expanded.lower()}")
        return [t for t in raw_tokens if t not in stop_words and len(t) > 1]

    @classmethod
    def _compute_interaction_score(
        cls,
        query: str,
        passage: str,
        rrf_score: float,
        dense_score: float,
    ) -> float:
        """Calibrated cross-term coverage, bigram proximity, and dense fusion score in [0, 1]."""
        q_tokens = cls._tokenize(query)
        p_tokens = cls._tokenize(passage)
        if not q_tokens or not p_tokens:
            return min(1.0, max(0.0, dense_score))

        p_set = set(p_tokens)
        matched = sum(1 for t in q_tokens if t in p_set)
        term_coverage = matched / len(q_tokens)

        q_bigrams = {f"{q_tokens[i]}_{q_tokens[i+1]}" for i in range(len(q_tokens) - 1)}
        p_bigrams = {f"{p_tokens[i]}_{p_tokens[i+1]}" for i in range(len(p_tokens) - 1)}
        bigram_overlap = len(q_bigrams & p_bigrams) / len(q_bigrams) if q_bigrams else term_coverage

        # Normalize RRF (typical max around 2 / (60 + 1) = 0.0328)
        normalized_rrf = min(1.0, rrf_score * 30.0)
        combined = (
            0.50 * term_coverage
            + 0.20 * bigram_overlap
            + 0.15 * min(1.0, max(0.0, dense_score))
            + 0.15 * normalized_rrf
        )
        return min(1.0, max(0.0, combined))
