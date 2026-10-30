"""Entity-Relation triple extractor with LLM structured extraction and heuristic NLP fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from src.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class KnowledgeTriple(BaseModel):
    """Directed knowledge graph triple (Entity -> RELATION -> Entity) with provenance."""

    triple_id: str = Field(..., description="Deterministic node/edge identifier")
    source_entity: str = Field(..., description="Normalized subject entity name")
    source_type: str = Field(default="Component", description="Subject entity category")
    relation: str = Field(..., description="UPPER_SNAKE_CASE relationship predicate")
    target_entity: str = Field(..., description="Normalized object entity name")
    target_type: str = Field(default="Component", description="Object entity category")
    chunk_id: str = Field(..., description="Source chunk ID")
    chunk_hash: str = Field(..., description="Source chunk SHA-256 hash")
    doc_id: str = Field(..., description="Source document ID")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    evidence: str = Field(default="", description="Supporting sentence snippet")


class GraphExtractor:
    """Extracts structured entity-relation triples from chunks for Neo4j ingestion."""

    RELATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9_\-\s]{1,40}?)\s+(?:depends on|requires|relies on)\s+([A-Z][A-Za-z0-9_\-\s]{1,40})\b"
            ),
            "DEPENDS_ON",
        ),
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9_\-\s]{1,40}?)\s+(?:connects to|communicates with|routes to|streams to|publishes to)\s+([A-Z][A-Za-z0-9_\-\s]{1,40})\b"
            ),
            "CONNECTS_TO",
        ),
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9_\-\s]{1,40}?)\s+(?:uses|leverages|utilizes|employs)\s+([A-Z][A-Za-z0-9_\-\s]{1,40})\b"
            ),
            "USES",
        ),
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9_\-\s]{1,40}?)\s+(?:stores data in|persists in|indexes into|writes to)\s+([A-Z][A-Za-z0-9_\-\s]{1,40})\b"
            ),
            "STORES_IN",
        ),
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9_\-\s]{1,40}?)\s+(?:authenticates via|validates with|is secured by)\s+([A-Z][A-Za-z0-9_\-\s]{1,40})\b"
            ),
            "AUTHENTICATES_VIA",
        ),
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9_\-\s]{1,40}?)\s+(?:orchestrates|manages|coordinates|governs)\s+([A-Z][A-Za-z0-9_\-\s]{1,40})\b"
            ),
            "ORCHESTRATES",
        ),
        (
            re.compile(
                r"\b([A-Z][A-Za-z0-9_\-\s]{1,40}?)\s+(?:mitigates|prevents|resolves)\s+([A-Z][A-Za-z0-9_\-\s]{1,40})\b"
            ),
            "MITIGATES",
        ),
    ]

    def __init__(self, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client

    @staticmethod
    def _make_triple_id(source: str, relation: str, target: str, chunk_hash: str) -> str:
        raw = f"{source.lower()}::{relation.upper()}::{target.lower()}::{chunk_hash}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
        return f"node_{digest}"

    @staticmethod
    def _clean_entity(entity: str) -> str:
        cleaned = re.sub(r"\s+", " ", entity).strip(" .,;:()[]{}\"'")
        stop_prefixes = ("The ", "A ", "An ", "In ", "When ", "While ", "Because ", "If ")
        for prefix in stop_prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned

    def extract_from_chunk(self, chunk: DocumentChunk) -> list[KnowledgeTriple]:
        """Extract KnowledgeTriple items from a single DocumentChunk."""
        if self.llm_client is not None:
            try:
                llm_triples = self._extract_via_llm(chunk)
                if llm_triples:
                    return llm_triples
            except Exception as exc:
                logger.warning("LLM triple extraction failed; falling back to pattern extractor: %s", exc)

        return self._extract_via_patterns(chunk)

    def extract_from_chunks(self, chunks: list[DocumentChunk]) -> list[KnowledgeTriple]:
        """Batch extract deduplicated KnowledgeTriple objects across multiple chunks."""
        all_triples: list[KnowledgeTriple] = []
        seen_keys: set[tuple[str, str, str, str]] = set()
        for chunk in chunks:
            for triple in self.extract_from_chunk(chunk):
                key = (
                    triple.source_entity.lower(),
                    triple.relation,
                    triple.target_entity.lower(),
                    triple.chunk_hash,
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_triples.append(triple)
        return all_triples

    def _extract_via_llm(self, chunk: DocumentChunk) -> list[KnowledgeTriple]:
        """Use configured LLM client to extract JSON array of triples."""
        prompt = (
            "Extract all factual technical entity-relation triples from the text below.\n"
            "Return ONLY a valid JSON list of objects with keys: "
            "'source_entity', 'source_type', 'relation', 'target_entity', 'target_type'.\n\n"
            f"Text:\n{chunk.text}"
        )
        raw_response = self.llm_client.complete(prompt)
        parsed = json.loads(raw_response)
        if not isinstance(parsed, list):
            return []

        triples: list[KnowledgeTriple] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            src = self._clean_entity(str(item.get("source_entity", "")))
            rel = re.sub(r"[^A-Z0-9_]", "_", str(item.get("relation", "RELATED_TO")).upper()).strip("_")
            tgt = self._clean_entity(str(item.get("target_entity", "")))
            if not src or not tgt or src.lower() == tgt.lower():
                continue
            triple_id = self._make_triple_id(src, rel or "RELATED_TO", tgt, chunk.chunk_hash)
            triples.append(
                KnowledgeTriple(
                    triple_id=triple_id,
                    source_entity=src,
                    source_type=str(item.get("source_type", "Entity")),
                    relation=rel or "RELATED_TO",
                    target_entity=tgt,
                    target_type=str(item.get("target_type", "Entity")),
                    chunk_id=chunk.chunk_id,
                    chunk_hash=chunk.chunk_hash,
                    doc_id=chunk.doc_id,
                    confidence=0.95,
                    evidence=chunk.text[:240],
                )
            )
        return triples

    def _extract_via_patterns(self, chunk: DocumentChunk) -> list[KnowledgeTriple]:
        """Deterministic linguistic & capitalized-entity extraction for high-speed or offline runs."""
        triples: list[KnowledgeTriple] = []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", chunk.text) if s.strip()]

        for sentence in sentences:
            matched_in_sentence = False
            for pattern, relation in self.RELATION_PATTERNS:
                for match in pattern.finditer(sentence):
                    src = self._clean_entity(match.group(1))
                    tgt = self._clean_entity(match.group(2))
                    if len(src) >= 2 and len(tgt) >= 2 and src.lower() != tgt.lower():
                        triple_id = self._make_triple_id(src, relation, tgt, chunk.chunk_hash)
                        triples.append(
                            KnowledgeTriple(
                                triple_id=triple_id,
                                source_entity=src,
                                relation=relation,
                                target_entity=tgt,
                                chunk_id=chunk.chunk_id,
                                chunk_hash=chunk.chunk_hash,
                                doc_id=chunk.doc_id,
                                confidence=0.88,
                                evidence=sentence[:240],
                            )
                        )
                        matched_in_sentence = True

            if not matched_in_sentence:
                # Extract co-occurring capitalized technical entities within the same sentence
                entities = self._extract_named_entities(sentence)
                if len(entities) >= 2:
                    rel = self._infer_predicate(sentence)
                    for i in range(len(entities) - 1):
                        src = entities[i]
                        tgt = entities[i + 1]
                        if src.lower() != tgt.lower():
                            triple_id = self._make_triple_id(src, rel, tgt, chunk.chunk_hash)
                            triples.append(
                                KnowledgeTriple(
                                    triple_id=triple_id,
                                    source_entity=src,
                                    relation=rel,
                                    target_entity=tgt,
                                    chunk_id=chunk.chunk_id,
                                    chunk_hash=chunk.chunk_hash,
                                    doc_id=chunk.doc_id,
                                    confidence=0.80,
                                    evidence=sentence[:240],
                                )
                            )
        return triples

    @classmethod
    def _extract_named_entities(cls, sentence: str) -> list[str]:
        """Identify multi-word capitalized technical phrases, services, or protocols."""
        candidates = re.findall(
            r"\b(?:[A-Z][A-Za-z0-9_\-]+(?:\s+[A-Z][A-Za-z0-9_\-]+){0,2})\b",
            sentence,
        )
        ignore = {
            "The",
            "This",
            "That",
            "These",
            "Those",
            "When",
            "Where",
            "While",
            "However",
            "Therefore",
            "Furthermore",
            "In",
            "On",
            "At",
            "For",
            "With",
            "By",
            "From",
            "To",
            "If",
            "Each",
            "All",
            "Every",
        }
        cleaned: list[str] = []
        for cand in candidates:
            ent = cls._clean_entity(cand)
            if ent and ent not in ignore and len(ent) >= 2 and ent not in cleaned:
                cleaned.append(ent)
        return cleaned

    @staticmethod
    def _infer_predicate(sentence: str) -> str:
        lower = sentence.lower()
        if any(w in lower for w in ("depend", "require", "rely", "prerequisite")):
            return "DEPENDS_ON"
        if any(w in lower for w in ("store", "index", "persist", "database", "cache")):
            return "STORES_IN"
        if any(w in lower for w in ("auth", "token", "jwt", "mtls", "certificate", "verify")):
            return "AUTHENTICATES_VIA"
        if any(w in lower for w in ("route", "stream", "publish", "connect", "replicate", "call")):
            return "CONNECTS_TO"
        if any(w in lower for w in ("orchestrate", "manage", "schedule", "coordinate")):
            return "ORCHESTRATES"
        if any(w in lower for w in ("use", "utilize", "leverage", "employ")):
            return "USES"
        return "ASSOCIATED_WITH"
