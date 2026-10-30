"""Recursive semantic document chunker with deterministic SHA-256 chunk hashes."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents an atomic, indexable text chunk with provenance metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_hash: str = Field(..., description="Deterministic SHA-256 content hash (12-char prefix)")
    doc_id: str = Field(..., description="Parent document identifier")
    text: str = Field(..., description="Normalized chunk text content")
    chunk_index: int = Field(..., ge=0, description="Zero-based index within parent document")
    start_char: int = Field(default=0, ge=0)
    end_char: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticChunker:
    """Splits technical documents into overlapping semantic chunks preserving paragraph/sentence boundaries."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_length: int = 30,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self._separators: list[str] = ["\n\n", "\n", ". ", " ", ""]

    @staticmethod
    def compute_chunk_hash(doc_id: str, text: str) -> str:
        """Compute a deterministic 12-character SHA-256 hex digest for citation tracking."""
        payload = f"{doc_id}::{text.strip()}".encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def chunk_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Split raw document text into structured DocumentChunk objects."""
        cleaned = re.sub(r"\r\n", "\n", text).strip()
        if not cleaned:
            return []

        base_meta = dict(metadata) if metadata else {}
        segments = self._recursive_split(cleaned, self._separators)
        merged_texts = self._merge_segments(segments)

        chunks: list[DocumentChunk] = []
        cursor = 0
        for idx, chunk_text in enumerate(merged_texts):
            normalized = chunk_text.strip()
            if len(normalized) < self.min_chunk_length and len(merged_texts) > 1:
                continue

            pos = cleaned.find(normalized, max(0, cursor - self.chunk_overlap))
            start_char = pos if pos != -1 else cursor
            end_char = start_char + len(normalized)
            cursor = end_char

            chunk_hash = self.compute_chunk_hash(doc_id, normalized)
            chunk_id = f"{doc_id}#c{idx:04d}_{chunk_hash}"
            chunk_meta = {
                **base_meta,
                "doc_id": doc_id,
                "chunk_index": idx,
                "chunk_hash": chunk_hash,
            }
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_hash=chunk_hash,
                    doc_id=doc_id,
                    text=normalized,
                    chunk_index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=chunk_meta,
                )
            )
        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using hierarchy of separators until under chunk_size."""
        if len(text) <= self.chunk_size or not separators:
            return [text]

        sep = separators[0]
        next_seps = separators[1:]

        if sep == "":
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        parts = text.split(sep)
        result: list[str] = []
        for idx, part in enumerate(parts):
            piece = part + (sep if idx < len(parts) - 1 and sep != "\n\n" else "")
            if len(piece) <= self.chunk_size:
                if piece.strip():
                    result.append(piece)
            else:
                result.extend(self._recursive_split(piece, next_seps))
        return result

    def _merge_segments(self, segments: list[str]) -> list[str]:
        """Merge small consecutive segments up to chunk_size with sliding overlap."""
        if not segments:
            return []

        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for seg in segments:
            seg_len = len(seg)
            if current_len + seg_len + (1 if current_parts else 0) <= self.chunk_size:
                current_parts.append(seg)
                current_len += seg_len + (1 if len(current_parts) > 1 else 0)
            else:
                if current_parts:
                    chunks.append(" ".join(p.strip() for p in current_parts if p.strip()))
                # Retain overlap from the tail of current_parts
                overlap_parts: list[str] = []
                overlap_len = 0
                for prev in reversed(current_parts):
                    if overlap_len + len(prev) <= self.chunk_overlap:
                        overlap_parts.insert(0, prev)
                        overlap_len += len(prev) + 1
                    else:
                        break
                current_parts = overlap_parts + [seg]
                current_len = sum(len(p) for p in current_parts) + max(0, len(current_parts) - 1)

        if current_parts:
            chunks.append(" ".join(p.strip() for p in current_parts if p.strip()))

        return chunks
