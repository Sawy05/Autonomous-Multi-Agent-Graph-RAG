"""Seed script to populate Qdrant, BM25, and Neo4j with enterprise technical documentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.graph import AutonomousGraphRAGAgent  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.eval.run_evals import SEED_ENTERPRISE_DOCS  # noqa: E402


def seed_all_documents() -> dict[str, int]:
    """Ingest seed enterprise architecture documents into vector and graph stores."""
    settings = get_settings()
    agent = AutonomousGraphRAGAgent(settings=settings)

    total_chunks = 0
    total_triples = 0

    for doc in SEED_ENTERPRISE_DOCS:
        result = agent.ingest_document(
            doc_id=doc["doc_id"],
            text=doc["text"],
            metadata={"title": doc["title"], "source": "enterprise_seed"},
        )
        total_chunks += int(result["chunks_indexed"])
        total_triples += int(result["triples_extracted"])

    summary = {
        "documents_processed": len(SEED_ENTERPRISE_DOCS),
        "total_chunks_indexed": total_chunks,
        "total_triples_extracted": total_triples,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    seed_all_documents()
