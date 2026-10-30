"""Neo4j Knowledge Graph storage and multi-hop Cypher / BFS traversal engine."""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from src.config import Neo4jConfig
from src.ingestion.graph_extractor import KnowledgeTriple

logger = logging.getLogger(__name__)


class GraphSearchResult(BaseModel):
    """Represents a traversed knowledge graph triple or multi-hop reasoning path."""

    node_id: str = Field(..., description="Unique triple/edge identifier for citation")
    source_entity: str
    relation: str
    target_entity: str
    hop_distance: int = Field(default=1, ge=1)
    chunk_hash: str
    doc_id: str
    evidence: str = Field(default="")
    path_summary: str = Field(default="")


class Neo4jGraphStore:
    """Manages entity-relation triples in Neo4j and executes multi-hop subgraph traversal."""

    def __init__(self, config: Neo4jConfig | None = None) -> None:
        self.config = config or Neo4jConfig()
        self._driver: Any | None = None
        self._triples: dict[str, KnowledgeTriple] = {}
        self._adjacency: dict[str, list[KnowledgeTriple]] = {}
        self._init_driver()

    def _init_driver(self) -> None:
        """Connect to Neo4j Bolt endpoint if reachable; maintain local graph replica."""
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                connection_timeout=2.0,
            )
            driver.verify_connectivity()
            self._driver = driver
            logger.info("Connected to Neo4j at %s", self.config.uri)
        except Exception as exc:
            logger.info("Neo4j server offline; using in-memory property graph engine (%s)", exc)
            self._driver = None

    def close(self) -> None:
        """Close the underlying Neo4j driver connection if active."""
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None

    def upsert_triples(self, triples: list[KnowledgeTriple]) -> int:
        """Persist entity-relation triples into Neo4j and local adjacency graph."""
        if not triples:
            return 0

        for triple in triples:
            self._triples[triple.triple_id] = triple
            src_key = triple.source_entity.lower()
            tgt_key = triple.target_entity.lower()
            self._adjacency.setdefault(src_key, []).append(triple)
            self._adjacency.setdefault(tgt_key, []).append(triple)

        if self._driver is not None:
            cypher = (
                "UNWIND $triples AS t "
                "MERGE (s:Entity {name_lower: toLower(t.source_entity)}) "
                "ON CREATE SET s.name = t.source_entity, s.type = t.source_type "
                "MERGE (o:Entity {name_lower: toLower(t.target_entity)}) "
                "ON CREATE SET o.name = t.target_entity, o.type = t.target_type "
                "MERGE (s)-[r:RELATION {triple_id: t.triple_id}]->(o) "
                "SET r.predicate = t.relation, "
                "    r.chunk_id = t.chunk_id, "
                "    r.chunk_hash = t.chunk_hash, "
                "    r.doc_id = t.doc_id, "
                "    r.confidence = t.confidence, "
                "    r.evidence = t.evidence"
            )
            payload = [t.model_dump() for t in triples]
            try:
                with self._driver.session(database=self.config.database) as session:
                    session.run(cypher, triples=payload)
            except Exception as exc:
                logger.warning("Neo4j Cypher upsert failed; triples retained in memory graph: %s", exc)

        return len(triples)

    def traverse_multi_hop(
        self,
        entities: list[str],
        max_hops: int | None = None,
        limit: int | None = None,
    ) -> list[GraphSearchResult]:
        """Perform up to k-hop graph traversal starting from seed entities or query terms."""
        hops = max_hops or self.config.max_hops
        max_results = limit or self.config.max_triples
        cleaned_seeds = [e.strip().lower() for e in entities if e.strip()]
        if not cleaned_seeds:
            return []

        if self._driver is not None:
            try:
                cypher_hits = self._traverse_neo4j_cypher(cleaned_seeds, hops, max_results)
                if cypher_hits:
                    return cypher_hits
            except Exception as exc:
                logger.warning("Neo4j Cypher traversal fallback to in-memory BFS: %s", exc)

        return self._traverse_memory_bfs(cleaned_seeds, hops, max_results)

    def _traverse_neo4j_cypher(
        self,
        seed_terms: list[str],
        max_hops: int,
        limit: int,
    ) -> list[GraphSearchResult]:
        """Execute parameterized multi-hop Cypher query against Neo4j."""
        cypher = (
            f"MATCH path = (s:Entity)-[rels:RELATION*1..{int(max_hops)}]-(t:Entity) "
            "WHERE any(seed IN $seeds WHERE s.name_lower CONTAINS seed OR seed CONTAINS s.name_lower) "
            "UNWIND range(0, size(rels)-1) AS idx "
            "WITH rels[idx] AS r, idx + 1 AS hop_dist, startNode(rels[idx]) AS u, endNode(rels[idx]) AS v "
            "RETURN DISTINCT r.triple_id AS node_id, "
            "       u.name AS source_entity, "
            "       r.predicate AS relation, "
            "       v.name AS target_entity, "
            "       hop_dist AS hop_distance, "
            "       r.chunk_hash AS chunk_hash, "
            "       r.doc_id AS doc_id, "
            "       r.evidence AS evidence "
            "ORDER BY hop_distance ASC "
            "LIMIT $limit"
        )
        results: list[GraphSearchResult] = []
        with self._driver.session(database=self.config.database) as session:
            records = session.run(cypher, seeds=seed_terms, limit=limit)
            for rec in records:
                src = str(rec["source_entity"])
                rel = str(rec["relation"])
                tgt = str(rec["target_entity"])
                results.append(
                    GraphSearchResult(
                        node_id=str(rec["node_id"]),
                        source_entity=src,
                        relation=rel,
                        target_entity=tgt,
                        hop_distance=int(rec["hop_distance"]),
                        chunk_hash=str(rec["chunk_hash"]),
                        doc_id=str(rec["doc_id"]),
                        evidence=str(rec["evidence"] or ""),
                        path_summary=f"({src}) -[:{rel}]-> ({tgt})",
                    )
                )
        return results

    def _traverse_memory_bfs(
        self,
        seed_terms: list[str],
        max_hops: int,
        limit: int,
    ) -> list[GraphSearchResult]:
        """Breadth-First Search (BFS) over in-memory entity graph supporting multi-hop chains."""
        if not self._triples:
            return []

        matched_nodes: set[str] = set()
        seed_tokens: set[str] = set()
        for term in seed_terms:
            for tok in re.findall(r"[a-z0-9_]+", term.lower()):
                if len(tok) > 2:
                    seed_tokens.add(tok)

        for node_key in self._adjacency:
            if any(seed in node_key or node_key in seed for seed in seed_terms):
                matched_nodes.add(node_key)
            elif any(tok in node_key for tok in seed_tokens):
                matched_nodes.add(node_key)

        queue: deque[tuple[str, int, list[str]]] = deque(
            (node, 1, [node]) for node in sorted(matched_nodes)
        )
        visited_nodes: set[str] = set(matched_nodes)
        visited_triples: set[str] = set()
        results: list[GraphSearchResult] = []

        while queue and len(results) < limit:
            current_node, hop, node_chain = queue.popleft()
            if hop > max_hops:
                continue

            for triple in self._adjacency.get(current_node, []):
                if triple.triple_id not in visited_triples:
                    visited_triples.add(triple.triple_id)
                    path_str = f"({triple.source_entity}) -[:{triple.relation}]-> ({triple.target_entity})"
                    results.append(
                        GraphSearchResult(
                            node_id=triple.triple_id,
                            source_entity=triple.source_entity,
                            relation=triple.relation,
                            target_entity=triple.target_entity,
                            hop_distance=hop,
                            chunk_hash=triple.chunk_hash,
                            doc_id=triple.doc_id,
                            evidence=triple.evidence,
                            path_summary=path_str,
                        )
                    )
                    if len(results) >= limit:
                        break

                neighbor = (
                    triple.target_entity.lower()
                    if triple.source_entity.lower() == current_node
                    else triple.source_entity.lower()
                )
                if neighbor not in visited_nodes and hop < max_hops:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, hop + 1, [*node_chain, neighbor]))

        return results

    @property
    def triple_count(self) -> int:
        """Total number of unique knowledge triples stored."""
        return len(self._triples)
