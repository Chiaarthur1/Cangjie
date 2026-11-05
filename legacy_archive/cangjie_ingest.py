"""
Project Cangjie (Цанцзе) — Phase 2 Production Neo4j Loader
Automated Graph Ontological Ingestion with Full AST Phonetic Extraction & Topology Wiring.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

try:
    from neo4j import GraphDatabase, Driver, Session
    HAS_NEO4J = True
except ImportError:
    GraphDatabase = None
    Driver = Any
    Session = Any
    HAS_NEO4J = False

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Neo4jLoader")

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "cangjie2026")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def extract_phonetic_component(ast: Dict[str, Any], semantic_radical_id: Optional[str] = None) -> Optional[str]:
    """
    Рекурсивно извлекает фонетический компонент из AST-дерева CJK:
    - Для горизонтального разделения (⿰, ⿲): правый компонент (последний потомок).
    - Для вертикального разделения (⿱, ⿳): нижний компонент (последний потомок).
    - Для объемлющих структур (⿴, ⿵, ⿶, ⿷, ⿸, ⿹, ⿺, ⿻): внутренний компонент.
    Если выявленный компонент совпадает со смысловым ключом, берется комплементарный узел.
    """
    if not isinstance(ast, dict) or not ast:
        return None

    node_type = ast.get("node_type")
    if node_type == "component":
        val = ast.get("value")
        return str(val) if val and str(val) != str(semantic_radical_id) else None

    if node_type != "operator":
        return None

    operator = ast.get("value", "")
    children = ast.get("children", [])
    if not isinstance(children, list) or not children:
        return None

    def resolve_leaf(node: Dict[str, Any]) -> Optional[str]:
        if not isinstance(node, dict):
            return None
        if node.get("node_type") == "component":
            return node.get("value")
        ch_list = node.get("children", [])
        if ch_list:
            for ch in ch_list:
                val = resolve_leaf(ch)
                if val:
                    return val
        return node.get("value")

    if operator in {"⿰", "⿲", "⿱", "⿳"}:
        primary_candidate = children[-1]
        secondary_candidate = children[0]
    elif operator in {"⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻"}:
        primary_candidate = children[-1] if len(children) > 1 else children[0]
        secondary_candidate = children[0]
    else:
        primary_candidate = children[-1]
        secondary_candidate = children[0]

    cand_val = resolve_leaf(primary_candidate)
    if cand_val and str(cand_val) != str(semantic_radical_id):
        return str(cand_val)

    alt_val = resolve_leaf(secondary_candidate)
    if alt_val and str(alt_val) != str(semantic_radical_id):
        return str(alt_val)

    return str(cand_val) if cand_val else None


def setup_schema(session: Session) -> None:
    logger.info("Initializing Neo4j Constraints and Indexes...")
    queries = [
        "CREATE CONSTRAINT character_id IF NOT EXISTS FOR (c:Character) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT radical_id IF NOT EXISTS FOR (r:Radical) REQUIRE r.id IS UNIQUE",
        "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (p:Component) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT neologism_id IF NOT EXISTS FOR (n:Neologism) REQUIRE n.id IS UNIQUE",
        """
        CREATE FULLTEXT INDEX ids_composition_idx IF NOT EXISTS
        FOR (c:Character) ON EACH [c.ids]
        OPTIONS { indexConfig: { `fulltext.analyzer`: 'cjk' } }
        """
    ]
    for q in queries:
        try:
            session.run(q)
        except Exception as e:
            logger.warning(f"Schema definition note: {e}")


def ingest_data(session: Session, records: List[Dict[str, Any]]) -> None:
    logger.info(f"Phase 1: Ingesting {len(records)} Character nodes...")
    phase1_query = """
    UNWIND $batch AS row
    CALL {
        WITH row
        MERGE (c:Character {id: row.id})
        SET c.glyph = row.glyph,
            c.ids = row.ids,
            c.is_fake = row.is_fake
    } IN TRANSACTIONS OF 2000 ROWS
    """
    session.run(phase1_query, batch=records)

    logger.info("Phase 2: Wiring Relationships (SEMANTIC, PHONETIC, COMPOSED_OF)...")
    phase2_query = """
    UNWIND $batch AS row
    CALL {
        WITH row
        MATCH (c:Character {id: row.id})

        FOREACH (_ IN CASE WHEN row.semantic_radical_id IS NOT NULL THEN [1] ELSE [] END |
            MERGE (sem:Radical {id: row.semantic_radical_id})
            MERGE (c)-[:SEMANTIC]->(sem)
            FOREACH (__ IN CASE WHEN row.operator IS NOT NULL THEN [1] ELSE [] END |
                MERGE (c)-[rel:COMPOSED_OF {role: 'semantic'}]->(sem)
                SET rel.operator = row.operator, rel.position = row.position
            )
        )

        FOREACH (_ IN CASE WHEN row.phonetic_component_id IS NOT NULL THEN [1] ELSE [] END |
            MERGE (phon:Component {id: row.phonetic_component_id})
            SET phon.glyph = row.phonetic_component_id
            MERGE (c)-[:PHONETIC]->(phon)
            FOREACH (__ IN CASE WHEN row.operator IS NOT NULL THEN [1] ELSE [] END |
                MERGE (c)-[rel:COMPOSED_OF {role: 'phonetic'}]->(phon)
                SET rel.operator = row.operator
            )
        )
    } IN TRANSACTIONS OF 1000 ROWS
    """
    session.run(phase2_query, batch=records)


def main():
    if not HAS_NEO4J:
        logger.error("neo4j driver not found. Install via 'pip install neo4j'.")
        sys.exit(1)

    logger.info(f"Connecting to {URI} as user '{USER}'...")
    try:
        driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        logger.error(f"Neo4j Connection Failed: {e}")
        sys.exit(1)

    data_path = os.getenv("CANGJIE_ONTOLOGY_DUMP", "./data/neo4j_ontology_dump.json")
    if not os.path.exists(data_path):
        logger.error(f"Ontology dump not found: {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for char in raw:
        ast = char.get("ast") if isinstance(char.get("ast"), dict) else {}
        krs = char.get("krs_unicode", [])
        sem_id = krs[0].get("radical_id") if isinstance(krs, list) and len(krs) > 0 else None
        sem_glyph = krs[0].get("glyph") if isinstance(krs, list) and len(krs) > 0 else None

        phonetic_id = extract_phonetic_component(ast, semantic_radical_id=str(sem_glyph or sem_id or ""))

        records.append({
            "id": char.get("codepoint", "UNKNOWN"),
            "glyph": char.get("glyph", ""),
            "ids": char.get("ids", char.get("glyph", "")),
            "is_fake": bool(char.get("is_fake", False)),
            "semantic_radical_id": str(sem_id) if sem_id else None,
            "operator": ast.get("value") if ast.get("node_type") == "operator" else None,
            "position": ast.get("position", "root") if ast.get("node_type") == "operator" else None,
            "phonetic_component_id": phonetic_id
        })

    with driver.session(database=DATABASE) as session:
        setup_schema(session)
        ingest_data(session, records)

    driver.close()
    logger.info("GRAPH INGESTION COMPLETE! 🚀")


if __name__ == "__main__":
    main()
