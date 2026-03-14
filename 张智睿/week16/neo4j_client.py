import os
from typing import Any, Dict, List, Optional, Tuple
from neo4j import GraphDatabase
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv is optional; env vars may already be set
    pass

def _get_env(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name, default)
    if val is None or val == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val

class Neo4jClient:
    """Small helper around the official Neo4j Python driver."""

    def __init__(
        self,
        uri: Optional[str] = None,
        auth: Optional[Tuple[str, str]] = None,
        database: Optional[str] = None,
    ) -> None:
        self.uri = uri or _get_env("NEO4J_URI", "bolt://localhost:7687")
        user = None
        pwd = None
        if auth is None:
            user = os.getenv("NEO4J_USER", "neo4j")
            pwd = _get_env("NEO4J_PASSWORD")
            auth = (user, pwd)
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver = GraphDatabase.driver(self.uri, auth=auth)

    def close(self) -> None:
        self._driver.close()

    def run(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a Cypher query and return list[dict] (like py2neo .data())."""
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, parameters or {})
            return result.data()

    def run_write(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Run a write Cypher query and consume the result."""
        with self._driver.session(database=self.database) as session:
            res = session.run(cypher, parameters or {})
            res.consume()

    def verify(self) -> bool:
        with self._driver.session(database=self.database) as session:
            rec = session.run("RETURN 1 AS ok").single()
            return bool(rec and rec.get("ok") == 1)
