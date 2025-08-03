# Copyright 2025
# Apache-2.0 License

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Tuple


@dataclass
class Document:
    id: str
    text: str
    score: float = 0.0
    metadata: Optional[dict] = None


class Retriever(Protocol):
    def search(self, query: str, k: int = 4) -> List[Document]:
        ...


class NoopRetriever:
    def search(self, query: str, k: int = 4) -> List[Document]:
        return []


class QdrantRetriever:
    def __init__(self, url: str, collection: str = "docs"):
        self.url = url
        self.collection = collection

    def search(self, query: str, k: int = 4) -> List[Document]:
        # TODO: integrate qdrant-client; placeholder returns empty
        return []


class PGVectorRetriever:
    def __init__(self, conn_str: str, table: str = "docs"):
        self.conn_str = conn_str
        self.table = table

    def search(self, query: str, k: int = 4) -> List[Document]:
        # TODO: integrate psycopg and pgvector; placeholder returns empty
        return []


def get_retriever(kind: str, url: Optional[str]) -> Retriever:
    if kind == "none":
        return NoopRetriever()
    if kind == "qdrant":
        return QdrantRetriever(url or "http://localhost:6333")
    if kind == "pgvector":
        return PGVectorRetriever(url or "postgresql://user:pass@localhost:5432/db")
    raise ValueError(f"Unknown retriever kind: {kind}")


def rerank_hook(query: str, docs: List[Document], reranker: Optional[str]) -> List[Document]:
    # TODO: integrate a real reranker; stable sort by score desc as placeholder
    return sorted(docs, key=lambda d: d.score, reverse=True)