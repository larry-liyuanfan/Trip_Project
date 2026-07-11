"""Hybrid retrieval facade that currently wraps the keyword baseline."""

from typing import Any

from src.retrieval.keyword_retriever import KeywordRetriever


class HybridRetriever:
    """Week-3 placeholder using keyword scores until embeddings are added."""

    def __init__(self, catalog: list[dict[str, Any]]) -> None:
        """Initialize the available retrieval backend for the provided catalog."""
        self.keyword = KeywordRetriever(catalog)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return baseline results labeled with their actual retrieval mode."""
        results = self.keyword.search(query, top_k=top_k)
        for result in results:
            result["retrieval_mode"] = "hybrid_keyword_baseline"
        return results

