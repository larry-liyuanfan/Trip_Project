import re
from collections.abc import Iterable
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class KeywordRetriever:
    def __init__(self, catalog: Iterable[dict[str, Any]]) -> None:
        self.catalog = list(catalog)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_tokens = set(_tokenize(query))
        ranked: list[dict[str, Any]] = []

        for item in self.catalog:
            item_tokens = set(_tokenize(_item_text(item)))
            matched = sorted(query_tokens & item_tokens)
            if not matched:
                continue

            score = len(matched) / max(len(query_tokens), 1)
            ranked.append(
                {
                    **item,
                    "score": round(score, 4),
                    "matched_reasons": matched,
                }
            )

        ranked.sort(key=lambda row: row["score"], reverse=True)
        return ranked[:top_k]


def _item_text(item: dict[str, Any]) -> str:
    fields = [
        item.get("name", ""),
        item.get("category", ""),
        item.get("description", ""),
        " ".join(item.get("tags", [])),
    ]
    return " ".join(fields)


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]

