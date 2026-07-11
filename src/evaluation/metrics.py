"""Small retrieval metrics used by baseline experiments."""


def recall_at_k(expected_ids: set[str], ranked_ids: list[str], k: int) -> float:
    """Compute the fraction of expected IDs present in the first k results."""
    if not expected_ids:
        return 0.0
    hits = expected_ids & set(ranked_ids[:k])
    return len(hits) / len(expected_ids)


def top_k_hit(expected_id: str, ranked_ids: list[str], k: int) -> int:
    """Return one when the expected ID appears in the first k results."""
    return int(expected_id in ranked_ids[:k])

