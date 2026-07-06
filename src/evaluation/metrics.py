def recall_at_k(expected_ids: set[str], ranked_ids: list[str], k: int) -> float:
    if not expected_ids:
        return 0.0
    hits = expected_ids & set(ranked_ids[:k])
    return len(hits) / len(expected_ids)


def top_k_hit(expected_id: str, ranked_ids: list[str], k: int) -> int:
    return int(expected_id in ranked_ids[:k])

