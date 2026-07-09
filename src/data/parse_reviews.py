import re
from collections import defaultdict
from typing import Any, Iterable


SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def parse_review_records(
    records: Iterable[dict[str, Any]],
    min_text_length: int = 20,
    reject_symbol_only: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    parsed: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    stars: dict[str, list[float]] = defaultdict(list)
    summary = {
        "input_reviews": 0,
        "valid_reviews": 0,
        "filtered_empty": 0,
        "filtered_too_short": 0,
        "filtered_symbol_only": 0,
    }
    for record in records:
        summary["input_reviews"] += 1
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            summary["filtered_empty"] += 1
            continue
        stripped = text.strip()
        if reject_symbol_only and SYMBOL_ONLY_RE.match(stripped):
            summary["filtered_symbol_only"] += 1
            continue
        if len(stripped) < min_text_length:
            summary["filtered_too_short"] += 1
            continue
        business_id = record.get("business_id")
        row = {
            "review_id": record.get("review_id"),
            "business_id": business_id,
            "user_id": record.get("user_id"),
            "stars": record.get("stars"),
            "useful": record.get("useful"),
            "funny": record.get("funny"),
            "cool": record.get("cool"),
            "text": stripped,
            "date": record.get("date"),
        }
        parsed.append(row)
        summary["valid_reviews"] += 1
        if business_id:
            counts[str(business_id)] += 1
            if isinstance(record.get("stars"), (int, float)):
                stars[str(business_id)].append(float(record["stars"]))
    stats = []
    for business_id, count in sorted(counts.items()):
        ratings = stars.get(business_id, [])
        stats.append(
            {
                "business_id": business_id,
                "valid_review_count": count,
                "average_review_stars": sum(ratings) / len(ratings) if ratings else None,
            }
        )
    return parsed, stats, summary
