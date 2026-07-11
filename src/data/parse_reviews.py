"""Filter Yelp reviews and aggregate bounded per-business review statistics."""

import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Iterable


SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def parse_review_records(
    records: Iterable[dict[str, Any]],
    min_text_length: int = 20,
    reject_symbol_only: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Collect parsed reviews while preserving filter and business summaries."""
    parsed: list[dict[str, Any]] = []
    stats, summary = stream_review_records(
        records,
        row_sink=parsed.append,
        min_text_length=min_text_length,
        reject_symbol_only=reject_symbol_only,
    )
    return parsed, stats, summary


def stream_review_records(
    records: Iterable[dict[str, Any]],
    row_sink: Callable[[dict[str, Any]], None],
    min_text_length: int = 20,
    reject_symbol_only: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Emit valid reviews immediately and record every rejection reason."""
    counts: dict[str, int] = defaultdict(int)
    star_sums: dict[str, float] = defaultdict(float)
    star_counts: dict[str, int] = defaultdict(int)
    summary = {
        "input_reviews": 0,
        "valid_reviews": 0,
        "filtered_empty": 0,
        "filtered_too_short": 0,
        "filtered_symbol_only": 0,
        "filtered_missing_identifier": 0,
    }
    for record in records:
        summary["input_reviews"] += 1
        row, filtered_reason = parse_review_record(
            record,
            min_text_length=min_text_length,
            reject_symbol_only=reject_symbol_only,
        )
        if filtered_reason:
            summary[filtered_reason] += 1
            continue
        if row is None:
            continue
        row_sink(row)
        summary["valid_reviews"] += 1
        business_id = row.get("business_id")
        if business_id:
            counts[str(business_id)] += 1
            if isinstance(row.get("stars"), (int, float)):
                star_sums[str(business_id)] += float(row["stars"])
                star_counts[str(business_id)] += 1
    stats = []
    for business_id, count in sorted(counts.items()):
        stats.append(
            {
                "business_id": business_id,
                "valid_review_count": count,
                "average_review_stars": (
                    star_sums[business_id] / star_counts[business_id]
                    if star_counts[business_id]
                    else None
                ),
            }
        )
    return stats, summary


def parse_review_record(
    record: dict[str, Any],
    min_text_length: int = 20,
    reject_symbol_only: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one review and return either its row or a named filter reason."""
    if not record.get("review_id") or not record.get("business_id"):
        return None, "filtered_missing_identifier"
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, "filtered_empty"
    stripped = text.strip()
    if reject_symbol_only and SYMBOL_ONLY_RE.match(stripped):
        return None, "filtered_symbol_only"
    if len(stripped) < min_text_length:
        return None, "filtered_too_short"
    return {
        "review_id": record.get("review_id"),
        "business_id": record.get("business_id"),
        "user_id": record.get("user_id"),
        "stars": record.get("stars"),
        "useful": record.get("useful"),
        "funny": record.get("funny"),
        "cool": record.get("cool"),
        "text": stripped,
        "date": record.get("date"),
    }, None
