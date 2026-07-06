from typing import Any

from src.planning.preference_parser import normalize_preferences


def build_itinerary(
    candidates: list[dict[str, Any]], preferences: dict[str, Any]
) -> dict[str, Any]:
    normalized = normalize_preferences(preferences)
    pace = str(normalized["pace"]).capitalize()
    duration = normalized["duration"]
    slots = _time_slots(len(candidates), normalized["pace"])

    itinerary = []
    for idx, candidate in enumerate(candidates):
        itinerary.append(
            {
                "time": slots[idx],
                "poi_name": candidate.get("name", "Unknown POI"),
                "poi_type": candidate.get("category", "POI"),
                "reason": _reason(candidate, normalized),
            }
        )

    return {
        "itinerary": itinerary,
        "summary": f"{pace} {duration} itinerary for {normalized['city']}.",
        "assumptions": ["Sample POI catalog is used until Yelp data is integrated."],
        "confidence": 0.72 if itinerary else 0.3,
    }


def _time_slots(count: int, pace: str) -> list[str]:
    relaxed = ["10:00-12:00", "14:00-15:30", "16:00-17:30", "19:00-20:30"]
    compact = ["09:00-10:30", "11:00-12:30", "14:00-15:30", "16:00-17:30"]
    slots = relaxed if pace == "relaxed" else compact
    return [slots[i % len(slots)] for i in range(count)]


def _reason(candidate: dict[str, Any], preferences: dict[str, Any]) -> str:
    interests = {str(item).lower() for item in preferences.get("interests", [])}
    category = str(candidate.get("category", "")).lower()
    if category and category in interests:
        return f"Matches user interest in {category}."
    return "Selected from visual search candidates and user preferences."

