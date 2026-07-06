from typing import Any


def normalize_preferences(preferences: dict[str, Any]) -> dict[str, Any]:
    return {
        "city": preferences.get("city", "Unknown city"),
        "duration": preferences.get("duration", "1 day"),
        "budget": preferences.get("budget", "medium"),
        "pace": preferences.get("pace", "relaxed"),
        "interests": preferences.get("interests", []),
    }

