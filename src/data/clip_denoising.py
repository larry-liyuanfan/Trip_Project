from typing import Any


def run_clip_denoising(
    weak_pairs: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not config.get("enabled", False):
        return {
            "status": "skipped",
            "reason": "clip_denoising.enabled is false",
            "input_pairs": len(weak_pairs),
            "retained_pairs": 0,
        }, []
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"CLIP dependencies unavailable: {exc}",
            "input_pairs": len(weak_pairs),
            "retained_pairs": 0,
        }, []
    return {
        "status": "skipped",
        "reason": "CLIP scoring interface is available but model loading is intentionally not automatic",
        "input_pairs": len(weak_pairs),
        "retained_pairs": 0,
    }, []
