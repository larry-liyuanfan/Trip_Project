import argparse
import json
from pathlib import Path
from typing import Any

import requests


DEFAULT_ENDPOINT = "http://localhost:8000/v1/image-understanding"


def collect_default_image_urls(project_root: Path | None = None) -> list[str]:
    root = project_root or Path.cwd()
    urls: list[str] = []
    multimodal_items = root / "data" / "yelp" / "processed" / "ota_subset_v1" / "multimodal_items.jsonl"
    if multimodal_items.exists():
        for line in multimodal_items.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_path = Path(json.loads(line).get("image_path", ""))
            if not image_path.is_absolute():
                image_path = root / image_path
            if image_path.exists():
                urls.append(path_to_api_file_url(image_path, root))
            if len(urls) >= 2:
                return urls

    sample_image = (root / "data" / "samples" / "images" / "cafe_001.jpg").resolve()
    if sample_image.exists():
        urls.extend([path_to_api_file_url(sample_image, root)] * (2 - len(urls)))
    return urls[:2]


def path_to_api_file_url(image_path: Path, project_root: Path) -> str:
    try:
        relative = image_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return image_path.resolve().as_uri()
    return f"file://{relative.as_posix()}"


def build_payload(image_urls: list[str]) -> dict[str, Any]:
    if len(image_urls) < 2:
        raise ValueError("multi-image live test requires at least two image URLs")
    return {
        "image_urls": image_urls[:2],
        "user_text": "multi-image live vLLM stretch check: compare the two OTA images and extract shared travel search signals.",
        "language": "zh",
        "prompt_version": "prompt_image_understanding_v1",
    }


def run_live_check(endpoint: str, image_urls: list[str], timeout_seconds: int) -> dict[str, Any]:
    response = requests.post(endpoint, json=build_payload(image_urls), timeout=timeout_seconds)
    response.raise_for_status()
    body = response.json()
    raw_model_output = str(body.get("raw_model_output") or "")
    if "fallback used" in raw_model_output or "vLLM request failed" in raw_model_output:
        raise RuntimeError(f"API responded with fallback instead of live vLLM output: {raw_model_output}")
    return body


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a stretch live vLLM smoke test with two images.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--image-url", action="append", dest="image_urls", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    image_urls = args.image_urls or collect_default_image_urls()
    result = run_live_check(args.endpoint, image_urls, args.timeout_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
