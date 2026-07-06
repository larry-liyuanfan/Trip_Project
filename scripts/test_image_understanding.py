import json

import requests


def main() -> None:
    payload = {
        "image_urls": ["file://data/samples/images/cafe_001.jpg"],
        "user_text": "这张图可能适合什么旅行场景？",
        "language": "zh",
        "prompt_version": "prompt_image_understanding_v1",
    }
    response = requests.post(
        "http://localhost:8000/v1/image-understanding",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

