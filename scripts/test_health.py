"""Smoke-test the local FastAPI health endpoint."""

import requests


def main() -> None:
    """Request health metadata and fail on transport or HTTP errors."""
    response = requests.get("http://localhost:8000/health", timeout=10)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
