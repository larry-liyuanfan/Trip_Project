import requests


def main() -> None:
    response = requests.get("http://localhost:8000/health", timeout=10)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()

