"""Create the lightweight sample-image directory used by API smoke tests."""

from pathlib import Path


def main() -> None:
    """Ensure the sample image directory exists without fabricating images."""
    Path("data/samples/images").mkdir(parents=True, exist_ok=True)
    print("Sample directories are ready. Add real OTA/Yelp images under data/samples/images/.")


if __name__ == "__main__":
    main()
