from pathlib import Path


def main() -> None:
    Path("data/samples/images").mkdir(parents=True, exist_ok=True)
    print("Sample directories are ready. Add real OTA/Yelp images under data/samples/images/.")


if __name__ == "__main__":
    main()

