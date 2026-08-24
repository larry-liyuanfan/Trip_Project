"""Upload a verified release directory to a private Alibaba OSS prefix."""

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument("destination", help="private oss://bucket/prefix")
    args = parser.parse_args()
    if not args.release_dir.joinpath("release_manifest.json").is_file():
        raise SystemExit("release_manifest.json is missing")
    if not args.destination.startswith("oss://"):
        raise SystemExit("destination must be an oss:// URI")
    subprocess.run(
        [
            "ossutil",
            "cp",
            "-r",
            "--acl",
            "private",
            str(args.release_dir),
            args.destination,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
