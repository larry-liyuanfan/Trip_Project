"""Expose the Week 1 Yelp subset builder as a repository-root command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.yelp_open_dataset import main


if __name__ == "__main__":
    main()
