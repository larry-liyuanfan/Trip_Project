"""Extract a bounded, label-blind source pool from an already-saved legal archive."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.week8_unlabeled_pool import build_pool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    print(json.dumps(build_pool(ROOT, parser.parse_args().config.resolve()), ensure_ascii=False))
