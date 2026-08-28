"""Build a label-blind, five-dimension isolated final image manifest."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.week8_visual_holdout import build_holdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/week8/visual_final_v2.json")
    print(json.dumps(build_holdout(ROOT, parser.parse_args().config.resolve())))
