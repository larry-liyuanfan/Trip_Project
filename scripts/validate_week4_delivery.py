"""统一只读验证 Week 4 运行、哈希、评分和比较产物。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.week4_runner import load_week4_config
from src.evaluation.week4_validation import validate_week4_delivery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week4.yaml")
    args = parser.parse_args()
    root = Path.cwd()
    config = load_week4_config(root, Path(args.config))
    summary = validate_week4_delivery(root, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
