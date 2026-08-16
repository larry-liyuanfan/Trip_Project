"""用同一确定性编码器比较 Week 3 baseline 与 Week 4 winner。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.common_semantic_comparison import compare_common_semantics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week3-config",
        default="configs/evaluation_week3_v2.yaml",
    )
    parser.add_argument(
        "--baseline-run-id",
        default="week3_v2_baseline_full_20260724_001",
    )
    parser.add_argument(
        "--winner-run-id",
        default="week4_winners_full_20260725_001",
    )
    parser.add_argument(
        "--winner-runs-dir",
        default="outputs/week4/runs",
    )
    parser.add_argument(
        "--semantic-coding-config",
        default="configs/evaluation/baseline_semantic_coding_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/week4/common_semantic/"
            "week4_common_semantic_coding_v1_20260726_001"
        ),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    args = parser.parse_args()
    summary = compare_common_semantics(
        root=Path.cwd(),
        week3_config_path=Path(args.week3_config),
        baseline_run_id=args.baseline_run_id,
        winner_run_id=args.winner_run_id,
        winner_runs_dir=Path(args.winner_runs_dir),
        semantic_coding_config=Path(args.semantic_coding_config),
        output_dir=Path(args.output_dir),
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
