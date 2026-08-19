"""Run one parameter-locked Week 6 adapter on frozen Week 3 v2 gold."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.week6_final_evaluation import run_final_scenario_evaluation
from src.training.week6_qlora import Week6TrainingError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_week6_final.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--scenario",
        required=True,
        choices=("image_product_search", "after_sales", "itinerary_planning"),
    )
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--prompt-version", default="standardized_v2")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    args = parser.parse_args()
    try:
        result = run_final_scenario_evaluation(
            root=Path.cwd(),
            config_path=Path(args.config),
            run_id=args.run_id,
            scenario=args.scenario,
            prompt_version=args.prompt_version,
            adapter_dir=Path(args.adapter_dir),
            expected_adapter_sha256=args.expected_adapter_sha256,
            max_input_tokens=args.max_input_tokens,
        )
    except (ValueError, Week6TrainingError) as exc:
        raise SystemExit(f"Week 6 final evaluation error: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
