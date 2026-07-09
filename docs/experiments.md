# Experiment Summary

Use this file as a high-level index of experiments. Detailed records remain in:

- `experiments/experiment_log.md`
- `experiments/results.csv`
- `experiments/failure_cases.md`

## Current Experiment Tracking Rules

Each experiment should record:

- date and git commit;
- model name and size, or `not_applicable_data_preparation` for data-only work;
- inference backend and serving command;
- prompt version and generation parameters;
- dataset version and task type;
- metrics, summary, failure cases, and next action.

## Recorded Experiments

| ID | Purpose | Main Result | Detailed Record |
|---|---|---|---|
| EXP-20260706-001 | Initial scaffold and fallback image understanding | Fallback path available before live vLLM startup | `experiments/experiment_log.md` |
| EXP-20260706-002 | Docker + vLLM live image smoke test | Qwen2-VL-2B served through vLLM Docker and returned image-understanding output | `experiments/experiment_log.md` |

## Verification Commands

```bash
python -m unittest discover -s tests
python scripts/test_health.py
python scripts/test_image_understanding.py
```

For Yelp subset preparation:

```bash
python scripts/prepare_yelp_subset.py --raw-dir data/yelp/raw --output-dir data/yelp/processed/ota_subset_v1
```

## Experiment Update Checklist

- Add or update an entry in `experiments/experiment_log.md`.
- Add numeric or tabular results to `experiments/results.csv` when applicable.
- Add reproducible failures to `experiments/failure_cases.md`.
- Summarize mentor-relevant progress in `docs/weekly_log.md`.
