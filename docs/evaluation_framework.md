# Week 3 Evaluation Framework

## Scope

The Stage 3 framework loads the approved Week 3 manifests and prompt contracts, renders OpenAI-compatible multimodal requests, optionally invokes a configured vLLM endpoint, parses raw text, validates parsed JSON against the scenario Schema, and writes immutable run artifacts. It does not compute Stage 4 metrics or perform a Stage 5 real evaluation.

## Configuration

`configs/evaluation_week3.yaml` is the entry point. Its `runtime` section references separate model and inference files instead of embedding model choices in Python:

- `configs/model_qwen2_vl.yaml`: `Qwen/Qwen2-VL-2B-Instruct`, the model used by the verified Week 1 Docker smoke path and current Compose default.
- `configs/inference_default.yaml`: temperature, top-p, and maximum output tokens.
- `runtime.live_base_url`: OpenAI-compatible vLLM base URL.
- `runtime.timeout_seconds`: HTTP timeout.
- `paths.runs_dir`: ignored local root for immutable run directories.

The runner does not download models, start containers, modify dependencies, or silently select another checkpoint.

## Eligibility and Modes

Only records satisfying these run-eligibility conditions are selected:

- `annotation_status == completed`
- `file_status == valid`
- `annotation` has the required scenario structure
- `review_status != rejected`

Modes have distinct behavior:

| Mode | Model call | Raw output source | Intended use |
| --- | --- | --- | --- |
| `dry-run` | Never | `null`; error is `dry_run` | Validate configuration, manifests, prompts, selection, and persistence without inference |
| `mock` | Never | Explicit `--mock-responses` JSONL | Deterministic parser, Schema, and persistence testing |
| `live` | Configured vLLM endpoint | OpenAI chat-completion response text | Real inference when service and validated data are available |

Mock JSONL rows contain exactly the lookup information needed by the runner:

```json
{"sample_id":"sample-001","raw_output":"{\"business_category\":\"hotel\"}"}
```

Duplicate sample IDs are rejected. Mock text is never treated as human annotation or measured model output.

If an eligible sample has no supplied mock row, the persisted category is `mock_fixture_missing`. This is distinct from `model_request_error`, which is reserved for failures while constructing or executing a live model request.

## Requests and Results

The selected prompt version is explicit for every run:

- `baseline_minimal_v1` uses the unchanged one-sentence baseline plus ordered multimodal input parts.
- `standardized_v1` uses the approved four layers and includes the complete scenario Schema in the model request.

For live requests, repository-relative `file://` image parts are resolved against the project root and encoded from actual local bytes as data URLs. Generation parameters and served model name come from referenced configuration files.

Every persisted JSONL row contains:

- `run_id`, `sample_id`, `scenario`, `mode`
- `model_name`, complete `model_config`, `prompt_version`
- normalized `input_metadata`, including ordered image paths/hashes and raw itinerary text constraints
- `raw_output`, `parsed_output`, `json_valid`, `schema_valid`
- `latency_ms`, `error`, and UTC `timestamp`

JSON parsing and Schema validation are separate states. Valid JSON that violates a Schema retains its parsed object and records a `schema_validation_error`. Invalid JSON retains the original `raw_output`, sets `parsed_output` to `null`, and records a `json_parse_error`. Model request failures persist no fabricated raw or parsed output.

JSON handling is strict at every boundary. `NaN`, `Infinity`, and `-Infinity` are rejected while parsing model text, while validating programmatically constructed Schema instances, and before any result JSONL row is written. Result and metadata serialization also use `allow_nan: false` as a final guard.

## Immutable Run Layout

Each run creates a new directory below `data/eval/runs/`:

```text
data/eval/runs/<run_id>/
  metadata.json
  results.jsonl
```

`results.jsonl` is written and flushed incrementally. `metadata.json` records mode, run scope, dataset version, prompt/model settings, selected count, persisted record count, completion status, and any run-level failure. It also stores SHA-256 values for all manifests, the exclusion registry, active Prompt assets, and all Schemas, plus a canonical hash of the ordered selected sample IDs. Every result row stores a canonical request hash. Scoring re-verifies these artifacts before reading annotations. A pre-existing `<run_id>` directory causes immediate rejection; the runner never appends to or overwrites an earlier run.

Before creating that directory, the configured runner loads all three manifests through the shared `src.evaluation.manifests.load_configured_manifests` API. The same API is used by the validation script and rejects any record whose `scenario` does not match the configured manifest slot. It also performs image-byte hash verification. The runner then performs one combined duplicate check for `source_id` and every image SHA-256 across scenarios, rebuilds the expected exclusion rows in memory, and requires the configured local exclusion manifest to match exactly. A missing, malformed, or stale registry aborts the run before any output directory exists. The runner never writes or automatically rebuilds the registry; rebuilding remains an explicit `prepare_week3_evaluation.py build-exclusion` operation.

The entire runs directory is ignored by Git. Run IDs accept only letters, numbers, dots, underscores, and hyphens, and must begin with an alphanumeric character.

## Commands

Initialize empty local manifests in a new clone, then validate them:

```powershell
python scripts/prepare_week3_evaluation.py init
python scripts/validate_week3_evaluation.py
```

Run a baseline dry-run without GPU or network access:

```powershell
python scripts/run_week3_evaluation.py --run-id stage3_dry_run_001 --mode dry-run --prompt-version baseline_minimal_v1
```

Run a standardized mock fixture:

```powershell
python scripts/run_week3_evaluation.py --run-id stage3_mock_001 --mode mock --prompt-version standardized_v1 --mock-responses path/to/mock_responses.jsonl
```

Live runs require an explicit scope. `pilot` permits a non-empty validated subset for operational verification. `full` requires every scenario target and configured stratum quota in the eligible set. `framework` is restricted to mock and dry-run modes.

Capture readiness-only evidence without sending Week 3 images:

```powershell
python scripts/capture_week3_readiness.py --config configs/evaluation_week3.yaml --evidence-id <evidence-id>
```

Reuse the validated completed baseline while its artifact hashes match. A new
live baseline would use:

```powershell
python scripts/run_week3_evaluation.py --config configs/evaluation_week3.yaml --run-id <baseline-run-id> --mode live --run-scope full --prompt-version baseline_minimal_v1
```

`--base-url` may explicitly override the configured endpoint for one run. The override is persisted in the per-record model configuration through the runtime used for that run.

## Current Limitations

- The frozen manifests contain 450 completed, structurally eligible records and the after-sales source mix is 76 public Yelp plus 74 business synthetic. Product unknown scalars, missing after-sales category coverage, and 100 empty itinerary style preferences remain metric/data limitations rather than run blockers.
- Stage 3 does not calculate task metrics or error aggregates.
- No live request is implied by a mock or dry-run status.
- Existing real run records are immutable and should not be repeated merely to recreate equivalent evidence.

Stage 4 scoring is separate from this runner. The mentor-required metric definitions and scoring command are documented in `docs/evaluation_metrics.md`.
