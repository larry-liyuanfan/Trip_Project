# Week 4 Prompt Optimization Report

## Tested Scope

The experiment uses the immutable `week3_evaluation_v2` manifests and gold.
Each scenario has five fixed positive and two fixed boundary examples. The
4-shot candidate uses three positives plus one boundary example; the 7-shot
candidate uses all seven. No labels, Week 3 Prompts, Schemas, runs, raw
outputs, or scores were modified.

Model/backend and generation settings remain
`Qwen/Qwen2-VL-2B-Instruct`, vLLM, temperature `0.1`, top-p `0.9`,
repetition penalty `1.05`, and maximum output `1280` tokens. The fixed pilot
contains five non-example samples per scenario. Every run records prompt,
input and dataset hashes, example IDs and collage hashes, raw output,
parse/Schema results, latency, token usage, and model settings under ignored
`outputs/week4/`.

## Pilot Selection

Selection score is reproducible:

`0.55 * business quality + 0.10 * JSON + 0.20 * Schema + 0.075 * token efficiency + 0.075 * latency efficiency`.

| Scenario | standardized_v2 | 4-shot | 7-shot | Winner |
| --- | ---: | ---: | ---: | --- |
| Product | 0.3450 | 0.0400 | 0.2665 | `standardized_v2` |
| After-sales | 0.5967 | 0.0800 | 0.5208 | `standardized_v2` |
| Itinerary | 0.4025 | 0.0750 | 0.0745 | `standardized_v2` |

The winner means only the best candidate in this pilot. Product
`standardized_v2` retained better business quality and Schema compliance.
After-sales 7-shot had slightly higher business quality but lost on Schema and
token efficiency. Both itinerary few-shot variants were rejected by the
backend before generation and therefore had zero JSON/Schema compliance.

## Full Winner and Baseline Comparison

The full winner run is `week4_winners_full_20260725_001`; all 450 records
completed with the same selected-sample hash as Week 3 v2.

| Scenario | Metric track | Week 3 baseline | Week 4 winner |
| --- | --- | ---: | ---: |
| Product | business quality / JSON / Schema | 0.3570 / 0% / 0% | 0.1565 / 77.5% / 75.5% |
| After-sales | business quality / JSON / Schema | 0.2500 / 0% / 0% | 0.2977 / 96.67% / 96.67% |
| Itinerary | business quality / JSON / Schema | 0.1930 / 0% / 0% | 0.0508 / 90.0% / 87.0% |

The corresponding business-quality deltas are -0.2005, +0.0477, and -0.1423.
Mean token use is 1211.70 / 1047.41 / 1944.39 and mean latency is
13658.11 / 6036.31 / 11062.42 ms for product, after-sales, and itinerary.
P95 latency is 49830.86 / 5663.14 / 52999.28 ms.

Week 3 baseline semantic values remain the independent,
gold-independent `baseline_semantic_coding_v1` lexical track. Week 4 uses the
same documented metric fields but does not present the comparison as a causal
Prompt effect or overwrite the Week 3 score files.

## Reproduction

```bash
python scripts/run_week4_prompt_evaluation.py --config configs/evaluation_week4.yaml --run-id <pilot-id> --stage pilot --product-variant <variant> --after-sales-variant <variant> --itinerary-variant <variant>
python scripts/analyze_week4_prompts.py --config configs/evaluation_week4.yaml --pilot-run-id <standardized-pilot> --pilot-run-id <4-shot-pilot> --pilot-run-id <7-shot-pilot>
python scripts/run_week4_prompt_evaluation.py --config configs/evaluation_week4.yaml --run-id <full-id> --stage full --product-variant standardized_v2 --after-sales-variant standardized_v2 --itinerary-variant standardized_v2
python scripts/analyze_week4_prompts.py --config configs/evaluation_week4.yaml --full-run-id <full-id>
```
