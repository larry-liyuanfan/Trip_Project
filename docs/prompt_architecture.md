# Week 3 Prompt Architecture

## Scope

Week 3 keeps the unoptimized zero-shot baseline physically and logically separate from the standardized prompt. Baseline results must use `baseline_minimal_v1`; a result produced with `standardized_v1` cannot be reported as the baseline.

## Minimal baseline

The three UTF-8 prompt files are stored under `configs/evaluation/prompts/baseline_minimal_v1/`:

| Scenario | Instruction |
| --- | --- |
| `image_product_search` | 识别图片所展示的旅游商品特征。 |
| `after_sales` | 识别图片中与旅游售后相关的问题。 |
| `itinerary_planning` | 根据参考图片和文字约束识别用户的行程需求。 |

Each baseline is one task sentence. It contains no role, output-field list, JSON or formatting instruction, reasoning request, chain-of-thought request, example, or prompt-optimization content. `render_baseline_request` loads that sentence unchanged and adds only the sample's multimodal inputs; it does not add standardized instructions or a Schema.

## Standardized prompt

`standardized_v1` combines `common.yaml` with one scenario YAML file. The renderer returns four named layers and OpenAI-compatible system/user messages:

1. `system_role`: professional OTA assistant plus the common language, evidence, privacy, and safety rules.
2. `task_instruction`: scenario-specific extraction and checking task.
3. `input_context`: compact UTF-8 JSON serialized from the manifest's top-level `input` object.
4. `output_constraint`: JSON-only output, scenario Schema name, the complete compact JSON Schema contract, no extra fields, and concise evidence rules.

The model-facing request receives the full Schema object, not only its filename. The renderer also returns the parsed Schema as `output_schema`, allowing callers to use exactly the same contract for constrained generation and validation.

Both baseline and standardized user messages use an explicit multimodal `content` array. Parts are ordered as follows:

1. The unchanged baseline task sentence, or the standardized task instruction.
2. For every input image in manifest order, one text part `参考图片占位符 <image_N>` immediately followed by its `image_url` part. Product-search and after-sales requests require exactly one image; itinerary requests require at least one image and preserve all reference images in manifest order.
3. For itinerary samples only, one `原始文字约束：...` text part after the final image. The text is copied from `input.text_constraints` after trimming only leading and trailing whitespace. It is required for itinerary samples and must be `null` for the two image-only scenarios.
4. For standardized requests only, the output-constraint text containing the complete Schema.

The common rules require Chinese natural-language values, explicit `unknown` or allowed `null` values, no fabrication, separation of observation from inference, privacy protection, and route/travel safety. Schema field names and controlled English enum values are machine-contract exceptions to the Chinese-language rule; this keeps output labels aligned with the Stage 1 manifest contract. Every evidence field contains only short observable facts or field sources. Prompts never request private chain-of-thought or narrated multi-step reasoning.

## Output schemas

The three Draft 2020-12 schemas live under `configs/evaluation/schemas/`:

- `image_product_search_v1.schema.json`
- `after_sales_v1.schema.json`
- `itinerary_planning_v1.schema.json`

All top-level fields are required and `additionalProperties` is false. Enums define controlled labels, arrays define item types, nullable fields use a type union with `null`, every evidence string is limited to 120 characters, and confidence is either `null` or a number from 0 to 1.

The itinerary contract requires a non-empty `itinerary` array. Each day requires `day_index`, nullable `date`, `summary`, and a non-empty `activities` array. Each activity requires nullable start/end time, nullable place, activity, nullable transport, and bounded `source_evidence`. `place_name` is `null` when neither the source text nor images identify a location; the model must not invent one. Day, activity, and `constraint_check` objects all reject extra properties, including nested `reasoning` fields.

`constraint_check.evidence`, activity `source_evidence`, and top-level `observed_evidence` use the same 120-character per-string limit. Every evidence array has `maxItems: 10`, preventing a response from bypassing concise-evidence rules by emitting many individually short entries.

`src/evaluation/schema_validation.py` loads these files and validates the project-supported JSON Schema subset: `type`, `required`, `properties`, `additionalProperties`, `enum`, `items`, `uniqueItems`, item bounds, string-length bounds, and numeric bounds. The schemas are inline and do not use unsupported `$ref` or composition keywords.

## Verification

```powershell
python -m unittest tests.test_evaluation_prompts tests.test_evaluation_schemas -v
python scripts/validate_week3_evaluation.py
python -m unittest discover -s tests -v
```
