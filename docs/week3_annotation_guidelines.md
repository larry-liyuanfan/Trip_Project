# Week 3 Human Annotation Guidelines

## Purpose and annotator role

These rules define the human ground-truth gate for the three Week 3 evaluation sets. Candidate discovery rules, source metadata, Yelp categories/reviews, synthetic recipe fields, Prompt outputs, and model suggestions are not gold labels.

One human annotator inspects the original image(s) and raw itinerary constraints
and submits the complete scenario annotation. The mentor requirement does not
require a second annotator; legacy review fields remain non-gating compatibility
metadata only.

Do not infer an unobservable fact. Use `unknown`, `null`, or an empty array according to the field contract. Reject unreadable, irrelevant, privacy-unsafe, or source-mismatched candidates instead of repairing them with guesses.

## Common evidence rules

- Label only visible image evidence and explicit source text. Yelp category/review text may help route a candidate but does not prove the image label.
- Preserve OCR exactly as visible, including identifiers, times, dates, amounts, and status words. Use `null` only when OCR is not applicable; use an empty array when OCR is applicable but no legible text exists.
- Use short normalized phrases in evidence arrays. Do not add explanations, hidden reasoning, or unsupported causes.
- Do not retain unnecessary personal information in OCR or key-information labels.
- Use `unknown` when a required field cannot reasonably be determined from the image; report these cases as dataset limitations instead of guessing.

## Image-to-product search

- `business_category`: choose `hotel`, `attraction`, `restaurant`, or `unknown` from the dominant visible OTA venue.
- `style_tags`: record visually supported ambience or design labels such as modern, historic, luxury, casual, family-friendly, or nature-oriented. Do not infer service quality.
- `visible_facilities`: include only visible facilities such as pool, bar, outdoor seating, parking sign, stage, or accessible entrance.
- `price_range`: choose `budget`, `mid_range`, `premium`, `luxury`, or `unknown`. Use a visible menu, price sign, rate board, or other direct price evidence. Otherwise use `unknown`; venue appearance alone is insufficient.

Counterexample: a polished lobby does not by itself justify `luxury` price range. A restaurant located inside a hotel is labelled by the dominant subject of the image, not automatically by the Yelp parent category.

## Intelligent after-sales

- `issue_type`: `hygiene_stain` for visible contamination, dirt, pests, or linen stains; `facility_damage` for broken or non-functional physical facilities; `attraction_closure` for explicit closure/unavailability evidence; `transport_delay` for explicit delay/cancellation evidence. Use `other` only for a clear in-scope problem outside these four groups; use `unknown` when evidence is inadequate.
- `severity`: `low` for cosmetic/local inconvenience; `medium` for material service degradation with a usable alternative; `high` for unusable booked service, significant safety concern, or major disruption; `critical` only for immediate serious health/safety risk or stranding requiring urgent intervention. Use `unknown` when the impact cannot be established. No deterministic suggestion is produced for this field.
- `key_information`: record observable entity, location/asset, status, date/time, booking/route identifiers, and impact when present.
- `ocr_ground_truth`: human transcription of legible evidence text. Do not normalize identifiers or invent occluded characters.

The weak Yelp review/photo pairs must be rejected if the image does not visibly support the routed issue. Synthetic closure/delay cards may be validated only when the displayed event details are internally coherent and legible.

## Multimodal itinerary planning

- `reference_images`: list the exact manifest image paths used for the preference judgement.
- `text_constraints`: split the raw text into atomic explicit constraints without changing meaning.
- `style_preferences`: describe visible preference cues, not a specific place identity unless visibly established.
- `hard_constraints`: mandatory dates, duration, budget caps, party needs, accessibility, fixed end time, required/forbidden activities, or transport restrictions.
- `soft_constraints`: preferences such as pace, ambience, preferred transport, cuisine, or optional activities.
- `required_itinerary_elements`: include the structural elements necessary to verify the request, such as daily schedule, POIs, meals, transport, duration, cost/budget, and constraint checks.

An itinerary place has value `null` when the source does not provide a defensible location. Never invent a city or venue to make the annotation look complete.

## Allowed values summary

| Field | Allowed values or type |
| --- | --- |
| `business_category` | `hotel`, `attraction`, `restaurant`, `unknown` |
| `price_range` | `budget`, `mid_range`, `premium`, `luxury`, `unknown` |
| `style_tags` | zero or more of `modern`, `traditional`, `historic`, `luxury`, `minimalist`, `industrial`, `rustic`, `casual`, `romantic`, `family_friendly`, `nature_oriented`, `business`, `resort`, `artistic`, `local_character`, `quiet` |
| `visible_facilities` | zero or more of `lobby`, `front_desk`, `guest_room`, `bed`, `pool`, `spa`, `gym`, `restaurant`, `bar`, `outdoor_seating`, `parking`, `accessible_entrance`, `elevator`, `stage`, `play_area`, `restroom`, `ticket_office`, `viewpoint`, `garden`, `beach`, `buffet`, `private_dining`, `charging_facility` |
| `issue_type` | `hygiene_stain`, `facility_damage`, `attraction_closure`, `transport_delay`, `other`, `unknown` |
| `severity` | `low`, `medium`, `high`, `critical`, `unknown` |
| `key_information` | array of concise observed facts |
| `ocr_ground_truth` | ordered string array, or JSON `null` when OCR is not applicable |
| `reference_images` | one or more registered image paths from the current sample |
| `text_constraints`, `hard_constraints`, `soft_constraints` | zero or more atomic constraints from the current raw constraint text; narrowly controlled exact entry is allowed when a necessary source constraint is missing from the choices |
| `style_preferences` | zero or more of `urban`, `nature`, `history_culture`, `food`, `leisure`, `romantic`, `family`, `adventure`, `photography`, `shopping`, `nightlife`, `quiet`, `luxury`, `budget`, `slow_paced` |
| `required_itinerary_elements` | zero or more of `daily_schedule`, `poi`, `accommodation`, `meals`, `transport`, `duration`, `budget_check`, `end_time_check`, `constraint_check` |
Legacy review or PII metadata may remain for compatibility, but it is not part of
the mentor-required annotation workflow and does not create a second-person gate.

## Optional deterministic suggestions

Use `--include-suggestions` only for annotation export. The tool calls no VLM and reads no model output. It uses the configured sampling stratum, source/provenance metadata, exact synthetic recipe text, ordered image paths, and deterministic itinerary text parsing.

Suggestions are stored only at `context.deterministic_suggestion` with:

- `contract_version=week3_deterministic_annotation_suggestion_v1`
- `method=source_metadata_rules_v1`
- `non_gold=true`
- per-field `value`, `confidence`, `basis`, and `requires_human_confirmation=true`
- `unsupported_fields` for facts that rules cannot establish

The annotation object remains unfilled. Product suggestions cover only the routed business category. After-sales suggestions cover the routed issue type and, for project-owned synthetic cards, the exact recipe text. Itinerary suggestions parse explicit text constraints and copy image paths, but never infer visual style. The apply command removes `context`, so suggestions cannot enter the manifest as gold labels.

## Synthetic after-sales evidence correction (2026-07-17)

The original `week3_after_sales_evidence_v1` cards used a small default font and
decorative high-contrast blocks that could distract from the evidence text.
They must not be used for annotation. All 74 still-pending business-synthetic
closure and delay samples were refreshed to
`week3_after_sales_evidence_v2` with four deterministic document-style layouts,
measured text bounds, non-overlapping text rows, and per-image perceptual
separation.

The correction preserved the exact evidence text and all stable sample/source
identities. No target annotation, review, or draft was overwritten. The local
audit and v1 backup remain under ignored `data/eval/logs/` and
`data/eval/backups/` respectively.

## Superseded synthetic after-sales recuration (2026-07-21)

An intermediate correction generated a pending project-owned v3 candidate set.
Project Control later superseded that route and froze the existing completed
human annotations for reuse. The active run-bound after-sales manifest is the
hash-verified mixed-source version (`public_yelp=76`,
`business_synthetic=74`); the pending v3 set remains only in ignored backup
storage. No new annotation, relabeling, or review work is required.

## Historical JSONL annotation workflow

The Week 3 Git deliverable intentionally contains no browser UI. The following
commands document the existing reusable packet mechanism; they are not a
current Week 3 labeling requirement. Do not edit a
full manifest directly; export a packet, inspect every registered image and raw
constraint, then apply a complete human submission transactionally.

Export one suggested annotation packet per scenario:

```powershell
python scripts/manage_week3_annotations.py --scenario image_product_search export --include-suggestions --output data/eval/codings/image_product_search_annotation_suggested.jsonl
python scripts/manage_week3_annotations.py --scenario after_sales export --include-suggestions --output data/eval/codings/after_sales_annotation_suggested.jsonl
python scripts/manage_week3_annotations.py --scenario itinerary_planning export --include-suggestions --output data/eval/codings/itinerary_planning_annotation_suggested.jsonl
```

For every row, open each repository-relative path in `context.input.images`, read the raw itinerary text when present, inspect the suggestion basis, then fill only `annotator` and the complete `annotation` object. Do not copy an unsupported suggestion from source metadata into a visual ground-truth field. Rows may be applied in smaller JSONL batches; validation is transactional for the submitted batch.

Apply completed annotations:

```powershell
python scripts/manage_week3_annotations.py --scenario <scenario> apply --input <completed-annotation-packet.jsonl>
```

Then rebuild and validate the exclusion registry:

```powershell
python scripts/prepare_week3_evaluation.py --config configs/evaluation_week3.yaml build-exclusion
python scripts/validate_week3_evaluation.py --config configs/evaluation_week3.yaml
```

Run the full baseline only after the three manifests reach the mentor-required
sizes, required scene/category coverage, readable-file checks, and completed
single-annotator labels.
