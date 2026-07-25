# Week 4 Bad Cases

Bad cases are exported only from persisted real model failures. The categories
are classification error, required-field omission, JSON format error, severity
error, and constraint omission. Raw outputs remain in the immutable Week 4 run
directory; the exporter records sample IDs and error evidence without repairing
or relabeling model output.

Run `week4_winners_full_20260725_001` produced:

| Category | Count | Representative sample |
| --- | ---: | --- |
| Classification error | 86 | `image_product_search-0ddcb7ef11f23afb` |
| Required-field or Schema error | 7 | `image_product_search-5d6823c5422a7bee` |
| JSON format error | 67 | `image_product_search-c0ed4d164daafeea` |
| Severity error | 105 | `after_sales-00b16886b9fe85f7` |
| Constraint omission | 100 | `itinerary_planning-b3ebfed1c8435fec` |

Representative format failures are truncated strings. Representative Schema
failures include duplicate array items and an out-of-enum `price_range`.
Counts are category counts, so one sample may appear in more than one row.
Machine-readable cases are stored under the ignored output
`outputs/week4/bad_cases/week4_bad_cases_v1.jsonl`.
