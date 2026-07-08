# Failure Cases

Use this file to record systematic failures.

| experiment_id | input | expected | actual | error_type | next_action |
|---|---|---|---|---|---|
| EXP-20260706-001 | local sample image path | live VLM result | fallback response | environment_not_started | start vLLM and rerun |
| EXP-20260708-001 | two Yelp subset images | valid structured JSON from live vLLM | truncated or malformed fenced JSON; parser fell back to unstructured response | malformed_model_json | tighten prompt constraints or adjust generation limits before accepting multi-image extraction |
