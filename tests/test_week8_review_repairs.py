import copy
import json
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from src.evaluation.product_semantics import audit_product_references, product_consistency_errors
from src.inference.system_runtime import TransformersPeftBackend
from src.retrieval.week8_relevance import Week8RetrievalError, claim_final_test, validate_development_selection
from src.training.week7_data import canonical_sha256, sha256_file, write_jsonl_new
from src.training.week8_product import (
    Week8ProductError, _unknown_and_slice_metrics, run_final_test_once,
    select_prompt, summarize_product_run, validate_week8_product_lock,
)
from src.training.week8_product_sft import _inspect_split_rows, Week8ProductSFTError
from src.training.week8_product_two_stage import (
    Week8TwoStageError, _generate_evidence, evidence_messages,
    load_two_stage_config, map_evidence_to_product, validate_observable_evidence,
)
from tests.test_week8_product_sft import product_row
from tests.test_week8_product_two_stage import evidence

ROOT = Path(__file__).resolve().parents[1]


class Vector(list):
    @property
    def shape(self):
        return (len(self),)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        return Vector(value) if isinstance(item, slice) else value


class Batch(list):
    @property
    def shape(self):
        return (len(self), len(self[0]))

    def to(self, _device):
        return self


class Processor:
    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, *_args, **_kwargs):
        self.calls += 1
        return {"input_ids": Batch([Vector([1, 2])])}

    def batch_decode(self, *_args, **_kwargs):
        return ['{"ok":true}']


def loaded_backend(factory=TransformersPeftBackend.from_loaded):
    processor = Processor()
    model = types.SimpleNamespace(
        parameters=lambda: iter([types.SimpleNamespace(device="cpu")]),
        generate=lambda **_kwargs: [Vector([1, 2, 3])],
    )
    return factory(model, processor, types.SimpleNamespace(inference_mode=nullcontext))


class Week8ReviewRepairTests(unittest.TestCase):
    def test_loaded_training_backend_readiness_never_reloads_weights(self):
        backend = loaded_backend()
        self.assertEqual(backend.ready(), (True, "ok"))
        backend._processor = None
        self.assertEqual(backend.ready(), (False, "in-memory backend is incomplete"))

    def test_selection_rejects_nonfinite_or_unbounded_metrics(self):
        from tests.test_week8_product import summary
        config = json.loads((ROOT / "configs/week8/product_understanding_v1.json").read_text(encoding="utf-8"))
        for value in (float("nan"), float("inf"), -0.1, 1.1, True):
            summaries = {role: summary(0.8) for role in config["prompts"]}
            summaries["compact_field_check"]["scenarios"]["image_product_search"]["composite"] = value
            with self.subTest(value=value), self.assertRaises(Week8ProductError):
                select_prompt(config, summaries)

    def test_both_training_backends_generate_with_complete_cache_state(self):
        from src.training.week8_product_sft import _in_memory_backend as direct
        from src.training.week8_product_two_stage import _in_memory_backend as two_stage
        for factory in (direct, two_stage):
            with self.subTest(factory=factory):
                backend = loaded_backend(factory)
                result = backend.generate_with_usage([], response_format=None, max_new_tokens=4)
                self.assertEqual(result.content, '{"ok":true}')
                self.assertEqual(backend.prepared_input_cache_snapshot()["entries"], 0)

    def test_cpu_and_device_caches_bypass_mutable_http_image(self):
        backend = loaded_backend()
        backend.configure_processor_cache(2)
        backend.configure_prepared_input_cache(2)
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/changing.jpg"}}]}]
        for _ in range(2):
            backend.generate_with_usage(messages, response_format=None, max_new_tokens=4)
        self.assertEqual(backend._processor.calls, 2)
        self.assertEqual(backend.processor_cache_snapshot()["entries"], 0)
        self.assertEqual(backend.prepared_input_cache_snapshot()["entries"], 0)

    def test_generation_is_serialized_per_model_instance(self):
        backend = loaded_backend()
        active = peak = 0
        lock = threading.Lock()

        def generate(*_args, **_kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1

        with patch.object(backend, "_generate_with_usage_locked", side_effect=generate):
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: backend.generate_with_usage([], response_format=None, max_new_tokens=1), range(8)))
        self.assertEqual(peak, 1)

    def test_schema_is_in_model_visible_prompt_not_only_decoder(self):
        config = load_two_stage_config(ROOT / "configs/week8/product_two_stage_v1.json")
        config["two_stage"]["include_schema_in_prompt"] = True
        messages = evidence_messages({"image_path": "image.jpg"}, config, root=ROOT)
        text = messages[-1]["content"][-1]["text"]
        for key in ("subject_category", "uncertainty_reasons", "no_facility_evidence", "uniqueItems"):
            self.assertIn(key, text)

    def test_training_messages_support_the_same_schema_visible_contract(self):
        from src.training.week8_product_two_stage import _evidence_training_messages
        config = load_two_stage_config(ROOT / "configs/week8/product_two_stage_v1.json")
        config["two_stage"]["include_schema_in_prompt"] = True
        row = {"image_path": "data/samples/images/cafe_001.jpg", "evidence_target": evidence()}
        messages = _evidence_training_messages(row, config, root=ROOT)
        self.assertIn("uniqueItems", messages[1]["content"][-1]["text"])
        self.assertEqual(json.loads(messages[-1]["content"]), row["evidence_target"])

    def test_negative_evidence_cannot_produce_positive_labels_in_v2(self):
        config = load_two_stage_config(ROOT / "configs/week8/product_two_stage_v1.json")
        config["two_stage"]["mapping_version"] = "evidence_consistent_v2"
        value = evidence(price_text=["premium"], uncertainty_reasons=["no_style_evidence", "no_facility_evidence", "no_price_evidence"])
        mapped = map_evidence_to_product(value, config)
        self.assertEqual(mapped["style_tags"], [])
        self.assertEqual(mapped["visible_facilities"], [])
        self.assertEqual(mapped["price_range"], "unknown")
        self.assertEqual(product_consistency_errors(mapped), [])

    def test_invalid_nested_evidence_fails_with_contract_error(self):
        config = load_two_stage_config(ROOT / "configs/week8/product_two_stage_v1.json")
        for field in ("subject_category", "subject_clarity", "style_cues"):
            with self.subTest(field=field), self.assertRaises(Week8TwoStageError):
                validate_observable_evidence(evidence(**{field: [{}]}), config)

    def test_retry_includes_actual_invalid_response_and_error(self):
        config = load_two_stage_config(ROOT / "configs/week8/product_two_stage_v1.json")
        responses = iter(['{"wrong":true}', json.dumps(evidence())])
        messages = []

        class Backend:
            def generate_with_usage(self, incoming, **_kwargs):
                messages.append(incoming)
                return types.SimpleNamespace(content=next(responses), input_tokens=10, output_tokens=20)

        result = _generate_evidence(ROOT, Backend(), {"sample_id": "s1", "image_path": "image.jpg"}, config, "review")
        self.assertFalse(result["failed"])
        self.assertEqual(messages[1][-2]["content"], '{"wrong":true}')
        self.assertIn("keys changed", messages[1][-1]["content"])

    def test_unconstrained_decoding_still_validates_full_evidence_schema(self):
        config = load_two_stage_config(ROOT / "configs/week8/product_two_stage_v1.json")
        config["two_stage"]["constrained_decoding"] = False
        formats = []

        class Backend:
            def generate_with_usage(self, _messages, **kwargs):
                formats.append(kwargs["response_format"])
                return types.SimpleNamespace(content='{"wrong":true}', input_tokens=10, output_tokens=5)

        result = _generate_evidence(ROOT, Backend(), {"sample_id": "s1", "image_path": "image.jpg"}, config, "review")
        self.assertEqual(formats, [None, None])
        self.assertTrue(result["failed"])
        self.assertFalse(result["evidence_schema_pass"])

    def test_unconstrained_review_preserves_data_and_evidence_token_budget(self):
        first = json.loads((ROOT / "configs/week8/product_review_v1.json").read_text(encoding="utf-8"))
        second = json.loads((ROOT / "configs/week8/product_review_v2.json").read_text(encoding="utf-8"))
        for field in ("product_config", "release_config", "two_stage_config", "dataset_lock_sha256", "development_count", "evidence_max_new_tokens"):
            self.assertEqual(first[field], second[field], field)

    def test_reference_audit_exposes_mislabeled_metadata_and_unknown_conflict(self):
        row = product_row("development", "s1")
        row["target"]["unknown_fields"].append("business_category")
        row["target"]["inferred_attributes"] = ["风格或设施包含 Yelp 商家元数据弱银标，不作为图片直接证据"]
        row["target_provenance"]["style_tags"] = "caption_lexical_silver"
        audit = audit_product_references([row])
        self.assertEqual(audit["metadata_proxy_samples"], 1)
        self.assertEqual(audit["issue_counts"]["known_value_marked_unknown:business_category"], 1)
        self.assertEqual(audit["issue_counts"]["metadata_mislabeled_as_caption_provenance"], 1)
        self.assertFalse(audit["visual_accuracy_claim_supported"])

    def test_visual_sft_rejects_mixed_metadata_instead_of_training_it(self):
        row = product_row("train", "s1")
        row["target"]["inferred_attributes"] = ["设施来自商家元数据"]
        with self.assertRaisesRegex(Week8ProductSFTError, "metadata proxies"):
            _inspect_split_rows([row], split="train", maximum_silver_weight=0.5)

    def test_failed_or_non_object_predictions_cannot_get_unknown_credit(self):
        row = product_row("development", "s1")
        for payload, failed in (([], False), (None, False), ({"unknown_fields": [{}]}, False), (row["target"], True)):
            with self.subTest(payload=payload):
                metrics = _unknown_and_slice_metrics([row], [{"sample_id": "s1", "raw_output": json.dumps(payload), "failed": failed}])
                self.assertEqual(metrics["unknown_usage_accuracy"], 0)
                self.assertEqual(metrics["price_unknown_accuracy"], 0)

    def test_price_support_is_derived_not_hard_coded(self):
        row = product_row("development", "s1")
        row["target"]["price_range"] = "budget"
        row["target"]["unknown_fields"] = []
        metrics = _unknown_and_slice_metrics([row], [{"sample_id": "s1", "raw_output": json.dumps(row["target"]), "failed": False}])
        self.assertEqual(metrics["known_price_support"], 1)
        self.assertEqual(metrics["price_unknown_support"], 0)
        self.assertIsNone(metrics["price_unknown_accuracy"])

    def test_metrics_reject_missing_or_duplicate_samples(self):
        rows = [product_row("development", "s1"), product_row("development", "s2")]
        for records in ([], [{"sample_id": "s1"}], [{"sample_id": "s1"}, {"sample_id": "s1"}]):
            with self.assertRaisesRegex(Week8ProductError, "exactly once"):
                summarize_product_run(ROOT, rows, records)

    def test_failed_placeholder_cannot_earn_format_or_semantic_score(self):
        row = product_row("development", "s1")
        result = summarize_product_run(ROOT, [row], [{
            "sample_id": "s1", "raw_output": json.dumps(row["target"]),
            "failed": True, "latency_ms": 10,
        }])
        product = result["scenarios"]["image_product_search"]
        self.assertEqual(product["composite"], 0)
        self.assertEqual(product["aggregate"]["json_compliance"], 0)
        self.assertEqual(product["aggregate"]["schema_pass"], 0)
        self.assertEqual(result["failure_count"], 1)

    def test_json_syntax_and_operational_failure_are_reported_separately(self):
        from scripts.review_week8_product import output_diagnostics
        result = output_diagnostics([{
            "raw_output": '{}', "evidence_raw_output": '{"wrong":true}',
            "evidence_schema_pass": False, "failed": True, "attempts": [],
        }])
        self.assertEqual(result["model_json_syntax_rate"], 1)
        self.assertEqual(result["model_schema_pass_rate"], 0)
        self.assertEqual(result["internal_consistency_rate"], 0)

    def test_offline_rescore_preserves_raw_evidence_and_rejects_changed_hash(self):
        from scripts.review_week8_product import rescore_review
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config_path = folder / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            source = folder / "original"
            source.mkdir()
            identity = {"config_sha256": sha256_file(config_path), "development_sha256": "dev", "dataset_lock_sha256": "lock", "test_rows_read": False, "git_commit": "source-commit"}
            (source / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
            row = product_row("development", "s1")
            record = {"sample_id": "s1", "raw_output": json.dumps(row["target"]), "failed": True, "latency_ms": 10}
            raw_path = source / "product_release/raw_outputs.jsonl"
            write_jsonl_new(raw_path, [record])
            original_hash = sha256_file(raw_path)
            (raw_path.parent / "metrics.json").write_text(json.dumps({"raw_sha256": original_hash}), encoding="utf-8")
            inputs = ({"profiles": ["product_release"]}, {}, [row], {"lock_sha256": "lock"}, "dev")
            with patch("scripts.review_week8_product.load_review_inputs", return_value=inputs), patch("scripts.review_week8_product.TransformersPeftBackend", side_effect=AssertionError("rescore must not run inference")):
                rescore_review(ROOT, config_path, source, folder / "rescored")
                result = json.loads((folder / "rescored/summary.json").read_text(encoding="utf-8"))
                self.assertEqual(result["new_model_requests"], 0)
                self.assertEqual(result["profiles"]["product_release"]["weighted_composite"], 0)
                self.assertEqual(sha256_file(raw_path), original_hash)
                raw_path.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "generations changed"):
                    rescore_review(ROOT, config_path, source, folder / "changed")
                self.assertFalse((folder / "changed").exists())

    def test_retrieval_development_only_rejected_before_read_or_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "marker.json"
            with self.assertRaisesRegex(Week8RetrievalError, "development-only"):
                claim_final_test(marker, {"schema_version": "week8_retrieval_latency_selection_v1"})
            with self.assertRaisesRegex(Week8RetrievalError, "development-only"):
                validate_development_selection({"split": {"development_only": True}}, Path(tmp) / "absent.json", lock_dir=Path(tmp), source_hashes={})
            self.assertFalse(marker.exists())

    def test_final_rejects_invalid_lock_before_selection_or_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.training.week8_product.validate_week8_product_lock", return_value={"status": "FAIL"}):
                with self.assertRaisesRegex(Week8ProductError, "intact data lock"):
                    run_final_test_once(ROOT, ROOT / "configs/week8/product_understanding_v1.json", Path(tmp) / "dev", Path(tmp) / "final")

    def test_development_validator_never_opens_test_labels_and_checks_all_identity_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads((ROOT / "configs/week8/product_understanding_v1.json").read_text(encoding="utf-8"))
            config["dataset"].update({"output_root": "lock", "continuation_train_count": 1, "development_count": 1, "test_count": 1})
            path = root / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            lock_root = root / "lock" / config["week8"]["dataset_version"]
            identities = []
            files = {}
            for split in ("train", "development", "test"):
                row = product_row(split, split)
                row.update({"source_id": split, "image_sha256": split, "group_id": split, "constraint_template_id": None})
                relative = f"{split}/image_product_search.jsonl"
                write_jsonl_new(lock_root / relative, [row])
                files[relative] = {"count": 1, "sha256": sha256_file(lock_root / relative)}
                identities.append({key: row[key] for key in ("sample_id", "source_id", "image_sha256", "group_id", "constraint_template_id", "split")})
            write_jsonl_new(lock_root / "identity_manifest.jsonl", identities)
            files["identity_manifest.jsonl"] = {"count": 3, "sha256": sha256_file(lock_root / "identity_manifest.jsonl")}
            lock = {"config_sha256": sha256_file(path), "files": files, "dataset_version": config["week8"]["dataset_version"], "test_status": "LOCKED_UNCONSUMED"}
            lock["lock_sha256"] = canonical_sha256(lock)
            (lock_root / "dataset_lock.json").write_text(json.dumps(lock), encoding="utf-8")
            original = Path.open

            def guarded_open(self, *args, **kwargs):
                if "test" in self.parts:
                    raise AssertionError("test labels were opened")
                return original(self, *args, **kwargs)

            with patch.object(Path, "open", guarded_open):
                result = validate_week8_product_lock(root, path, include_test=False)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(result["isolation"]["dimensions"]), 5)


if __name__ == "__main__":
    unittest.main()
