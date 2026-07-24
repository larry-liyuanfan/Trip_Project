import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.build_week3_candidate_manifests import (
    collect_synthetic_after_sales_candidates,
)

from src.evaluation.candidates import (
    CandidateDeduplicator,
    classify_after_sales_issue,
    classify_product_coverage,
    image_fingerprints,
    render_synthetic_evidence,
    render_synthetic_evidence_image,
    render_synthetic_visual_evidence,
    retain_best_group_row,
    synthetic_evidence_template_name,
)


class EvaluationCandidateTest(unittest.TestCase):
    def test_image_fingerprints_are_derived_from_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "sample.png"
            Image.new("RGB", (12, 12), color=(20, 40, 60)).save(image_path)

            first = image_fingerprints(root, "sample.png")
            image_path.write_bytes(image_path.read_bytes() + b"trailing")
            second = image_fingerprints(root, "sample.png")

            self.assertNotEqual(first.sha256, second.sha256)
            self.assertEqual(len(first.perceptual_hash), 16)
            self.assertEqual(first.perceptual_hash, second.perceptual_hash)

    def test_deduplicator_rejects_cross_scene_group_and_near_duplicate(self):
        deduplicator = CandidateDeduplicator(max_perceptual_distance=4)
        self.assertTrue(
            deduplicator.accept(
                source_id="source-a",
                group_id="group-a",
                image_hashes=["a" * 64],
                perceptual_hashes=["0000000000000000"],
            )
        )
        self.assertFalse(
            deduplicator.accept(
                source_id="source-b",
                group_id="group-a",
                image_hashes=["b" * 64],
                perceptual_hashes=["ffffffffffffffff"],
            )
        )
        self.assertFalse(
            deduplicator.accept(
                source_id="source-c",
                group_id="group-c",
                image_hashes=["c" * 64],
                perceptual_hashes=["0000000000000001"],
            )
        )

    def test_product_coverage_uses_explicit_ota_categories(self):
        self.assertEqual(classify_product_coverage(["Hotels", "Travel Services"]), "hotel")
        self.assertEqual(classify_product_coverage(["Hotels", "Restaurants"]), "hotel")
        self.assertEqual(classify_product_coverage(["Museums", "Arts & Entertainment"]), "attraction")
        self.assertEqual(classify_product_coverage(["Museums", "Restaurants"]), "attraction")
        self.assertEqual(classify_product_coverage(["Restaurants", "Seafood"]), "restaurant")
        self.assertIsNone(classify_product_coverage(["Dentists", "Health & Medical"]))

    def test_after_sales_issue_classifier_is_conservative_and_deterministic(self):
        self.assertEqual(
            classify_after_sales_issue("The bathroom had mold and dirty sheets."),
            "hygiene_stain",
        )
        self.assertEqual(
            classify_after_sales_issue("The elevator was broken and the door was damaged."),
            "facility_damage",
        )
        self.assertEqual(
            classify_after_sales_issue("The museum was closed and our ticket was cancelled."),
            "attraction_closure",
        )
        self.assertEqual(
            classify_after_sales_issue("Our flight was delayed for four hours."),
            "transport_delay",
        )
        self.assertIsNone(classify_after_sales_issue("The food was fine."))

    def test_synthetic_evidence_is_reproducible_and_visually_distinct(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.png"
            second = root / "second.png"
            render_synthetic_evidence(first, issue_type="transport_delay", index=1)
            render_synthetic_evidence(second, issue_type="transport_delay", index=2)

            first_fingerprint = image_fingerprints(root, "first.png")
            second_fingerprint = image_fingerprints(root, "second.png")
            self.assertNotEqual(first_fingerprint.sha256, second_fingerprint.sha256)
            self.assertGreater(
                (int(first_fingerprint.perceptual_hash, 16) ^ int(second_fingerprint.perceptual_hash, 16)).bit_count(),
                4,
            )

    def test_visual_synthetic_hygiene_and_damage_are_reproducible_and_distinct(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = []
            for issue_type in ("hygiene_stain", "facility_damage"):
                for index in range(4):
                    path = root / f"{issue_type}_{index}.png"
                    render_synthetic_visual_evidence(
                        path,
                        issue_type=issue_type,
                        index=index,
                    )
                    paths.append(path)
            fingerprints = [
                image_fingerprints(root, path.name) for path in paths
            ]
            self.assertEqual(len({item.sha256 for item in fingerprints}), len(paths))
            for index, first in enumerate(fingerprints):
                for second in fingerprints[index + 1 :]:
                    distance = (
                        int(first.perceptual_hash, 16)
                        ^ int(second.perceptual_hash, 16)
                    ).bit_count()
                    self.assertGreater(distance, 4)

            repeated = root / "repeated.png"
            render_synthetic_visual_evidence(
                repeated,
                issue_type="hygiene_stain",
                index=0,
            )
            self.assertEqual(repeated.read_bytes(), paths[0].read_bytes())

    def test_after_sales_candidate_pool_covers_all_four_gold_types(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = {
                "sampling": {
                    "quotas": {
                        "hygiene_stain": 2,
                        "facility_damage": 2,
                        "attraction_closure": 2,
                        "transport_delay": 2,
                    }
                }
            }
            candidates = collect_synthetic_after_sales_candidates(
                root=root,
                settings=settings,
                recipe_version="fixture-v3",
                deduplicator=CandidateDeduplicator(max_perceptual_distance=4),
            )

            self.assertEqual(len(candidates), 8)
            self.assertEqual(
                {item["coverage_group"] for item in candidates},
                {
                    "hygiene_stain",
                    "facility_damage",
                    "attraction_closure",
                    "transport_delay",
                },
            )
            self.assertTrue(
                all(item["provenance"]["pii_review_status"] == "not_applicable" for item in candidates)
            )

    def test_synthetic_evidence_v2_cycles_four_templates(self):
        expected = [
            "official_notice",
            "booking_status",
            "app_notification",
            "ticket_status",
        ]

        actual = [
            synthetic_evidence_template_name("attraction_closure", index)
            for index in range(4)
        ]

        self.assertEqual(actual, expected)

    def test_synthetic_evidence_v2_is_deterministic_and_text_stays_in_safe_bounds(self):
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first = Path(first_directory) / "card.png"
            second = Path(second_directory) / "card.png"

            first_audit = render_synthetic_evidence(
                first,
                issue_type="transport_delay",
                index=3,
            )
            second_audit = render_synthetic_evidence(
                second,
                issue_type="transport_delay",
                index=3,
            )

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_audit.template_name, "ticket_status")
            self.assertEqual(first_audit, second_audit)
            self.assertEqual(len(first_audit.text_boxes), 3)
            self.assertTrue(
                all(40 <= left < right <= 920 for left, _, right, _ in first_audit.text_boxes)
            )
            self.assertTrue(
                all(40 <= top < bottom <= 600 for _, top, _, bottom in first_audit.text_boxes)
            )
            with Image.open(first) as image:
                self.assertEqual(image.size, (960, 640))
                self.assertEqual(image.mode, "RGB")

    def test_synthetic_evidence_v2_examples_have_distinct_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            digests = set()
            for issue_type in ("attraction_closure", "transport_delay"):
                for index in range(4):
                    path = root / f"{issue_type}_{index}.png"
                    render_synthetic_evidence(path, issue_type=issue_type, index=index)
                    digests.add(hashlib.sha256(path.read_bytes()).hexdigest())

            self.assertEqual(len(digests), 8)

    def test_synthetic_evidence_v2_text_rows_do_not_overlap(self):
        for issue_type in ("attraction_closure", "transport_delay"):
            for index in range(4):
                _, audit = render_synthetic_evidence_image(
                    issue_type=issue_type,
                    index=index,
                )
                heading, status, detail = audit.text_boxes
                self.assertLessEqual(heading[3], status[1])
                self.assertLessEqual(status[3], detail[1])

    def test_app_notification_detail_stays_inside_notification_card(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "card.png"

            audit = render_synthetic_evidence(
                path,
                issue_type="attraction_closure",
                index=2,
            )

            detail_left, _, detail_right, _ = audit.text_boxes[2]
            self.assertGreaterEqual(detail_left, 202)
            self.assertLessEqual(detail_right, 738)

    def test_synthetic_evidence_can_be_planned_in_memory_without_a_file(self):
        image, audit = render_synthetic_evidence_image(
            issue_type="transport_delay",
            index=0,
        )
        output = io.BytesIO()

        image.save(output, format="PNG", optimize=False)

        self.assertGreater(len(output.getvalue()), 0)
        self.assertEqual(image.size, (960, 640))
        self.assertEqual(audit.template_name, "official_notice")

    def test_synthetic_evidence_v2_supports_configured_perceptual_dedup_quotas(self):
        deduplicator = CandidateDeduplicator(max_perceptual_distance=4)
        accepted_counts = {}
        for issue_type in ("attraction_closure", "transport_delay"):
            accepted = 0
            for index in range(40):
                image, _ = render_synthetic_evidence_image(
                    issue_type=issue_type,
                    index=index,
                )
                grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                pixels = list(grayscale.get_flattened_data())
                bits = 0
                for row in range(8):
                    offset = row * 9
                    for column in range(8):
                        bits = (bits << 1) | int(
                            pixels[offset + column] > pixels[offset + column + 1]
                        )
                perceptual_hash = f"{bits:016x}"
                image_hash = hashlib.sha256(image.tobytes()).hexdigest()
                if deduplicator.accept(
                    source_id=f"synthetic:{issue_type}:{index:04d}",
                    group_id=f"synthetic-event:{issue_type}:{index:04d}",
                    image_hashes=[image_hash],
                    perceptual_hashes=[perceptual_hash],
                ):
                    accepted += 1
            accepted_counts[issue_type] = accepted

        self.assertEqual(
            accepted_counts,
            {"attraction_closure": 40, "transport_delay": 40},
        )

    def test_grouped_photo_selection_retains_only_best_rank_per_business(self):
        grouped = {}
        retain_best_group_row(grouped, group_id="business-a", rank=20, row={"photo_id": "late"})
        retain_best_group_row(grouped, group_id="business-a", rank=5, row={"photo_id": "best"})
        retain_best_group_row(grouped, group_id="business-b", rank=8, row={"photo_id": "other"})

        self.assertEqual(grouped["business-a"], (5, {"photo_id": "best"}))
        self.assertEqual(len(grouped), 2)


if __name__ == "__main__":
    unittest.main()
