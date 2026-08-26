import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.data.week8_product_silver_source_v8 import (
    Week8SilverSourceAuditError,
    confirmed_visible_price_tokens,
    load_silver_source_audit_config,
    map_confirmed_tier,
    price_tokens,
    strict_ota_category,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/week8/product_silver_source_audit_v8.json"


class Week8ProductSilverSourceV8Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_silver_source_audit_config(CONFIG)

    def test_config_forbids_human_and_final_test_access(self):
        policy = self.config["policy"]
        self.assertEqual(policy["label_provenance"], "programmatic_silver")
        self.assertFalse(policy["human_annotation"])
        self.assertFalse(policy["human_review"])
        self.assertFalse(policy["human_acceptance"])
        self.assertFalse(policy["read_final_test_rows"])
        self.assertFalse(policy["read_final_test_outputs"])
        self.assertFalse(policy["metadata_is_visual_evidence"])

    def test_config_rejects_test_row_path(self):
        payload = copy.deepcopy(self.config)
        payload["source"]["identity_manifest_path"] = "outputs/week8/test/final.jsonl"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(Week8SilverSourceAuditError, "final-test"):
                load_silver_source_audit_config(path)

    def test_existing_category_wins_before_strict_expansion(self):
        category, provenance, term = strict_ota_category(
            ["Restaurants", "Art Galleries"], self.config
        )
        self.assertEqual((category, term), ("restaurant", "restaurants"))
        self.assertEqual(provenance, "existing_strict_yelp_category_silver")

    def test_strict_expansion_accepts_stadium_but_not_broad_travel(self):
        self.assertEqual(
            strict_ota_category(["Stadiums & Arenas"], self.config),
            (
                "attraction",
                "expanded_strict_yelp_category_silver",
                "stadiums & arenas",
            ),
        )
        self.assertEqual(
            strict_ota_category(["Hotels & Travel"], self.config),
            (None, None, None),
        )

    def test_numeric_amount_requires_exact_ocr_agreement_and_stays_unmapped(self):
        confirmed = confirmed_visible_price_tokens(
            "Dinner special $25", "DINNER SPECIAL $25.00"
        )
        self.assertEqual(confirmed["amounts"], ["dollar:25"])
        self.assertEqual(map_confirmed_tier(confirmed, self.config), "unknown")
        self.assertEqual(
            confirmed_visible_price_tokens("Dinner $25", "DINNER 35")["amounts"],
            [],
        )

    def test_explicit_tier_requires_caption_and_ocr_agreement(self):
        confirmed = confirmed_visible_price_tokens(
            "Affordable lunch menu", "AFFORDABLE LUNCH MENU"
        )
        self.assertEqual(confirmed["tiers"], ["affordable"])
        self.assertEqual(map_confirmed_tier(confirmed, self.config), "budget")
        self.assertEqual(price_tokens("premium menu")["amounts"], [])


if __name__ == "__main__":
    unittest.main()
