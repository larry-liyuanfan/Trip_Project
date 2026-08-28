import os
import unittest
from unittest.mock import Mock, patch

from src.api.routes import visual_search
from src.inference.schemas import VisualSearchRequest
from src.retrieval.query_inputs import unapplied_query_text, user_query_attributes


class QueryConstraintCoverageTests(unittest.TestCase):
    def test_applied_simple_constraints_remain_complete(self):
        for text in ("推荐便宜餐厅", "find a budget restaurant", "推荐酒店"):
            self.assertEqual(unapplied_query_text(text, user_query_attributes(text)), "")

    def test_ambiguous_categories_are_not_claimed_as_applied(self):
        text = "推荐酒店或餐厅"
        attrs = user_query_attributes(text)
        self.assertNotIn("business_category", attrs)
        remainder = unapplied_query_text(text, attrs)
        self.assertIn("酒店", remainder)
        self.assertIn("餐厅", remainder)

    def test_conflicting_price_is_preserved_after_explicit_filter(self):
        attrs = user_query_attributes("推荐奢华餐厅", {"price_range": "budget"})
        self.assertEqual(attrs, {"price_range": "budget", "business_category": "restaurant"})
        self.assertEqual(unapplied_query_text("推荐奢华餐厅", attrs), "奢华")

    def test_ambiguous_prices_are_not_discarded(self):
        text = "推荐便宜或高档餐厅"
        attrs = user_query_attributes(text)
        self.assertNotIn("price_range", attrs)
        self.assertEqual(unapplied_query_text(text, attrs), "便宜或高档")

    def test_city_substring_does_not_erase_unapplied_words(self):
        self.assertEqual(unapplied_query_text("find calm restaurant", {"city": "a", "business_category": "restaurant"}), "calm")
        self.assertEqual(unapplied_query_text("find restaurant in LA", {"city": "LA", "business_category": "restaurant"}), "")

    def test_plural_categories_apply_the_same_filters_without_substring_matches(self):
        for term, category in (("restaurants", "restaurant"), ("hotels", "hotel"), ("cafes", "restaurant"), ("museums", "attraction"), ("parks", "attraction")):
            with self.subTest(term=term):
                text = "find cheap " + term
                attrs = user_query_attributes(text)
                self.assertEqual(attrs, {"business_category": category, "price_range": "budget"})
                self.assertEqual(unapplied_query_text(text, attrs), "")
                self.assertNotIn("business_category", user_query_attributes("no " + term))
        self.assertEqual(user_query_attributes("find sparks"), {})

    def test_production_api_does_not_report_conflicting_query_as_completed(self):
        service = Mock()
        service.search.return_value = [{"business_id": "b1", "price_range": "budget"}]
        with patch.dict(os.environ, {"APP_ENV": "production"}), patch("src.api.routes.get_visual_search_service", return_value=service):
            result = visual_search(VisualSearchRequest(query_text="推荐奢华餐厅", price_range="budget", retrieval_mode="keyword"))
        self.assertEqual(result["query_status"], "PARTIAL_UNSUPPORTED_CONSTRAINTS")
        self.assertEqual(result["unapplied_query_text"], "奢华")
        self.assertEqual(len(result["results"]), 1)


if __name__ == "__main__":
    unittest.main()
