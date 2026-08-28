import unittest
from types import SimpleNamespace

from src.inference.processor_cache import ProcessorInputCache, processor_signature
from src.inference.visual_limits import configure_visual_pixel_limit


class VisualPixelLimitTests(unittest.TestCase):
    def fast(self):
        return SimpleNamespace(image_processor=SimpleNamespace(
            size={"shortest_edge": 65536, "longest_edge": 16777216},
            min_pixels=None, max_pixels=None, patch_size=16, merge_size=2))

    def test_fast_size_and_paired_bounds_are_all_updated(self):
        processor = self.fast()
        old_size = processor.image_processor.size
        configure_visual_pixel_limit(processor, 131072)
        image = processor.image_processor
        self.assertEqual(image.size, {"shortest_edge": 65536, "longest_edge": 131072})
        self.assertEqual((image.min_pixels, image.max_pixels), (65536, 131072))
        self.assertEqual(old_size["longest_edge"], 16777216)

    def test_limit_below_default_minimum_remains_consistent(self):
        processor = self.fast()
        configure_visual_pixel_limit(processor, 32768)
        self.assertEqual(processor.image_processor.min_pixels, 32768)
        self.assertEqual(processor.image_processor.size["shortest_edge"], 32768)

    def test_legacy_processor_remains_supported(self):
        processor = SimpleNamespace(image_processor=SimpleNamespace(min_pixels=65536, max_pixels=1048576))
        configure_visual_pixel_limit(processor, 32768)
        self.assertEqual(processor.image_processor.min_pixels, 32768)
        self.assertEqual(processor.image_processor.max_pixels, 32768)

    def test_none_is_a_noop_for_frozen_v9(self):
        processor = self.fast()
        original = processor_signature(processor)
        configure_visual_pixel_limit(processor, None)
        self.assertEqual(processor_signature(processor), original)
        configure_visual_pixel_limit(object(), None)

    def test_invalid_limit_and_unsupported_processor_fail(self):
        for value in (True, 0, -1, 3.5, 100):
            with self.subTest(value=value), self.assertRaises(ValueError):
                configure_visual_pixel_limit(self.fast(), value)
        with self.assertRaises(ValueError):
            configure_visual_pixel_limit(object(), 65536)
        with self.assertRaises(ValueError):
            configure_visual_pixel_limit(SimpleNamespace(image_processor=object()), 65536)

    def test_processor_cache_identity_changes_with_effective_bound(self):
        processor = self.fast()
        before = ProcessorInputCache.key([], processor_signature(processor))
        configure_visual_pixel_limit(processor, 131072)
        after = ProcessorInputCache.key([], processor_signature(processor))
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
