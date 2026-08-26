import unittest

from src.inference.processor_cache import ProcessorInputCache, processor_signature


class ProcessorInputCacheTests(unittest.TestCase):
    def test_disabled_cache_never_retains_values(self) -> None:
        cache = ProcessorInputCache()
        cache.put("key", {"input_ids": object()})
        self.assertIsNone(cache.get("key"))
        self.assertEqual(cache.snapshot()["entries"], 0)

    def test_lru_cache_reuses_and_evicts_bounded_entries(self) -> None:
        first = object()
        cache = ProcessorInputCache(max_entries=2)
        cache.put("first", {"input_ids": first})
        cache.put("second", {"input_ids": object()})
        self.assertIs(cache.get("first")["input_ids"], first)
        cache.put("third", {"input_ids": object()})
        self.assertIsNone(cache.get("second"))
        self.assertEqual(cache.snapshot()["entries"], 2)
        self.assertEqual(cache.snapshot()["hits"], 1)

    def test_key_changes_with_visual_processor_settings(self) -> None:
        messages = [{"role": "user", "content": "same image"}]
        low = ProcessorInputCache.key(messages, {"max_pixels": 100})
        high = ProcessorInputCache.key(messages, {"max_pixels": 200})
        self.assertNotEqual(low, high)
        self.assertEqual(
            low,
            ProcessorInputCache.key(messages, {"max_pixels": 100}),
        )

    def test_clear_can_reconfigure_capacity_and_reset_metrics(self) -> None:
        cache = ProcessorInputCache(max_entries=1)
        cache.put("first", {"input_ids": object()})
        self.assertIsNotNone(cache.get("first"))
        cache.clear(max_entries=3)
        self.assertEqual(
            cache.snapshot(),
            {
                "max_entries": 3,
                "entries": 0,
                "hits": 0,
                "misses": 0,
                "hit_rate": None,
            },
        )

    def test_processor_signature_tracks_image_parameters(self) -> None:
        class ImageProcessor:
            max_pixels = 1024
            min_pixels = 64
            size = {"longest_edge": 1024}
            do_resize = True

        class Processor:
            image_processor = ImageProcessor()

        signature = processor_signature(Processor())
        self.assertEqual(signature["max_pixels"], 1024)
        self.assertEqual(signature["min_pixels"], 64)
        self.assertTrue(signature["do_resize"])


if __name__ == "__main__":
    unittest.main()
