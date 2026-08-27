import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from src.retrieval.clip_embeddings import CLIPImageEncoder


class CLIPReviewRepairTests(unittest.TestCase):
    def dependency_stubs(self):
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        factory, processor = Mock(), Mock()
        factory.from_pretrained.return_value = model
        modules = {
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
            "transformers": types.SimpleNamespace(CLIPModel=factory, CLIPProcessor=processor),
        }
        return modules, factory, processor

    def test_failed_processor_does_not_publish_half_loaded_model(self):
        encoder = CLIPImageEncoder(device="cpu")
        modules, factory, processor = self.dependency_stubs()
        processor.from_pretrained.side_effect = [RuntimeError("processor offline"), object()]
        with patch.dict("sys.modules", modules):
            self.assertFalse(encoder.ready()[0])
            self.assertIsNone(encoder._model)
            self.assertEqual(encoder.device, "unloaded")
            self.assertEqual(encoder.ready(), (True, "ok"))
        self.assertEqual(factory.from_pretrained.call_count, 2)

    def test_concurrent_readiness_loads_once(self):
        encoder = CLIPImageEncoder(device="cpu")
        modules, factory, _ = self.dependency_stubs()
        with patch.dict("sys.modules", modules):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: encoder.ready(), range(8)))
        self.assertEqual(results, [(True, "ok")] * 8)
        factory.from_pretrained.assert_called_once()

    def test_encoding_is_serialized_per_encoder(self):
        encoder = CLIPImageEncoder(device="cpu")
        active = peak = 0

        def encode(_paths):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            time.sleep(0.01)
            active -= 1
            return []

        with patch.object(encoder, "_encode_locked", side_effect=encode):
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: encoder.encode([]), range(8)))
        self.assertEqual(peak, 1)


if __name__ == "__main__":
    unittest.main()
