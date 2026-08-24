import hashlib
import tempfile
import unittest
from pathlib import Path

from src.training.week7_data import sha256_file


class CanonicalConfigHashingTests(unittest.TestCase):
    def test_config_hash_is_stable_across_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs" / "week7"
            config_dir.mkdir(parents=True)
            config = config_dir / "runtime.json"
            config.write_bytes(b'{\r\n  "model": "qwen3-vl"\r\n}\r\n')

            expected = hashlib.sha256(
                b'{\n  "model": "qwen3-vl"\n}\n'
            ).hexdigest()
            self.assertEqual(sha256_file(config), expected)

    def test_non_config_artifacts_keep_raw_byte_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "results.json"
            artifact.write_bytes(b'{\r\n  "status": "ok"\r\n}\r\n')

            expected = hashlib.sha256(artifact.read_bytes()).hexdigest()
            self.assertEqual(sha256_file(artifact), expected)


if __name__ == "__main__":
    unittest.main()
