import json
import unittest
from pathlib import Path


class DockerComposeConfigTest(unittest.TestCase):
    def test_vllm_compose_allows_two_images_for_dev_multi_image_stretch(self):
        lines = Path("docker/docker-compose.yml").read_text(encoding="utf-8").splitlines()
        limit_line_index = lines.index("      - --limit-mm-per-prompt")
        limit_arg = lines[limit_line_index + 1].strip().removeprefix("- ").strip("'")

        self.assertEqual(json.loads(limit_arg)["image"], 2)


if __name__ == "__main__":
    unittest.main()
