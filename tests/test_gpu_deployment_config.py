import re
import unittest
from pathlib import Path


class GpuDeploymentConfigTests(unittest.TestCase):
    def setUp(self):
        self.compose = Path("docker/gpu/docker-compose.yml").read_text(encoding="utf-8")
        self.example = Path("docker/gpu/.env.example").read_text(encoding="utf-8")
        self.example_4b = Path("docker/gpu/.env.qwen3-vl-4b.example").read_text(encoding="utf-8")
        self.script = Path("scripts/deploy_gpu_vllm.sh").read_text(encoding="utf-8")

    def test_service_is_loopback_only_and_gpu_only(self):
        self.assertIn("${VLLM_BIND_ADDRESS:-127.0.0.1}", self.compose)
        self.assertIn("VLLM_BIND_ADDRESS=127.0.0.1", self.example)
        self.assertIn("gpus: all", self.compose)
        self.assertNotRegex(self.compose, re.compile(r'^\s*-\s*"?8001:8000', re.MULTILINE))

    def test_qwen3_vl_approved_sizes_and_compatible_vllm_are_pinned(self):
        tracked_text = "\n".join((self.compose, self.example, self.example_4b, self.script))
        self.assertIn("vllm/vllm-openai:v0.11.0", tracked_text)
        self.assertIn("Qwen/Qwen3-VL-2B-Instruct", tracked_text)
        self.assertIn("Qwen/Qwen3-VL-4B-Instruct", tracked_text)
        self.assertNotIn("Qwen3-VL-7B", tracked_text)
        self.assertNotIn("Qwen3-VL-8B", tracked_text)
        self.assertNotIn("Qwen2-VL-2B-Instruct", tracked_text)

    def test_cache_is_on_data_disk_and_no_business_services_are_defined(self):
        self.assertIn("HF_HOME=/data/huggingface", self.example)
        self.assertIn("${HF_HOME:-/data/huggingface}", self.compose)
        self.assertIn('HF_HUB_ENABLE_HF_TRANSFER: "0"', self.compose)
        self.assertIn('HF_HUB_DISABLE_XET: "1"', self.compose)
        self.assertNotRegex(self.compose, re.compile(r"^\s+(api|milvus|etcd|minio):", re.MULTILINE))

    def test_multimodal_limit_is_explicit_and_configurable(self):
        self.assertIn("VLLM_LIMIT_IMAGES:-1", self.compose)
        self.assertIn("VLLM_LIMIT_IMAGES=1", self.example)
        self.assertIn("VLLM_LIMIT_IMAGES=8", self.example_4b)

    def test_launcher_rejects_unsafe_overrides(self):
        self.assertIn('VLLM_BIND_ADDRESS:-}" != "127.0.0.1"', self.script)
        self.assertIn("Qwen/Qwen3-VL-2B-Instruct|Qwen/Qwen3-VL-4B-Instruct", self.script)
        self.assertIn('HF_HOME:-}" != /data/*', self.script)

    def test_launcher_handles_sudo_only_docker_without_changing_groups(self):
        self.assertIn("sudo -n docker info", self.script)
        self.assertIn("docker_command=(sudo -n docker)", self.script)
        self.assertNotIn("usermod", self.script)

    def test_qwen3_vl_4b_evaluation_configs_use_loopback_and_separate_outputs(self):
        week3 = Path("configs/evaluation_week3_qwen3_vl_4b_gpu.yaml").read_text(encoding="utf-8")
        week4 = Path("configs/evaluation_week4_qwen3_vl_4b_gpu.yaml").read_text(encoding="utf-8")
        model = Path("configs/model_qwen3_vl_4b_gpu.yaml").read_text(encoding="utf-8")
        self.assertIn("Qwen3-VL-4B-Instruct", model)
        self.assertIn("http://127.0.0.1:18001/v1", week3)
        self.assertIn("configs/model_qwen3_vl_4b_gpu.yaml", week3)
        self.assertIn("week3_qwen3vl4b_baseline_full_20260809_001", week4)
        self.assertIn("outputs/week4_qwen3_vl_4b", week4)


if __name__ == "__main__":
    unittest.main()
