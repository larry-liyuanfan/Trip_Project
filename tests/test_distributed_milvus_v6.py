"""Unit tests for the fail-closed distributed Milvus v6 path."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from scripts.prepare_distributed_milvus_runtime_v6 import configure_milvus_text
from scripts.run_distributed_milvus_cluster_v6 import (
    build_minio_environment,
    candidate_port_bases,
    find_local_port_base,
    prepare_node,
    serve_milvus,
)
from scripts.run_http_milvus_service_benchmark_v4 import _validate_external_cluster_identity
from scripts.smoke_distributed_milvus_runtime_v6 import validate_dependencies


class DistributedMilvusV6Tests(unittest.TestCase):
    def test_runtime_config_uses_remote_dependencies_without_default_secret(self) -> None:
        template = "\n".join(
            [
                "localhost:2379",
                "    embed: true # Whether to enable embedded Etcd (an in-process EtcdServer).",
                "  address: localhost:9000",
                "  port: 9000 # Port of MinIO or S3 service.",
                "  accessKeyID: " + "minio" + "admin",
                "  secretAccessKey: " + "minio" + "admin",
                "  type: default",
                "  storageType: local # please adjust in embedded Milvus: local, available values are [local, remote, opendal], value minio is deprecated, use remote instead",
                "/var/lib/milvus/data/",
                "/var/lib/milvus/rdb_data",
                "/tmp/milvus_access",
                "  port: 22125 # TCP port of rootCoord",
                "  port: 19530 # TCP port of proxy",
                "  internalPort: 19529",
                "  port: 19531 # TCP port of queryCoord",
                "  port: 21123 # TCP port of queryNode",
                "  port: 13333 # TCP port of dataCoord",
                "  port: 21124 # TCP port of dataNode",
                "  port: 22222 # TCP port of streamingNode",
                "  minSegmentSizeToEnableIndex: 1024",
                "    enabled: true # Whether to enable the http server",
                "",
            ]
        )
        configured = configure_milvus_text(
            template,
            control_node="node-a",
            port_base=28000,
            access_key="trip0123456789abcd",
            secret_key="s" * 40,
            output_dir=Path("/tmp/trip-distributed-milvus-1/runtime"),
            component_port_base=28100,
        )
        self.assertIn("node-a:28000", configured)
        self.assertIn("embed: false", configured)
        self.assertNotIn("embed: true", configured)
        self.assertIn("node-a:28001", configured)
        self.assertIn("port: 28104 # TCP port of proxy", configured)
        self.assertIn("storageType: remote", configured)
        self.assertIn("type: woodpecker", configured)
        self.assertNotIn("minio" + "admin", configured)
        self.assertNotIn("port: 19530", configured)

    def test_external_identity_requires_two_nodes_and_exact_roles(self) -> None:
        expected_server = {
            "version": "2.6.18",
            "package_sha256": "a" * 64,
            "multi_node_distributed_cluster": True,
        }
        identity = {
            "schema_version": "distributed_milvus_cluster_identity_v6",
            "status": "READY",
            "nodes": ["node-a", "node-b"],
            "roles": {
                "mixcoord": "node-a",
                "proxy": "node-a",
                "querynode": "node-b",
                "datanode": "node-b",
                "streamingnode": "node-b",
            },
            "milvus_server": expected_server,
        }
        _validate_external_cluster_identity(identity, {"milvus_server": expected_server})
        identity["nodes"] = ["node-a"]
        with self.assertRaisesRegex(ValueError, "at least two nodes"):
            _validate_external_cluster_identity(identity, {"milvus_server": expected_server})

    def test_smoke_dependency_validation_is_fail_closed(self) -> None:
        config = {
            "performance": {
                "milvus_server": {
                    "multi_node_distributed_cluster": False,
                    "package_sha256": "a" * 64,
                },
                "dependencies": {},
            }
        }
        args = type("Args", (), {})()
        with self.assertRaisesRegex(ValueError, "locked distributed"):
            validate_dependencies(args, config)

    @mock.patch("scripts.run_distributed_milvus_cluster_v6.prepare_runtime")
    def test_worker_preparation_creates_separate_query_streaming_and_data_runtimes(
        self,
        prepare_runtime: mock.Mock,
    ) -> None:
        output_dir = Path("/tmp/trip-distributed-milvus-test")
        prepare_node(
            rpm=Path("milvus.rpm"),
            output_dir=output_dir,
            control_node="node-a",
            port_base=28000,
            expected_rpm_sha256="a" * 64,
            access_key="trip0123456789abcd",
            secret_key="s" * 40,
            placement="worker",
        )
        self.assertEqual(prepare_runtime.call_count, 2)
        calls = prepare_runtime.call_args_list
        self.assertEqual(calls[0].kwargs["output_dir"], output_dir / "runtime-query-streaming")
        self.assertEqual(calls[0].kwargs["component_port_base"], 28000)
        self.assertEqual(calls[1].kwargs["output_dir"], output_dir / "runtime-data")
        self.assertEqual(calls[1].kwargs["component_port_base"], 28020)

    def test_unknown_placements_are_rejected(self) -> None:
        common = {
            "rpm": Path("milvus.rpm"),
            "output_dir": Path("/tmp/trip-distributed-milvus-test"),
            "control_node": "node-a",
            "port_base": 28000,
            "expected_rpm_sha256": "a" * 64,
            "access_key": "trip0123456789abcd",
            "secret_key": "s" * 40,
        }
        with self.assertRaisesRegex(ValueError, "unsupported node placement"):
            prepare_node(**common, placement="invalid")
        with self.assertRaisesRegex(ValueError, "unsupported Milvus placement"):
            serve_milvus(Path("/tmp/runtime"), "invalid", 28013)

    def test_minio_uses_job_local_config_without_replacing_home(self) -> None:
        base_env = {"HOME": "/home/yzhang3504", "EXISTING": "value"}
        config_dir = Path.cwd() / "trip-minio-config"
        env = build_minio_environment(
            base_env,
            access_key="trip0123456789abcd",
            secret_key="s" * 40,
            config_dir=config_dir,
        )
        self.assertEqual(env["HOME"], base_env["HOME"])
        self.assertEqual(env["MINIO_CONFIG_DIR"], str(config_dir))
        self.assertEqual(env["MINIO_ROOT_USER"], "trip0123456789abcd")
        self.assertNotIn("MINIO_CONFIG_DIR", base_env)

    def test_candidate_port_blocks_are_deterministic_bounded_and_non_overlapping(self) -> None:
        candidates = candidate_port_bases("30004341")
        self.assertEqual(candidates, candidate_port_bases("30004341"))
        self.assertEqual(len(candidates), len(set(candidates)))
        self.assertGreaterEqual(min(candidates), 24000)
        self.assertLessEqual(max(candidates) + 53, 65535)
        ordered = sorted(candidates)
        self.assertTrue(
            all(right - left >= 64 for left, right in zip(ordered, ordered[1:]))
        )

    @mock.patch("scripts.run_distributed_milvus_cluster_v6.check_port_blocks")
    def test_local_port_selection_skips_a_colliding_block(self, check_blocks: mock.Mock) -> None:
        check_blocks.side_effect = [OSError("occupied"), None]
        candidates = candidate_port_bases("30004341")
        selected = find_local_port_base("30004341", (0, 20, 40))
        self.assertEqual(selected, candidates[1])
        self.assertEqual(check_blocks.call_args_list[0].args, (candidates[0], (0, 20, 40)))


if __name__ == "__main__":
    unittest.main()
