"""Unit tests for the fail-closed distributed Milvus v6 path."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.prepare_distributed_milvus_runtime_v6 import configure_milvus_text
from scripts.run_http_milvus_service_benchmark_v4 import _validate_external_cluster_identity
from scripts.smoke_distributed_milvus_runtime_v6 import validate_dependencies


class DistributedMilvusV6Tests(unittest.TestCase):
    def test_runtime_config_uses_remote_dependencies_without_default_secret(self) -> None:
        template = "\n".join(
            [
                "localhost:2379",
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
        )
        self.assertIn("node-a:28000", configured)
        self.assertIn("node-a:28001", configured)
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


if __name__ == "__main__":
    unittest.main()
