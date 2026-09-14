"""Record CPU-only scorer audit verification; outputs are append-only."""

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.relevance_evidence import canonical_json_sha256, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    commands = [
        ("full_cpu_unittest", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]),
        ("tripctl_validate", [sys.executable, "scripts/tripctl.py", "validate"]),
        ("formal_package_read_only", [sys.executable, "scripts/verify_final_delivery.py", str(args.release_dir)]),
        ("compose_config_only", ["docker", "compose", "-f", "docker/system/docker-compose.yml", "--env-file", "docker/system/.env.example", "config", "--quiet"]),
        ("diff_check", ["git", "diff", "--check"]),
    ]
    report = {"schema_version": "search_scorer_v2_cpu_verification", "checks": [],
              "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "hardware": {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(), "gpu_used": False}}
    for name, command in commands:
        print(f"Running {name}", flush=True)
        start = time.monotonic()
        result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace")
        log = result.stdout + "\n" + result.stderr
        path = args.output_dir / f"{name}.log"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(log)
        display = ["python" if token == sys.executable else "<formal-package>" if token == str(args.release_dir) else token for token in command]
        item = {"name": name, "command": display, "exit_code": result.returncode,
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "elapsed_seconds": time.monotonic() - start, "log_sha256": file_sha256(path)}
        if name == "full_cpu_unittest":
            count = re.search(r"Ran (\d+) tests", log)
            skipped = re.search(r"skipped=(\d+)", log)
            item["test_count"] = int(count[1]) if count else None
            item["skipped"] = int(skipped[1]) if skipped else 0
            item["skip_reasons"] = re.findall(r"\.\.\. skipped (.+)", log)
        if "WARNING" in log:
            item["warnings_present"] = True
        report["checks"].append(item)
        print(f"{name}: {item['status']}", flush=True)
    report["status"] = "PASS" if all(item["exit_code"] == 0 for item in report["checks"]) else "FAIL"
    report["content_sha256"] = canonical_json_sha256(report)
    with (args.output_dir / "verification.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
