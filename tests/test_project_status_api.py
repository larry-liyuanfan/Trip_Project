from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from src.api.project_status import project_status


class ProjectStatusApiTests(unittest.TestCase):
    def test_status_requires_configured_versioned_document(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROJECT_STATUS_FILE", None)
            with self.assertRaises(HTTPException) as caught:
                project_status()
        self.assertEqual(caught.exception.status_code, 503)

    def test_status_returns_precomputed_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            path.write_text(json.dumps({
                "schema_version": "trip_project_status_v1",
                "week5": {"preannotated": 15166, "human_accepted": 0},
            }), encoding="utf-8")
            with patch.dict(os.environ, {"PROJECT_STATUS_FILE": str(path)}):
                response = project_status()
            self.assertEqual(response["week5"]["human_accepted"], 0)


if __name__ == "__main__":
    unittest.main()
