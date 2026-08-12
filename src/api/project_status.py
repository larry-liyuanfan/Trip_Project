"""Read-only project delivery status for the lightweight Aliyun display host."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.get("/v1/project-status")
def project_status() -> dict[str, Any]:
    """Return a versioned, precomputed status document without GPU inference."""
    configured = os.getenv("PROJECT_STATUS_FILE")
    if not configured:
        raise HTTPException(status_code=503, detail="project status is not configured")
    path = Path(configured)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="project status is unavailable") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("schema_version"), str):
        raise HTTPException(status_code=503, detail="project status document is invalid")
    return payload
