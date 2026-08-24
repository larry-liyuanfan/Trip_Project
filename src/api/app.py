"""FastAPI application factory for the OTA multimodal service."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api.project_status import router as project_status_router
from src.api.routes import router


def create_app() -> FastAPI:
    """Create the API application and register all business routes."""
    app = FastAPI(
        title="OTA Multimodal Search and Travel Planning System",
        version="1.0.0-rc1",
        description="VLM-based OTA multimodal search and travel planning API.",
    )
    app.include_router(router)
    app.include_router(project_status_router)
    report_dir = os.getenv("PROJECT_REPORT_DIR")
    if report_dir and Path(report_dir).is_dir():
        app.mount("/reports", StaticFiles(directory=report_dir, html=True), name="reports")
    return app


app = create_app()
