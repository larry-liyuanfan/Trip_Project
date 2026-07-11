"""FastAPI application factory for the OTA multimodal service."""

from fastapi import FastAPI

from src.api.routes import router


def create_app() -> FastAPI:
    """Create the API application and register all business routes."""
    app = FastAPI(
        title="OTA Multimodal Search and Travel Planning System",
        version="0.1.0",
        description="VLM-based OTA multimodal search and travel planning API.",
    )
    app.include_router(router)
    return app


app = create_app()
