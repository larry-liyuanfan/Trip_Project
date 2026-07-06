from fastapi import FastAPI

from src.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="OTA Multimodal Search and Travel Planning System",
        version="0.1.0",
        description="VLM-based OTA multimodal search and travel planning API.",
    )
    app.include_router(router)
    return app


app = create_app()

