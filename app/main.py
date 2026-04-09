from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.container import ServiceContainer
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)

    container = ServiceContainer(settings)
    await container.qdrant.ensure_collection()
    app.state.container = container

    if settings.scheduler_enabled:
        container.scheduler.start()

    yield

    await container.scheduler.stop()
    await container.qdrant.close()


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router, prefix=settings.api_prefix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=(settings.cors_allow_credentials if settings.cors_origins != ['*'] else False),
    allow_methods=['*'],
    allow_headers=['*'],
)
