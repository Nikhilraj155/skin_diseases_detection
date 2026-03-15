from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.history import router as history_router
from app.api.routes.prediction import router as prediction_router
from app.core.config import get_settings
from app.core.database import close_mongo_connection, connect_to_mongo
from app.services.inference_service import inference_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    await connect_to_mongo()
    inference_service.load()
    yield
    await close_mongo_connection()


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(prediction_router, prefix=settings.api_prefix)
app.include_router(history_router, prefix=settings.api_prefix)
