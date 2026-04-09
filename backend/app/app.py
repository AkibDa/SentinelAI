# backend/app.py

from fastapi import FastAPI
from app.core.config import settings
from app.api.routes import router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION
)

# Include the endpoints from the routes file
app.include_router(router)
