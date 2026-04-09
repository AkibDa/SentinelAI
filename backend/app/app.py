from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Import this
from .core.config import settings
from .api.routes import router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION
)

# --- Add CORS Middleware ---
origins = [
    "http://localhost:3000",    # React/Next.js default port
    "http://localhost:5173",    # Vite default port
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],              # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],              # Allows all headers
)
# ---------------------------

app.include_router(router)