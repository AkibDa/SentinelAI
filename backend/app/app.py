from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Import this
from .core.config import settings
from .api.routes import router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION
)

origins = [
    "http://localhost:3000",    
    "http://localhost:5173",    
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      
    allow_credentials=True,
    allow_methods=["*"],        
    allow_headers=["*"],        
)

app.include_router(router)
