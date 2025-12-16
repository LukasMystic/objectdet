from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.core.config import settings
from app.core.database import db

from app.routers import auth, inference

from app.services.ml_service import ml_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    db.connect()
   
    from app.routers.auth import cleanup_unverified_users
    if db.db is not None:
        await cleanup_unverified_users(db.db)
    
    ml_service.load_models()
    
    yield
    
    print("Shutting down...")
    db.close()

app = FastAPI(title="Object Detection API", lifespan=lifespan)

origins = [
    "http://localhost:5173",   
    "http://127.0.0.1:8000",    
    "https://huggingface.co",  
    "https://bembeng123-objectdet.hf.space", 
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(inference.router, prefix="/api", tags=["Inference"])

@app.get("/health") 
def health_check():
    return {
        "status": "online",
        "database": "connected" if db.client else "disconnected",
        "config_loaded": bool(settings.SECRET_KEY),
        "docs": "/docs"
    }

if os.path.isdir("static/assets"):
    app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):

    file_path = f"static/{full_path}"
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)

    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")

    return {"message": "Frontend not found. Make sure 'static' folder exists."}