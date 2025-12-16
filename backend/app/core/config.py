from pydantic_settings import BaseSettings
from pydantic import ValidationError
import os

class Settings(BaseSettings):
    # --- App Security ---
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # --- Database ---
    MONGODB_URL: str
    DB_NAME: str = "object_detection_db"

    # --- Cloudinary ---
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str

    # --- Brevo (Email) ---
    BREVO_API_KEY: str
    SENDER_EMAIL: str
    SENDER_NAME: str = "Object Detection App"

    class Config:
        env_file = ".env"
        # This allows you to use the same .env file when running from root or backend/
        env_file_encoding = 'utf-8'

# Try to load settings
try:
    settings = Settings()
except ValidationError as e:
    print("CRITICAL: Missing environment variables!")
    print(e)
    # In production, you might want to exit here, 
    # but for dev we let it pass to see the error clearly in logs