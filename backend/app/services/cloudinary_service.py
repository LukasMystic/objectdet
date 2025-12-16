import cloudinary
import cloudinary.uploader
from app.core.config import settings

# Initialize Configuration
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET
)

async def upload_image(file_obj, folder="inference_images"):
    """
    Uploads a file-like object to Cloudinary using credentials from .env
    """
    try:
        response = cloudinary.uploader.upload(file_obj, folder=folder)
        return {
            "url": response.get("secure_url"),
            "public_id": response.get("public_id")
        }
    except Exception as e:
        print(f"Cloudinary upload error: {e}")
        return None