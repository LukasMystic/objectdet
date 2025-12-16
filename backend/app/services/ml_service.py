import sys
import cv2
import numpy as np
import traceback

# Import the class we just updated above
from app.services.object_detection import ObjectDetectionSystem

class MLService:
    def __init__(self):
        self.detector = None
    
    def load_models(self):
        """
        Called when the server starts up (in main.py)
        """
        print("Loading ML Models...")
        try:
            self.detector = ObjectDetectionSystem()
            print("✓ Service ready.")
        except Exception as e:
            print(f"CRITICAL ERROR LOADING MODELS: {e}")
            traceback.print_exc()

    async def predict(self, image_bytes):
        """
        Called when a user hits the /predict endpoint
        """
        if not self.detector:
            self.load_models()
        
        # Convert raw bytes (from upload) to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")

        # Run the actual detection
        results = self.detector.detect_from_memory(image)
        
        return results['detections']

# Create a single instance to be used across the app
ml_service = MLService()