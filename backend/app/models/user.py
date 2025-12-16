from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# --- Auth Models ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str = Field(alias="_id")
    email: EmailStr
    full_name: Optional[str] = None
    is_verified: bool
    profile_picture: Optional[str] = None
    
class Token(BaseModel):
    access_token: str
    token_type: str

class VerifyAccount(BaseModel):
    email: EmailStr
    code: str

# --- New Auth Features ---
class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

class UserUpdate(BaseModel):
    full_name: Optional[str] = None

# --- Inference Models ---
class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class DetectionItem(BaseModel):
    box: List[int] # [x, y, w, h]
    label: str
    score: float

class InferenceResult(BaseModel):
    model_name: str
    prediction: int  # 0 or 1
    label: str
    confidence: float
    has_object: bool
    probabilities: Dict[str, float]
    inference_time_ms: float
    # List of detections (Bounding Boxes) specifically for this model
    detections: Optional[List[DetectionItem]] = None 

class InferenceRecord(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    user_id: str
    image_url: str
    # New field to store the URL of the image with drawn bounding boxes
    annotated_image_url: Optional[str] = None 
    results: Dict[str, InferenceResult] 
    created_at: datetime = Field(default_factory=datetime.utcnow)