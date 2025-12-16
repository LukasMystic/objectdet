from fastapi import APIRouter, HTTPException, Depends, status, Header, UploadFile, File
from app.models.user import (
    UserCreate, UserLogin, Token, VerifyAccount, 
    PasswordResetRequest, PasswordResetConfirm, PasswordChange, UserResponse
)
from app.core.database import get_db
from app.core.security import get_password_hash, verify_password, create_access_token
from app.services.email_service import send_verification_email, send_reset_password_email
from app.services.cloudinary_service import upload_image
from app.core.config import settings
from datetime import datetime, timedelta
from jose import jwt, JWTError
import secrets
from io import BytesIO
from bson import ObjectId  # <--- IMPORT ADDED

router = APIRouter()

# --- UTILITY: Cleanup Unverified Users ---
async def cleanup_unverified_users(db):
    """Removes users who are unverified and created > 24 hours ago"""
    cutoff = datetime.utcnow() - timedelta(hours=24)
    result = await db.users.delete_many({
        "is_verified": False,
        "created_at": {"$lt": cutoff}
    })
    if result.deleted_count > 0:
        print(f"Cleaned up {result.deleted_count} unverified users.")

# --- DEPENDENCY: Get Current User ---
async def get_current_user(authorization: str = Header(None), db=Depends(get_db)):
    """Shared dependency to extract user from JWT"""
    if not authorization:
        raise HTTPException(401, "No auth token")
    
    try:
        token = authorization.replace("Bearer ", "")
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email = payload.get("sub")
        if email is None:
             raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(401, "Invalid token")
        
    user = await db.users.find_one({"email": email})
    if not user:
        raise HTTPException(401, "User not found")
    
    user["_id"] = str(user["_id"])
    return user

# --- AUTH ENDPOINTS ---

@router.post("/register", response_model=Token)
async def register(user: UserCreate, db=Depends(get_db)):
    # 1. Lazy Cleanup of old unverified accounts
    await cleanup_unverified_users(db)

    # 2. Check existing
    existing = await db.users.find_one({"email": user.email})
    if existing:
        if not existing.get("is_verified"):
            # If exists but unverified, maybe we allow overwriting or resending?
            # For now, let's strictly say registered.
            pass 
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # 3. Create User
    verification_code = secrets.token_hex(3).upper()
    user_doc = {
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": get_password_hash(user.password),
        "is_verified": False,
        "verification_code": verification_code,
        "created_at": datetime.utcnow(),
        "profile_picture": f"https://api.dicebear.com/7.x/initials/svg?seed={user.full_name or user.email}"
    }
    await db.users.insert_one(user_doc)
    
    # 4. Send Email
    send_verification_email(user.email, verification_code)
    
    # 5. Return Token
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=Token)
async def login(user_in: UserLogin, db=Depends(get_db)):
    user = await db.users.find_one({"email": user_in.email})
    if not user or not verify_password(user_in.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    # Optional: Enforce verification before login? 
    # For now, we allow login but maybe frontend restricts access.
        
    access_token = create_access_token(data={"sub": user["email"]})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/verify")
async def verify(data: VerifyAccount, db=Depends(get_db)):
    user = await db.users.find_one({"email": data.email})
    if not user:
        raise HTTPException(404, "User not found")
    
    if user.get("verification_code") == data.code:
        await db.users.update_one({"email": data.email}, {"$set": {"is_verified": True}})
        return {"message": "Account verified successfully"}
    
    raise HTTPException(400, "Invalid verification code")

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user=Depends(get_current_user)):
    return current_user

# --- PASSWORD MANAGEMENT ---

@router.post("/forgot-password")
async def forgot_password(data: PasswordResetRequest, db=Depends(get_db)):
    user = await db.users.find_one({"email": data.email})
    if not user:
        # Don't reveal if user exists
        return {"message": "If that email exists, a reset link has been sent."}
    
    # Generate a reset token (valid for 30 mins)
    reset_token = create_access_token(
        data={"sub": user["email"], "type": "reset"}, 
        expires_delta=timedelta(minutes=30)
    )
    
    # Save token hash or just rely on stateless JWT? 
    # Stateless is fine for simple apps, but saving allows revocation.
    # For simplicity, we send the JWT directly.
    send_reset_password_email(data.email, reset_token)
    return {"message": "If that email exists, a reset link has been sent."}

@router.post("/reset-password")
async def reset_password(data: PasswordResetConfirm, db=Depends(get_db)):
    try:
        payload = jwt.decode(data.token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email = payload.get("sub")
        token_type = payload.get("type")
        
        if not email or token_type != "reset":
            raise HTTPException(400, "Invalid token")
            
        user = await db.users.find_one({"email": email})
        if not user:
            raise HTTPException(404, "User not found")
            
        # Update password
        new_hash = get_password_hash(data.new_password)
        await db.users.update_one({"email": email}, {"$set": {"hashed_password": new_hash}})
        
        return {"message": "Password updated successfully"}
        
    except JWTError:
        raise HTTPException(400, "Invalid or expired token")

@router.post("/change-password")
async def change_password(data: PasswordChange, db=Depends(get_db), current_user=Depends(get_current_user)):
    # Verify old password
    if not verify_password(data.old_password, current_user["hashed_password"]):
        raise HTTPException(400, "Incorrect old password")
    
    # Update
    new_hash = get_password_hash(data.new_password)
    # FIX: Convert string ID back to ObjectId
    await db.users.update_one(
        {"_id": ObjectId(current_user["_id"])}, 
        {"$set": {"hashed_password": new_hash}}
    )
    return {"message": "Password changed successfully"}

# --- PROFILE CUSTOMIZATION ---

@router.post("/me/avatar")
async def update_avatar(
    file: UploadFile = File(...),
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    # Upload to Cloudinary
    content = await file.read()
    upload_result = await upload_image(BytesIO(content), folder="profile_pictures")
    
    if not upload_result:
        raise HTTPException(500, "Image upload failed")
        
    # Update DB
    new_url = upload_result["url"]
    # FIX: Convert string ID back to ObjectId
    await db.users.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": {"profile_picture": new_url}}
    )
    
    return {"message": "Profile picture updated", "url": new_url}