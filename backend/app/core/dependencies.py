from fastapi import Header, Depends, HTTPException, status
from jose import jwt, JWTError
from app.core.database import get_db
from app.core.config import settings

async def get_current_user(authorization: str = Header(None), db=Depends(get_db)):
    """
    Shared dependency to extract user from JWT.
    Used by both Auth and Inference routers.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="No auth token")
    
    try:
        token = authorization.replace("Bearer ", "")
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email = payload.get("sub")
        if email is None:
             raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
        
    user = await db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Convert ObjectId to string for easy handling downstream
    user["_id"] = str(user["_id"])
    return user