from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from app.services.object_detection import ObjectDetectionSystem
from app.services.cloudinary_service import upload_image
from app.core.database import db
from app.routers.auth import get_current_user
from app.models.user import InferenceRecord, InferenceResult, DetectionItem
import numpy as np
import cv2
import io
import time

router = APIRouter()
detector = ObjectDetectionSystem()

@router.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    current_user: dict = Depends(get_current_user)
):
    
    # --- 1. PREPARE IMAGE ---
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process image file")
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    # --- 2. UPLOAD ORIGINAL ---
    is_success, buffer = cv2.imencode(".jpg", image)
    if not is_success:
        raise HTTPException(status_code=500, detail="Could not encode image")
    
    file_obj = io.BytesIO(buffer)
    upload_result = await upload_image(file_obj, folder="inference_original")
    
    if not upload_result:
        raise HTTPException(status_code=500, detail="Failed to upload original image")
    
    original_image_url = upload_result['url']

    # --- 3. RUN CLASSIFICATION (ALL MODELS) ---
    classification_results = detector.detect_from_memory(image)
    
    final_results_dict = {}
    annotated_image_url = None
    
    if 'detections' in classification_results:
        for model_name, res in classification_results['detections'].items():

            final_results_dict[model_name] = InferenceResult(
                model_name=model_name,
                prediction=int(res['prediction']),
                label=str(res['label']),
                confidence=float(res['confidence']),
                has_object=bool(res['has_object']),
                probabilities=res['probabilities'],
                inference_time_ms=float(res['inference_time_ms']),
                detections=[] 
            )
    aktp_key = 'AKTP-SVM'
    
    if aktp_key in final_results_dict and final_results_dict[aktp_key].has_object:
        
        height, width = image.shape[:2]
        max_dim = 400 
        scale = 1.0
        
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_w = int(width * scale)
            new_h = int(height * scale)
            detection_image = cv2.resize(image, (new_w, new_h))
        else:
            detection_image = image.copy()

        # Run Selective Search + Detection Pipeline
        box_results = detector.detect_objects(detection_image, model_name=aktp_key, max_proposals=100)
        
        if 'detections' in box_results and box_results['detections']:
            detections_list = box_results['detections']
            
            # Sort by confidence score (descending) and take ONLY the top 1
            detections_list.sort(key=lambda x: x['score'], reverse=True)
            detections_list = detections_list[:1]

            formatted_detections = []
            for d in detections_list:
                x, y, w, h = d['box']
                if scale != 1.0:
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)

                formatted_detections.append(DetectionItem(
                    box=[x, y, w, h],
                    label=d['label'],
                    score=d['score']
                ))
            
            # Update the existing AKTP result in our dictionary
            final_results_dict[aktp_key].detections = formatted_detections
            
            # Update box_results with the filtered list so the image only shows the single best box
            box_results['detections'] = detections_list
            box_results['model'] = aktp_key 
            
            annotated_buffer = detector.generate_annotated_image(detection_image, box_results)
            
            if annotated_buffer:
                annotated_upload = await upload_image(annotated_buffer, folder="inference_annotated")
                if annotated_upload:
                    annotated_image_url = annotated_upload['url']

    # --- 5. SAVE TO DATABASE ---
    record = InferenceRecord(
        user_id=str(current_user["_id"]),
        image_url=original_image_url,
        annotated_image_url=annotated_image_url,
        results=final_results_dict, 
    )
    
    new_record = await db.db["history"].insert_one(record.model_dump(by_alias=True, exclude=["id"]))
    
    # --- 6. RETURN RESPONSE ---
    # We must serialize the Pydantic models to dicts for the JSON response
    response_data = {
        "results": {k: v.model_dump() for k, v in final_results_dict.items()},
        "original_image_url": original_image_url,
        "annotated_image_url": annotated_image_url,
        "record_id": str(new_record.inserted_id)
    }
    
    return JSONResponse(content=response_data)

@router.get("/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    """Fetch user's inference history"""
    cursor = db.db["history"].find({"user_id": str(current_user["_id"])}).sort("created_at", -1)
    history = await cursor.to_list(length=100)
    
    for rec in history:
        rec["_id"] = str(rec["_id"])
        
    return history