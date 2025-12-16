import cv2
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import time
import os
import math
import warnings
import sys
import io

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path(".").resolve()

MODEL_DIR = BASE_DIR / "ml_models"

DEVICE = torch.device('cpu')
torch.set_num_threads(2) 


LABEL_NAMES = {
    0: "Negative (No Object)",
    1: "Positive",
    2: "Dark",
    3: "Angle Change"
}

LABEL_COLORS = {
    "Positive": (0, 255, 0),      
    "Dark": (255, 0, 0),          
    "Angle Change": (0, 165, 255) 
}

# --- CONFIGURATION ---
HARRIS_PARAMS = {'blockSize': 2, 'ksize': 3, 'k': 0.04}
SIFT_PARAMS = {'nfeatures': 0, 'nOctaveLayers': 3, 'contrastThreshold': 0.04, 'edgeThreshold': 10, 'sigma': 1.6}

# --- OPTIMIZED MODEL DEFINITIONS ---
class SharedResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.resnet18(weights=None) 
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x): return self.backbone(x)
    
    def forward_optimized(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.backbone.fc(features)
        return features, logits

class SharedEfficientNetB0(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
    
    def forward(self, x): return self.backbone(x)

    def forward_optimized(self, x):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        logits = self.backbone.classifier(features)
        return features, logits

# --- HELPERS ---
def calculate_probabilities(model, features, prediction, num_classes):
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(features)
            if num_classes == 2:
                score = decision[0] if isinstance(decision, (list, np.ndarray)) else decision
                try: prob_pos = 1 / (1 + np.exp(-score))
                except: prob_pos = 0.0 if score < 0 else 1.0
                probabilities = np.array([1 - prob_pos, prob_pos])
            else:
                try:
                    exp_scores = np.exp(decision - np.max(decision))
                    probabilities = exp_scores / exp_scores.sum()
                except:
                    probabilities = np.zeros(num_classes)
                    probabilities[int(prediction)] = 1.0
        else:
            probabilities = np.zeros(num_classes)
            probabilities[int(prediction)] = 1.0
        
        if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)
            if np.sum(probabilities) == 0:
                probabilities = np.zeros(num_classes)
                probabilities[int(prediction)] = 1.0
            else:
                probabilities /= np.sum(probabilities)
        
        confidence = probabilities[int(prediction)]
        return confidence, probabilities
    except Exception:
        probs = np.zeros(num_classes)
        try: probs[int(prediction)] = 1.0
        except: probs[0] = 1.0
        return 1.0, probs

def process_logits(logits):
    try:
        probs = torch.softmax(logits, dim=1).detach().numpy()[0]
        if np.any(np.isnan(probs)):
             probs = np.nan_to_num(probs, nan=0.0)
             if np.sum(probs) > 0: probs /= np.sum(probs)
             else: probs[0] = 1.0 
        pred = np.argmax(probs)
        return {'prediction': int(pred), 'confidence': float(probs[pred]), 'probabilities': probs}
    except Exception as e:
        return {'prediction': 0, 'confidence': 0.0, 'probabilities': np.zeros(4)}

def non_max_suppression(boxes, scores, labels, iou_threshold=0.3):
    if len(boxes) == 0: return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / (union + 1e-6)
        inds = np.where((iou <= iou_threshold) | (labels[order[1:]] != labels[i]))[0]
        order = order[inds + 1]
    return keep

# --- MAIN SYSTEM CLASS ---
class ObjectDetectionSystem:
    def __init__(self):
        print("="*60)
        print("INITIALIZING DETECTION SYSTEM (HF SPACES READY)")
        print(f"Loading models from: {MODEL_DIR}")
        self.models_loaded = False
        self.vanilla_data = None
        self.sift = None
        self.resnet = None
        self.effnet = None
        self.distilled_svm_data = None
        self.aktp_svm_data = None
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        try:
            self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            self.has_ss = True
        except (AttributeError, ImportError):
            print("WARNING: cv2.ximgproc not available. Bounding boxes disabled.")
            self.has_ss = False

        try:
            self._load_models()
            self.models_loaded = True
            print("System Ready")
        except Exception as e:
            print(f"CRITICAL ERROR LOADING MODELS: {e}")

    def _load_models(self):
        # --- A. Vanilla SVM ---
        vanilla_path = MODEL_DIR / 'vanilla_svm_model.pkl'
        if vanilla_path.exists():
            with open(vanilla_path, 'rb') as f: self.vanilla_data = pickle.load(f)
            self.sift = cv2.SIFT_create(**SIFT_PARAMS)

        # --- B. ResNet18 (Quantized + Fallback) ---
        teacher_path = MODEL_DIR / 'teacher_resnet18_best.pth'
        if teacher_path.exists():
            try:
                model = SharedResNet18(num_classes=4)
                model.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
            except RuntimeError:
                model = SharedResNet18(num_classes=2)
                model.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
            if model:
                model.eval()
                try: self.resnet = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
                except: self.resnet = model

        # --- C. EfficientNet (Quantized + Fallback) ---
        teacher_effnet_path = MODEL_DIR / 'teacher_efficientnet-b0_best.pth'
        if teacher_effnet_path.exists():
            try:
                model = SharedEfficientNetB0(num_classes=4)
                model.load_state_dict(torch.load(teacher_effnet_path, map_location=DEVICE))
            except RuntimeError:
                model = SharedEfficientNetB0(num_classes=2)
                model.load_state_dict(torch.load(teacher_effnet_path, map_location=DEVICE))
            if model:
                model.eval()
                try: self.effnet = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
                except: self.effnet = model

        # --- D. SVM Heads ---
        distilled_path = MODEL_DIR / 'distilled_svm_model.pkl'
        if distilled_path.exists():
            with open(distilled_path, 'rb') as f: self.distilled_svm_data = pickle.load(f)

        aktp_svm_path = MODEL_DIR / 'aktp_svm_student.pkl'
        if not aktp_svm_path.exists(): aktp_svm_path = MODEL_DIR / 'aktp_svm_model.pkl'
        if aktp_svm_path.exists():
            with open(aktp_svm_path, 'rb') as f: self.aktp_svm_data = pickle.load(f)

    def _preprocess_numpy(self, image):
        try:
            img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            img_norm = (img_float - self.mean) / self.std
            img_chw = img_norm.transpose(2, 0, 1)
            tensor = torch.from_numpy(img_chw).unsqueeze(0).float()
            return tensor, img_resized
        except Exception: return None, None

    def _preprocess_batch(self, images_list):
        tensors = []
        for img in images_list:
            t, _ = self._preprocess_numpy(img)
            if t is not None: tensors.append(t)
        if not tensors: return None
        return torch.cat(tensors, dim=0)

    # --- MAIN INFERENCE METHODS ---

    def detect_from_memory(self, image):
        """Classification Only (For Legacy/Fast Path)"""
        final_results = {'image_shape': image.shape, 'detections': {}}
        
        # 1. Preprocessing (We do NOT count this towards model inference time)
        dl_tensor, resized_image = self._preprocess_numpy(image)

        # 2. Vanilla SVM (Includes SIFT extraction because SIFT is part of the model logic)
        if self.vanilla_data:
            gray_small = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            # Vanilla calculates its own total time internally
            vanilla_res, t_vanilla = self._process_vanilla_optimized(gray_small)
            if vanilla_res: 
                self._format_result(final_results, 'Vanilla SVM', vanilla_res, t_vanilla)

        # 3. Deep Learning Models (Pass 0.0 as prep time to isolate model speed)
        self._run_dl_inference(dl_tensor, final_results, 0.0) 
        
        return final_results

    def detect_objects(self, image, model_name='AKTP-SVM', max_proposals=500, conf_thresh=0.4, batch_size=32):
       
        if not self.has_ss: return {'error': 'Selective Search unavailable'}

        results = {
            'image_shape': image.shape, 
            'model': model_name, 
            'detections': []
        }
        
        # 1. Generate Proposals
        self.ss.setBaseImage(image)
        self.ss.switchToSelectiveSearchFast()
        rects = self.ss.process()
        
        proposals = []
        h_img, w_img = image.shape[:2]
        count = 0
        for x, y, w, h in rects:
            if count >= max_proposals: break
            if w < 30 or h < 30: continue
            if w > w_img*0.95 or h > h_img*0.95: continue
            proposals.append((x, y, w, h))
            count += 1
            
        if not proposals: return results

        all_boxes, all_scores, all_labels = [], [], []
        
        # Select SVM data based on requested model
        svm_data = self.aktp_svm_data if model_name == 'AKTP-SVM' else self.distilled_svm_data
        if not svm_data: return {'error': f'Model {model_name} not loaded'}

        # 2. Batch Inference
        for i in range(0, len(proposals), batch_size):
            batch_rects = proposals[i : i + batch_size]
            batch_crops = [image[y:y+h, x:x+w] for x, y, w, h in batch_rects]
            batch_tensor = self._preprocess_batch(batch_crops)
            
            if batch_tensor is None: continue
            
            with torch.no_grad():
                features = None
                if model_name == 'AKTP-SVM' and self.resnet and self.effnet:
                    res_f, _ = self.resnet.forward_optimized(batch_tensor)
                    eff_f, _ = self.effnet.forward_optimized(batch_tensor)
                    features = torch.cat([eff_f, res_f], dim=1).numpy()
                elif model_name == 'Distilled SVM' and self.resnet:
                    res_f, _ = self.resnet.forward_optimized(batch_tensor)
                    features = res_f.numpy()
                
                if features is not None:
                    scaler = svm_data['scaler']
                    clf = svm_data['svm']
                    feat_scaled = scaler.transform(features)
                    preds = clf.predict(feat_scaled)
                    
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(feat_scaled)
                        confs = np.max(probs, axis=1)
                    else:
                        decs = clf.decision_function(feat_scaled)
                        confs = 1.0 / (1.0 + np.exp(-np.abs(decs)))
                        if len(confs.shape) > 1: confs = np.max(confs, axis=1)
                    
                    for j, pred in enumerate(preds):
                        if pred != 0 and confs[j] >= conf_thresh:
                            all_boxes.append(batch_rects[j])
                            all_scores.append(float(confs[j]))
                            all_labels.append(int(pred))

        # 3. NMS
        if all_boxes:
            keep_idx = non_max_suppression(all_boxes, all_scores, all_labels)
            for idx in keep_idx:
                x, y, w, h = all_boxes[idx]
                lbl = all_labels[idx]
                results['detections'].append({
                    'box': [int(x), int(y), int(w), int(h)],
                    'label': LABEL_NAMES.get(lbl, "Unknown"),
                    'score': float(all_scores[idx])
                })

        return results

    def generate_annotated_image(self, original_image, detection_results):

        if detection_results.get('model') != 'AKTP-SVM':
            return None

        detections = detection_results.get('detections', [])
        if not detections:
            return None

        annotated_img = original_image.copy()

        for det in detections:
            x, y, w, h = det['box']
            label_text = det['label']
            score = det['score']
            
        
            color = LABEL_COLORS.get(label_text, (0, 255, 0)) # BGR format
            
       
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
            
         
            text = f"{label_text} {score:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (x, y - text_h - 10), (x + text_w, y), color, -1)
            
         
            cv2.putText(annotated_img, text, (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        is_success, buffer = cv2.imencode(".jpg", annotated_img)
        if not is_success:
            return None
        
        return io.BytesIO(buffer)

    # --- INTERNAL HELPERS ---
    def _process_vanilla_optimized(self, gray_image):
        if not self.vanilla_data or not self.sift: return None, 0.0
        try:
            t_start = time.perf_counter()
            keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)
            if descriptors is None or len(descriptors) == 0:
                return {'prediction': 0, 'confidence': 0.0, 'probabilities': {}}, (time.perf_counter() - t_start)*1000
            
            stats = [np.mean(descriptors, 0), np.std(descriptors, 0), np.max(descriptors, 0), np.min(descriptors, 0)]
            feat = np.concatenate(stats + [[len(keypoints)]])
            dst = cv2.cornerHarris(np.float32(gray_image), **HARRIS_PARAMS)
            corners = np.argwhere(dst > 0.01 * dst.max())
            feat = np.append(feat, len(corners))
            feat = feat.reshape(1, -1)
            
            scaler = self.vanilla_data['scaler']
            model = self.vanilla_data['model']
            feat_scaled = scaler.transform(feat)
            pred = model.predict(feat_scaled)[0]
            num_classes = len(model.classes_)
            conf, probs = calculate_probabilities(model, feat_scaled, pred, num_classes)
            return {'prediction': int(pred), 'confidence': float(conf), 'probabilities': probs}, (time.perf_counter() - t_start)*1000
        except Exception: return None, 0.0

    def _run_dl_inference(self, dl_tensor, results_dict, prep_time_ms):
      
        if dl_tensor is None: return
        try:
            resnet_features = None
            effnet_features = None
            
         
            t_resnet_ms = 0.0
            t_effnet_ms = 0.0
            
            with torch.no_grad():
              
                if self.resnet:
                    t0 = time.perf_counter()
                    resnet_features, resnet_logits = self.resnet.forward_optimized(dl_tensor)
                    t_resnet = time.perf_counter() - t0
                    t_resnet_ms = t_resnet * 1000
                    
                    if 'detections' in results_dict: 
                         
                         self._format_dl_output(results_dict, 'Teacher (ResNet18)', process_logits(resnet_logits), t_resnet_ms)
                    
                    if self.distilled_svm_data:
                  
                        self._run_svm_head(self.distilled_svm_data, resnet_features.numpy(), 'Distilled SVM', results_dict, base_time_ms=t_resnet_ms)

              
                if self.effnet:
                    t0 = time.perf_counter()
                    effnet_features, effnet_logits = self.effnet.forward_optimized(dl_tensor)
                    t_effnet = time.perf_counter() - t0
                    t_effnet_ms = t_effnet * 1000
                    
                    if 'detections' in results_dict:
                         self._format_dl_output(results_dict, 'Teacher (EfficientNet)', process_logits(effnet_logits), t_effnet_ms)

            
                if self.aktp_svm_data and resnet_features is not None and effnet_features is not None:
                    
                    t0_cat = time.perf_counter()
                    fused_features = torch.cat([effnet_features, resnet_features], dim=1)
                    t_cat_ms = (time.perf_counter() - t0_cat) * 1000
                    
                    total_backbone_time = t_resnet_ms + t_effnet_ms + t_cat_ms
                    
                    self._run_svm_head(self.aktp_svm_data, fused_features.numpy(), 'AKTP-SVM', results_dict, base_time_ms=total_backbone_time)
                    
        except Exception as e: print(f"DL Inference Error: {e}")

    def _run_svm_head(self, model_data, features_np, name, results_dict, base_time_ms):
        try:
           
            t0 = time.perf_counter()
            
            scaler = model_data['scaler']
            svm = model_data['svm']
     
            feat_scaled = scaler.transform(features_np)
      
            pred = svm.predict(feat_scaled)[0]
            
          
            num_classes = len(svm.classes_)
            conf, probs = calculate_probabilities(svm, feat_scaled, pred, num_classes)
            
        
            t_svm_ms = (time.perf_counter() - t0) * 1000
            
            raw_res = {'prediction': int(pred), 'confidence': float(conf), 'probabilities': probs}
            
        
            final_time = base_time_ms + t_svm_ms
            
            self._format_dl_output(results_dict, name, raw_res, final_time)
        except Exception: pass

    def _format_dl_output(self, results_dict, name, raw_res, time_ms):
        is_positive = (raw_res['prediction'] != 0)
        try: confidence = float(raw_res['confidence'])
        except: confidence = 0.0
        probs_formatted = {}
        if isinstance(raw_res['probabilities'], (np.ndarray, list)):
            for i, p in enumerate(raw_res['probabilities']):
                label_name = LABEL_NAMES.get(i, str(i))
                probs_formatted[label_name] = float(p) if not math.isnan(p) else 0.0
        results_dict['detections'][name] = {
            'prediction': int(raw_res['prediction']),
            'confidence': confidence,
            'label': "Positive" if is_positive else "Negative",
            'has_object': is_positive,
            'probabilities': probs_formatted,
            'inference_time_ms': round(time_ms, 2)
        }
    
    def _format_result(self, results_dict, name, raw_res, time_ms):
        self._format_dl_output(results_dict, name, raw_res, time_ms)