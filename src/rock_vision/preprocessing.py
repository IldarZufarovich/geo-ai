from __future__ import annotations
import cv2, numpy as np

def quality_metrics(rgb: np.ndarray) -> dict:
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    return {
        'blur_score': float(cv2.Laplacian(gray,cv2.CV_64F).var()),
        'brightness': float(gray.mean()),
        'contrast': float(gray.std()),
        'height': int(rgb.shape[0]), 'width': int(rgb.shape[1])
    }

def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB); l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(2.0,(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]),cv2.COLOR_LAB2RGB)
