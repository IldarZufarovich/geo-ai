from __future__ import annotations
import cv2
import numpy as np


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Conservative deskew. Returns the original image when angle evidence is weak."""
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.25 or abs(angle) > 12:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _illumination_normalize(gray: np.ndarray) -> np.ndarray:
    """Reduce shadows/uneven lighting common in smartphone document photos."""
    k = max(31, (min(gray.shape) // 18) | 1)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def preprocess_scan(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = _deskew(gray)
    illum = _illumination_normalize(gray)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(illum)
    den = cv2.fastNlMeansDenoising(clahe, None, 7, 7, 21)
    bw = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 12)
    return gray, clahe, bw


def ocr_variants(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """
    Generate OCR candidates for scans and smartphone photos.

    IMPORTANT:
    The untouched RGB image is always preserved as a candidate because
    clean/high-contrast documents may OCR better without preprocessing.
    """
    gray, clahe, bw = preprocess_scan(rgb)

    h, w = gray.shape

    # OCR benefits from sufficient character pixels,
    # but avoid excessive enlargement on old workstations.
    scale = 2.2 if max(h, w) < 1800 else (
        1.55 if max(h, w) < 2800 else 1.15
    )

    def up(img):
        return cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    up_gray = up(gray)
    up_clahe = up(clahe)

    blur = cv2.GaussianBlur(
        up_clahe,
        (0, 0),
        1.0
    )

    sharp = cv2.addWeighted(
        up_clahe,
        1.8,
        blur,
        -0.8,
        0
    )

    _, otsu = cv2.threshold(
        sharp,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        11
    )

    # Mild morphology reconnects broken character strokes.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 1)
    )

    repaired = cv2.morphologyEx(
        otsu,
        cv2.MORPH_CLOSE,
        kernel
    )

    def to_rgb(img):
        return cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

    return {
        # Critical fallback: never discard the original input.
        "original_rgb": rgb.copy(),

        # Simple grayscale candidate.
        "gray": to_rgb(gray),

        # Escalation candidates for difficult scans/photos.
        "upscaled_gray": to_rgb(up_gray),
        "illumination_clahe": to_rgb(up_clahe),
        "clahe_sharpen": to_rgb(sharp),
        "otsu": to_rgb(otsu),
        "adaptive": to_rgb(adaptive),
        "stroke_repair": to_rgb(repaired),
    }


