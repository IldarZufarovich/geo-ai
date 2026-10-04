from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def _component_filter(mask: np.ndarray, min_length: int, max_width: float = 10.0) -> np.ndarray:
    """Keep thin connected networks. This is a conservative visual fracture-candidate detector."""
    labels, n = ndi.label(mask > 0)
    out = np.zeros(mask.shape, dtype=bool)
    for lab in range(1, n + 1):
        comp = labels == lab
        area = int(comp.sum())
        if area < 12:
            continue
        skel = skeletonize(comp)
        length = int(skel.sum())
        if length < min_length:
            continue
        width = area / max(length, 1)
        ys, xs = np.where(comp)
        if len(xs) == 0:
            continue
        span = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
        if width <= max_width and span >= min_length * 0.55:
            out |= comp
    return out


def fracture_candidates(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Detect obvious elongated blue/red/dark fracture candidates.

    This is intentionally conservative and is NOT a trained fracture model. It is useful
    for a portfolio demo because it highlights line-like candidates without claiming that
    every boundary is a fracture.
    """
    h, w = rgb.shape[:2]
    min_dim = min(h, w)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]

    # Colored epoxy/dye or manually highlighted fracture traces.
    blue = (b > 115) & (b > 1.20 * r) & (b > 1.08 * g) & (hsv[..., 1] > 45)
    red = (r > 135) & (r > 1.18 * g) & (r > 1.15 * b) & (hsv[..., 1] > 55)
    colored = cv2.morphologyEx((blue | red).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    colored = _component_filter(colored, max(60, int(min_dim * 0.16)), max_width=max(8.0, min_dim * 0.018))

    # Dark, thin crack-like ridges. Black-hat suppresses broad dark mineral patches.
    k = max(7, int(min_dim * 0.025))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    thr = max(18, int(np.percentile(blackhat, 91)))
    dark = blackhat >= thr
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    dark = _component_filter(dark, max(70, int(min_dim * 0.16)), max_width=max(6.0, min_dim * 0.012))

    mask = colored | dark
    labels, n = ndi.label(mask)
    lengths = []
    for lab in range(1, n + 1):
        lengths.append(int(skeletonize(labels == lab).sum()))
    return mask.astype(bool), {
        'candidate_networks': int(n),
        'skeleton_length_px': int(sum(lengths)),
        'method': 'conservative line/ridge + color fracture-candidate detector',
    }
