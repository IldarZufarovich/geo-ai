from __future__ import annotations
import cv2
import numpy as np
from .fractures import fracture_candidates

CLASS_NAMES = {0: 'matrix', 1: 'grain', 2: 'pore', 3: 'fracture', 4: 'cement/clay'}
COLORS = {0: (45, 52, 60), 1: (226, 185, 92), 2: (52, 152, 219), 3: (231, 76, 60), 4: (145, 95, 165)}


def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == i] = 1
    return out.astype(bool)


def _keep_elongated(mask: np.ndarray, min_length: int, min_aspect: float = 4.0) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        rw, rh = rect[1]
        long_side, short_side = max(rw, rh), max(1.0, min(rw, rh))
        if long_side >= min_length and long_side / short_side >= min_aspect:
            out[labels == i] = 1
    return out.astype(bool)


def _kmeans_material_mask(
    rgb: np.ndarray,
    k: int = 8,
    sample_side: int = 360,
    attempts: int = 3,
    max_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cluster map, RGB centers and pixel counts.

    Parameters are controlled by Analysis Detail. Higher detail
    uses a larger clustering sample and more material clusters.
    This remains the deterministic CPU fallback, not a trained
    geological segmentation model.
    """
    h, w = rgb.shape[:2]

    scale = min(
        1.0,
        float(sample_side) / max(h, w)
    )

    sample = cv2.resize(
        rgb,
        (
            max(32, int(w * scale)),
            max(32, int(h * scale))
        ),
        interpolation=cv2.INTER_AREA
    )

    sample_lab = cv2.cvtColor(
        sample,
        cv2.COLOR_RGB2LAB
    ).reshape(-1, 3).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        int(max_iter),
        0.5
    )

    _, _, centers = cv2.kmeans(
        sample_lab,
        int(k),
        None,
        criteria,
        int(attempts),
        cv2.KMEANS_PP_CENTERS
    )
    full_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    # Chunked distance evaluation keeps memory bounded for large uploads.
    labels = np.empty(len(full_lab), dtype=np.int16)
    for start in range(0, len(full_lab), 120_000):
        chunk = full_lab[start:start + 120_000]
        d2 = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + len(chunk)] = d2.argmin(axis=1)
    cluster_map = labels.reshape(h, w)
    counts = np.bincount(labels, minlength=k)
    centers_rgb = np.array([
        cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_LAB2RGB)[0, 0] for c in centers
    ]).astype(np.int16)
    return cluster_map, centers_rgb, counts


DETAIL_PROFILES = {
    "Standard": {
        "k": 8,
        "sample_side": 360,
        "attempts": 3,
        "max_iter": 25,
        "area_factor": 1.00,
    },
    "Detailed": {
        "k": 10,
        "sample_side": 520,
        "attempts": 4,
        "max_iter": 35,
        "area_factor": 0.70,
    },
    "Deep": {
        "k": 12,
        "sample_side": 720,
        "attempts": 5,
        "max_iter": 45,
        "area_factor": 0.45,
    },
}


def get_detail_profile(detail: str = "Standard") -> dict:
    """Return a defensive copy of the requested detail profile."""
    key = detail if detail in DETAIL_PROFILES else "Standard"
    return dict(DETAIL_PROFILES[key])


def demo_semantic_segmentation(
    rgb: np.ndarray,
    detail: str = "Standard"
) -> tuple[np.ndarray, dict]:
    """CPU-safe deterministic baseline.

    This is intentionally labelled as a demo fallback, not a trained geological model.
    It is tuned to make synthetic/demo samples interpretable while degrading honestly on
    unfamiliar real imagery. Validated model weights should replace this function later.
    """
    profile = get_detail_profile(detail)

    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sem = np.zeros((h, w), np.uint8)

    # Strong color cues used by the synthetic petrographic generator.
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    blue = (b > 125) & (b > 1.28 * r) & (b > 1.12 * g) & (hsv[..., 1] > 55)
    red = (r > 145) & (r > 1.22 * g) & (r > 1.18 * b) & (hsv[..., 1] > 65)
    fracture_mask, fracture_meta = fracture_candidates(rgb)

    # Material clustering: the dominant cluster is treated as matrix/background.
    clusters, centers, counts = _kmeans_material_mask(
        rgb,
        k=profile["k"],
        sample_side=profile["sample_side"],
        attempts=profile["attempts"],
        max_iter=profile["max_iter"],
    )
    background_cluster = int(np.argmax(counts))
    candidate_clusters = set(range(len(centers))) - {background_cluster}

    # Exclude obvious blue/red centers from grain candidates.
    for i, c in enumerate(centers):
        cr, cg, cb = map(float, c)
        if (cb > 1.28 * cr and cb > 1.12 * cg) or (cr > 1.22 * cg and cr > 1.18 * cb):
            candidate_clusters.discard(i)

    # Very dark/brown compact regions are treated as cement/clay; remaining non-matrix
    # material becomes a grain candidate. This is much less aggressive than raw thresholding.
    center_mean = centers.mean(axis=1)
    dark_clusters = {int(i) for i in np.where(center_mean < 112)[0] if int(i) != background_cluster}
    candidate_clusters -= dark_clusters

    grain = np.isin(clusters, list(candidate_clusters))
    cement = np.isin(clusters, list(dark_clusters))

    # Morphological cleanup removes texture speckle before instance separation.
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    grain = cv2.morphologyEx(grain.astype(np.uint8), cv2.MORPH_OPEN, kernel3)
    grain = cv2.morphologyEx(grain, cv2.MORPH_CLOSE, kernel5)
    blue_m = cv2.morphologyEx(blue.astype(np.uint8), cv2.MORPH_OPEN, kernel3)
    blue_m = cv2.morphologyEx(blue_m, cv2.MORPH_CLOSE, kernel3)
    red_m = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE, kernel3)
    cement_m = cv2.morphologyEx(cement.astype(np.uint8), cv2.MORPH_OPEN, kernel3)

    px = h * w
    area_factor = float(profile["area_factor"])

    grain = _remove_small(
        grain,
        max(
            24,
            int(max(80, px * 0.00018) * area_factor)
        )
    )

    blue = _remove_small(
        blue_m,
        max(
            8,
            int(max(20, px * 0.00004) * area_factor)
        )
    )

    red = _remove_small(
        red_m,
        max(
            8,
            int(max(18, px * 0.00003) * area_factor)
        )
    )

    red = _keep_elongated(
        red,
        min_length=max(
            20,
            int(
                max(
                    28,
                    min(h, w) * 0.055
                )
                * (
                    0.88
                    if detail == "Detailed"
                    else 0.75
                    if detail == "Deep"
                    else 1.0
                )
            )
        ),
        min_aspect=3.2
    )

    red = red | fracture_mask

    cement = _remove_small(
        cement_m,
        max(
            24,
            int(max(80, px * 0.00015) * area_factor)
        )
    )

    # Priority: matrix < grain/cement < pore < fracture.
    sem[grain] = 1
    sem[cement & ~grain] = 4
    sem[blue & ~red] = 2
    sem[red] = 3

    # A confidence score here describes demo-method confidence, not calibrated probability.
    recognized = float((sem > 0).mean())
    mean_conf = 0.74 if 0.03 <= recognized <= 0.60 else 0.55
    confidence = {
        'method': 'deterministic-material-clustering-fallback-v0.2',
        'mean_confidence': mean_conf,
        'fracture_candidates': fracture_meta,
        'recognized_fraction': recognized,
        'analysis_detail': detail,
        'detail_parameters': profile,
    }
    return sem, confidence


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), np.uint8)
    for k, c in COLORS.items():
        out[mask == k] = c
    return out


def overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.34) -> np.ndarray:
    """Overlay only predicted geological classes; preserve matrix pixels unchanged."""
    out = rgb.copy()
    col = colorize_mask(mask)
    active = mask > 0
    if np.any(active):
        blended = cv2.addWeighted(rgb, 1 - alpha, col, alpha, 0)
        out[active] = blended[active]
    return out
