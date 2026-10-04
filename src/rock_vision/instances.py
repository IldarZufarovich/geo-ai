from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def _watershed_class(binary: np.ndarray, min_distance: int, min_area: int) -> list[np.ndarray]:
    if not binary.any():
        return []
    dist = ndi.distance_transform_edt(binary)
    coords = peak_local_max(
        dist,
        min_distance=max(2, int(min_distance)),
        labels=binary,
        exclude_border=False,
        threshold_abs=2,
    )
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    if markers.max() == 0:
        markers, _ = ndi.label(binary)
    labels = watershed(-dist, markers, mask=binary)
    return [(labels == lab) for lab in range(1, int(labels.max()) + 1) if int((labels == lab).sum()) >= min_area]


def _components(binary: np.ndarray, min_area: int) -> list[np.ndarray]:
    labels, n = ndi.label(binary)
    return [(labels == lab) for lab in range(1, n + 1) if int((labels == lab).sum()) >= min_area]


def semantic_to_instances(
    sem: np.ndarray,
    min_area: int | None = None,
    detail: str = "Standard"
) -> np.ndarray:
    """Convert semantic classes to meaningful instances with class-specific rules.

    Grains/pores use marker-controlled watershed; fractures/cement use connected
    components. Area thresholds scale with image size to suppress texture speckle.
    """
    h, w = sem.shape
    px = h * w

    detail_settings = {
        "Standard": {
            "area_factor": 1.00,
            "distance_factor": 1.00,
        },
        "Detailed": {
            "area_factor": 0.70,
            "distance_factor": 0.78,
        },
        "Deep": {
            "area_factor": 0.45,
            "distance_factor": 0.60,
        },
    }

    cfg = detail_settings.get(
        detail,
        detail_settings["Standard"]
    )

    area_factor = float(
        cfg["area_factor"]
    )

    distance_factor = float(
        cfg["distance_factor"]
    )

    default_base = max(
        25,
        int(px * 0.00005)
    )

    base = (
        int(min_area)
        if min_area is not None
        else max(
            8,
            int(default_base * area_factor)
        )
    )

    inst = np.zeros_like(
        sem,
        dtype=np.int32
    )
    oid = 1

    class_rules = {
        1: (
            'watershed',
            max(
                3,
                int(
                    max(
                        8,
                        min(h, w) * 0.025
                    )
                    * distance_factor
                )
            ),
            max(
                base,
                int(
                    px
                    * 0.00020
                    * area_factor
                )
            )
        ),

        2: (
            'watershed',
            max(
                2,
                int(
                    max(
                        4,
                        min(h, w) * 0.010
                    )
                    * distance_factor
                )
            ),
            max(
                6,
                int(
                    px
                    * 0.000035
                    * area_factor
                )
            )
        ),

        3: (
            'components',
            0,
            max(
                6,
                int(
                    px
                    * 0.000025
                    * area_factor
                )
            )
        ),

        4: (
            'components',
            0,
            max(
                16,
                int(
                    px
                    * 0.00010
                    * area_factor
                )
            )
        ),
    }

    for cls, (method, distance, area) in class_rules.items():
        binary = sem == cls
        masks = _watershed_class(binary, distance, area) if method == 'watershed' else _components(binary, area)
        for m in masks:
            inst[m] = oid
            oid += 1
    return inst
