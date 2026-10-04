
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def _projection_splits(
    gray: np.ndarray,
    axis: int,
    min_gap_fraction: float = 0.012
) -> list[tuple[int, int]]:
    """
    Detect broad light gutters using grayscale projections.

    axis=0 -> vertical bands / columns
    axis=1 -> horizontal bands / rows
    """

    h, w = gray.shape

    # White / near-white pixels are likely panel gutters.
    white = gray > 238

    if axis == 0:
        score = white.mean(axis=0)
        total = w
        orthogonal = h
    else:
        score = white.mean(axis=1)
        total = h
        orthogonal = w

    # Require a substantial fraction of the line to be light.
    gap_mask = score > 0.72

    min_gap = max(
        3,
        int(total * min_gap_fraction)
    )

    gaps = []
    start = None

    for i, flag in enumerate(gap_mask):

        if flag and start is None:
            start = i

        elif not flag and start is not None:

            if i - start >= min_gap:
                gaps.append((start, i))

            start = None

    if start is not None and total - start >= min_gap:
        gaps.append((start, total))

    # Convert gutters into content intervals.
    boundaries = [0]

    for a, b in gaps:
        center = (a + b) // 2

        # Ignore page-edge whitespace.
        if center > total * 0.08 and center < total * 0.92:
            boundaries.append(center)

    boundaries.append(total)
    boundaries = sorted(set(boundaries))

    intervals = []

    min_panel = total * 0.18

    for a, b in zip(boundaries[:-1], boundaries[1:]):

        if b - a >= min_panel:
            intervals.append((a, b))

    return intervals


def _grid_candidates(rgb: np.ndarray):
    """
    Try whitespace/gutter based grid decomposition.
    """

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    rows = _projection_splits(
        gray,
        axis=1
    )

    cols = _projection_splits(
        gray,
        axis=0
    )

    return rows, cols


def _equal_grid_candidates(
    rgb: np.ndarray,
    rows: int,
    cols: int
):
    """
    Conservative fallback for obvious multi-panel collages.
    """

    h, w = rgb.shape[:2]

    result = []

    for r in range(rows):
        for c in range(cols):

            y0 = round(r * h / rows)
            y1 = round((r + 1) * h / rows)

            x0 = round(c * w / cols)
            x1 = round((c + 1) * w / cols)

            result.append(
                (x0, y0, x1, y1)
            )

    return result


def _panel_quality(crop: np.ndarray) -> float:
    """
    Lightweight image-likeness score.
    """

    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_RGB2GRAY
    )

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_RGB2HSV
    )

    std = float(gray.std())
    sat = float(
        (hsv[..., 1] > 30).mean()
    )

    dark = float(
        (gray < 230).mean()
    )

    score = (
        min(std / 55.0, 1.0) * 0.45
        + min(sat / 0.30, 1.0) * 0.30
        + min(dark / 0.55, 1.0) * 0.25
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )


def split_figure_panels(
    rgb: np.ndarray,
    parent_id: str = "FIG-001",
    max_panels: int = 12
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """
    Split a detected multi-panel geological figure/collage.

    CPU-safe deterministic fallback:
      1. whitespace/gutter projection
      2. conservative regular-grid fallback
      3. quality filtering

    Returns panel metadata and panel RGB crops.
    """

    h, w = rgb.shape[:2]

    rows, cols = _grid_candidates(rgb)

    boxes = []

    # Strongest case: real gutters create a plausible grid.
    if (
        1 <= len(rows) <= 4
        and 1 <= len(cols) <= 4
        and len(rows) * len(cols) >= 2
    ):

        for y0, y1 in rows:
            for x0, x1 in cols:
                boxes.append(
                    (x0, y0, x1, y1)
                )

    # If no useful gutters were found, inspect common
    # scientific multi-panel layouts.
    if len(boxes) < 2:

        aspect = w / max(h, 1)

        candidates = []

        if aspect >= 1.25:
            candidates.extend([
                (2, 3),
                (2, 2),
                (1, 3),
                (1, 2)
            ])
        else:
            candidates.extend([
                (3, 2),
                (2, 2),
                (3, 1),
                (2, 1)
            ])

        best = None
        best_score = -1

        for nr, nc in candidates:

            candidate_boxes = _equal_grid_candidates(
                rgb,
                nr,
                nc
            )

            qualities = []

            for x0, y0, x1, y1 in candidate_boxes:

                crop = rgb[
                    y0:y1,
                    x0:x1
                ]

                qualities.append(
                    _panel_quality(crop)
                )

            if not qualities:
                continue

            mean_q = float(
                np.mean(qualities)
            )

            accepted = sum(
                q >= 0.28
                for q in qualities
            )

            coverage = (
                accepted /
                len(qualities)
            )

            score = (
                mean_q * 0.65
                + coverage * 0.35
            )

            if (
                accepted >= 2
                and score > best_score
            ):
                best_score = score
                best = candidate_boxes

        if best is not None:
            boxes = best

    rows_out = []
    crops_out = []

    for box in boxes[:max_panels]:

        x0, y0, x1, y1 = box

        crop = rgb[
            y0:y1,
            x0:x1
        ]

        q = _panel_quality(crop)

        if q < 0.28:
            continue

        panel_no = len(rows_out) + 1

        panel_id = (
            f"{parent_id}-P{panel_no:02d}"
        )

        rows_out.append({
            "panel_id": panel_id,
            "parent_figure": parent_id,
            "x0": int(x0),
            "y0": int(y0),
            "x1": int(x1),
            "y1": int(y1),
            "width": int(x1 - x0),
            "height": int(y1 - y0),
            "panel_fraction": round(
                ((x1-x0)*(y1-y0))
                / max(h*w, 1),
                4
            ),
            "image_quality": round(
                q,
                3
            )
        })

        crops_out.append(
            crop.copy()
        )

    # If decomposition was not convincing,
    # preserve the original figure as one panel.
    if len(rows_out) < 2:

        rows_out = [{
            "panel_id": f"{parent_id}-P01",
            "parent_figure": parent_id,
            "x0": 0,
            "y0": 0,
            "x1": w,
            "y1": h,
            "width": w,
            "height": h,
            "panel_fraction": 1.0,
            "image_quality": round(
                _panel_quality(rgb),
                3
            )
        }]

        crops_out = [
            rgb.copy()
        ]

    return (
        pd.DataFrame(rows_out),
        crops_out
    )
