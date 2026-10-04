from __future__ import annotations
import hashlib, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageOps

ALLOWED_IMAGES={'.jpg','.jpeg','.png','.tif','.tiff','.webp'}

def sha256_file(path: str|Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def load_rgb(path: str|Path, max_side: int=1280) -> np.ndarray:
    p=Path(path)
    if p.suffix.lower() not in ALLOWED_IMAGES: raise ValueError(f"Unsupported image type: {p.suffix}")
    with Image.open(p) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        if max(im.size)>max_side:
            scale=max_side/max(im.size); im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))))
        return np.asarray(im)

def save_json(data, path):
    """Safely save JSON and create the destination directory if required."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(data, indent=2, default=conv),
        encoding="utf-8"
    )


def utc_now(): return datetime.now(timezone.utc).isoformat()
