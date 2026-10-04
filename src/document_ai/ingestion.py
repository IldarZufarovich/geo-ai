from __future__ import annotations
from pathlib import Path
import pymupdf as fitz, numpy as np
from PIL import Image
from src.common.io import load_rgb

def ingest(path,max_pages=5):
    p=Path(path); ext=p.suffix.lower(); pages=[]
    if ext=='.pdf':
        doc=fitz.open(p)
        for i,page in enumerate(doc):
            if i>=max_pages: break
            text=page.get_text('text').strip(); pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False); rgb=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)[...,:3]
            pages.append({'page':i+1,'text_layer':text,'searchable':len(text)>=30,'rgb':rgb})
        return pages, len(doc)
    if ext in {'.jpg','.jpeg','.png','.tif','.tiff','.webp'}:
        return [{'page':1,'text_layer':'','searchable':False,'rgb':load_rgb(p)}],1
    raise ValueError('Live demo supports PDF and image scans. DOCX/XLSX remain supported in the research notebook.')
