from __future__ import annotations
import cv2
import numpy as np
import pandas as pd
from .preprocessing import ocr_variants
from .ocr import best_ocr


def page_content_profile(rgb: np.ndarray) -> dict:
    """Lightweight page-vs-rock router using OCR evidence + image statistics."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ocr = best_ocr({'router': ocr_variants(rgb)['upscaled_gray']})
    text_chars = sum(ch.isalnum() for ch in ocr.get('text',''))
    white_ratio = float((gray > 225).mean())
    sat_ratio = float((hsv[...,1] > 45).mean())
    edges = cv2.Canny(gray, 80, 180)
    edge_ratio = float((edges > 0).mean())
    document_score = min(1.0, text_chars/180.0) * 0.62 + white_ratio * 0.23 + min(edge_ratio/0.12,1.0)*0.15
    is_document = bool(text_chars >= 35 or (white_ratio > 0.52 and edge_ratio > 0.015))
    return {'is_document':is_document, 'document_score':round(document_score,3), 'ocr_text':ocr.get('text',''),
            'ocr_confidence':ocr.get('mean_confidence',0.0), 'ocr_language':ocr.get('language',''),
            'white_ratio':round(white_ratio,3), 'saturation_ratio':round(sat_ratio,3), 'edge_ratio':round(edge_ratio,3)}


def detect_figure_regions(rgb: np.ndarray) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Detect large image-like regions on a document page.

    This is a CPU-safe layout fallback, not a learned layout detector. It targets large
    photographic/petrographic panels and deliberately rejects text-sized components.
    """
    h,w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Photographic figures tend to have sustained saturation or non-white tonal variation.
    sat = hsv[...,1] > 38
    nonwhite = gray < 225
    mask = (sat & nonwhite).astype(np.uint8)*255
    k = max(5, int(min(h,w)*0.012)//2*2+1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rows=[]; crops=[]
    page_area=h*w
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c); area=bw*bh
        if area < page_area*0.035 or bw < w*0.16 or bh < h*0.12:
            continue
        crop=rgb[y:y+bh,x:x+bw]
        chsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV); cgray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
        sat_ratio=float((chsv[...,1]>38).mean()); std=float(cgray.std())
        # Reject mostly-white/text blocks and very flat colored headers.
        if sat_ratio < 0.10 or std < 12:
            continue
        rows.append({'region_id':f'FIG-{len(rows)+1:03d}','region_type':'geological_figure_candidate',
                     'x0':x,'y0':y,'x1':x+bw,'y1':y+bh,'width':bw,'height':bh,
                     'page_fraction':round(area/page_area,3),'saturation_ratio':round(sat_ratio,3),'tonal_std':round(std,1),
                     'confidence':round(min(0.94,0.55+sat_ratio*0.35+min(std/100,0.2)),3)})
        crops.append(crop)
    # Largest first, cap for demo safety.
    if rows:
        order=np.argsort([-r['page_fraction'] for r in rows])[:12]
        rows=[rows[i] for i in order]; crops=[crops[i] for i in order]
    return pd.DataFrame(rows), crops


def draw_regions(rgb: np.ndarray, regions: pd.DataFrame) -> np.ndarray:
    out=rgb.copy()
    for _,r in regions.iterrows():
        x0,y0,x1,y1=map(int,[r.x0,r.y0,r.x1,r.y1])
        cv2.rectangle(out,(x0,y0),(x1,y1),(45,210,180),3)
        cv2.putText(out,str(r.region_id),(x0,max(18,y0-7)),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,30,40),3,cv2.LINE_AA)
        cv2.putText(out,str(r.region_id),(x0,max(18,y0-7)),cv2.FONT_HERSHEY_SIMPLEX,.65,(90,245,220),1,cv2.LINE_AA)
    return out


def detect_text_regions(rgb: np.ndarray) -> pd.DataFrame:
    """Approximate text blocks for visualization/routing only (CPU-safe fallback)."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); h,w=gray.shape
    bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,15)
    # Connect characters into line/block structures.
    kw=max(15,int(w*0.018)); kh=max(3,int(h*0.003))
    merged=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(kw,kh)),iterations=2)
    contours,_=cv2.findContours(merged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rows=[]; page=h*w
    for c in contours:
        x,y,bh_w,bh_h=cv2.boundingRect(c); area=bh_w*bh_h
        if area<page*0.00035 or bh_w<w*0.06 or bh_h<h*0.008: continue
        if area>page*0.35: continue
        rows.append({'region_id':f'TXT-{len(rows)+1:03d}','region_type':'text_to_ocr','x0':x,'y0':y,'x1':x+bh_w,'y1':y+bh_h,'confidence':0.82})
    return pd.DataFrame(rows[:40])


def draw_routing_map(rgb: np.ndarray, figure_regions: pd.DataFrame, text_regions: pd.DataFrame | None = None) -> np.ndarray:
    """Visualize first-stage routing: green TEXT->OCR, teal FIGURE->Rock Vision."""
    out=rgb.copy()
    if text_regions is not None and len(text_regions):
        for _,r in text_regions.iterrows():
            x0,y0,x1,y1=map(int,[r.x0,r.y0,r.x1,r.y1])
            cv2.rectangle(out,(x0,y0),(x1,y1),(65,180,90),2)
    if figure_regions is not None and len(figure_regions):
        for _,r in figure_regions.iterrows():
            x0,y0,x1,y1=map(int,[r.x0,r.y0,r.x1,r.y1])
            cv2.rectangle(out,(x0,y0),(x1,y1),(30,190,210),4)
            cv2.rectangle(out,(x0,max(0,y0-28)),(min(out.shape[1]-1,x0+235),y0),(30,190,210),-1)
            cv2.putText(out,'FIGURE -> ROCK VISION',(x0+5,max(18,y0-8)),cv2.FONT_HERSHEY_SIMPLEX,.52,(10,25,30),1,cv2.LINE_AA)
    # Legend is intentionally part of the image so screenshots remain self-explanatory.
    cv2.rectangle(out,(10,10),(370,72),(248,250,252),-1)
    cv2.rectangle(out,(10,10),(370,72),(60,70,80),1)
    cv2.rectangle(out,(22,24),(42,40),(65,180,90),-1); cv2.putText(out,'TEXT -> OCR / entity extraction',(52,38),cv2.FONT_HERSHEY_SIMPLEX,.48,(25,35,45),1,cv2.LINE_AA)
    cv2.rectangle(out,(22,48),(42,64),(30,190,210),-1); cv2.putText(out,'FIGURE -> semantic + instance segmentation',(52,62),cv2.FONT_HERSHEY_SIMPLEX,.45,(25,35,45),1,cv2.LINE_AA)
    return out


def classify_input(rgb: np.ndarray) -> dict:
    """Conservative top-level router: document page vs geological visual.

    OCR garbage on textured rock images should not make the whole image a document.
    A document decision therefore requires either credible OCR confidence/text on a light
    page, or a strongly page-like white background.
    """
    p = page_content_profile(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    white_ratio = float((gray > 225).mean())
    text_chars = sum(ch.isalnum() for ch in p.get('ocr_text', ''))
    ocr_conf = float(p.get('ocr_confidence', 0.0))
    # Real document pages are usually light or have credible OCR. Rock collages often
    # generate nonsense OCR at low confidence; route those to Rock Vision.
    credible_text = text_chars >= 45 and ocr_conf >= 0.56
    page_like = white_ratio >= 0.46 and text_chars >= 18
    is_document = bool(credible_text or page_like)
    kind = 'document_page' if is_document else 'geological_image_or_collage'
    confidence = max(0.55, min(0.95, (ocr_conf if is_document else 1.0 - min(0.8, white_ratio))))
    return {**p, 'input_kind': kind, 'is_document': is_document, 'routing_confidence': round(confidence, 3)}
