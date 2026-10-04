from __future__ import annotations
import math, cv2, numpy as np, pandas as pd
from skimage.measure import regionprops
from .segmentation import CLASS_NAMES

def measure_objects(rgb, sem, inst, source_file='', page=None, image_id='image', model_version='geo-ai-lightweight-1.1.0'):
    rows=[]; h,w=sem.shape; image_area=h*w
    for r in regionprops(inst,intensity_image=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)):
        cy,cx=r.centroid; y0,x0,y1,x1=r.bbox; cls=int(np.bincount(sem[inst==r.label]).argmax())
        per=max(float(r.perimeter),1e-6); area=float(r.area)
        circularity=float(4*math.pi*area/(per*per)); aspect=float(r.axis_major_length/max(r.axis_minor_length,1e-6))
        name=CLASS_NAMES.get(cls,'other')
        # Large blue pore objects are presented separately as vugs in the object-level result.
        if name == 'pore' and (area/image_area > 0.0025 or float(r.equivalent_diameter_area) > 0.055*min(h,w)):
            name='vug'
        rows.append(dict(source_document=source_file,page_number=page,image_id=image_id,object_id=int(r.label),predicted_class=name,confidence=0.67,model_version=model_version,area_px=area,perimeter_px=per,equivalent_diameter_px=float(r.equivalent_diameter_area),major_axis_px=float(r.axis_major_length),minor_axis_px=float(r.axis_minor_length),aspect_ratio=aspect,eccentricity=float(r.eccentricity),solidity=float(r.solidity),circularity=circularity,orientation_deg=float(np.degrees(r.orientation)),centroid_x=float(cx),centroid_y=float(cy),bbox_x0=int(x0),bbox_y0=int(y0),bbox_x1=int(x1),bbox_y1=int(y1)))
    return pd.DataFrame(rows)

def summarize(sem,objects):
    n=max(1,sem.size); por=float((sem==2).sum()/n*100)
    vc=objects['predicted_class'].value_counts() if len(objects) else pd.Series(dtype=int)
    fr=objects[objects.predicted_class=='fracture'] if len(objects) else objects
    return {'visible_2d_porosity_pct':round(por,2),'objects':int(len(objects)),'grains':int(vc.get('grain',0)),'pores_vugs':int(vc.get('pore',0)+vc.get('vug',0)),'pores':int(vc.get('pore',0)),'vugs':int(vc.get('vug',0)),'fractures':int(vc.get('fracture',0)),'cement_clay':int(vc.get('cement/clay',0)),'dominant_fracture_orientation_deg':None if len(fr)==0 else round(float(fr.orientation_deg.median()),1)}
