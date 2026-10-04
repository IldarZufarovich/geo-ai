from pathlib import Path
import random, numpy as np, cv2, fitz
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parent
random.seed(42); np.random.seed(42)

def rock(name,fractures=False,vugs=False):
    h=w=720; arr=np.zeros((h,w,3),np.uint8); arr[:]=[150,125,100]
    # irregular grains
    for i in range(130):
        cx,cy=np.random.randint(25,w-25),np.random.randint(25,h-25); rx,ry=np.random.randint(12,42),np.random.randint(10,34); ang=np.random.randint(0,180)
        col=tuple(int(x) for x in np.random.randint([145,120,85],[225,205,175]))
        cv2.ellipse(arr,(cx,cy),(rx,ry),ang,0,360,col,-1); cv2.ellipse(arr,(cx,cy),(rx,ry),ang,0,360,(80,70,60),1)
    # blue epoxy pores
    for i in range(35 if not vugs else 55):
        cx,cy=np.random.randint(20,w-20),np.random.randint(20,h-20); r=np.random.randint(5,18 if not vugs else 34)
        cv2.ellipse(arr,(cx,cy),(r,max(3,int(r*.65))),np.random.randint(180),0,360,(40,120,205),-1)
    if fractures:
        for x in [150,410,570]:
            pts=np.array([[x,20],[x+10,180],[x-8,350],[x+20,520],[x+8,700]],np.int32); cv2.polylines(arr,[pts],False,(35,35,35),4)
    noise=np.random.normal(0,7,arr.shape).astype(np.int16); arr=np.clip(arr.astype(np.int16)+noise,0,255).astype(np.uint8)
    Image.fromarray(arr).save(ROOT/name)

rock('carbonate_thin_section.png')
rock('fractured_rock.png',fractures=True)
rock('vuggy_carbonate.png',vugs=True)
# scanned page image
img=Image.new('RGB',(1240,1754),'#eee9df'); d=ImageDraw.Draw(img)
lines=['LEGACY GEOLOGICAL REPORT','Well: AB-143','Field: North Dome','Formation: Arab-D','Depth: 7,450 - 7,820 ft','Lithology: Dolomitic limestone','Porosity: 18.4%','Permeability: 47 mD','Water Saturation: 31%','Sample ID: C-17','Core Number: 12','Pore Type: interparticle pores and isolated vugs','Fracture: thin open microfractures']
y=120
for i,t in enumerate(lines): d.text((110,y),t,fill=(35,35,35)); y+=90 if i==0 else 70
img=img.rotate(1.2,expand=False,fillcolor='#eee9df'); img.save(ROOT/'legacy_report_scan.png')
# searchable PDF
doc=fitz.open(); page=doc.new_page(width=612,height=792); y=72
for t in lines: page.insert_text((72,y),t,fontsize=13); y+=34
doc.save(ROOT/'legacy_report_searchable.pdf')
