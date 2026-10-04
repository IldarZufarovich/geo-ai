from __future__ import annotations
from pathlib import Path
import os
import platform
import shutil as _runtime_shutil
import html, time, platform
import cv2
import pandas as pd
import gradio as gr
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether
)
from config import SETTINGS
from inference import analyze_rock, analyze_document
from utils import ensure_path
from src.document_ai.ingestion import ingest
from src.document_ai.router import classify_input, detect_figure_regions, detect_text_regions, draw_routing_map
from src.document_ai.ocr import configure_tesseract, available_languages, russian_ocr_ready

EX = SETTINGS.examples_dir
DISCLAIMER = 'Portfolio prototype · lightweight CPU baseline · not validated for operational geological interpretation.'
CSS = r"""
:root{--bg:#f4f7fb;--card:#fff;--text:#111827;--muted:#526174;--accent:#0f9f91;--border:#d8e1eb;--code:#eef3f8;--ok:#087f5b}
html[data-geo-theme='dark']{--bg:#08111f;--card:#101c2d;--text:#f7fafc;--muted:#c1ccda;--accent:#3bd4c0;--border:#2b4058;--code:#0b1626;--ok:#6ee7b7}
html,body,.gradio-container{background:var(--bg)!important;color:var(--text)!important;font-family:Inter,ui-sans-serif,system-ui,-apple-system,'Segoe UI',Arial,sans-serif!important}
.gradio-container{
width:100%!important;
max-width:1500px!important;
min-width:0!important;
margin:0 auto!important;
padding:18px!important;
box-sizing:border-box!important;
}.hero{padding:24px 28px;border-radius:20px;background:linear-gradient(135deg,#102d49,#176d70);margin-bottom:10px}.hero h1{color:white!important;margin:0!important;font-size:2rem!important}.hero p{color:#eafffb!important;margin:7px 0 0!important}.themebar{justify-content:flex-end}.themebar button{min-width:auto!important;border-radius:999px!important;background:var(--card)!important;color:var(--text)!important;border:1px solid var(--border)!important}.card,.exec-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:15px 18px;color:var(--text)!important}
.version-badge{
    display:inline-block!important;
    margin-top:8px!important;
    padding:5px 11px!important;
    border-radius:999px!important;
    background:rgba(255,255,255,.16)!important;
    border:1px solid rgba(255,255,255,.35)!important;
    color:#ffffff!important;
    font-size:.78rem!important;
    font-weight:800!important;
    letter-spacing:.04em!important;
}
.badge{display:inline-block;background:#d8f7f1;color:#07695f;border-radius:999px;padding:4px 9px;font-size:.76rem;font-weight:850}.executed{display:inline-block;background:#d1fae5;color:#065f46;border:1px solid #6ee7b7;border-radius:999px;padding:4px 9px;font-size:.75rem;font-weight:900}.muted{color:var(--muted)!important}.legend{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}.lg{padding:5px 9px;border-radius:8px;color:#111827;font-weight:850}.g{background:#EEBE46}.p{background:#2887E6;color:#fff}.v{background:#1ECDCF}.f{background:#EB4141;color:#fff}.c{background:#9B5ACD;color:#fff}.flow{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--code);border:1px solid var(--border);border-radius:14px;padding:16px;white-space:pre-wrap;line-height:1.65;color:var(--text)!important}.gradio-container label,.gradio-container p,.gradio-container span,.gradio-container h1,.gradio-container h2,.gradio-container h3,.gradio-container h4{color:var(--text)!important}.gradio-container input,.gradio-container textarea,.gradio-container .block,.gradio-container .form{background:var(--card)!important;color:var(--text)!important;border-color:var(--border)!important}.gradio-container table{color:var(--text)!important;background:var(--card)!important}.gradio-container th{background:var(--code)!important;color:var(--text)!important}.gradio-container td{background:var(--card)!important;color:var(--text)!important}.gradio-container pre,.gradio-container code{background:var(--code)!important;color:var(--text)!important}.gradio-container .tab-nav button{color:var(--text)!important;font-weight:850!important}.gradio-container .tab-nav button.selected{color:var(--accent)!important;border-color:var(--accent)!important}

/* ==========================================================
   VISUAL AI ROUTING MAP
   ========================================================== */

.routing-shell{
    width:100%;
    margin:12px 0 18px 0;
    padding:20px;
    border:1px solid var(--border);
    border-radius:18px;
    background:var(--card);
    box-sizing:border-box;
}

.routing-title{
    text-align:center;
    margin-bottom:18px;
}

.routing-title h3{
    margin:3px 0 4px 0!important;
    font-size:1.15rem!important;
}

.routing-title p{
    margin:0!important;
    color:var(--muted)!important;
    font-size:.84rem;
}

.routing-kicker{
    display:inline-block;
    color:var(--accent)!important;
    font-size:.68rem;
    font-weight:900;
    letter-spacing:.13em;
}

.route-flow{
    width:100%;
    max-width:1050px;
    margin:0 auto;
}

.route-node{
    position:relative;
    display:flex;
    align-items:center;
    gap:11px;
    width:100%;
    padding:12px 14px;
    border:1px solid var(--border);
    border-radius:14px;
    background:var(--code);
    color:var(--text)!important;
    box-sizing:border-box;
    box-shadow:0 5px 16px rgba(15,23,42,.06);
}

.route-node b{
    display:block;
    color:var(--text)!important;
    font-size:.82rem;
    letter-spacing:.025em;
}

.route-node small{
    display:block;
    margin-top:2px;
    color:var(--muted)!important;
    font-size:.72rem;
}

.route-icon{
    display:flex;
    align-items:center;
    justify-content:center;
    flex:0 0 38px;
    width:38px;
    height:38px;
    border-radius:11px;
    font-size:.72rem;
    font-weight:900;
    background:rgba(30,205,207,.13);
    border:1px solid rgba(30,205,207,.35);
    color:var(--accent)!important;
}

.route-ok{
    margin-left:auto;
    flex:0 0 auto;
    padding:4px 7px;
    border-radius:999px;
    font-size:.61rem;
    font-weight:900;
    color:#047857!important;
    background:#d1fae5;
    border:1px solid #6ee7b7;
}

.route-arrow,
.route-merge{
    text-align:center;
    color:var(--accent)!important;
    font-size:1.35rem;
    font-weight:900;
    line-height:1.25;
    padding:4px 0;
}

.split-arrow{
    letter-spacing:4rem;
    padding-left:4rem;
}

.route-branches{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:18px;
    align-items:start;
}

.route-branch{
    padding:12px;
    border-radius:16px;
    border:1px solid var(--border);
    background:rgba(148,163,184,.04);
}

.doc-node{
    border-color:#60a5fa;
    background:linear-gradient(
        135deg,
        rgba(59,130,246,.13),
        rgba(59,130,246,.04)
    );
}

.doc-node .route-icon{
    color:#2563eb!important;
    border-color:#93c5fd;
    background:#dbeafe;
}

.doc-soft{
    border-left:4px solid #60a5fa;
}

.vision-node{
    border-color:#f2c14e;
    background:linear-gradient(
        135deg,
        rgba(242,193,78,.16),
        rgba(242,193,78,.04)
    );
}

.vision-node .route-icon{
    color:#8a5b00!important;
    border-color:#f5cf70;
    background:#fef3c7;
}

.vision-soft{
    border-left:4px solid #f2c14e;
}

.router-node{
    border-color:#2dd4bf;
    background:linear-gradient(
        135deg,
        rgba(45,212,191,.15),
        rgba(45,212,191,.04)
    );
}

.input-node{
    border-left:5px solid #64748b;
}

.result-node{
    border-color:#34d399;
    background:linear-gradient(
        135deg,
        rgba(52,211,153,.16),
        rgba(52,211,153,.04)
    );
}

.result-node .route-icon{
    color:#047857!important;
    border-color:#6ee7b7;
    background:#d1fae5;
}

.route-branch.skipped{
    opacity:.46;
    filter:grayscale(.65);
    border-style:dashed;
}

.skipped-node{
    border-style:dashed;
}

html[data-geo-theme='dark'] .route-ok{
    color:#d1fae5!important;
    background:rgba(6,95,70,.45);
}

html[data-geo-theme='dark'] .doc-node .route-icon{
    background:rgba(37,99,235,.25);
    color:#bfdbfe!important;
}

html[data-geo-theme='dark'] .vision-node .route-icon{
    background:rgba(146,95,0,.30);
    color:#fde68a!important;
}

html[data-geo-theme='dark'] .result-node .route-icon{
    background:rgba(4,120,87,.30);
    color:#a7f3d0!important;
}

@media(max-width:768px){

    .routing-shell{
        padding:12px;
    }

    .route-branches{
        grid-template-columns:1fr;
        gap:10px;
    }

    .split-arrow,
    .route-merge{
        letter-spacing:0;
        padding-left:0;
    }

    .route-node{
        padding:10px;
    }

    .route-ok{
        font-size:.55rem;
    }
}



/* ==========================================================
   STAGE 2C - GEO-AI VISUAL LANGUAGE
   Inspired by the supplied Academy UI reference
   ========================================================== */

/* Soft ambient background */
html:not([data-geo-theme='dark']) body{
    background:
        radial-gradient(
            ellipse 52% 28% at 16% 3%,
            rgba(27,122,69,.075),
            transparent 68%
        ),
        radial-gradient(
            ellipse 48% 25% at 84% 7%,
            rgba(184,134,43,.075),
            transparent 68%
        ),
        #f5f7fa!important;
}

/* ----------------------------------------------------------
   Multimodal intro
   ---------------------------------------------------------- */

.asset-intro{
    position:relative;
    overflow:hidden;
    padding:18px 21px;
    margin:5px 0 12px;
    border-radius:20px;

    background:
        linear-gradient(160deg,#f2f8f4,#ffffff);

    border:1px solid rgba(27,122,69,.18);

    box-shadow:
        0 18px 45px -34px rgba(15,23,42,.32);
}

.asset-intro::after{
    content:"";
    position:absolute;
    width:180px;
    height:180px;
    border-radius:50%;
    right:-70px;
    top:-95px;

    background:
        radial-gradient(
            circle,
            rgba(184,134,43,.14),
            transparent 68%
        );
}

.asset-kicker{
    position:relative;
    z-index:1;

    font-size:.66rem;
    font-weight:900;
    letter-spacing:.14em;

    color:#1b7a45!important;
}

.asset-intro h3{
    position:relative;
    z-index:1;

    margin:4px 0 3px!important;
    font-size:1.12rem!important;
    letter-spacing:-.015em;
}

.asset-intro p{
    position:relative;
    z-index:1;

    margin:0!important;
    color:var(--muted)!important;
    font-size:.82rem;
}

/* ----------------------------------------------------------
   Upload / preview
   ---------------------------------------------------------- */

.gradio-container [data-testid="file"]{
    border-radius:18px!important;
}

.gradio-container .image-container{
    border-radius:20px!important;
    overflow:hidden!important;
}

/* ----------------------------------------------------------
   Buttons
   ---------------------------------------------------------- */

.academy-primary button,
button.academy-primary{
    border:none!important;
    border-radius:999px!important;

    background:
        linear-gradient(
            135deg,
            #1b7a45,
            #20a875
        )!important;

    color:#fff!important;
    font-weight:850!important;

    box-shadow:
        0 14px 30px -16px rgba(27,122,69,.58)!important;

    transition:
        transform .22s ease,
        box-shadow .22s ease!important;
}

.academy-primary button:hover,
button.academy-primary:hover{
    transform:translateY(-2px)!important;

    box-shadow:
        0 18px 34px -15px rgba(27,122,69,.66)!important;
}

.academy-secondary button,
button.academy-secondary{
    border-radius:999px!important;

    background:
        linear-gradient(
            160deg,
            #ffffff,
            #f7f4ec
        )!important;

    border:1px solid rgba(184,134,43,.25)!important;

    color:var(--text)!important;
    font-weight:800!important;

    transition:
        transform .22s ease,
        box-shadow .22s ease!important;
}

.academy-secondary button:hover,
button.academy-secondary:hover{
    transform:translateY(-2px)!important;

    box-shadow:
        0 15px 30px -20px rgba(0,0,0,.30)!important;
}

/* ----------------------------------------------------------
   Route status
   ---------------------------------------------------------- */

.routing-shell{
    max-width:980px!important;
    margin:10px auto 16px!important;

    padding:13px 15px!important;

    border-radius:20px!important;

    border:1.5px solid transparent!important;

    background-image:
        linear-gradient(180deg,#fff,#fafbf9),
        linear-gradient(
            135deg,
            rgba(27,122,69,.60),
            rgba(184,134,43,.55),
            rgba(198,54,47,.40)
        )!important;

    background-origin:border-box!important;
    background-clip:padding-box,border-box!important;

    box-shadow:
        0 24px 55px -38px rgba(0,0,0,.36)!important;
}

/* Compact title */
.routing-title{
    margin-bottom:10px!important;
}

.routing-title h3{
    margin:2px 0!important;
    font-size:.98rem!important;
}

.routing-title p{
    font-size:.70rem!important;
}

.routing-kicker{
    font-size:.60rem!important;
}

/* Compact nodes */
.route-flow{
    max-width:850px!important;
}

.route-node{
    padding:7px 9px!important;
    gap:8px!important;

    border-radius:11px!important;

    box-shadow:
        0 8px 18px -16px rgba(0,0,0,.35)!important;
}

.route-node b{
    font-size:.70rem!important;
}

.route-node small{
    font-size:.61rem!important;
}

.route-icon{
    flex-basis:29px!important;
    width:29px!important;
    height:29px!important;

    border-radius:9px!important;
    font-size:.60rem!important;
}

.route-ok{
    padding:3px 6px!important;
    font-size:.52rem!important;
}

.route-arrow,
.route-merge{
    padding:1px 0!important;
    font-size:.96rem!important;
    line-height:1!important;
}

.route-branches{
    gap:10px!important;
}

.route-branch{
    padding:7px!important;
    border-radius:13px!important;
}

/* Academy-like branch gradients */
.document-branch{
    background:
        linear-gradient(
            155deg,
            rgba(59,130,246,.055),
            rgba(255,255,255,.80)
        )!important;
}

.vision-branch{
    background:
        linear-gradient(
            155deg,
            rgba(184,134,43,.075),
            rgba(255,255,255,.82)
        )!important;
}

.doc-node{
    background:
        linear-gradient(
            155deg,
            #eef5ff,
            #ffffff
        )!important;
}

.vision-node{
    background:
        linear-gradient(
            155deg,
            #fdf6ea,
            #ffffff
        )!important;
}

.result-node{
    background:
        linear-gradient(
            155deg,
            #edf9f2,
            #ffffff
        )!important;
}

/* ----------------------------------------------------------
   Main cards
   ---------------------------------------------------------- */

.card,
.exec-card{
    border-radius:18px!important;

    box-shadow:
        0 18px 42px -38px rgba(0,0,0,.30)!important;
}

/* ----------------------------------------------------------
   Tabs - more product-like
   ---------------------------------------------------------- */

.gradio-container .tab-nav{
    border-radius:999px!important;
    padding:5px!important;

    box-shadow:
        0 12px 28px -26px rgba(0,0,0,.40)!important;
}

.gradio-container .tab-nav button{
    border-radius:999px!important;
}

.gradio-container .tab-nav button.selected{
    background:
        linear-gradient(
            135deg,
            rgba(27,122,69,.12),
            rgba(184,134,43,.09)
        )!important;

    color:#1b7a45!important;
}

/* ----------------------------------------------------------
   Dark mode
   ---------------------------------------------------------- */

html[data-geo-theme='dark'] .asset-intro{
    background:
        radial-gradient(
            ellipse 80% 120% at 100% 0%,
            rgba(184,134,43,.12),
            transparent 62%
        ),
        linear-gradient(
            155deg,
            #111d2b,
            #101827
        )!important;

    border-color:rgba(94,234,212,.16)!important;
}

html[data-geo-theme='dark'] .asset-kicker{
    color:#67e8d4!important;
}

html[data-geo-theme='dark'] .routing-shell{
    background-image:
        linear-gradient(180deg,#111b2a,#0f1826),
        linear-gradient(
            135deg,
            rgba(45,212,191,.65),
            rgba(184,134,43,.50),
            rgba(198,54,47,.40)
        )!important;
}

html[data-geo-theme='dark'] .document-branch,
html[data-geo-theme='dark'] .vision-branch{
    background:rgba(255,255,255,.025)!important;
}

html[data-geo-theme='dark'] .doc-node,
html[data-geo-theme='dark'] .vision-node,
html[data-geo-theme='dark'] .result-node{
    background:var(--code)!important;
}

/* ----------------------------------------------------------
   Mobile
   ---------------------------------------------------------- */

@media(max-width:768px){

    .asset-intro{
        padding:14px 15px;
        border-radius:16px;
    }

    .routing-shell{
        padding:10px!important;
        border-radius:16px!important;
    }

    .route-node{
        padding:7px!important;
    }

    .route-node small{
        font-size:.58rem!important;
    }
}

footer{display:none!important}

/* Stable desktop geometry */
.gradio-container .tabs,
.gradio-container [role="tabpanel"]{
    width:100%!important;
    max-width:100%!important;
    min-width:0!important;
    box-sizing:border-box!important;
}

.gradio-container .row{
    width:100%!important;
    max-width:100%!important;
    min-width:0!important;
    box-sizing:border-box!important;
}

.gradio-container .column{
    min-width:0!important;
    max-width:100%!important;
    box-sizing:border-box!important;
}

.gradio-container .block{
    max-width:100%!important;
    min-width:0!important;
    box-sizing:border-box!important;
}

/* Keep wide tables inside their own scroll area */
.gradio-container .table-wrap,
.gradio-container [data-testid="dataframe"]{
    width:100%!important;
    max-width:100%!important;
    overflow-x:auto!important;
}

/* Profile controls */
.profile-shell{
    width:100%!important;
    max-width:100%!important;
}

.profile-shell .row{
    align-items:flex-end!important;
}

.profile-shell .block{
    border-radius:14px!important;
}

/* Dropdown visual treatment */
.profile-shell input,
.profile-shell select{
    font-weight:750!important;
}



.gradio-container .tab-nav{
    width:100%!important;
    display:flex!important;
    gap:8px!important;
    padding:6px!important;
    margin:8px 0 14px 0!important;
    border:1px solid var(--border)!important;
    border-radius:14px!important;
    background:var(--card)!important;
    box-sizing:border-box!important;
}

.gradio-container .tab-nav button{
    flex:1 1 0!important;
    min-width:0!important;
    min-height:44px!important;
    border-radius:10px!important;
    font-weight:850!important;
    transition:
        background .18s ease,
        border-color .18s ease,
        transform .18s ease!important;
}

.gradio-container .tab-nav button:hover{
    transform:translateY(-1px);
}

.gradio-container .tab-nav button.selected{
    background:linear-gradient(
        135deg,
        rgba(30,205,207,.18),
        rgba(40,135,230,.13)
    )!important;
    border:1px solid var(--accent)!important;
    color:var(--accent)!important;
}


/* Mobile-first corrections for phone demo */
@media (max-width: 768px){
  .gradio-container{padding:8px!important;max-width:100%!important;overflow-x:hidden!important}
  .hero{padding:16px!important;border-radius:14px!important}.hero h1{font-size:1.35rem!important}.hero p{font-size:.92rem!important}
  .gradio-container .row{display:flex!important;flex-direction:column!important;gap:10px!important}
  .gradio-container .column{min-width:0!important;width:100%!important}
  .gradio-container button{width:100%!important;min-height:46px!important;font-size:1rem!important}
  .gradio-container img,.gradio-container video{max-width:100%!important;height:auto!important;object-fit:contain!important}
  .gradio-container .tab-nav{overflow-x:auto!important;white-space:nowrap!important;display:flex!important;scrollbar-width:thin}
  .gradio-container .tab-nav button{flex:0 0 auto!important;padding:10px 12px!important;font-size:.92rem!important}
  .gradio-container table{font-size:.78rem!important;min-width:680px!important}
  .gradio-container .table-wrap,.gradio-container [data-testid='dataframe']{overflow-x:auto!important;max-width:100%!important}
  .card,.exec-card{padding:12px!important;border-radius:12px!important}.flow{font-size:.78rem!important;padding:10px!important;overflow-x:auto!important}
  .legend{gap:5px!important}.lg{font-size:.75rem!important;padding:4px 6px!important}
}
"""
LEGEND="""<div class='legend'><span class='lg g'>G · Grain</span><span class='lg p'>P · Pore</span><span class='lg v'>V · Vug</span><span class='lg f'>F · Fracture candidate</span><span class='lg c'>C · Cement / clay</span></div>"""

def _safe_df(df,n=30): return df.head(n).copy() if isinstance(df,pd.DataFrame) else pd.DataFrame()
def _runtime_ms(t0): return round((time.perf_counter()-t0)*1000,1)
def _exec_df(rows): return pd.DataFrame(rows, columns=['stage','model / algorithm','version','device','status','output'])
def _graph(title, lines): return f"<div class='exec-card'><span class='executed'>✓ EXECUTED IN THIS RUN</span><h3>{html.escape(title)}</h3><div class='flow'>{html.escape(chr(10).join(lines))}</div></div>"

def rock_summary(s):
    return f"""<div class='card'><span class='badge'>ROCK VISION · {html.escape(str(s.get('image_type','image')))}</span><br><b>Visible 2D porosity:</b> {s.get('visible_2d_porosity_pct',0)}% · <b>Objects:</b> {s.get('objects',0)} · <b>Grains:</b> {s.get('grains',0)} · <b>Pores/vugs:</b> {s.get('pores_vugs',0)} · <b>Fracture candidates:</b> {s.get('fractures',0)}<br><b>Version:</b> {s.get('model_version')}<br><span class='muted'>{DISCLAIMER}</span></div>"""

def rock_execution(ms):
    rows=[
      ['Input QC & preprocessing','OpenCV','runtime','CPU','EXECUTED','normalized RGB / quality metrics'],
      ['Semantic segmentation','K-means material clustering + morphology','lightweight-baseline','CPU','EXECUTED','grain / pore / vug / cement-clay pixel classes'],
      ['Instance segmentation','Distance transform + watershed + connected components','lightweight-baseline','CPU','EXECUTED','individual object IDs'],
      ['Fracture candidates','thin-ridge / color / skeleton network detector','lightweight-baseline','CPU','EXECUTED','candidate fracture networks'],
      ['Quantification','scikit-image regionprops','runtime','CPU','EXECUTED','area / diameter / shape / orientation'],
      ['Total inference',f'{ms} ms','this run','CPU','EXECUTED','annotated image + object table']]
    graph=_graph('Rock Vision · execution graph',['Geological image','  ↓  OpenCV QC / normalization','Semantic segmentation · K-means + morphology','  ↓','Instance segmentation · watershed + connected components','  ↓','Fracture candidate detector · ridge/color/skeleton','  ↓','Object quantification · scikit-image regionprops'])
    return _exec_df(rows),graph

def run_rock(file):
    if not file: raise gr.Error('Upload a geological image or choose a demo sample.')
    path=ensure_path(file); t0=time.perf_counter()
    try:
        r=analyze_rock(path); ms=_runtime_ms(t0); edf,graph=rock_execution(ms)
        return r['original'],r['annotated'],rock_summary(r['summary']),_safe_df(r['objects'].sort_values('area_px',ascending=False),35),edf,graph,r['files']['annotated'],r['files']['objects_csv'],r['files']['json']
    except Exception as exc:
        try: rgb=ingest(path,1)[0][0]['rgb']
        except Exception: rgb=None
        msg=f"<div class='card'><span class='badge'>ANALYSIS NOT COMPLETED</span><br>{html.escape(str(exc))}<br><span class='muted'>Input preserved; no geological claim is made.</span></div>"
        return rgb,rgb,msg,pd.DataFrame(),pd.DataFrame(),_graph('Rock Vision · stopped',['Input received','  ↓','Analysis stopped safely · no geological claim']),None,None,None

def demo_rock(kind):
    m={'Carbonate thin section':'carbonate_thin_section.png','Fractured rock':'fractured_rock.png','Vuggy carbonate':'vuggy_carbonate.png'}; return str(EX/m[kind])
def preview_document(file):
    if not file:return None
    try:return ingest(ensure_path(file),1)[0][0]['rgb']
    except Exception:return None

def doc_summary(s):
    conf=float(s.get('mean_confidence',0)); langs=available_languages(); rus='READY' if russian_ocr_ready() else 'NOT INSTALLED'
    return f"""<div class='card'><span class='badge'>{'TEXT / OCR READY' if conf>=.60 else 'LOW OCR CONFIDENCE'}</span><br><b>Document type:</b> {html.escape(str(s.get('document_type','unknown')))} · <b>Text/OCR confidence:</b> {conf:.0%}<br><b>OCR languages available:</b> {html.escape(', '.join(langs) or 'none')} · <b>Russian OCR:</b> {rus} · <b>Version:</b> {SETTINGS.model_version}<br><span class='muted'>{DISCLAIMER}</span></div>"""

def doc_execution(result, ms):
    pages=result.get('pages',pd.DataFrame()); methods=[]
    if isinstance(pages,pd.DataFrame) and 'extraction_method' in pages:
        methods=[str(x) for x in pages['extraction_method'].dropna().unique()]
    searchable=any('text' in x.lower() and 'tesseract' not in x.lower() for x in methods)
    ocr=any('tesseract' in x.lower() for x in methods)
    rows=[['File/page ingestion','PyMuPDF + Pillow/OpenCV','runtime','CPU','EXECUTED','page image + metadata']]
    if searchable: rows.append(['Text extraction','PyMuPDF active text layer','runtime','CPU','EXECUTED','native PDF text'])
    if ocr:
        langs='+'.join([x for x in available_languages() if x in ('eng','rus')]) or 'eng'
        rows += [['Scan preprocessing','OpenCV multi-variant preprocessing','runtime','CPU','EXECUTED','OCR-ready page variants'],['OCR',f'Tesseract OCR ({langs})','local installation','CPU','EXECUTED','text + OCR confidence']]
    rows += [['Geological entity extraction','Regex dictionaries + unit normalization','lightweight-baseline','CPU','EXECUTED','structured entities + provenance'],['Total document inference',f'{ms} ms','this run','CPU','EXECUTED','text / entities / page traceability']]
    branch='PyMuPDF active text extraction' if searchable and not ocr else 'OpenCV preprocessing → Tesseract OCR'
    graph=_graph('Legacy Document AI · execution graph',['PDF / scan / photographed page','  ↓  file & page inspection',branch,'  ↓','Geological entity extraction · regex + normalization','  ↓','Traceability · source file → page → extracted value'])
    return _exec_df(rows),graph

def run_doc(file):
    if not file: raise gr.Error('Upload a PDF or scanned/photographed page.')
    t0=time.perf_counter(); r=analyze_document(ensure_path(file)); ms=_runtime_ms(t0); edf,graph=doc_execution(r,ms)
    return preview_document(file),doc_summary(r['summary']),r['text'],r['entities'],r['pages'],edf,graph,r['files']['entities_csv'],r['files']['text'],r['files']['json']

def demo_doc(kind):
    p=str(EX/('legacy_report_searchable.pdf' if kind=='Searchable geological PDF' else 'legacy_report_scan.png')); return p,preview_document(p)

def mm_exec(rows): return _exec_df(rows)




def build_document_ai_result(doc_text, entities):
    """Presentation layer for OCR / active-text results."""

    raw = str(doc_text or "").strip()

    if not raw:
        return (
            "<div class='doc-ai-shell'>"
            "<div class='doc-ai-head'>"
            "<div><span class='doc-ai-kicker'>DOCUMENT AI</span>"
            "<h3>Recognized Geological Content</h3>"
            "<p>No reliable document text was detected for this input.</p>"
            "</div></div></div>"
        )

    # Remove internal page markers from presentation.
    clean_lines = []

    for line in raw.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("--- Page"):
            continue

        clean_lines.append(line)

    # --------------------------------------------------------
    # Extract structured entity cards
    # --------------------------------------------------------

    cards = []

    if isinstance(entities, pd.DataFrame) and not entities.empty:
        if "entity" in entities.columns:

            preferred = [
                "formation",
                "lithology",
                "sample_id",
                "depth",
                "porosity",
                "permeability",
                "water_saturation",
                "pore_description",
                "fracture_description",
            ]

            labels = {
                "formation": "Formation",
                "lithology": "Lithology",
                "sample_id": "Sample",
                "depth": "Depth",
                "porosity": "Porosity",
                "permeability": "Permeability",
                "water_saturation": "Water saturation",
                "pore_description": "Pore system",
                "fracture_description": "Fractures",
            }

            for key in preferred:

                rows = entities[
                    entities["entity"]
                    .astype(str)
                    .str.lower()
                    == key
                ]

                if rows.empty:
                    continue

                value = None

                for col in [
                    "normalized_value",
                    "original_value"
                ]:
                    if col in rows.columns:
                        v = rows.iloc[0][col]

                        if pd.notna(v):
                            value = str(v)
                            break

                if not value:
                    continue

                cards.append(
                    "<div class='doc-ai-fact'>"
                    "<span>"
                    + html.escape(labels.get(key, key))
                    + "</span>"
                    "<b>"
                    + html.escape(value)
                    + "</b>"
                    "</div>"
                )

    # --------------------------------------------------------
    # Structure recognized content
    # --------------------------------------------------------

    content_blocks = []

    current = []

    def flush():
        nonlocal current

        if current:
            content_blocks.append(
                " ".join(current)
            )
            current = []

    for line in clean_lines:

        # Short all-caps / heading-like lines
        heading_like = (
            len(line) <= 90
            and (
                line.isupper()
                or line.endswith(":")
            )
        )

        if heading_like:
            flush()

            content_blocks.append(
                ("__HEADING__", line)
            )

            continue

        current.append(line)

        # Keep paragraphs reasonably compact.
        if len(" ".join(current)) > 420:
            flush()

    flush()

    rendered = []

    for block in content_blocks:

        if (
            isinstance(block, tuple)
            and block[0] == "__HEADING__"
        ):
            rendered.append(
                "<div class='doc-ai-section-title'>"
                + html.escape(block[1])
                + "</div>"
            )

        else:
            rendered.append(
                "<p>"
                + html.escape(str(block))
                + "</p>"
            )

    facts_html = "".join(cards)

    if not facts_html:
        facts_html = (
            "<div class='doc-ai-empty'>"
            "No structured geological entities were "
            "reliably extracted from this text."
            "</div>"
        )

    return (
        "<div class='doc-ai-shell'>"

        "<div class='doc-ai-head'>"
        "<div>"
        "<span class='doc-ai-kicker'>DOCUMENT AI RESULT</span>"
        "<h3>Recognized Geological Content</h3>"
        "<p>OCR / active text transformed into structured, "
        "human-readable geological information.</p>"
        "</div>"
        "<span class='doc-ai-status'>✓ RECOGNIZED</span>"
        "</div>"

        "<div class='doc-ai-facts-title'>"
        "KEY GEOLOGICAL INFORMATION"
        "</div>"

        "<div class='doc-ai-facts'>"
        + facts_html +
        "</div>"

        "<div class='doc-ai-content-title'>"
        "RECOGNIZED CONTENT"
        "</div>"

        "<div class='doc-ai-content'>"
        + "".join(rendered) +
        "</div>"

        "</div>"
    )


def build_geological_interpretation(summary, entities):
    s = summary or {}

    def ent(name):
        if not isinstance(entities, pd.DataFrame) or entities.empty or 'entity' not in entities.columns:
            return None
        rows = entities[entities['entity'].astype(str).str.lower() == name.lower()]
        if rows.empty:
            return None
        for col in ['normalized_value', 'original_value']:
            if col in rows.columns:
                v = rows.iloc[0][col]
                if pd.notna(v):
                    return str(v)
        return None

    def esc(v):
        return html.escape(str(v))

    def num(v, digits=1):
        try:
            return f'{float(v):.{digits}f}'
        except Exception:
            return '—'

    reported = []
    fields = [
        ('porosity', 'Reported porosity', '%'),
        ('permeability', 'Reported permeability', ' mD'),
        ('water_saturation', 'Water saturation', '%'),
        ('lithology', 'Lithology', ''),
        ('formation', 'Formation', ''),
    ]

    for key, label, unit in fields:
        value = ent(key)
        if value:
            reported.append((label, value + unit))

    por = s.get('visible_2d_porosity_pct')
    objects = int(s.get('objects', 0) or 0)
    grains = int(s.get('grains', 0) or 0)
    pores = int(s.get('pores', 0) or 0)
    vugs = int(s.get('vugs', 0) or 0)
    fractures = int(s.get('fractures', 0) or 0)
    cement = int(s.get('cement_clay', 0) or 0)
    orient = s.get('dominant_fracture_orientation_deg')
    conf = s.get('prototype_confidence')
    image_type = s.get('image_type', 'geological image')

    metrics = [
        ('Visible 2D pore area', (num(por,2) + '%') if por is not None else '—', 'blue'),
        ('Objects', objects, 'green'),
        ('Grains', grains, 'gold'),
        ('Pores', pores, 'blue'),
        ('Vugs', vugs, 'cyan'),
        ('Fractures', fractures, 'red'),
        ('Cement / clay', cement, 'purple'),
    ]

    if orient is not None:
        metrics.append(('Fracture orientation', num(orient,1) + '°', 'red'))

    if conf is not None:
        metrics.append(('CV confidence', f'{float(conf):.0%}', 'green'))

    obs = []
    if fractures: obs.append(f'{fractures} fracture candidate(s)')
    if vugs: obs.append(f'{vugs} vug-like object(s)')
    if pores: obs.append(f'{pores} pore object(s)')
    if grains: obs.append(f'{grains} grain object(s)')

    observed = ', '.join(obs) if obs else 'no dominant classified object population'

    interpretation = (
        f'The routed {image_type} contains {observed}. '
        + (f'The segmented pore-class area occupies approximately {num(por,2)}% of the analyzed 2D image. ' if por is not None else '')
        + (f'The median detected fracture orientation is approximately {num(orient,1)}° in image coordinates.' if orient is not None else '')
    )

    if reported:
        reported_html = ''.join(
            "<div class='geo-report-row'><span>" + esc(k) + "</span><b>" + esc(v) + "</b></div>"
            for k, v in reported
        )
    else:
        reported_html = "<div class='geo-empty'>No petrophysical properties reliably extracted from document text.</div>"

    metric_html = ''.join(
        "<div class='geo-kpi geo-kpi-" + c + "'><span>" + esc(k) + "</span><b>" + esc(v) + "</b></div>"
        for k, v, c in metrics
    )

    return (
        "<div class='geo-ai-summary'>"
        "<div class='geo-ai-head'><div><span class='geo-ai-kicker'>MULTIMODAL RESULT</span>"
        "<h3>AI Geological Interpretation</h3>"
        "<p>Reported document properties and image-derived measurements are shown separately.</p></div>"
        "<span class='geo-ai-status'>✓ TRACEABLE</span></div>"
        "<div class='geo-ai-grid'>"
        "<div class='geo-ai-panel'><span class='geo-panel-kicker'>DOCUMENT AI</span><h4>Reported Properties</h4>"
        + reported_html + "</div>"
        "<div class='geo-ai-panel'><span class='geo-panel-kicker'>COMPUTER VISION</span><h4>Image-Derived Measurements</h4>"
        "<div class='geo-kpis'>" + metric_html + "</div></div></div>"
        "<div class='geo-ai-text'><span>PRELIMINARY INTERPRETATION</span><p>"
        + esc(interpretation) + "</p></div>"
        "<div class='geo-ai-disclaimer'><b>Scope:</b> Visible 2D pore area is an image-segmentation measurement, not laboratory porosity. "
        "Permeability is displayed only when extracted from the source document and is not inferred by the current Computer Vision model.</div>"
        "</div>"
    )

def build_routing_map(top, figure_count=0, text_detected=True):
    """
    Build a responsive visual routing map for the current inference run.
    Presentation only: does not alter inference decisions.
    """

    is_document = bool(top.get("is_document", False))

    try:
        route_conf = float(top.get("routing_confidence", 0) or 0)
    except Exception:
        route_conf = 0.0

    try:
        ocr_conf = float(top.get("ocr_confidence", 0) or 0)
    except Exception:
        ocr_conf = 0.0

    route_pct = f"{route_conf * 100:.0f}%"
    ocr_pct = f"{ocr_conf * 100:.0f}%"

    if is_document:

        doc_state = "executed"
        cv_state = "executed" if figure_count > 0 else "skipped"

        cv_status = (
            f"FIGURE DETECTED · {figure_count}"
            if figure_count > 0
            else "NO FIGURE ROUTED"
        )

        return f"""
        <div class="routing-shell">

          <div class="routing-title">
            <span class="routing-kicker">AI ROUTING MAP</span>
            <h3>Legacy geological asset → multimodal analysis</h3>
            <p>
              The toolkit inspected the uploaded asset and selected
              the analysis branches shown below.
            </p>
          </div>

          <div class="route-flow">

            <div class="route-node input-node">
              <div class="route-icon">01</div>
              <div>
                <b>LEGACY ASSET</b>
                <small>Report / scan / photographed page</small>
              </div>
              <span class="route-ok">✓ INPUT</span>
            </div>

            <div class="route-arrow">↓</div>

            <div class="route-node router-node">
              <div class="route-icon">02</div>
              <div>
                <b>CONTENT ROUTER</b>
                <small>Document route · confidence {route_pct}</small>
              </div>
              <span class="route-ok">✓ EXECUTED</span>
            </div>

            <div class="route-arrow split-arrow">↙ &nbsp;&nbsp;&nbsp; ↘</div>

            <div class="route-branches">

              <div class="route-branch document-branch">

                <div class="route-node doc-node">
                  <div class="route-icon">Aa</div>
                  <div>
                    <b>DOCUMENT AI</b>
                    <small>Text / OCR / entities</small>
                  </div>
                  <span class="route-ok">✓ EXECUTED</span>
                </div>

                <div class="route-arrow">↓</div>

                <div class="route-node doc-soft">
                  <div>
                    <b>TEXT RECOGNITION</b>
                    <small>
                      Tesseract / active text · OCR evidence {ocr_pct}
                    </small>
                  </div>
                </div>

                <div class="route-arrow">↓</div>

                <div class="route-node doc-soft">
                  <div>
                    <b>GEOLOGICAL ENTITIES</b>
                    <small>Structured text + provenance</small>
                  </div>
                </div>

              </div>

              <div class="route-branch vision-branch {cv_state}">

                <div class="route-node vision-node">
                  <div class="route-icon">CV</div>
                  <div>
                    <b>COMPUTER VISION</b>
                    <small>{cv_status}</small>
                  </div>
                  <span class="route-ok">
                    {"✓ EXECUTED" if figure_count > 0 else "○ SKIPPED"}
                  </span>
                </div>

                <div class="route-arrow">↓</div>

                <div class="route-node vision-soft">
                  <div>
                    <b>ROCK VISION</b>
                    <small>Geological image interpretation</small>
                  </div>
                </div>

                <div class="route-arrow">↓</div>

                <div class="route-node vision-soft">
                  <div>
                    <b>OBJECT ANALYSIS</b>
                    <small>
                      Segmentation · instances · measurements
                    </small>
                  </div>
                </div>

              </div>

            </div>

            <div class="route-merge">↘ &nbsp;&nbsp;&nbsp; ↙</div>

            <div class="route-node result-node">
              <div class="route-icon">✓</div>
              <div>
                <b>STRUCTURED GEOSCIENCE DATA</b>
                <small>
                  Unified results · provenance · traceability
                </small>
              </div>
              <span class="route-ok">✓ RESULT</span>
            </div>

          </div>
        </div>
        """

    # --------------------------------------------------------
    # ROCK / GEOLOGICAL IMAGE ROUTE
    # --------------------------------------------------------

    return f"""
    <div class="routing-shell">

      <div class="routing-title">
        <span class="routing-kicker">AI ROUTING MAP</span>
        <h3>Geological image → Computer Vision</h3>
        <p>
          The input was classified as geological imagery,
          so the Document AI branch was skipped.
        </p>
      </div>

      <div class="route-flow">

        <div class="route-node input-node">
          <div class="route-icon">01</div>
          <div>
            <b>GEOLOGICAL IMAGE</b>
            <small>Rock / petrographic / collage input</small>
          </div>
          <span class="route-ok">✓ INPUT</span>
        </div>

        <div class="route-arrow">↓</div>

        <div class="route-node router-node">
          <div class="route-icon">02</div>
          <div>
            <b>CONTENT ROUTER</b>
            <small>Image route · confidence {route_pct}</small>
          </div>
          <span class="route-ok">✓ EXECUTED</span>
        </div>

        <div class="route-arrow">↓</div>

        <div class="route-branches">

          <div class="route-branch skipped">

            <div class="route-node skipped-node">
              <div class="route-icon">Aa</div>
              <div>
                <b>DOCUMENT AI</b>
                <small>No primary text-document route</small>
              </div>
              <span>○ SKIPPED</span>
            </div>

          </div>

          <div class="route-branch vision-branch">

            <div class="route-node vision-node">
              <div class="route-icon">CV</div>
              <div>
                <b>COMPUTER VISION</b>
                <small>Rock Vision selected</small>
              </div>
              <span class="route-ok">✓ EXECUTED</span>
            </div>

            <div class="route-arrow">↓</div>

            <div class="route-node vision-soft">
              <div>
                <b>GEOLOGICAL INTERPRETATION</b>
                <small>
                  Semantic classes · instances · fractures
                </small>
              </div>
            </div>

            <div class="route-arrow">↓</div>

            <div class="route-node vision-soft">
              <div>
                <b>OBJECT MEASUREMENTS</b>
                <small>
                  Geometry · area · shape · provenance
                </small>
              </div>
            </div>

          </div>

        </div>

        <div class="route-arrow">↓</div>

        <div class="route-node result-node">
          <div class="route-icon">✓</div>
          <div>
            <b>STRUCTURED GEOSCIENCE DATA</b>
            <small>Traceable Computer Vision result</small>
          </div>
          <span class="route-ok">✓ RESULT</span>
        </div>

      </div>
    </div>
    """



# ============================================================
# PROFESSIONAL MULTIMODAL PDF REPORT
# ============================================================

def _pdf_rgb_image(
    rgb,
    output_path,
    max_width_mm=170,
    max_height_mm=105
):
    """Convert RGB ndarray to a ReportLab image preserving aspect."""

    if rgb is None:
        return None

    arr = rgb

    if not hasattr(arr, "shape"):
        return None

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            arr,
            cv2.COLOR_RGB2BGR
        )
    )

    h, w = arr.shape[:2]

    max_w = max_width_mm * mm
    max_h = max_height_mm * mm

    scale = min(
        max_w / max(1, w),
        max_h / max(1, h)
    )

    return RLImage(
        str(output_path),
        width=w * scale,
        height=h * scale
    )


def _pdf_entity_value(
    entities,
    name
):
    if (
        not isinstance(entities, pd.DataFrame)
        or entities.empty
        or "entity" not in entities.columns
    ):
        return None

    rows = entities[
        entities["entity"]
        .astype(str)
        .str.lower()
        == str(name).lower()
    ]

    if rows.empty:
        return None

    for col in [
        "normalized_value",
        "original_value"
    ]:
        if col in rows.columns:

            value = rows.iloc[0][col]

            if pd.notna(value):
                return str(value)

    return None



def generate_multimodal_pdf_report(
    source_path,
    source_rgb,
    routed_rgb,
    doc_text,
    entities,
    routes,
    crop,
    interp,
    rock_summary,
    trace,
    exec_df,
    top,
    runtime_meta=None
):
    """
    GEO AI Professional Report v4.0.

    Uses ONLY outputs produced by the current multimodal
    inference run. No additional inference is performed.

    PAGE 1
        Executive Summary + Multimodal AI Routing

    PAGE 2
        Content Routing Map + Before / After

    PAGE 3
        Document AI + Geological Interpretation

    PAGE 4
        Traceability + Methodology
    """

    output_dir = Path(
        SETTINGS.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = Path(
        str(source_path)
    ).stem

    safe_stem = "".join(
        c if c.isalnum() or c in "-_"
        else "_"
        for c in stem
    )[:60]

    pdf_path = (
        output_dir
        / f"{safe_stem}_GEO_AI_Report.pdf"
    )

    asset_dir = (
        output_dir
        / "_pdf_assets"
    )

    asset_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # COLORS
    # ========================================================

    DARK = colors.HexColor("#172033")
    GREEN = colors.HexColor("#1B7A45")
    TEAL = colors.HexColor("#16A89A")
    BLUE = colors.HexColor("#4B82E6")
    GOLD = colors.HexColor("#B8862B")
    PURPLE = colors.HexColor("#8B5CF6")
    RED = colors.HexColor("#D45C50")

    MUTED = colors.HexColor("#687386")

    LINE = colors.HexColor("#DCE3E8")
    LIGHT = colors.HexColor("#F8FAFC")
    GREEN_BG = colors.HexColor("#EFF8F4")
    BLUE_BG = colors.HexColor("#EFF5FD")
    GOLD_BG = colors.HexColor("#FDF7EB")

    # ========================================================
    # DOCUMENT
    # ========================================================

    doc = SimpleDocTemplate(
        str(pdf_path),

        pagesize=A4,

        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=16 * mm,
        bottomMargin=15 * mm,

        title=(
            "GEO AI — Geoscience Computer Vision "
            "& Document AI Report"
        ),

        author=(
            "GEO AI — Geoscience Computer Vision "
            "& Document AI Toolkit"
        )
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "GeoTitleV2",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        textColor=DARK,
        spaceAfter=2.5 * mm
    )

    page_title_style = ParagraphStyle(
        "GeoPageTitleV2",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=DARK,
        spaceAfter=2 * mm
    )

    section_style = ParagraphStyle(
        "GeoSectionV2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=DARK,
        spaceBefore=2 * mm,
        spaceAfter=2.5 * mm
    )

    kicker_style = ParagraphStyle(
        "GeoKickerV2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.8,
        leading=8,
        textColor=GREEN,
        spaceAfter=1.5 * mm
    )

    body_style = ParagraphStyle(
        "GeoBodyV2",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=11.2,
        textColor=colors.HexColor("#253044")
    )

    body_bold_style = ParagraphStyle(
        "GeoBodyBoldV2",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    small_style = ParagraphStyle(
        "GeoSmallV2",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.8,
        leading=9.3,
        textColor=MUTED
    )

    tiny_style = ParagraphStyle(
        "GeoTinyV2",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=5.8,
        leading=7.6,
        textColor=MUTED
    )

    node_title = ParagraphStyle(
        "GeoNodeTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.4,
        leading=9,
        textColor=DARK
    )

    node_small = ParagraphStyle(
        "GeoNodeSmall",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=5.8,
        leading=7.2,
        textColor=MUTED
    )

    story = []

    rs = rock_summary or {}
    runtime_meta = runtime_meta or {}

    requested_profile = runtime_meta.get(
        "requested_profile",
        "AUTO"
    )

    resolved_profile = runtime_meta.get(
        "resolved_profile",
        "Lightweight CPU"
    )

    analysis_detail_level = runtime_meta.get(
        "analysis_detail",
        "Standard"
    )

    runtime_device = runtime_meta.get(
        "device",
        "CPU"
    )

    runtime_fallback = bool(
        runtime_meta.get(
            "fallback",
            False
        )
    )

    # ========================================================
    # HELPERS
    # ========================================================

    def esc(value):
        return html.escape(
            str(value)
        )

    def add_page_heading(
        number,
        kicker,
        title,
        subtitle
    ):
        story.append(
            Paragraph(
                f"{esc(number)} · {esc(kicker)}",
                kicker_style
            )
        )

        story.append(
            Paragraph(
                esc(title),
                page_title_style
            )
        )

        story.append(
            Paragraph(
                esc(subtitle),
                small_style
            )
        )

        story.append(
            Spacer(
                1,
                4 * mm
            )
        )

    def add_section(title):
        story.append(
            Paragraph(
                esc(title),
                section_style
            )
        )

    def entity(name):

        value = _pdf_entity_value(
            entities,
            name
        )

        return value

    def fmt_percent(value):

        if value is None:
            return "—"

        try:
            return (
                f"{float(value):.2f}%"
            )
        except Exception:
            return str(value)

    def fmt_orientation(value):

        if value is None:
            return "—"

        try:
            return (
                f"{float(value):.1f}°"
            )
        except Exception:
            return str(value)

    def make_image(
        arr,
        name,
        max_w_mm,
        max_h_mm
    ):

        return _pdf_rgb_image(
            arr,
            asset_dir / name,
            max_width_mm=max_w_mm,
            max_height_mm=max_h_mm
        )

    def kv_table(rows):

        data = []

        for key, value in rows:

            if value is None:
                continue

            data.append([
                Paragraph(
                    esc(key),
                    small_style
                ),
                Paragraph(
                    f"<b>{esc(value)}</b>",
                    body_style
                )
            ])

        if not data:

            data = [[
                Paragraph(
                    "No reliable value",
                    small_style
                ),
                Paragraph(
                    "—",
                    body_style
                )
            ]]

        table = Table(
            data,
            colWidths=[
                52 * mm,
                120 * mm
            ]
        )

        table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    .45,
                    LINE
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    .30,
                    LINE
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
            ])
        )

        story.append(table)

    def metric_box(
        label,
        value,
        bg=LIGHT
    ):

        return Table(
            [[
                Paragraph(
                    esc(label).upper(),
                    tiny_style
                )
            ], [
                Paragraph(
                    f"<b>{esc(value)}</b>",
                    body_bold_style
                )
            ]],
            colWidths=[
                41 * mm
            ],
            style=[
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    bg
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    .4,
                    LINE
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
            ]
        )

    def route_node(
        title,
        subtitle,
        bg,
        border
    ):

        t = Table(
            [[
                Paragraph(
                    esc(title),
                    node_title
                ),
                Paragraph(
                    "✓",
                    node_title
                )
            ], [
                Paragraph(
                    esc(subtitle),
                    node_small
                ),
                ""
            ]],
            colWidths=[
                68 * mm,
                8 * mm
            ]
        )

        t.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    bg
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    .7,
                    border
                ),
                (
                    "SPAN",
                    (0, 1),
                    (1, 1)
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
            ])
        )

        return t

    def arrow():

        return Paragraph(
            "<para alignment='center'>↓</para>",
            ParagraphStyle(
                "Arrow",
                parent=body_bold_style,
                fontSize=12,
                textColor=TEAL,
                leading=13
            )
        )

    # ========================================================
    # PAGE 1
    # EXECUTIVE SUMMARY + MULTIMODAL ROUTING
    # ========================================================

    story.append(
        Paragraph(
            "GEO AI",
            kicker_style
        )
    )

    story.append(
        Paragraph(
            "Geoscience Computer Vision & Document AI Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Traceable multimodal analysis of legacy geological "
            "documents and geological imagery",
            small_style
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm
        )
    )

    brand_bar = Table(
        [[
            Paragraph(
                "<b>GEO AI</b><br/>v4.0 · Lightweight CPU",
                body_style
            ),
            Paragraph(
                "<b>DOCUMENT AI</b> &nbsp; • &nbsp; "
                "<b>COMPUTER VISION</b> &nbsp; • &nbsp; "
                "<b>MULTIMODAL ROUTING</b> &nbsp; • &nbsp; "
                "<b>TRACEABILITY</b>",
                small_style
            )
        ]],
        colWidths=[
            43 * mm,
            129 * mm
        ]
    )

    brand_bar.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                GREEN_BG
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                .7,
                colors.HexColor("#A7D7C4")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(
        brand_bar
    )

    story.append(
        Spacer(
            1,
            5 * mm
        )
    )

    add_section(
        "Executive Summary"
    )

    route_kind = (
        "Legacy document / multimodal route"
        if bool(
            top.get(
                "is_document",
                False
            )
        )
        else
        "Direct geological image / Computer Vision route"
    )

    kv_table([
        (
            "Source asset",
            Path(
                str(source_path)
            ).name
        ),
        (
            "Analysis route",
            route_kind
        ),
        (
            "Model version",
            str(
                SETTINGS.model_version
            )
        ),
        (
            "Requested profile",
            requested_profile
        ),
        (
            "Resolved profile",
            resolved_profile
        ),
        (
            "Analysis detail",
            analysis_detail_level
        ),
        (
            "Runtime device",
            runtime_device
        ),
        (
            "Profile fallback",
            "YES" if runtime_fallback else "NO"
        ),
        (
            "Routing confidence",
            (
                f"{float(top.get('routing_confidence', 0) or 0):.0%}"
            )
        ),
        (
            "Reported porosity",
            entity("porosity")
        ),
        (
            "Visible 2D pore area",
            (
                fmt_percent(
                    rs.get(
                        "visible_2d_porosity_pct"
                    )
                )
            )
        ),
        (
            "Detected objects",
            rs.get("objects")
        ),
        (
            "Fracture candidates",
            rs.get("fractures")
        ),
    ])

    story.append(
        Spacer(
            1,
            5 * mm
        )
    )

    add_section(
        "Multimodal AI Routing"
    )

    story.append(
        Paragraph(
            "The workflow below represents the branches actually "
            "selected for this analysis run.",
            small_style
        )
    )

    story.append(
        Spacer(
            1,
            2.5 * mm
        )
    )

    # Source
    story.append(
        route_node(
            "01 · LEGACY ASSET",
            Path(
                str(source_path)
            ).name,
            colors.HexColor("#F1F5F9"),
            colors.HexColor("#A9B7C8")
        )
    )

    story.append(
        arrow()
    )

    # Router
    router_conf = (
        f"{float(top.get('routing_confidence', 0) or 0):.0%}"
    )

    story.append(
        route_node(
            "02 · CONTENT ROUTER",
            f"{route_kind} · confidence {router_conf}",
            colors.HexColor("#EAF9F7"),
            TEAL
        )
    )

    story.append(
        arrow()
    )

    # Parallel branches
    doc_branch = route_node(
        "DOCUMENT AI",
        (
            "OCR / active text → geological entities"
            if bool(doc_text)
            else
            "No primary document-text branch"
        ),
        BLUE_BG,
        BLUE
    )

    cv_branch = route_node(
        "COMPUTER VISION",
        (
            "Rock Vision → segmentation → objects → measurements"
            if interp is not None
            else
            "No accepted geological figure"
        ),
        GOLD_BG,
        GOLD
    )

    branches = Table(
        [[
            doc_branch,
            cv_branch
        ]],
        colWidths=[
            83 * mm,
            83 * mm
        ]
    )

    branches.setStyle(
        TableStyle([
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                0
            ),
        ])
    )

    story.append(
        branches
    )

    story.append(
        arrow()
    )

    story.append(
        route_node(
            "STRUCTURED GEOSCIENCE DATA",
            "Unified results · provenance · traceability",
            GREEN_BG,
            GREEN
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm
        )
    )

    story.append(
        Paragraph(
            "<b>Scope.</b> Reported document properties and "
            "image-derived measurements are intentionally kept "
            "separate. Visible 2D pore area is an image-segmentation "
            "measurement and is not laboratory porosity.",
            small_style
        )
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # PAGE 2
    # CONTENT ROUTING + BEFORE / AFTER
    # ========================================================

    add_page_heading(
        "02",
        "VISUAL AI",
        "Content Routing & Geological Image Analysis",
        "Document-level routing followed by routed geological "
        "image interpretation"
    )

    add_section(
        "Content Routing Map"
    )

    routed_img = make_image(
        routed_rgb,
        f"{safe_stem}_routing.png",
        170,
        91
    )

    if routed_img is not None:

        routing_table = Table(
            [[routed_img]],
            colWidths=[
                172 * mm
            ]
        )

        routing_table.setStyle(
            TableStyle([
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    .55,
                    LINE
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.white
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
            ])
        )

        story.append(
            routing_table
        )

    else:

        story.append(
            Paragraph(
                "Content routing visualization unavailable.",
                body_style
            )
        )

    story.append(
        Spacer(
            1,
            5 * mm
        )
    )

    add_section(
        "Routed Geological Image → AI Interpretation"
    )

    before_arr = (
        crop
        if crop is not None
        else source_rgb
    )

    before_img = make_image(
        before_arr,
        f"{safe_stem}_before.png",
        80,
        78
    )

    after_img = make_image(
        interp,
        f"{safe_stem}_after.png",
        80,
        78
    )

    before_content = [
        Paragraph(
            "<b>01 · BEFORE</b><br/>"
            "Routed geological image",
            body_style
        )
    ]

    if before_img is not None:
        before_content.append(
            before_img
        )

    after_content = [
        Paragraph(
            "<b>02 · AFTER</b><br/>"
            "AI geological interpretation",
            body_style
        )
    ]

    if after_img is not None:
        after_content.append(
            after_img
        )

    visual_table = Table(
        [[
            before_content,
            after_content
        ]],
        colWidths=[
            84 * mm,
            84 * mm
        ]
    )

    visual_table.setStyle(
        TableStyle([
            (
                "BOX",
                (0, 0),
                (-1, -1),
                .55,
                LINE
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                .40,
                LINE
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.white
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(
        visual_table
    )

    story.append(
        Spacer(
            1,
            4 * mm
        )
    )

    legend = Table(
        [[
            Paragraph(
                "<b>G</b> · Grain",
                small_style
            ),
            Paragraph(
                "<b>P</b> · Pore",
                small_style
            ),
            Paragraph(
                "<b>V</b> · Vug",
                small_style
            ),
            Paragraph(
                "<b>F</b> · Fracture candidate",
                small_style
            ),
            Paragraph(
                "<b>C</b> · Cement / clay",
                small_style
            )
        ]],
        colWidths=[
            30 * mm,
            27 * mm,
            25 * mm,
            48 * mm,
            39 * mm
        ]
    )

    legend.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, 0),
                colors.HexColor("#FFF4C9")
            ),
            (
                "BACKGROUND",
                (1, 0),
                (1, 0),
                colors.HexColor("#E8F0FF")
            ),
            (
                "BACKGROUND",
                (2, 0),
                (2, 0),
                colors.HexColor("#DFF8F5")
            ),
            (
                "BACKGROUND",
                (3, 0),
                (3, 0),
                colors.HexColor("#FDE8E6")
            ),
            (
                "BACKGROUND",
                (4, 0),
                (4, 0),
                colors.HexColor("#F1E8FF")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                .4,
                LINE
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                .25,
                LINE
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4
            ),
        ])
    )

    story.append(
        legend
    )

    story.append(
        PageBreak()
    )

    # ========================================================
    # PAGE 3
    # DOCUMENT AI + GEOLOGICAL INTERPRETATION
    # ========================================================

    add_page_heading(
        "03",
        "DOCUMENT AI + COMPUTER VISION",
        "Recognized Geological Content & AI Interpretation",
        "Reported geological information and image-derived "
        "measurements shown separately"
    )

    add_section(
        "Reported Geological Properties"
    )

    properties = [
        (
            "Sample",
            entity("sample_id")
        ),
        (
            "Formation",
            entity("formation")
        ),
        (
            "Lithology",
            entity("lithology")
        ),
        (
            "Depth",
            entity("depth")
        ),
        (
            "Reported porosity",
            entity("porosity")
        ),
        (
            "Reported permeability",
            entity("permeability")
        ),
        (
            "Water saturation",
            entity("water_saturation")
        ),
        (
            "Fracture description",
            entity("fracture_description")
        ),
    ]

    kv_table(
        properties
    )

    story.append(
        Spacer(
            1,
            4 * mm
        )
    )

    add_section(
        "Image-Derived Measurements"
    )

    metrics = Table(
        [[
            metric_box(
                "Visible 2D pore area",
                fmt_percent(
                    rs.get(
                        "visible_2d_porosity_pct"
                    )
                ),
                BLUE_BG
            ),
            metric_box(
                "Objects",
                rs.get(
                    "objects",
                    "—"
                ),
                GREEN_BG
            ),
            metric_box(
                "Grains",
                rs.get(
                    "grains",
                    "—"
                ),
                GOLD_BG
            ),
            metric_box(
                "Pores",
                rs.get(
                    "pores",
                    "—"
                ),
                BLUE_BG
            )
        ], [
            metric_box(
                "Vugs",
                rs.get(
                    "vugs",
                    "—"
                ),
                colors.HexColor("#EAF9F7")
            ),
            metric_box(
                "Fractures",
                rs.get(
                    "fractures",
                    "—"
                ),
                colors.HexColor("#FDECEA")
            ),
            metric_box(
                "Cement / clay",
                rs.get(
                    "cement_clay",
                    "—"
                ),
                colors.HexColor("#F3ECFF")
            ),
            metric_box(
                "Fracture orientation",
                fmt_orientation(
                    rs.get(
                        "dominant_fracture_orientation_deg"
                    )
                ),
                colors.HexColor("#FDECEA")
            )
        ]],
        colWidths=[
            43 * mm,
            43 * mm,
            43 * mm,
            43 * mm
        ]
    )

    metrics.setStyle(
        TableStyle([
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                1.5
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                1.5
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                1.5
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                1.5
            ),
        ])
    )

    story.append(
        metrics
    )

    story.append(
        Spacer(
            1,
            4 * mm
        )
    )

    add_section(
        "Preliminary Geological Interpretation"
    )

    interpretation_parts = []

    if rs.get("fractures") is not None:

        interpretation_parts.append(
            f"{rs.get('fractures')} fracture candidate(s)"
        )

    if rs.get("pores") is not None:

        interpretation_parts.append(
            f"{rs.get('pores')} pore object(s)"
        )

    if rs.get("vugs") is not None:

        interpretation_parts.append(
            f"{rs.get('vugs')} vug object(s)"
        )

    if rs.get("grains") is not None:

        interpretation_parts.append(
            f"{rs.get('grains')} grain object(s)"
        )

    if interpretation_parts:

        interpretation_text = (
            "The routed geological image contains "
            + ", ".join(
                interpretation_parts
            )
            + ". "
        )

    else:

        interpretation_text = (
            "No quantitative Computer Vision summary "
            "was available for this input. "
        )

    if rs.get(
        "visible_2d_porosity_pct"
    ) is not None:

        interpretation_text += (
            "The segmented pore-class area occupies "
            f"approximately "
            f"{float(rs.get('visible_2d_porosity_pct')):.2f}% "
            "of the analyzed 2D image. "
        )

    if rs.get(
        "dominant_fracture_orientation_deg"
    ) is not None:

        interpretation_text += (
            "The median detected fracture orientation is "
            f"approximately "
            f"{float(rs.get('dominant_fracture_orientation_deg')):.1f}° "
            "in image coordinates."
        )

    interpretation_box = Table(
        [[
            Paragraph(
                esc(
                    interpretation_text
                ),
                body_style
            )
        ]],
        colWidths=[
            172 * mm
        ]
    )

    interpretation_box.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                GREEN_BG
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                .5,
                colors.HexColor("#CFE5DA")
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
        ])
    )

    story.append(
        interpretation_box
    )

    story.append(
        Spacer(
            1,
            4 * mm
        )
    )

    add_section(
        "Recognized Content"
    )

    clean_text = str(
        doc_text or ""
    ).strip()

    if clean_text:

        clean_text = clean_text.replace(
            "--- Page 1 ---",
            ""
        ).strip()

        paragraphs = [
            p.strip()
            for p in clean_text.split("\n")
            if p.strip()
        ]

        # Deliberately bounded excerpt.
        # Full raw OCR remains available in the web UI.
        for paragraph in paragraphs[:8]:

            story.append(
                Paragraph(
                    esc(
                        paragraph[:900]
                    ),
                    small_style
                )
            )

            story.append(
                Spacer(
                    1,
                    1.2 * mm
                )
            )

    else:

        story.append(
            Paragraph(
                "No recognized document text was used "
                "as the primary route for this input.",
                body_style
            )
        )

    story.append(
        PageBreak()
    )

    # ========================================================
    # PAGE 4
    # TRACEABILITY + METHODOLOGY
    # ========================================================

    add_page_heading(
        "04",
        "TRACEABILITY",
        "Routing, Provenance & Methodology",
        "Analysis decisions and algorithms executed in this run"
    )

    add_section(
        "Routing Decisions"
    )

    if (
        isinstance(
            routes,
            pd.DataFrame
        )
        and not routes.empty
    ):

        route_rows = []

        route_rows.append([
            Paragraph(
                f"<b>{esc(col)}</b>",
                tiny_style
            )
            for col in routes.columns
        ])

        for row in routes.head(
            10
        ).itertuples(
            index=False
        ):

            route_rows.append([
                Paragraph(
                    esc(str(value)[:120]),
                    tiny_style
                )
                for value in row
            ])

        route_table = Table(
            route_rows,
            repeatRows=1
        )

        route_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    GREEN_BG
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    .35,
                    LINE
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4
                ),
            ])
        )

        story.append(
            route_table
        )

    else:

        story.append(
            Paragraph(
                "No routing table was produced.",
                body_style
            )
        )

    story.append(
        Spacer(
            1,
            5 * mm
        )
    )

    add_section(
        "Algorithms Actually Executed"
    )

    if (
        isinstance(
            exec_df,
            pd.DataFrame
        )
        and not exec_df.empty
    ):

        exec_rows_pdf = []

        exec_rows_pdf.append([
            Paragraph(
                f"<b>{esc(col)}</b>",
                tiny_style
            )
            for col in exec_df.columns
        ])

        for row in exec_df.head(
            16
        ).itertuples(
            index=False
        ):

            exec_rows_pdf.append([
                Paragraph(
                    esc(str(value)[:95]),
                    tiny_style
                )
                for value in row
            ])

        exec_table = Table(
            exec_rows_pdf,
            repeatRows=1
        )

        exec_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    BLUE_BG
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    .30,
                    LINE
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    3
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    3
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    3
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    3
                ),
            ])
        )

        story.append(
            exec_table
        )

    story.append(
        Spacer(
            1,
            5 * mm
        )
    )

    add_section(
        "Object Provenance"
    )

    if (
        isinstance(
            trace,
            pd.DataFrame
        )
        and not trace.empty
    ):

        preferred_cols = [
            c
            for c in [
                "source_document",
                "page_number",
                "image_id",
                "object_id",
                "predicted_class",
                "confidence",
                "area_px"
            ]
            if c in trace.columns
        ]

        if preferred_cols:

            trace_rows = [[
                Paragraph(
                    f"<b>{esc(col)}</b>",
                    tiny_style
                )
                for col in preferred_cols
            ]]

            for _, row in trace[
                preferred_cols
            ].head(8).iterrows():

                trace_rows.append([
                    Paragraph(
                        esc(str(row[col])[:70]),
                        tiny_style
                    )
                    for col in preferred_cols
                ])

            trace_table = Table(
                trace_rows,
                repeatRows=1
            )

            trace_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        GOLD_BG
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        .28,
                        LINE
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP"
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        5.4
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        3
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        3
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        3
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        3
                    ),
                ])
            )

            story.append(
                trace_table
            )

    story.append(
        Spacer(
            1,
            5 * mm
        )
    )

    limitation = Table(
        [[
            Paragraph(
                "<b>Prototype limitation.</b> "
                "This report records outputs produced by the "
                "Lightweight CPU prototype. Results are intended "
                "for research, portfolio demonstration, and workflow "
                "prototyping. They are not validated for operational "
                "geological or petrophysical decision-making.",
                small_style
            )
        ]],
        colWidths=[
            172 * mm
        ]
    )

    limitation.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#FFF8EE")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                .45,
                colors.HexColor("#E9D4AF")
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
        ])
    )

    story.append(
        limitation
    )

    # ========================================================
    # PAGE DECORATION
    # ========================================================

    def decorate_page(
        canvas,
        document
    ):

        canvas.saveState()

        width, height = A4

        # top GEO AI brand line
        canvas.setStrokeColor(
            GREEN
        )

        canvas.setLineWidth(
            2.0
        )

        canvas.line(
            15 * mm,
            height - 9.5 * mm,
            width - 15 * mm,
            height - 9.5 * mm
        )

        # small secondary blue segment
        canvas.setStrokeColor(
            BLUE
        )

        canvas.setLineWidth(
            2.0
        )

        canvas.line(
            width - 55 * mm,
            height - 9.5 * mm,
            width - 15 * mm,
            height - 9.5 * mm
        )

        # footer
        canvas.setFillColor(
            GREEN
        )

        canvas.setFont(
            "Helvetica-Bold",
            6.5
        )

        canvas.drawString(
            15 * mm,
            7.5 * mm,
            "GEO AI"
        )

        canvas.setFillColor(
            MUTED
        )

        canvas.setFont(
            "Helvetica",
            6.2
        )

        canvas.drawString(
            29 * mm,
            7.5 * mm,
            "Geoscience Computer Vision & Document AI Toolkit"
        )

        canvas.drawRightString(
            width - 15 * mm,
            7.5 * mm,
            f"v4.0  ·  Page {document.page}"
        )

        canvas.restoreState()

    # ========================================================
    # BUILD
    # ========================================================

    doc.build(
        story,
        onFirstPage=decorate_page,
        onLaterPages=decorate_page
    )

    return str(
        pdf_path
    )



def multimodal_pipeline(
    file,
    model_profile="AUTO",
    analysis_detail="Standard"
):
    if not file:
        raise gr.Error(
            'Upload a PDF/page or geological image.'
        )

    t0 = time.perf_counter()

    runtime_resolution = resolve_model_profile(
        model_profile
    )

    caps = runtime_resolution[
        "capabilities"
    ]

    requested_profile = runtime_resolution[
        "requested"
    ]

    resolved_profile = runtime_resolution[
        "resolved"
    ]

    profile_fallback = bool(
        runtime_resolution[
            "fallback"
        ]
    )

    detail_level = (
        analysis_detail
        if analysis_detail in (
            "Standard",
            "Detailed",
            "Deep"
        )
        else "Standard"
    )

    if resolved_profile == "Lightweight CPU":
        runtime_device = "CPU"

    else:
        runtime_device = (
            caps.get("gpu_name")
            or "CUDA GPU"
        )

    runtime_meta = {
        "requested_profile": requested_profile,
        "resolved_profile": resolved_profile,
        "analysis_detail": detail_level,
        "device": runtime_device,
        "fallback": profile_fallback,
        "resolution_reason": runtime_resolution.get(
            "reason",
            ""
        ),
        "cuda_available": bool(
            caps.get(
                "cuda_available",
                False
            )
        ),
        "gpu_name": caps.get(
            "gpu_name"
        ),
    }

    path = ensure_path(file)

    rgb = ingest(
        path,
        1
    )[0][0]['rgb']

    top = classify_input(rgb)

    doc_text = ''
    entities = pd.DataFrame()
    crop = None
    interp = None
    trace = pd.DataFrame()
    routes = []
    exec_rows = []
    rock_summary_data = {}

    exec_rows.append([
        'Runtime profile',
        requested_profile,
        resolved_profile,
        runtime_device,
        'FALLBACK' if profile_fallback else 'RESOLVED',
        (
            f'analysis_detail={detail_level}; '
            f'{runtime_resolution.get("reason", "")}'
        )
    ])

    exec_rows.append([
        'Input router',
        'OCR evidence + page/image statistics',
        'lightweight-router',
        runtime_device,
        'EXECUTED',
        top['input_kind']
    ])
    if top['is_document']:
        doc=analyze_document(path,max_pages=1); doc_text=doc['text'][:7000]; entities=doc['entities']; regions,crops=detect_figure_regions(rgb); texts=detect_text_regions(rgb); routed=draw_routing_map(rgb,regions,None)
        routes.append({'region':'PAGE-TEXT','type':'TEXT','route':'OCR / active text → geological entities','confidence':round(top['ocr_confidence'],3)})
        exec_rows += [['Layout approximation','OpenCV contours + OCR evidence','lightweight-layout','CPU','EXECUTED','TEXT / FIGURE candidate regions']]
        ddf,_=doc_execution(doc,0); exec_rows += ddf.iloc[:-1].values.tolist()[1:]
        chosen=None
        for i,c in enumerate(crops):
            if not classify_input(c)['is_document']: chosen=(i,c); break
        if chosen:
            i,crop=chosen; tmp=SETTINGS.output_dir/'_routed_figure.png'; cv2.imwrite(str(tmp),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR)); rr=analyze_rock(
                tmp,
                detail=detail_level,
                backend=resolved_profile
            ); rock_summary_data=rr.get('summary',{}); interp=rr['annotated']; trace=_safe_df(rr['objects'],40); trace['source_document']=Path(path).name; trace['page_number']=1; trace['image_id']=str(regions.iloc[i].region_id); routes.append({'region':str(regions.iloc[i].region_id),'type':'FIGURE','route':'semantic → instance → measurements','confidence':float(regions.iloc[i].confidence)}); rdf,_=rock_execution(0); exec_rows += rdf.iloc[:-1].values.tolist()[1:]
        note="<div class='card'><span class='badge'>DOCUMENT ROUTE</span> Green = TEXT → text extraction/OCR. Teal = FIGURE candidate → Rock Vision only after the content gate accepts it.</div>"
        graph_lines=['SOURCE DOCUMENT / PAGE','  ↓','Input router · document route','  ↓','Layout approximation · TEXT / FIGURE candidates','  ├── TEXT → active text or Tesseract OCR → geological entities','  └── FIGURE → Rock Vision','                 ↓ semantic segmentation','                 ↓ instance segmentation','                 ↓ fracture candidates / measurements','  ↓','Unified traceability · document → page → figure → object → measurement']
    else:
        routed=rgb.copy(); rr=analyze_rock(
            path,
            detail=detail_level,
            backend=resolved_profile
        ); rock_summary_data=rr.get('summary',{}); crop=rr['original']; interp=rr['annotated']; trace=_safe_df(rr['objects'],40); trace['source_document']=Path(path).name; trace['page_number']=1; routes.append({'region':'WHOLE IMAGE','type':'GEOLOGICAL IMAGE / COLLAGE','route':'Rock Vision directly · OCR skipped','confidence':top['routing_confidence']}); note="<div class='card'><span class='badge'>GEOLOGICAL IMAGE ROUTE</span> Input treated as geological visual/collage. OCR skipped as the primary route.</div>"; rdf,_=rock_execution(0); exec_rows += rdf.iloc[:-1].values.tolist()[1:]
        graph_lines=['SOURCE GEOLOGICAL IMAGE / COLLAGE','  ↓','Input router · geological-image route','  ↓','Rock Vision directly · OCR skipped','  ↓ semantic segmentation','  ↓ instance segmentation','  ↓ fracture candidates','  ↓ object quantification','  ↓','Traceability · source → image → object → measurement']
    if (
        rock_summary_data
        and rock_summary_data.get(
            "executed_backend"
        ) == "Advanced GPU"
    ):

        exec_rows.append([
            "Advanced GPU backend",
            rock_summary_data.get(
                "gpu_backend",
                "geo-ai-advanced-gpu-v1"
            ),
            "CUDA runtime",
            rock_summary_data.get(
                "gpu_device_name",
                runtime_device
            ),
            "EXECUTED",
            (
                f'CUDA preprocessing / feature preparation · '
                f'{rock_summary_data.get("gpu_runtime_ms")} ms · '
                f'{rock_summary_data.get("gpu_peak_vram_mb")} MB peak VRAM'
            )
        ])

    ms = _runtime_ms(t0)

    exec_rows.append([
        'Total multimodal inference',
        f'{ms} ms',
        'this run',
        runtime_device,
        'EXECUTED',
        (
            f'routed structured result · '
            f'{resolved_profile} · {detail_level}'
        )
    ])
    exec_df = mm_exec(exec_rows)
    graph=_graph('Multimodal Pipeline · execution graph',graph_lines)
    routing_html = build_routing_map(
        top,
        figure_count=len(regions) if top['is_document'] and 'regions' in locals() else 0,
        text_detected=bool(doc_text.strip())
    )
    document_html = build_document_ai_result(doc_text,entities) if doc_text else ''
    interpretation_html = build_geological_interpretation(rock_summary_data,entities) if rock_summary_data else ''

    routes_df = pd.DataFrame(routes)

    if rock_summary_data:

        actual_backend = (
            rock_summary_data.get(
                "executed_backend"
            )
        )

        if actual_backend:

            runtime_meta[
                "resolved_profile"
            ] = actual_backend

        gpu_name = (
            rock_summary_data.get(
                "gpu_device_name"
            )
        )

        if gpu_name:

            runtime_meta[
                "device"
            ] = gpu_name

        runtime_meta[
            "gpu_runtime_ms"
        ] = rock_summary_data.get(
            "gpu_runtime_ms"
        )

        runtime_meta[
            "gpu_peak_vram_mb"
        ] = rock_summary_data.get(
            "gpu_peak_vram_mb"
        )

    try:
        report_path = generate_multimodal_pdf_report(
            source_path=path,
            source_rgb=rgb,
            routed_rgb=routed,
            doc_text=doc_text,
            entities=entities,
            routes=routes_df,
            crop=crop,
            interp=interp,
            rock_summary=rock_summary_data,
            trace=trace,
            exec_df=exec_df,
            top=top,
            runtime_meta=runtime_meta
        )

        report_status = (
            "<div class='report-ready-card'>"
            "<div class='report-ready-icon'>✓</div>"
            "<div class='report-ready-copy'>"
            "<span>ANALYSIS COMPLETE</span>"
            "<h3>Professional PDF Report Ready</h3>"
            "<p>Generated from this exact multimodal inference run · "
            "Document AI · Computer Vision · routing · traceability</p>"
            "</div>"
            "<div class='report-ready-badge'>PDF · 4 PAGES</div>"
            "</div>"
        )

    except Exception as exc:

        report_path = None

        report_status = (
            "<div class='report-ready-card report-generation-error'>"
            "<div class='report-ready-icon'>!</div>"
            "<div class='report-ready-copy'>"
            "<span>ANALYSIS COMPLETE</span>"
            "<h3>Analysis succeeded · PDF generation unavailable</h3>"
            "<p>"
            + html.escape(str(exc))
            + "</p>"
            "</div>"
            "</div>"
        )

    return (
        rgb,
        routed,
        note,
        routing_html,
        document_html,
        doc_text,
        entities,
        routes_df,
        crop,
        interp,
        interpretation_html,
        trace,
        exec_df,
        graph,
        report_status,
        report_path
    )

def demo_mm(): return str(EX/'multimodal_legacy_report.png')
def info():
    ok,_=configure_tesseract(); langs=available_languages() if ok else []
    return f"""<div class='card'><b>Project information</b><br>Version: <code>{SETTINGS.model_version}</code> · Device: CPU lightweight edition · Tesseract: {'READY' if ok else 'NOT FOUND'} · OCR languages: {html.escape(', '.join(langs) or 'none')}<br><span class='muted'>Only algorithms actually executed are labeled EXECUTED IN THIS RUN. SAM 2, U-Net, YOLO-Seg and transformer models belong to the planned GPU/research edition and are not claimed here.</span></div>"""


# ============================================================
# PREMIUM UI - FINAL OVERRIDE LAYER
# ============================================================

_PREMIUM_UI = Path(__file__).resolve().parent / "assets" / "premium_ui.css"

if _PREMIUM_UI.exists():
    CSS += "\n" + _PREMIUM_UI.read_text(
        encoding="utf-8"
    )

theme=gr.themes.Base(primary_hue='teal',secondary_hue='slate',neutral_hue='gray')

# ============================================================
# v4.0 UI OVERRIDES
# ============================================================

CSS += r"""

/* ---------- MODEL PROFILE ---------- */

.profile-shell{
    margin:12px 0 16px 0;
    padding:14px 16px;
    border:1px solid var(--border);
    border-radius:16px;
    background:var(--card);
}

.profile-status{
    padding:11px 14px;
    border:1px solid var(--border);
    border-radius:12px;
    background:var(--code);
    color:var(--text)!important;
    line-height:1.45;
}

.profile-status.unavailable{
    border-style:dashed;
}

.profile-help{
    margin-top:8px;
    color:var(--muted)!important;
    font-size:.82rem;
    line-height:1.55;
}

.class-caption{
    margin:7px 0 3px 0;
    color:var(--muted)!important;
    font-size:.78rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.04em;
}

/* ---------- DARK MODE ---------- */

html[data-geo-theme='dark'] body,
html[data-geo-theme='dark'] .gradio-container{
    background:var(--bg)!important;
    color:var(--text)!important;
}

html[data-geo-theme='dark'] .gradio-container .block,
html[data-geo-theme='dark'] .gradio-container .form,
html[data-geo-theme='dark'] .gradio-container .wrap,
html[data-geo-theme='dark'] .gradio-container .panel{
    background:var(--card)!important;
    color:var(--text)!important;
    border-color:var(--border)!important;
}

html[data-geo-theme='dark'] .gradio-container label,
html[data-geo-theme='dark'] .gradio-container .label-wrap,
html[data-geo-theme='dark'] .gradio-container .label-wrap span{
    background:var(--card)!important;
    color:var(--text)!important;
    border-color:var(--border)!important;
}

html[data-geo-theme='dark'] .gradio-container input,
html[data-geo-theme='dark'] .gradio-container textarea,
html[data-geo-theme='dark'] .gradio-container select{
    background:var(--code)!important;
    color:var(--text)!important;
    border-color:var(--border)!important;
}

html[data-geo-theme='dark'] .gradio-container table,
html[data-geo-theme='dark'] .gradio-container thead,
html[data-geo-theme='dark'] .gradio-container tbody,
html[data-geo-theme='dark'] .gradio-container tr,
html[data-geo-theme='dark'] .gradio-container th,
html[data-geo-theme='dark'] .gradio-container td{
    background:var(--card)!important;
    color:var(--text)!important;
    border-color:var(--border)!important;
}

html[data-geo-theme='dark'] .gradio-container .tab-nav button{
    color:#cbd5e1!important;
}

html[data-geo-theme='dark'] .gradio-container .tab-nav button.selected{
    color:var(--accent)!important;
    border-color:var(--accent)!important;
}

html[data-geo-theme='dark'] .gradio-container .file-preview,
html[data-geo-theme='dark'] .gradio-container .file-preview-holder{
    background:var(--card)!important;
    color:var(--text)!important;
}

/* Keep geological class legend colors unchanged */
.legend{
    background:transparent!important;
}

@media (max-width:768px){
    .profile-shell{
        padding:10px!important;
    }

    .profile-help{
        font-size:.76rem!important;
    }
}

"""



# ============================================================
# STAGE 3B — RUNTIME / HARDWARE DETECTOR
# ============================================================

def detect_runtime_capabilities():
    """
    Detect capabilities available to this running process.

    Important:
    GPU hardware availability and GPU-model availability are
    intentionally treated as separate concepts.

    A CUDA GPU does NOT imply that Advanced / Research model
    weights are installed.
    """

    info = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count() or 1,

        "torch_installed": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "gpu_name": None,
        "cuda_version": None,

        "advanced_weights": False,
        "research_weights": False,

        "advanced_backend_installed": False,
        "advanced_backend_executable": False,

        "lightweight_available": True,
    }

    # --------------------------------------------------------
    # PyTorch / CUDA
    # --------------------------------------------------------

    try:
        import torch

        info["torch_installed"] = True

        info["cuda_available"] = bool(
            torch.cuda.is_available()
        )

        info["cuda_version"] = (
            getattr(
                torch.version,
                "cuda",
                None
            )
        )

        if info["cuda_available"]:

            info["cuda_device_count"] = int(
                torch.cuda.device_count()
            )

            if info["cuda_device_count"] > 0:

                info["gpu_name"] = str(
                    torch.cuda.get_device_name(0)
                )

    except Exception:
        # PyTorch is optional in the Lightweight deployment.
        pass


    # --------------------------------------------------------
    # Advanced GPU backend
    # --------------------------------------------------------

    try:

        from src.rock_vision.backends.advanced_gpu import (
            advanced_gpu_available
        )

        info[
            "advanced_backend_installed"
        ] = True

        gpu_backend_info = (
            advanced_gpu_available()
        )

        info[
            "advanced_backend_executable"
        ] = bool(
            gpu_backend_info.available
        )

        if (
            gpu_backend_info.device_name
            and not info.get("gpu_name")
        ):
            info["gpu_name"] = (
                gpu_backend_info.device_name
            )

    except Exception:

        pass


    # --------------------------------------------------------
    # Model weight directories
    # --------------------------------------------------------

    model_root = (
        PROJECT_ROOT / "models"
        if "PROJECT_ROOT" in globals()
        else Path(__file__).resolve().parent / "models"
    )

    advanced_candidates = [
        model_root / "advanced",
        model_root / "unet",
        model_root / "sam2",
        model_root / "yolo_seg",
    ]

    research_candidates = [
        model_root / "research",
        model_root / "vit",
        model_root / "transformers",
    ]


    def has_model_files(paths):

        extensions = {
            ".pt",
            ".pth",
            ".onnx",
            ".ckpt",
            ".safetensors",
        }

        for path in paths:

            if not path.exists():
                continue

            if path.is_file():

                if path.suffix.lower() in extensions:
                    return True

            else:

                try:

                    for candidate in path.rglob("*"):

                        if (
                            candidate.is_file()
                            and candidate.suffix.lower()
                            in extensions
                        ):
                            return True

                except Exception:
                    pass

        return False


    info["advanced_weights"] = has_model_files(
        advanced_candidates
    )

    info["research_weights"] = has_model_files(
        research_candidates
    )


    # --------------------------------------------------------
    # Executable profile availability
    # --------------------------------------------------------

    info["advanced_gpu_available"] = bool(
        info["cuda_available"]
        and info["advanced_backend_executable"]
    )

    info["research_gpu_available"] = bool(
        info["cuda_available"]
        and info["research_weights"]
    )


    # --------------------------------------------------------
    # AUTO resolution
    # --------------------------------------------------------

    if info["research_gpu_available"]:

        resolved = "Research GPU"

    elif info["advanced_gpu_available"]:

        resolved = "Advanced GPU"

    else:

        resolved = "Lightweight CPU"


    info["auto_resolved_profile"] = resolved

    return info



def resolve_model_profile(
    requested_profile,
    capabilities=None
):
    """
    Resolve requested Model Profile to the best executable
    backend currently installed.

    Profiles
    --------
    AUTO
        Hardware-aware automatic selection.

    Lightweight CPU
        Portable deterministic baseline. Always available.

    Advanced GPU
        CUDA-accelerated learned segmentation backend.
        Requires CUDA + validated advanced model weights/backend.

    Research GPU
        Experimental foundation/research backend.
        Requires CUDA + research model weights/backend.

    Important
    ---------
    GPU hardware availability alone does not imply that a
    geological GPU model is installed.
    """

    caps = (
        capabilities
        if capabilities is not None
        else detect_runtime_capabilities()
    )

    requested = (
        requested_profile
        if requested_profile
        else "AUTO"
    )

    advanced_ready = bool(
        caps.get(
            "advanced_gpu_available",
            False
        )
    )

    research_ready = bool(
        caps.get(
            "research_gpu_available",
            False
        )
    )

    cuda_ready = bool(
        caps.get(
            "cuda_available",
            False
        )
    )


    # --------------------------------------------------------
    # AUTO
    # --------------------------------------------------------

    if requested == "AUTO":

        if research_ready:

            resolved = "Research GPU"

            reason = (
                "Research GPU backend and validated weights "
                "are available."
            )

        elif advanced_ready:

            resolved = "Advanced GPU"

            reason = (
                "Advanced GPU backend and validated weights "
                "are available."
            )

        else:

            resolved = "Lightweight CPU"

            if cuda_ready:

                reason = (
                    "CUDA GPU detected, but no validated "
                    "geological GPU backend/weights are installed. "
                    "Using portable CPU pipeline."
                )

            else:

                reason = (
                    "CUDA GPU is unavailable. "
                    "Using portable CPU pipeline."
                )

        return {
            "requested": "AUTO",
            "resolved": resolved,
            "fallback": False,
            "availability": "AVAILABLE",
            "reason": reason,
            "capabilities": caps,
        }


    # --------------------------------------------------------
    # HARDWARE-AWARE
    # --------------------------------------------------------

    if requested == "Lightweight CPU":

        return {
            "requested": requested,
            "resolved": "Lightweight CPU",
            "fallback": False,
            "availability": "AVAILABLE",
            "reason": (
                "Portable CPU inference profile."
            ),
            "capabilities": caps,
        }


    # --------------------------------------------------------
    # ADVANCED GPU
    # --------------------------------------------------------

    if requested == "Advanced GPU":

        if advanced_ready:

            return {
                "requested": requested,
                "resolved": "Advanced GPU",
                "fallback": False,
                "availability": "AVAILABLE",
                "reason": (
                    "CUDA-accelerated Advanced GPU backend "
                    "is executable."
                ),
                "capabilities": caps,
            }

        return {
            "requested": requested,
            "resolved": "Lightweight CPU",
            "fallback": True,
            "availability": "NOT INSTALLED",
            "reason": (
                "Advanced GPU is an optional accelerated "
                "segmentation profile. CUDA hardware may be "
                "available, but a validated geological GPU "
                "checkpoint/backend is not installed."
            ),
            "capabilities": caps,
        }


    # --------------------------------------------------------
    # RESEARCH GPU
    # --------------------------------------------------------

    if requested == "Research GPU":

        if research_ready:

            return {
                "requested": requested,
                "resolved": "Research GPU",
                "fallback": False,
                "availability": "EXPERIMENTAL",
                "reason": (
                    "Experimental research GPU backend and "
                    "research weights are available."
                ),
                "capabilities": caps,
            }

        return {
            "requested": requested,
            "resolved": "Lightweight CPU",
            "fallback": True,
            "availability": "EXPERIMENTAL · NOT INSTALLED",
            "reason": (
                "Research GPU is an optional experimental "
                "foundation-model extension and is not installed "
                "in this deployment."
            ),
            "capabilities": caps,
        }


    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return {
        "requested": requested,
        "resolved": "Lightweight CPU",
        "fallback": True,
        "availability": "UNKNOWN",
        "reason": (
            "Unknown profile. Using Lightweight CPU."
        ),
        "capabilities": caps,
    }



def runtime_profile_message(profile):
    """
    Human-readable runtime status for the Model Profile UI.
    """

    result = resolve_model_profile(
        profile
    )

    caps = result[
        "capabilities"
    ]

    requested = result[
        "requested"
    ]

    resolved = result[
        "resolved"
    ]


    gpu_text = (
        html.escape(
            caps["gpu_name"]
        )
        if caps.get("gpu_name")
        else "No CUDA GPU available to this process"
    )


    if resolved == "Lightweight CPU":

        stack = (
            "Tesseract · OpenCV · clustering · morphology · "
            "watershed · connected components · regionprops"
        )

    elif resolved == "Advanced GPU":

        stack = (
            "CUDA · advanced segmentation runtime · "
            "validated GPU model weights"
        )

    else:

        stack = (
            "CUDA · research GPU runtime · "
            "research model weights"
        )


    if result["fallback"]:

        badge = "FALLBACK"

        status_line = (
            f"{html.escape(requested.upper())} "
            f"→ {html.escape(resolved.upper())}"
        )

    elif requested == "AUTO":

        badge = "AUTO"

        status_line = (
            f"AUTO → {html.escape(resolved.upper())}"
        )

    else:

        badge = "AVAILABLE"

        status_line = (
            f"{html.escape(resolved.upper())} · AVAILABLE"
        )


    return f"""
    <div class='profile-status'>
        <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap;'>
            <b>{status_line}</b>
            <span class='badge'>{badge}</span>
        </div>

        <div class='muted' style='margin-top:4px;'>
            {html.escape(stack)}
        </div>

        <div class='muted' style='margin-top:4px;font-size:.72rem;'>
            Device: {gpu_text}
            · CPU threads: {int(caps['cpu_count'])}
            · CUDA: {'READY' if caps['cuda_available'] else 'NOT AVAILABLE'}
            · Advanced weights: {'READY' if caps['advanced_weights'] else 'NOT FOUND'}
            · Research weights: {'READY' if caps['research_weights'] else 'NOT FOUND'}
        </div>

        {
            "<div class='muted' style='margin-top:4px;'>"
            + html.escape(result["reason"])
            + "</div>"
            if result["fallback"]
            else ""
        }
    </div>
    """



# ============================================================
# STAGE 3E — PROFILE × DETAIL MATRIX
# ============================================================

DETAIL_UI_PROFILES = {

    "Standard": {
        "title": "STANDARD",
        "description": (
            "Fast quantitative analysis with conservative "
            "object separation."
        ),
        "k": 8,
        "sample_side": 360,
        "area_factor": 1.00,
        "watershed": "conservative",
        "labels": 24,
    },

    "Detailed": {
        "title": "DETAILED",
        "description": (
            "Higher object-level granularity with finer "
            "segmentation and watershed separation."
        ),
        "k": 10,
        "sample_side": 520,
        "area_factor": 0.70,
        "watershed": "finer",
        "labels": 40,
    },

    "Deep": {
        "title": "DEEP",
        "description": (
            "Maximum available CPU detail with the finest "
            "current segmentation and object separation."
        ),
        "k": 12,
        "sample_side": 720,
        "area_factor": 0.45,
        "watershed": "finest",
        "labels": 64,
    },
}


def profile_detail_status(
    profile="AUTO",
    detail="Standard"
):
    """
    Dynamic UI status for requested Model Profile × Analysis Detail.

    This reports requested and executable runtime separately.
    It never claims unavailable GPU models were executed.
    """

    resolution = resolve_model_profile(
        profile
    )

    caps = resolution[
        "capabilities"
    ]

    resolved = resolution[
        "resolved"
    ]

    requested = resolution[
        "requested"
    ]

    fallback = bool(
        resolution[
            "fallback"
        ]
    )

    availability = resolution.get(
        "availability",
        "AVAILABLE"
    )

    cfg = DETAIL_UI_PROFILES.get(
        detail,
        DETAIL_UI_PROFILES["Standard"]
    )

    detail = (
        detail
        if detail in DETAIL_UI_PROFILES
        else "Standard"
    )


    # --------------------------------------------------------
    # Device / runtime
    # --------------------------------------------------------

    if resolved == "Lightweight CPU":

        device = "CPU"

        runtime_stack = (
            "OpenCV · clustering · morphology · watershed · "
            "connected components · regionprops"
        )

    else:

        device = (
            caps.get("gpu_name")
            or "CUDA GPU"
        )

        runtime_stack = (
            "CUDA GPU runtime · validated model weights"
        )


    # --------------------------------------------------------
    # Status badge
    # --------------------------------------------------------

    if fallback:

        if requested == "Research GPU":
            badge = "EXPERIMENTAL · NOT INSTALLED"
        else:
            badge = "NOT INSTALLED · CPU FALLBACK"

        badge_class = "profile-matrix-fallback"

    elif requested == "AUTO":

        badge = "AUTO RESOLVED"
        badge_class = "profile-matrix-auto"

    elif requested == "Research GPU":

        badge = availability
        badge_class = "profile-matrix-ready"

    else:

        badge = availability
        badge_class = "profile-matrix-ready"


    # --------------------------------------------------------
    # Availability text
    # --------------------------------------------------------

    cuda_state = (
        "READY"
        if caps.get("cuda_available")
        else "NOT AVAILABLE"
    )

    advanced_state = (
        "READY"
        if caps.get("advanced_weights")
        else "NOT FOUND"
    )

    research_state = (
        "READY"
        if caps.get("research_weights")
        else "NOT FOUND"
    )


    requested_text = html.escape(
        str(requested)
    )

    resolved_text = html.escape(
        str(resolved)
    )

    detail_text = html.escape(
        detail.upper()
    )


    # --------------------------------------------------------
    # Model / algorithm stack shown in Execution Profile
    # --------------------------------------------------------

    if requested == "Advanced GPU":

        if resolved == "Advanced GPU":

            model_stack = (
                "<div class='profile-matrix-models'>"
                "<span>AI / CV MODELS</span>"
                "<b>PyTorch CUDA</b>"
                "<i></i>"
                "<b>U-Net / U-Net++</b>"
                "<i></i>"
                "<b>DeepLabV3+</b>"
                "<i></i>"
                "<b>YOLO-Seg</b>"
                "<i></i>"
                "<b>SAM 2</b>"
                "<em>GPU-ready architecture</em>"
                "</div>"
            )

        else:

            model_stack = (
                "<div class='profile-matrix-models'>"
                "<span>AI / CV MODELS</span>"
                "<b>U-Net / U-Net++</b>"
                "<i></i>"
                "<b>DeepLabV3+</b>"
                "<i></i>"
                "<b>YOLO-Seg</b>"
                "<i></i>"
                "<b>SAM 2</b>"
                "<em>Advanced GPU · optional</em>"
                "</div>"
            )


    elif requested == "Research GPU":

        model_stack = (
            "<div class='profile-matrix-models'>"
            "<span>RESEARCH MODELS</span>"
            "<b>SAM 2</b>"
            "<i></i>"
            "<b>YOLO-Seg</b>"
            "<i></i>"
            "<b>U-Net++</b>"
            "<i></i>"
            "<b>DeepLabV3+</b>"
            "<i></i>"
            "<b>Ensemble / Fusion</b>"
            "<em>Experimental architecture</em>"
            "</div>"
        )


    elif requested == "AUTO":

        if resolved == "Advanced GPU":

            model_stack = (
                "<div class='profile-matrix-models'>"
                "<span>AUTO MODEL STACK</span>"
                "<b>PyTorch CUDA</b>"
                "<i></i>"
                "<b>U-Net / U-Net++</b>"
                "<i></i>"
                "<b>DeepLabV3+</b>"
                "<i></i>"
                "<b>YOLO-Seg</b>"
                "<i></i>"
                "<b>SAM 2</b>"
                "<em>AUTO → GPU-ready</em>"
                "</div>"
            )

        else:

            model_stack = (
                "<div class='profile-matrix-models'>"
                "<span>AUTO MODEL STACK</span>"
                "<b>OpenCV</b>"
                "<i></i>"
                "<b>K-means</b>"
                "<i></i>"
                "<b>Watershed</b>"
                "<i></i>"
                "<b>Regionprops</b>"
                "<em>AUTO → portable CPU</em>"
                "</div>"
            )


    else:

        model_stack = (
            "<div class='profile-matrix-models'>"
            "<span>ACTIVE ALGORITHMS</span>"
            "<b>OpenCV</b>"
            "<i></i>"
            "<b>K-means</b>"
            "<i></i>"
            "<b>Watershed</b>"
            "<i></i>"
            "<b>Connected Components</b>"
            "<i></i>"
            "<b>Regionprops</b>"
            "<em>Portable CPU stack</em>"
            "</div>"
        )



    # --------------------------------------------------------
    # Compact model stack for Execution Profile
    # --------------------------------------------------------

    if requested == "Advanced GPU":

        model_stack = (
            "<div class='profile-matrix-models'>"
            "<strong>AI / CV MODELS:</strong> "
            "PyTorch CUDA · "
            "U-Net / U-Net++ · "
            "DeepLabV3+ · "
            "YOLO-Seg · "
            "SAM 2"
            "<span>GPU-ready architecture</span>"
            "</div>"
        )


    elif requested == "Research GPU":

        model_stack = (
            "<div class='profile-matrix-models'>"
            "<strong>RESEARCH MODELS:</strong> "
            "SAM 2 · "
            "YOLO-Seg · "
            "U-Net++ · "
            "DeepLabV3+ · "
            "Ensemble / Fusion"
            "<span>Experimental architecture</span>"
            "</div>"
        )


    elif requested == "AUTO":

        if resolved == "Advanced GPU":

            model_stack = (
                "<div class='profile-matrix-models'>"
                "<strong>AUTO MODEL STACK:</strong> "
                "PyTorch CUDA · "
                "U-Net / U-Net++ · "
                "DeepLabV3+ · "
                "YOLO-Seg · "
                "SAM 2"
                "<span>AUTO → Advanced GPU</span>"
                "</div>"
            )

        else:

            model_stack = (
                "<div class='profile-matrix-models'>"
                "<strong>AUTO MODEL STACK:</strong> "
                "OpenCV · "
                "K-means · "
                "Watershed · "
                "Connected Components · "
                "Regionprops"
                "<span>AUTO → Lightweight CPU</span>"
                "</div>"
            )


    else:

        model_stack = (
            "<div class='profile-matrix-models'>"
            "<strong>ACTIVE ALGORITHMS:</strong> "
            "OpenCV · "
            "K-means · "
            "Watershed · "
            "Connected Components · "
            "Regionprops"
            "<span>Portable CPU stack</span>"
            "</div>"
        )


    fallback_note = ""

    if fallback:

        fallback_note = (
            "<div class='profile-matrix-warning'>"
            "<b>"
            + html.escape(availability)
            + ".</b> "
            + html.escape(
                resolution.get(
                    "reason",
                    ""
                )
            )
            + " For this run the executable backend is "
            "<b>Lightweight CPU</b>; the selected "
            "Analysis Detail is preserved."
            "</div>"
        )


    return f"""
    <div class="profile-matrix-card">

        <div class="profile-matrix-top">

            <div>
                <span class="profile-matrix-kicker">
                    EXECUTION PROFILE
                </span>

                <h3>
                    {requested_text}
                    <span>→</span>
                    {resolved_text}
                    <em>·</em>
                    {detail_text}
                </h3>
            </div>

            <span class="profile-matrix-badge {badge_class}">
                {badge}
            </span>

        </div>

        <div class="profile-matrix-description">
            {html.escape(cfg["description"])}
        </div>

        <div class="profile-matrix-grid">

            <div class="profile-matrix-item">
                <span>DEVICE</span>
                <b>{html.escape(device)}</b>
            </div>

            <div class="profile-matrix-item">
                <span>MATERIAL CLUSTERS</span>
                <b>K = {cfg["k"]}</b>
            </div>

            <div class="profile-matrix-item">
                <span>CLUSTER SAMPLE</span>
                <b>{cfg["sample_side"]} px</b>
            </div>

            <div class="profile-matrix-item">
                <span>OBJECT AREA FACTOR</span>
                <b>{cfg["area_factor"]:.2f}</b>
            </div>

            <div class="profile-matrix-item">
                <span>WATERSHED</span>
                <b>{html.escape(cfg["watershed"].upper())}</b>
            </div>

            <div class="profile-matrix-item">
                <span>VISIBLE LABEL LIMIT</span>
                <b>{cfg["labels"]}</b>
            </div>

        </div>

        <div class="profile-matrix-runtime">
            <b>Runtime:</b>
            {html.escape(runtime_stack)}
        </div>

        <div class="profile-matrix-capabilities">
            CUDA: <b>{cuda_state}</b>
            <i></i>
            Advanced weights: <b>{advanced_state}</b>
            <i></i>
            Research weights: <b>{research_state}</b>
        </div>

        {model_stack}

        

        {fallback_note}

    </div>
    """


def profile_message(profile):
    if profile == "AUTO":
        return """<div class='profile-status'>
        <b>AUTO → LIGHTWEIGHT CPU</b><br>
        <span class='muted'>
        Portable CPU inference selected automatically.
        </span></div>"""

    if profile == "Lightweight CPU":
        return """<div class='profile-status'>
        <b>LIGHTWEIGHT CPU · AVAILABLE</b><br>
        <span class='muted'>
        Tesseract · OpenCV · clustering · morphology ·
        watershed · connected components · regionprops
        </span></div>"""

    if profile == "Advanced GPU":
        return """<div class='profile-status unavailable'>
        <b>ADVANCED GPU · GPU RUNTIME REQUIRED</b><br>
        <span class='muted'>
        U-Net · SAM 2 · YOLO-Seg · ResNet / EfficientNet.
        Not executed by this Lightweight deployment.
        </span></div>"""

    return """<div class='profile-status unavailable'>
    <b>RESEARCH GPU · RESEARCH RUNTIME REQUIRED</b><br>
    <span class='muted'>
    Advanced GPU stack + Vision Transformers · XAI / Grad-CAM ·
    benchmarking. Not executed by this Lightweight deployment.
    </span></div>"""


with gr.Blocks(title="Multimodal Geological AI") as demo:

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    gr.HTML("""
    <header class="geo-product-header">

        <div class="geo-product-header-main">

            <div class="geo-brand-mark">
                <span class="geo-brand-geo">GEO</span>
                <span class="geo-brand-ai">AI</span>
            </div>

            <div class="geo-product-heading">

                <span class="geo-product-kicker">
                    GEOSCIENCE AI
                </span>

                <h1>
                    GEOSCIENCE COMPUTER VISION
                    &amp; DOCUMENT AI TOOLKIT
                </h1>

                <p>
                    Multimodal AI for legacy geological
                    documents and images
                </p>

            </div>

            <div class="geo-product-version">
                <b>v4.0</b>
                
                <span>Traceable AI Prototype</span>
            </div>

        </div>

        <div class="geo-product-capabilities">

            <span>DOCUMENT AI</span>
            <i></i>

            <span>COMPUTER VISION</span>
            <i></i>

            <span>MULTIMODAL ROUTING</span>
            <i></i>

            <span>STRUCTURED GEOSCIENCE DATA</span>

        </div>

    </header>
    """)

    with gr.Row(elem_classes=["themebar"]):
        light = gr.Button("☀ Light", size="sm")
        dark = gr.Button("🌙 Dark", size="sm")

    light.click(
        None, None, None,
        js="() => {document.documentElement.setAttribute('data-geo-theme','light');}"
    )

    dark.click(
        None, None, None,
        js="() => {document.documentElement.setAttribute('data-geo-theme','dark');}"
    )

    gr.HTML("""
    <div class="geo-ai-model-architecture">

        <div class="geo-ai-model-architecture-head">

            <div>
                <span class="geo-ai-model-kicker">
                    AI / COMPUTER VISION ARCHITECTURE
                </span>

                <h3>
                    Runtime &amp; Model Stack
                </h3>

                <p>
                    Portable CPU analysis → CUDA acceleration →
                    advanced segmentation → research-grade model extensions
                </p>
            </div>

            <span class="geo-ai-model-architecture-badge">
                HARDWARE-AWARE
            </span>

        </div>


        <div class="geo-ai-model-layers">


            <div class="geo-ai-model-layer executed">

                <div class="geo-ai-model-layer-head">
                    <span>EXECUTED / AVAILABLE</span>
                    <b>CURRENT PROTOTYPE</b>
                </div>

                <div class="geo-ai-model-pills">

                    <span>OpenCV</span>
                    <span>K-means Semantic Segmentation</span>
                    <span>Morphology</span>
                    <span>Watershed</span>
                    <span>Connected Components</span>
                    <span>Fracture Detection</span>
                    <span>scikit-image Regionprops</span>
                    <span>PyTorch CUDA</span>

                </div>

            </div>


            <div class="geo-ai-model-layer advanced">

                <div class="geo-ai-model-layer-head">
                    <span>ADVANCED AI ARCHITECTURE</span>
                    <b>EXTENSIBLE</b>
                </div>

                <div class="geo-ai-model-pills">

                    <span>U-Net</span>
                    <span>U-Net++</span>
                    <span>DeepLabV3+</span>
                    <span>YOLO-Seg</span>
                    <span>SAM 2</span>

                </div>

                <p>
                    Trainable semantic / instance segmentation and
                    foundation-model refinement for GPU deployments.
                </p>

            </div>


            <div class="geo-ai-model-layer research">

                <div class="geo-ai-model-layer-head">
                    <span>RESEARCH EXTENSIONS</span>
                    <b>EXPERIMENTAL</b>
                </div>

                <div class="geo-ai-model-pills">

                    <span>Model Ensemble / Fusion</span>
                    <span>Geological Fine-Tuning</span>
                    <span>High-Resolution Refinement</span>
                    <span>Foundation-Model Assistance</span>

                </div>

            </div>

        </div>


        <div class="geo-ai-model-disclaimer">
            <b>Execution transparency.</b>
            Advanced models are shown as supported / extensible architecture.
            Only algorithms explicitly listed in the execution trace are
            reported as executed for the current analysis.
        </div>

    </div>
    """)






    # --------------------------------------------------------
    # MODEL PROFILE
    # --------------------------------------------------------

    with gr.Group(
        elem_classes=["profile-shell"]
    ):

        with gr.Row(
            equal_height=False,
            elem_classes=["profile-control-row"]
        ):

            with gr.Column(
                scale=5,
                min_width=420,
                elem_classes=["profile-selectors"]
            ):

                with gr.Row(
                    equal_height=False,
                    elem_classes=["profile-selector-row"]
                ):

                    model_profile = gr.Radio(
                        choices=[
                            "AUTO",
                            "Lightweight CPU",
                            "Advanced GPU",
                            "Research GPU"
                        ],
                        value="AUTO",
                        label="Compute Profile",
                        interactive=True,
                        elem_id="geo-model-profile",
                        elem_classes=["geo-profile-radio"]
                    )

                    analysis_detail = gr.Radio(
                        choices=[
                            "Standard",
                            "Detailed",
                            "Deep"
                        ],
                        value="Standard",
                        label="Analysis Quality",
                        interactive=True,
                        elem_id="geo-analysis-detail",
                        elem_classes=["geo-profile-radio"]
                    )

            with gr.Column(
                scale=5,
                min_width=460,
                elem_classes=["profile-status-column"]
            ):

                profile_status = gr.HTML(
                    profile_detail_status(
                        "AUTO",
                        "Standard"
                    ),
                    elem_id="geo-profile-status"
                )




    model_profile.change(
        profile_detail_status,
        [
            model_profile,
            analysis_detail
        ],
        profile_status
    )

    analysis_detail.change(
        profile_detail_status,
        [
            model_profile,
            analysis_detail
        ],
        profile_status
    )


    # ========================================================
    # AI / CV MODEL ARCHITECTURE
    # Static presentation block — no callbacks.
    # ========================================================







    gr.HTML("""
    <div class="geo-start-analysis">
        <span class="geo-start-analysis-kicker">
            START ANALYSIS
        </span>

        <h2>Analyze a Geological Asset</h2>

        <p>
            Choose a workflow below, then upload a geological
            document, scan, photographed page, or rock image.
        </p>

        <div class="geo-start-analysis-arrow">↓</div>
    </div>
    """)

    # ========================================================
    # TABS
    # ========================================================

    with gr.Tabs():

        # ====================================================
        # 1. MULTIMODAL AI
        # ====================================================

        with gr.Tab("Multimodal AI"):

            gr.Markdown(
                "### One input → automatic content understanding → appropriate analysis branch"
            )

            gr.Markdown(
                "Upload a legacy report, scan, photographed page, "
                "or geological image. The Lightweight router determines "
                "whether the content is text-focused, geological imagery, "
                "or a mixed legacy page."
            )

            gr.HTML("""
            <div class="asset-intro">
                <span class="asset-kicker">LEGACY MULTIMODAL</span>
                <h3>Upload a geological asset</h3>
                <p>
                    Report · scan · photographed page · geological image
                </p>
            </div>
            """)

            mm_in = gr.File(
                file_types=[
                    ".pdf",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tif",
                    ".tiff"
                ],
                label="Upload geological asset",
                elem_id="geo-main-upload",
                elem_classes=["geo-main-upload"]
            )

            gr.HTML("""
            <div class="pipeline-section-head">
                <span class="pipeline-number">01</span>
                <div>
                    <span class="pipeline-kicker">INPUT ASSET</span>
                    <h3>Geological Source</h3>
                    <p>Legacy report, scan, photograph, PDF, or geological image</p>
                </div>
            </div>
            """)

            with gr.Group(
                elem_classes=[
                    "pipeline-section",
                    "geo-input-workspace"
                ]
            ):

                mm_preview = gr.Image(
                    label="Uploaded asset · preview",
                    interactive=False,
                    height=420,
                    elem_id="geo-main-preview",
                    elem_classes=[
                        "clean-scientific-image",
                        "geo-main-preview"
                    ]
                )

                with gr.Row(
                    elem_classes=["geo-input-actions"]
                ):
                    mm_demo = gr.Button(
                        "USE DEMO REPORT",
                        elem_classes=["academy-secondary"]
                    )
                    mm_run = gr.Button(
                        "ANALYZE & ROUTE →",
                        variant="primary",
                        elem_classes=["academy-primary"]
                    )

            mm_report_status = gr.HTML(
                visible=False
            )

            gr.HTML("""
            <div class="geo-download-report-head">

                <div class="geo-download-report-icon">
                    <span>PDF</span>
                    <b>↓</b>
                </div>

                <div class="geo-download-report-copy">

                    <span class="geo-download-report-kicker">
                        FINAL OUTPUT
                    </span>

                    <h2>
                        Download AI Report
                    </h2>

                    <p>
                        Analysis, measurements, geological interpretation
                        and execution traceability are compiled into the
                        downloadable PDF report below.
                    </p>

                </div>

                <div class="geo-download-report-badge">
                    PDF REPORT
                </div>

            </div>
            """)

            mm_report_file = gr.File(
                label="DOWNLOAD PDF REPORT ↓",
                interactive=False,
                elem_id="geo-pdf-download",
                elem_classes=["geo-pdf-download"]
            )

            gr.HTML("""
            <div class="pipeline-section-head">
                <span class="pipeline-number">02</span>
                <div>
                    <span class="pipeline-kicker">AI ROUTING</span>
                    <h3>Multimodal Content Routing</h3>
                    <p>Automatic separation of document text and geological visual content</p>
                </div>
            </div>
            """)

            with gr.Group(elem_classes=["pipeline-section"]):

                mm_note = gr.HTML()

                mm_routing_map = gr.HTML()

                mm_regions = gr.Image(
                    label="Content routing map",
                    elem_id="geo-routing-image",
                    elem_classes=[
                        "clean-scientific-image",
                        "geo-routing-image"
                    ]
                )

            gr.HTML("""
            <div class="pipeline-section-head">
                <span class="pipeline-number">03</span>
                <div>
                    <span class="pipeline-kicker">DOCUMENT AI</span>
                    <h3>Text Recognition & Geological Information</h3>
                    <p>Recognized text → structured geological entities → readable content</p>
                </div>
            </div>
            """)

            with gr.Group(elem_classes=["pipeline-section"]):
                mm_document_result = gr.HTML()

            with gr.Accordion(
                "Raw OCR text",
                open=False
            ):
                mm_text = gr.Textbox(
                    label="Raw OCR / active text",
                    lines=10,
                    interactive=False
                )

            with gr.Accordion(
                "Extracted geological entities · traceability",
                open=False
            ):
                mm_entities = gr.Dataframe(
                    interactive=False
                )

            with gr.Accordion(
                "Routing decisions",
                open=False
            ):
                mm_routes = gr.Dataframe(
                    interactive=False
                )

            gr.HTML("""
            <div class="pipeline-section-head">
                <span class="pipeline-number">04</span>
                <div>
                    <span class="pipeline-kicker">COMPUTER VISION</span>
                    <h3>Geological Image Analysis</h3>
                    <p>Original routed figure → AI interpretation → quantitative geological summary</p>
                </div>
                <span class="analysis-ready">✓ ANALYZED</span>
            </div>
            """)

            with gr.Group(
                elem_classes=[
                    "pipeline-section",
                    "analysis-workspace"
                ]
            ):

                with gr.Row(elem_classes=["before-after-row"]):

                    with gr.Column(elem_classes=["analysis-image-column"]):

                        gr.HTML(
                            "<div class='analysis-image-label'>"
                            "<span>01</span>"
                            "<div><b>BEFORE</b><small>Routed geological image</small></div>"
                            "</div>"
                        )

                        mm_crop = gr.Image(
                            label=None,
                            show_label=False,
                            elem_classes=["scientific-viewer"]
                        )

                    with gr.Column(elem_classes=["analysis-image-column"]):

                        gr.HTML(
                            "<div class='analysis-image-label'>"
                            "<span>02</span>"
                            "<div><b>AFTER</b><small>AI geological interpretation</small></div>"
                            "</div>"
                        )

                        mm_interp = gr.Image(
                            label=None,
                            show_label=False,
                            elem_classes=["scientific-viewer"]
                        )

                        gr.HTML(
                            "<div class='detected-classes-shell'>"
                            "<div class='class-caption'>Detected classes</div>"
                            + LEGEND +
                            "</div>"
                        )

                mm_interpretation = gr.HTML()

            with gr.Accordion(
                "Object provenance & traceability",
                open=False
            ):
                mm_trace = gr.Dataframe(
                    interactive=False
                )

            with gr.Accordion(
                "Execution trace",
                open=False
            ):
                mm_graph = gr.HTML()

            with gr.Accordion(
                "Technical execution details",
                open=False
            ):
                mm_exec_table = gr.Dataframe(
                    interactive=False,
                    label="Models / algorithms actually executed"
                )

            mm_in.change(
                preview_document,
                mm_in,
                mm_preview
            )

            mm_demo.click(
                demo_mm,
                None,
                mm_in
            ).then(
                preview_document,
                mm_in,
                mm_preview
            )

            mm_run.click(
                multimodal_pipeline,
                [
                    mm_in,
                    model_profile,
                    analysis_detail
                ],
                [
                    mm_preview,
                    mm_regions,
                    mm_note,
                    mm_routing_map,
                    mm_document_result,
                    mm_text,
                    mm_entities,
                    mm_routes,
                    mm_crop,
                    mm_interp,
                    mm_interpretation,
                    mm_trace,
                    mm_exec_table,
                    mm_graph,
                    mm_report_status,
                    mm_report_file
                ]
            )

        # ====================================================
        # 2. ROCK VISION
        # ====================================================

        with gr.Tab("Rock Vision"):

            gr.Markdown(
                "### Standalone geological image → interpretation → objects → measurements"
            )

            with gr.Row():

                rock_in = gr.Image(
                    type="filepath",
                    sources=["upload", "webcam"],
                    label="Geological image / collage"
                )

                original = gr.Image(
                    label="Original"
                )

                with gr.Column():

                    rock_out = gr.Image(
                        label="AI geological interpretation"
                    )

                    gr.HTML(
                        "<div class='class-caption'>Detected classes</div>"
                        + LEGEND
                    )

            with gr.Row():

                choice = gr.Dropdown(
                    [
                        "Carbonate thin section",
                        "Fractured rock",
                        "Vuggy carbonate"
                    ],
                    value="Fractured rock",
                    label="Demo"
                )

                use = gr.Button("USE DEMO SAMPLE")

                analyze = gr.Button(
                    "ANALYZE",
                    variant="primary"
                )

            stats = gr.HTML()

            objects = gr.Dataframe(
                label="Largest objects / candidates",
                interactive=False
            )

            gr.Markdown("### Execution trace")

            rock_graph = gr.HTML()

            with gr.Accordion(
                "Technical execution details",
                open=False
            ):
                rock_exec = gr.Dataframe(
                    interactive=False
                )

            with gr.Row():
                ann = gr.File(label="Annotated image")
                objcsv = gr.File(label="Objects CSV")
                rockjson = gr.File(label="JSON result")

            use.click(
                demo_rock,
                choice,
                rock_in
            )

            analyze.click(
                run_rock,
                rock_in,
                [
                    original,
                    rock_out,
                    stats,
                    objects,
                    rock_exec,
                    rock_graph,
                    ann,
                    objcsv,
                    rockjson
                ]
            )

        # ====================================================
        # 3. LEGACY TEXT RECOGNITION
        # ====================================================

        with gr.Tab("Document AI"):

            gr.Markdown(
                "### Legacy document → text recognition / OCR → geological entities → traceability"
            )

            gr.Markdown(
                "Direct text-focused mode for searchable PDFs, "
                "scanned reports and photographed legacy documents."
            )

            with gr.Row():

                with gr.Column():

                    doc_in = gr.File(
                        file_types=[
                            ".pdf",
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".tif",
                            ".tiff"
                        ],
                        label="PDF / scanned / photographed document"
                    )

                    dc = gr.Dropdown(
                        [
                            "Searchable geological PDF",
                            "Scanned geological page"
                        ],
                        value="Scanned geological page",
                        label="Demo"
                    )

                    du = gr.Button("USE DEMO SAMPLE")

                    da = gr.Button(
                        "RECOGNIZE TEXT",
                        variant="primary"
                    )

                doc_preview = gr.Image(
                    label="Preview · page 1",
                    interactive=False
                )

            ds = gr.HTML()

            text = gr.Textbox(
                label="Recognized / active text",
                lines=10
            )

            entities = gr.Dataframe(
                label="Geological entities",
                interactive=False
            )

            pages = gr.Dataframe(
                label="Page traceability",
                interactive=False
            )

            gr.Markdown("### Execution trace")

            doc_graph = gr.HTML()

            with gr.Accordion(
                "Technical execution details",
                open=False
            ):
                doc_exec = gr.Dataframe(
                    interactive=False
                )

            with gr.Row():
                ecsv = gr.File(label="Entities CSV")
                otxt = gr.File(label="OCR text")
                djson = gr.File(label="JSON result")

            doc_in.change(
                preview_document,
                doc_in,
                doc_preview
            )

            du.click(
                demo_doc,
                dc,
                [doc_in, doc_preview]
            )

            da.click(
                run_doc,
                doc_in,
                [
                    doc_preview,
                    ds,
                    text,
                    entities,
                    pages,
                    doc_exec,
                    doc_graph,
                    ecsv,
                    otxt,
                    djson
                ]
            )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML("""
    <footer class="geo-footer-signature">
        <span class="geo-footer-brand">GEO AI</span>
        <span>
            Geoscience Computer Vision &amp; Document AI Toolkit
        </span>
        <b>v4.0</b>
    </footer>
    """)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        theme=theme,
        css=CSS,
        server_name=os.environ.get("GEOAI_HOST", "0.0.0.0"),
        server_port=int(
            os.environ.get(
                "PORT",
                os.environ.get("GEOAI_PORT", "7871")
            )
        ),
        inbrowser=False
    )
