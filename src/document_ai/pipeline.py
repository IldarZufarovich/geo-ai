from __future__ import annotations
from pathlib import Path
import pandas as pd, cv2
from config import SETTINGS
from src.common.io import save_json, utc_now
from .ingestion import ingest
from .preprocessing import preprocess_scan, ocr_variants
from .ocr import ocr_rgb, best_ocr
from .entities import extract_entities

def classify_text(text):
    t=text.lower()
    if 'porosity' in t and 'permeability' in t: return 'laboratory/petrographic report'
    if 'core' in t and ('description' in t or 'sample' in t): return 'core description'
    if 'production' in t: return 'production table/report'
    if 'well log' in t or 'gamma ray' in t: return 'well log'
    return 'geological report' if len(t)>40 else 'unknown'

def analyze_document(path,output_dir=None,max_pages=None):
    outdir=Path(output_dir or SETTINGS.output_dir); outdir.mkdir(parents=True,exist_ok=True); max_pages=max_pages or SETTINGS.max_pdf_pages
    pages,total=ingest(path,max_pages); page_rows=[]; ents=[]; texts=[]; warnings=[]
    for rec in pages:
        if rec['searchable']:
            text=rec['text_layer']; conf=1.0; method='pdf_text_layer'
        else:
            o=best_ocr(ocr_variants(rec['rgb'])); text=o['text']; conf=o['mean_confidence']; method=f"tesseract_ocr:{o.get('variant','default')}:psm{o.get('psm','')}:lang{o.get('language','')}:attempts{o.get('attempts_run',1)}";
            if o['warning']: warnings.append(o['warning'])
        texts.append(f"--- Page {rec['page']} ---\n{text}")
        e=extract_entities(text,str(path),rec['page'],method); ents.append(e)
        page_rows.append({'source_file':str(path),'page':rec['page'],'searchable_pdf_page':rec['searchable'],'extraction_method':method,'ocr_confidence':conf,'document_class':classify_text(text),'text':text[:1000], 'ocr_language': (o.get('language','') if not rec['searchable'] else 'text-layer')})
    ent=pd.concat(ents,ignore_index=True) if ents and any(len(x) for x in ents) else pd.DataFrame(columns=['entity','original_value','normalized_value','confidence','source_file','page','extraction_method'])
    pages_df=pd.DataFrame(page_rows); stem=Path(path).stem
    # Use short stable output names.
    # The original filename remains preserved inside provenance metadata.
    # This avoids Windows MAX_PATH failures with long uploaded filenames.
    ent_csv = outdir / 'entities.csv'
    page_csv = outdir / 'pages.csv'
    txt = outdir / 'ocr.txt'
    js = outdir / 'document_result.json'
    ent.to_csv(ent_csv,index=False); pages_df.to_csv(page_csv,index=False); txt.write_text('\n\n'.join(texts),encoding='utf-8')
    summary={'source_file':str(path),'pages_in_file':total,'pages_processed':len(pages),'document_type':pages_df.document_class.mode().iat[0] if len(pages_df) else 'unknown','mean_confidence':round(float(pages_df.ocr_confidence.mean()),3) if len(pages_df) else 0,'warnings':sorted(set(warnings)),'model_version':SETTINGS.model_version,'timestamp':utc_now()}
    save_json({'summary':summary,'entities':ent.to_dict('records'),'pages':pages_df.to_dict('records')},js)
    report = outdir / 'report.html'; report.write_text('<html><body><h1>Legacy Document AI Report</h1><pre>'+str(summary)+'</pre>'+ent.to_html(index=False)+'</body></html>',encoding='utf-8')
    return {'summary':summary,'entities':ent,'pages':pages_df,'text':'\n\n'.join(texts),'files':{'entities_csv':str(ent_csv),'pages_csv':str(page_csv),'text':str(txt),'json':str(js),'report':str(report)}}
