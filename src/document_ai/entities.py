from __future__ import annotations
import re
import pandas as pd

# Field labels are used as boundaries because OCR often reconstructs a scanned page as one line.
NEXT_FIELD = r'(?=\s+(?:well|field|formation|depth|interval|lithology|porosity|permeability|water\s+saturation|sw|sample|core|age|geological\s+age|pore\s+type|porosity\s+type|fracture)\b|$)'
PATTERNS = {
    'well': r'(?i)\bwell\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{1,20})',
    'field': rf'(?i)\bfield\s*[:#-]?\s*(.+?){NEXT_FIELD}',
    'formation': rf'(?i)\bformation\s*[:#-]?\s*(.+?){NEXT_FIELD}',
    'depth_interval': r'(?i)\b(?:depth|interval)\s*[:#-]?\s*([0-9,.]+)\s*[-–]\s*([0-9,.]+)\s*(ft|m)?',
    'lithology': rf'(?i)\blithology\s*[,.:#-]?\s*(.+?){NEXT_FIELD}',
    'porosity': r'(?i)\bporosity\s*[:#-]?\s*([0-9]+(?:[.,][0-9]+)?)\s*%',
    'permeability': r'(?i)\bpermeability\s*[:#-]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(mD|D)?',
    'water_saturation': r'(?i)\b(?:water\s+saturation|Sw)\s*[.:#=-]?\s*([0-9]+(?:[.,][0-9]+)?)\s*%',
    'sample_id': r'(?i)\bsample\s*(?:id|no\.?|#)?\s*[.:#-]?\s*([A-Z0-9][A-Z0-9.-]{1,30})',
    'core_number': r'(?i)\bcore\s*(?:number|no\.?|#)\s*[.:#-]?\s*([A-Z0-9-]+)',
    'geological_age': rf'(?i)\b(?:geological\s+age|age)\s*[:#-]?\s*(.+?){NEXT_FIELD}',
    'pore_description': rf'(?i)\b(?:pore\s+type|porosity\s+type)\s*[:#-]?\s*(.+?){NEXT_FIELD}',
    'fracture_description': rf'(?i)\bfracture(?:\s+description)?\s*[:#-]?\s*(.+?){NEXT_FIELD}',
}


def _light_ocr_cleanup(text: str) -> str:
    # Generic label-level cleanup only; never invent geological values.
    text = re.sub(r'(?i)\bWelt\b', 'Well', text)
    text = re.sub(r'(?i)\bSample\s+10[.:]', 'Sample ID: ', text)  # common OCR I/D -> 1/0 confusion in labels
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_entities(text, source_file, page, method='regex'):
    clean = _light_ocr_cleanup(text)
    rows = []
    for entity, pat in PATTERNS.items():
        for m in re.finditer(pat, clean):
            vals = m.groups()
            raw = ' '.join(v for v in vals if v) if len(vals) > 1 else vals[0]
            raw = raw.strip(' ,.;:')
            norm = raw.replace(',', '.') if entity in {'porosity', 'permeability', 'water_saturation'} else raw.strip()
            rows.append({
                'entity': entity, 'original_value': raw, 'normalized_value': norm,
                'confidence': 0.88, 'source_file': source_file, 'page': page,
                'extraction_method': method,
            })
    return pd.DataFrame(rows)
