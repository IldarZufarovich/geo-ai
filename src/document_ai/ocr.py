from __future__ import annotations
import os
import re
import shutil
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_TESSDATA = Path(__file__).resolve().parents[2] / 'assets' / 'tessdata'


def configure_tesseract() -> tuple[bool, str | None]:
    if importlib.util.find_spec('pytesseract') is None:
        return False, None
    import pytesseract
    exe = os.environ.get('TESSERACT_CMD') or shutil.which('tesseract')
    if not exe and os.name == 'nt':
        local = os.environ.get('LOCALAPPDATA', '')
        candidates = [
            Path(local) / 'Programs' / 'Tesseract-OCR' / 'tesseract.exe' if local else None,
            Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe'),
            Path(r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'),
        ]
        for p in candidates:
            if p is not None and p.exists():
                exe = str(p)
                break
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
        exe_dir = str(Path(exe).parent)
        if exe_dir not in os.environ.get('PATH', ''):
            os.environ['PATH'] = exe_dir + os.pathsep + os.environ.get('PATH', '')
        return True, exe
    return False, None


def _tessdata_dir(exe: str | None = None) -> Path | None:
    if PROJECT_TESSDATA.exists() and any(PROJECT_TESSDATA.glob('*.traineddata')):
        return PROJECT_TESSDATA
    if exe:
        p = Path(exe).parent / 'tessdata'
        if p.exists():
            return p
    return None


def available_languages() -> list[str]:
    ok, exe = configure_tesseract()
    if not ok:
        return []
    import pytesseract
    td = _tessdata_dir(exe)
    if td:
        langs = sorted(p.stem for p in td.glob('*.traineddata'))
        if langs:
            return langs
    try:
        return sorted(pytesseract.get_languages(config=''))
    except Exception:
        return []


def preferred_language() -> str:
    langs = set(available_languages())
    if {'eng', 'rus'} <= langs:
        return 'eng+rus'
    if 'eng' in langs:
        return 'eng'
    if 'rus' in langs:
        return 'rus'
    return ''


def russian_ocr_ready() -> bool:
    return 'rus' in set(available_languages())


def ocr_rgb(rgb: np.ndarray, psm: int = 3, lang: str | None = None):
    ok, exe = configure_tesseract()
    if not ok:
        return {'text': '', 'mean_confidence': 0.0, 'tokens': pd.DataFrame(),
                'warning': 'Tesseract executable was not found', 'language': '',
                'tesseract_path': None}
    import pytesseract
    lang = preferred_language() if lang is None else lang
    cfg = f'--oem 3 --psm {int(psm)} -c preserve_interword_spaces=1'
    td = _tessdata_dir(exe)
    if td:
        os.environ["TESSDATA_PREFIX"] = str(td)
    kwargs = {'output_type': pytesseract.Output.DATAFRAME, 'config': cfg}
    if lang:
        kwargs['lang'] = lang
    try:
        d = pytesseract.image_to_data(rgb, **kwargs)
    except Exception as exc:
        return {'text': '', 'mean_confidence': 0.0, 'tokens': pd.DataFrame(),
                'warning': f'OCR failed: {exc}', 'language': lang,
                'tesseract_path': exe}
    d = d.dropna(subset=['text']).copy()
    d = d[d.text.astype(str).str.strip() != '']
    conf = pd.to_numeric(d.conf, errors='coerce')
    good = conf[conf >= 0]
    lines = []
    if len(d):
        group_cols = [c for c in ['page_num', 'block_num', 'par_num', 'line_num'] if c in d.columns]
        for _, g in d.groupby(group_cols, sort=False):
            line = ' '.join(g.text.astype(str).tolist()).strip()
            if line:
                lines.append(line)
    text = '\n'.join(lines)
    return {'text': text,
            'mean_confidence': float(good.mean() / 100 if len(good) else 0),
            'tokens': d, 'warning': None, 'language': lang,
            'tesseract_path': exe}


def _text_metrics(text: str) -> dict:
    tokens = re.findall(r"[A-Za-z\u0400-\u04FF0-9]+(?:[-/.][A-Za-z\u0400-\u04FF0-9]+)*", text or '')
    alpha_tokens = [t for t in tokens if any(ch.isalpha() for ch in t)]
    meaningful = [t for t in alpha_tokens if len(t) >= 2]
    single_alpha = [t for t in alpha_tokens if len(t) == 1 and t.lower() not in {'a', 'i'}]
    chars = [c for c in (text or '') if not c.isspace()]
    alnum = sum(c.isalnum() for c in chars)
    weird = sum(not (c.isalnum() or c in "-_/.,:%()+&'\"[]") for c in chars)
    return {
        'tokens': len(tokens),
        'meaningful_tokens': len(meaningful),
        'single_alpha_ratio': len(single_alpha) / max(1, len(alpha_tokens)),
        'alnum_ratio': alnum / max(1, len(chars)),
        'weird_ratio': weird / max(1, len(chars)),
        'text_chars': len(text or ''),
    }


def _quality_score(out: dict) -> float:
    """Quality-first OCR score. Length is capped so garbage cannot win by verbosity."""
    m = _text_metrics(out.get('text', ''))
    conf = float(out.get('mean_confidence', 0.0))
    token_bonus = min(m['meaningful_tokens'], 24) / 24 * 0.10
    alnum_bonus = min(max((m['alnum_ratio'] - 0.45) / 0.45, 0.0), 1.0) * 0.06
    short_penalty = 0.08 if m['meaningful_tokens'] == 0 else 0.0
    garbage_penalty = 0.22 * m['single_alpha_ratio'] + 0.18 * m['weird_ratio']
    return conf * 0.82 + token_bonus + alnum_bonus - garbage_penalty - short_penalty


def _candidate(out: dict, variant: str, psm: int) -> dict:
    rec = {**out, 'variant': variant, 'psm': int(psm)}
    rec['selection_score'] = _quality_score(rec)
    rec.update({f'metric_{k}': v for k, v in _text_metrics(rec.get('text', '')).items()})
    return rec


def _is_strong(rec: dict) -> bool:
    return (rec.get('mean_confidence', 0.0) >= 0.78 and
            rec.get('metric_meaningful_tokens', 0) >= 3 and
            rec.get('metric_single_alpha_ratio', 1.0) <= 0.30 and
            rec.get('metric_alnum_ratio', 0.0) >= 0.60)


def best_ocr(variants: dict[str, np.ndarray]):
    """Adaptive OCR: direct/original first, then progressively escalate only if needed."""
    if not variants:
        return ocr_rgb(np.zeros((10, 10, 3), np.uint8))

    candidates: list[dict] = []
    # The caller supplies original_rgb. It is intentionally tested first: preprocessing
    # must never destroy a clean, high-confidence OCR result.
    original_key = 'original_rgb' if 'original_rgb' in variants else next(iter(variants))
    direct = _candidate(ocr_rgb(variants[original_key], psm=3), original_key, 3)
    candidates.append(direct)
    winner = direct

    if not _is_strong(direct):
        # Cheap alternatives first. Sparse-text PSM 11 is useful for posters/headlines.
        plan = []
        for name in ('original_rgb', 'gray', 'upscaled_gray', 'illumination_clahe',
                     'clahe_sharpen', 'otsu', 'stroke_repair', 'adaptive'):
            if name not in variants:
                continue
            psms = (3, 6, 11) if name in ('original_rgb', 'gray', 'upscaled_gray') else (3, 6)
            if name == 'adaptive':
                psms = (3, 6, 11)  # last resort only
            for psm in psms:
                if name == original_key and psm == 3:
                    continue
                plan.append((name, psm))

        for name, psm in plan:
            rec = _candidate(ocr_rgb(variants[name], psm=psm), name, psm)
            candidates.append(rec)
            if rec['selection_score'] > winner['selection_score']:
                winner = rec
            if _is_strong(rec) and rec['selection_score'] >= direct['selection_score']:
                break

    # If both packs exist, refine the selected image with the dominant script only.
    langs = set(available_languages())
    txt = winner.get('text', '')
    latin = sum(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in txt)
    cyr = sum('\u0400' <= c <= '\u04FF' for c in txt)
    letters = latin + cyr
    refine_lang = None
    if {'eng', 'rus'} <= langs and letters >= 20:
        if latin / letters >= 0.72:
            refine_lang = 'eng'
        elif cyr / letters >= 0.72:
            refine_lang = 'rus'
    if refine_lang:
        refined = _candidate(ocr_rgb(variants[winner['variant']], psm=winner['psm'], lang=refine_lang),
                             winner['variant'], winner['psm'])
        if refined['selection_score'] >= winner['selection_score'] - 0.025:
            candidates.append(refined)
            winner = refined

    ranked = sorted(candidates, key=lambda z: z['selection_score'], reverse=True)
    winner['candidates'] = [{
        'variant': x['variant'], 'psm': x['psm'],
        'mean_confidence': round(x.get('mean_confidence', 0.0), 3),
        'selection_score': round(x.get('selection_score', 0.0), 3),
        'text_chars': len(x.get('text', '')), 'language': x.get('language', '')
    } for x in ranked[:8]]
    winner['attempts_run'] = len(candidates)
    if winner.get('mean_confidence', 0) < 0.45:
        winner['warning'] = ('Low OCR confidence: use a flatter, sharper, higher-contrast photo '
                             'and fill the frame with the document.')
    return winner
