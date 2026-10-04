import numpy as np
from src.document_ai.ocr import _candidate, _is_strong, _quality_score


def test_clean_high_confidence_beats_long_garbage():
    clean = _candidate({
        'text': 'AI & DIGITAL SUBSURFACE FOR OIL & GAS',
        'mean_confidence': 0.935,
        'tokens': None,
        'warning': None,
        'language': 'eng',
        'tesseract_path': 'tesseract',
    }, 'original_rgb', 3)
    garbage = _candidate({
        'text': ('a LL fo i tf Sc Fl ae ip Ge on he Ae ee ' * 35),
        'mean_confidence': 0.33,
        'tokens': None,
        'warning': None,
        'language': 'eng',
        'tesseract_path': 'tesseract',
    }, 'adaptive', 11)
    assert _is_strong(clean)
    assert clean['selection_score'] > garbage['selection_score']
