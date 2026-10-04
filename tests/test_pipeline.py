from pathlib import Path
import numpy as np
from src.rock_vision.pipeline import analyze_rock
from src.document_ai.pipeline import analyze_document
from src.document_ai.entities import extract_entities

ROOT=Path(__file__).resolve().parents[1]

def test_rock_demo():
    r=analyze_rock(ROOT/'assets/examples/carbonate_thin_section.png',ROOT/'outputs')
    assert r['semantic_mask'].shape==r['instance_mask'].shape
    assert r['annotated'].shape[:2]==r['semantic_mask'].shape
    assert {'object_id','predicted_class','model_version'}.issubset(r['objects'].columns)
    assert Path(r['files']['json']).exists()

def test_searchable_pdf():
    r=analyze_document(ROOT/'assets/examples/legacy_report_searchable.pdf',ROOT/'outputs')
    assert r['summary']['pages_processed']==1
    assert len(r['entities'])>=5
    assert 'Arab-D' in r['text']

def test_scan_image_does_not_crash():
    r=analyze_document(ROOT/'assets/examples/legacy_report_scan.png',ROOT/'outputs')
    assert r['summary']['pages_processed']==1
    assert 'model_version' in r['summary']

def test_entity_provenance():
    e=extract_entities('Well: AB-143\nFormation: Arab-D\nPorosity: 18.4%', 'x.pdf', 7)
    assert len(e)>=3
    assert set(e.page)=={7}
    assert set(e.source_file)=={'x.pdf'}

def test_corrupted_pdf():
    p=ROOT/'outputs/bad.pdf'; p.write_bytes(b'not a pdf')
    try:
        analyze_document(p,ROOT/'outputs')
    except Exception:
        return
    raise AssertionError('Corrupted PDF should raise a controlled exception at pipeline level')
