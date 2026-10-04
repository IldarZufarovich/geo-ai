# v1.0 — Final realistic portfolio scope

This build deliberately stops before pretending to be a production geological interpreter.

## Reliable demo scope
- Three workflows only: Rock Vision, Legacy Document AI, Model Pipeline.
- Rock Vision always accepts a geological image/collage; it no longer rejects it because OCR sees labels/scale bars.
- Model Pipeline uses a conservative top-level router: document pages go to OCR/layout; geological visuals/collages go directly to Rock Vision.
- Legacy Document AI shows page-1 preview before analysis.
- OCR supports English and Russian when Tesseract `eng`/`rus` language packs are installed. A Windows helper is included.
- Rock overlays use fixed class colors and compact labels.
- Fractures are presented as **fracture candidates** from a conservative line/ridge/color detector, not as validated fracture truth.
- All outputs retain source/model provenance.

## Intentionally not claimed
- No field-validated grain/mineral classifier.
- No guaranteed segmentation of arbitrary real thin sections.
- No learned layout transformer or SAM/Mask R-CNN in the live CPU demo.
- No production-grade multilingual OCR guarantee.

The research notebook remains the place for advanced/trained models. The live app prioritizes stability, traceability, visual clarity, and scientific honesty.
