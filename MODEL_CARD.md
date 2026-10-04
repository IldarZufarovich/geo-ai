# Model Card — Multimodal Geological AI Demo 0.1.0

## Intended use
Public, non-confidential portfolio demonstration of a traceable geological AI workflow.

## Rock Vision live method
Deterministic CPU fallback combining HSV/color cues, grayscale texture/edge cues, Hough-line fracture candidates, and watershed instance separation. It is designed to demonstrate software architecture and object-level outputs when validated weights are unavailable.

## Document AI live method
- Searchable PDF: native text layer.
- Raster scan/image: CLAHE/denoising + Tesseract OCR when installed.
- Geological entities: transparent regex/rule extraction.

## Research models
The accompanying notebook contains training paths for a lightweight U-Net, classical object classifier, CNN classifier and simplified CNN–Transformer hybrid.

## Metrics
No field accuracy is claimed for the public fallback. Accuracy metrics in the notebook are meaningful only for held-out synthetic/source-level splits.

## Known limitations
Domain shift, stain/illumination differences, scale calibration, geological ambiguity, touching objects, OCR degradation, handwriting and complex tables. Object dimensions are pixel-based unless scale metadata is supplied.

## Ethical/scientific use
Not for reserves estimation, operational decisions, safety-critical interpretation, or confidential data. Expert geological validation is required.
