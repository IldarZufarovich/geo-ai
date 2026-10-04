# Project structure and notebook audit

## Audit of the supplied notebook

### Reused directly or conceptually
- Reproducible configuration and seed strategy.
- Synthetic petrographic/document generation concept.
- File discovery and PDF text-layer detection.
- Scan preprocessing/QC concepts.
- OCR and transparent geological entity extraction.
- Semantic → instance segmentation distinction.
- Connected-components/distance-transform/watershed baseline.
- Object-level measurements and provenance fields.
- Scientific limitations and interview framing.

### Training-only components retained in the notebook
- Synthetic dataset generation at scale.
- Leakage-safe train/validation/test split.
- U-Net training and held-out segmentation metrics.
- Classical ML/CNN/hybrid CNN–Transformer training and comparison.
- Permutation importance / saliency diagnostics.

### Refactored into live inference
- PDF/image ingestion.
- Text-layer vs scan routing.
- OCR + entity extraction.
- Rock preprocessing, semantic fallback, instance separation and measurements.
- Export/provenance logic.

### Too heavy/risky for P0 free deployment
- SAM, Mask R-CNN, YOLO-seg downloads.
- Large Vision Transformers/PoreViT-like inference without exported weights.
- LayoutLM/Docling/PaddleOCR as mandatory dependencies.
- Public LLM/RAG/vector database.
- 3D micro-CT inference.

### Functionality requiring validated saved weights
The notebook trains U-Net/CNN/hybrid models in-session; those weights are not embedded in the supplied notebook file. The public app therefore does **not** pretend to use trained weights. It exposes a deterministic CPU fallback and documents how validated weights can replace it.

### Best live-demo outputs
- Original vs colored rock overlay.
- Individual object IDs and measurements.
- Visible 2D porosity with explicit limitation.
- Searchable-vs-scanned document routing.
- OCR/extracted geological entities with page provenance.
- Multimodal relational traceability.

## P0/P1/P2

**P0 implemented:** rock upload/camera, demo samples, segmentation fallback, instance separation, measurements, annotated output, PDF/image upload, OCR/text layer, entity extraction, clean Gradio UI, exports.

**P1 partial:** document class heuristic, confidence, downloadable outputs, provenance. Table extraction and PDF embedded-figure extraction remain research-notebook capabilities/roadmap.

**P2 intentionally postponed:** SAM, Mask R-CNN, YOLO-seg, production ViT/PoreViT, RAG/LLM, vector DB, advanced layout transformer, 3D CT.
