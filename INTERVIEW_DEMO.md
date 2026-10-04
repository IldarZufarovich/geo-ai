# Interview / conference demo

## 30 seconds
“I built one end-to-end geological AI product rather than a standalone notebook. The research layer trains and evaluates models; the inference layer separates document AI from rock-image AI; and this public interface lets a user upload a scan or take a rock photo. Every output retains provenance and confidence. The live version is deliberately CPU-safe and clearly distinguishes demo fallbacks from validated models.”

## 90 seconds
1. Open **Rock Vision** and click **Use demo sample**.
2. Analyze. Point to semantic colors, then individual object IDs.
3. Show the object table and visible 2D porosity; state that it is not laboratory effective porosity.
4. Switch to **Legacy Document AI**. Use the searchable PDF or scan.
5. Show text-layer/OCR routing and extracted well/formation/depth/porosity with page provenance.
6. State: “OCR creates machine-readable evidence first; RAG would only come after document understanding.”

## 3 minutes
“First, the project starts from two data families: legacy documents and physical-rock imagery. On the document side, I detect whether a PDF already has an active text layer. If not, I treat the page as an image, preprocess it, run OCR, and extract geological entities with source-page traceability. On the rock side, I separate semantic segmentation from instance segmentation. The semantic mask identifies pore, grain, fracture and matrix pixels; watershed then separates individual objects so I can calculate size, shape and orientation. The research notebook contains train/test logic and more advanced models, while this live app uses a lightweight CPU fallback unless validated weights are present. Finally, the multimodal tab connects source document, page, image, object, prediction, confidence and model version. The key point is scientific honesty: synthetic or demo performance is not field validation, and 2D visible porosity is not automatically effective porosity.”

## Failure-safe conference flow
If camera/network conditions are poor, use the built-in demo sample. Never make the presentation dependent on a live external file.
