# Validation status

## TESTED in the build environment
- Python syntax/bytecode compilation across the repository.
- Rock Vision synthetic demo end-to-end.
- Searchable PDF ingestion, text extraction, entity extraction and provenance.
- Scanned-page OCR path with local Tesseract.
- Corrupted PDF behavior at pipeline level.
- Segmentation/instance output dimension checks.
- CSV/JSON/PNG output generation.
- Critical imports.
- Gradio application import.
- Local Gradio HTTP launch: returned HTTP 200 on `127.0.0.1:7860`.
- Automated tests: **5 passed**.

## NOT TESTED / requires your cloud account
- Actual Hugging Face Spaces build and public URL.
- Smartphone camera permissions on the final Space URL.
- System Tesseract availability on the chosen Hugging Face runtime.
- GitHub Pages links using your final GitHub username/repository.
- QR code to final public URL (URL does not exist yet).

## Scientific validation not claimed
- No independent real-field geological validation of the live heuristic segmentation.
- No calibrated field accuracy for rock classes.
- No operational validation of OCR/entity extraction on confidential corporate archives.
