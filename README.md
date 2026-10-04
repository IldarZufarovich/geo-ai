# GEO AI v4.0

### Multimodal Geological Intelligence

**From Legacy Geological Assets to Traceable Digital Intelligence**

GEO AI v4.0 is a hardware-aware multimodal AI prototype for
legacy geological documents, geological imagery, Document AI,
Computer Vision, structured interpretation and traceable reporting.

> **Release status:** v4.0 · 51/51 final smoke checks passed · Functional freeze approved

---

## Overview

Subsurface archives contain heterogeneous information:

- scanned legacy reports;
- searchable PDFs;
- photographed pages;
- geological text and tables;
- core and rock photographs;
- thin sections and petrographic imagery;
- mixed document/image pages.

GEO AI provides one workflow for routing these assets into the
appropriate analysis branch while retaining execution provenance.

```text
Legacy Geological Asset
          |
          v
   Multimodal Router
          |
    +-----+-----+
    |           |
    v           v
Document AI   Rock Vision
    |           |
    +-----+-----+
          |
          v
Structured Geoscience Data
          |
          v
Traceable PDF Report
```

---

## Core Capabilities

### Multimodal AI

A single uploaded asset is inspected and routed automatically.

The system distinguishes:

- document-oriented content;
- geological imagery;
- mixed legacy pages containing text and geological figures.

### Document AI

The document pipeline supports:

- searchable PDF text extraction;
- OCR for raster/scanned pages;
- English OCR;
- optional Russian OCR;
- geological entity extraction;
- source/page/extraction provenance;
- OCR confidence and reliability information.

### Rock Vision

The Computer Vision pipeline provides:

- semantic geological segmentation;
- instance separation;
- grain, pore, vug and fracture interpretation;
- object-level measurements;
- fracture analysis;
- visible 2D pore-area estimation;
- annotated visual outputs;
- execution provenance.

### Traceable PDF Reporting

A multimodal analysis can generate a downloadable PDF containing:

- executive summary;
- multimodal routing;
- document-derived properties;
- image-derived measurements;
- before/after visualization;
- geological interpretation;
- object provenance;
- runtime metadata.

---

## Hardware-Aware Runtime

GEO AI uses one codebase across different hardware configurations.

```text
                       GEO AI v4.0
                            |
                       AUTO DETECT
                            |
          +-----------------+-----------------+
          |                 |                 |
          v                 v                 v
   Lightweight CPU     Advanced GPU      Research GPU
          |                 |                 |
    Portable CV        PyTorch CUDA       Experimental
      pipeline          GPU backend        extension
```

| Compute Profile | Purpose |
|---|---|
| `AUTO` | Detect hardware and resolve an executable backend |
| `Lightweight CPU` | Portable CPU-compatible analysis |
| `Advanced GPU` | CUDA-enabled execution on supported NVIDIA hardware |
| `Research GPU` | Experimental extension point for future research backends |

Analysis quality is selected independently:

**Standard · Detailed · Deep**

Higher detail increases segmentation and instance-analysis granularity.

---

## AI / Computer Vision Architecture

GEO AI distinguishes between algorithms that are executed today and
advanced model families that represent extensible architecture.

### Current Prototype Stack

- OpenCV
- K-means semantic segmentation
- morphology
- watershed
- connected components
- fracture detection
- scikit-image Regionprops
- PyTorch CUDA GPU feature processing

### Advanced AI Architecture

The architecture is designed for integration with:

- U-Net
- U-Net++
- DeepLabV3+
- YOLO-Seg
- SAM 2

### Research Extensions

- model ensemble / fusion;
- geological fine-tuning;
- high-resolution refinement;
- foundation-model-assisted segmentation.

> **Execution transparency:** advanced models are architectural extension
> targets unless validated weights and execution backends are installed.
> Only algorithms explicitly reported in the execution trace are considered
> executed for a particular inference run.

---

## Quick Start

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd GEO-AI
```

### 2. Create a virtual environment

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run GEO AI

```bash
python app.py
```

Open the local Gradio URL printed in the terminal.

---

## NVIDIA GPU Execution

The same repository can run on a CUDA-capable workstation.

The validated development workstation used:

- NVIDIA RTX 4500 Ada Generation;
- approximately 24 GB VRAM;
- PyTorch 2.6.0+cu124;
- CUDA build 12.4.

When CUDA and the Advanced GPU backend are available:

```text
AUTO -> Advanced GPU
```

Otherwise:

```text
AUTO -> Lightweight CPU
```

Install a PyTorch build compatible with the local NVIDIA driver and
CUDA environment before using the GPU execution path.

---

## OCR

Tesseract is optional for searchable PDFs but recommended for raster scans.

Ubuntu / Debian example:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

Russian OCR additionally requires the corresponding Tesseract language data.
Windows helper scripts are included in the repository.

---

## Repository Structure

```text
GEO-AI/
|-- app.py
|-- config.py
|-- inference.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|
|-- src/
|   |-- document_ai/
|   `-- rock_vision/
|       `-- backends/
|           `-- advanced_gpu.py
|
|-- notebooks/
|   `-- Multimodal_Geological_AI_Pipeline.ipynb
|
|-- assets/
|   |-- examples/
|   `-- premium_ui.css
|
|-- models/
|-- outputs/
|-- tests/
`-- docs/
```

---

## Research to Deployment

The included notebook represents the research/training side of the project,
while the application represents the inference/product side.

```text
RESEARCH
   |
TRAINING / EXPERIMENTATION
   |
MODEL ARCHITECTURE
   |
INFERENCE
   |
WEB APPLICATION
   |
DEPLOYMENT
   |
END USER
```

---

## Example Assets

The repository includes reproducible demonstration assets:

- carbonate thin section;
- fractured rock;
- vuggy carbonate;
- legacy report scan;
- multimodal legacy report;
- searchable PDF.

These examples allow the application to be demonstrated without
proprietary field data.

---

## Deployment Modes

One repository supports three deployment patterns.

### Public Live Demo

Browser-accessible deployment for demonstrations from desktop or mobile.

### Local Full Version

Clone the repository and run GEO AI locally. `AUTO` resolves an
appropriate runtime according to the available hardware.

### GPU Workstation Deployment

The same application can run on a CUDA workstation and expose the
Advanced GPU execution path.

---

## Scientific Scope and Limitations

GEO AI v4.0 is a **research and portfolio prototype**, not a certified
geological interpretation or reserves-evaluation system.

Important limitations:

- prototype segmentation performance does not establish field accuracy;
- synthetic/demo validation does not replace independent real-data validation;
- visible 2D pore area is not laboratory porosity or 3D effective porosity;
- permeability is not inferred from image appearance unless supported by
  an independently validated model;
- advanced neural/foundation models require validated weights before they
  can be reported as executed.

---

## Validation

Final GEO AI v4.0 release smoke test:

```text
PASS: 51
FAIL: 0
SUCCESS RATE: 100.0%
FUNCTIONAL FREEZE: APPROVED
```

The Advanced GPU backend was additionally validated through real CUDA
execution on an NVIDIA RTX 4500 Ada Generation GPU.

---

## Documentation

- `ARCHITECTURE.md` — system architecture;
- `DEPLOYMENT.md` — deployment guidance;
- `MODEL_CARD.md` — model/prototype scope;
- `PROJECT_STRUCTURE.md` — repository organization;
- `docs/history/` — historical development notes.

---

## License

See `LICENSE`.

---

## Author

**Ildar Z. Farkhutdinov**

Petroleum Engineering · Applied AI · Computer Vision · Digital Rock

---

### GEO AI v4.0

**From legacy geological assets to traceable digital intelligence.**
