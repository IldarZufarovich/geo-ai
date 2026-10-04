# Deployment guide

## 1. Local

```bash
git clone <YOUR_REPOSITORY_URL>
cd multimodal-geological-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio.

For scan OCR install Tesseract separately. Searchable PDFs do not need it.

## 2. GitHub

1. Create a public repository, e.g. `multimodal-geological-ai`.
2. Upload the project files.
3. Commit/push.
4. Do not commit private documents or large unvalidated weights.
5. Add screenshots after the app is deployed.

## 3. Hugging Face Spaces (free CPU)

1. Create/log in to Hugging Face.
2. Select **New Space**.
3. Name it, e.g. `multimodal-geological-ai`.
4. Choose **Gradio** SDK.
5. Choose public visibility for a portfolio demo.
6. Select the available free CPU hardware tier.
7. Upload the repository files or push them to the Space git repository.
8. Ensure `app.py` and `requirements.txt` are at repository root.
9. Wait for dependency build and launch.
10. Test `Use demo sample` first.
11. Test image upload on desktop.
12. Open the public Space URL on your phone and test camera input.

### OCR note
The Space base image may not contain the Tesseract system binary. The app degrades safely: searchable PDFs still work, while raster OCR reports a warning. For guaranteed scan OCR, add a Space Docker configuration or use a Python-only OCR model later. This is a P1 deployment refinement, not a P0 blocker.

## 4. GitHub Pages integration

Keep GitHub Pages as the portfolio landing page and link to the Space.

```html
<a class="btn" href="https://huggingface.co/spaces/YOUR_USER/multimodal-geological-ai" target="_blank">Try Live Demo</a>
<a class="btn" href="https://github.com/YOUR_USER/multimodal-geological-ai" target="_blank">View Source</a>
<a class="btn" href="https://colab.research.google.com/github/YOUR_USER/multimodal-geological-ai/blob/main/notebooks/Multimodal_Geological_AI_Pipeline.ipynb" target="_blank">Open Colab</a>
```

## 5. QR code

After deployment:

```bash
python generate_qr.py https://huggingface.co/spaces/YOUR_USER/multimodal-geological-ai
```

Output: `assets/qr/live_demo_qr.png`.

## 6. Mobile/conference test checklist

- Open QR URL over cellular network.
- Camera permission works.
- Demo sample works even without a suitable geological image.
- Analysis completes on CPU.
- Overlay is legible on a small screen.
- Disclaimer is visible.
- No confidential data is used.

## 7. What still requires cloud validation
Actual Hugging Face build/runtime, mobile camera behavior on the final public URL, and system-level Tesseract availability must be verified after you create the Space.
