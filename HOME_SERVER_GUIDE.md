# Home Server Guide — Lightweight v1.2

This edition is designed to run inference on a home Windows workstation while users access it through a browser.

## Architecture

GitHub / portfolio → public HTTPS URL → Cloudflare Tunnel → home PC → Gradio → OCR/CV inference.

The home PC must be powered on, connected to the internet, and the app plus tunnel must be running.

## 1. Install prerequisites

Recommended: Python 3.10 or 3.11, Git, Tesseract OCR 5.x, and the project dependencies.

From Anaconda Prompt:

```bat
conda create -n geoai_home python=3.11 -y
conda activate geoai_home
pip install -r requirements.txt
```

Install Tesseract for Windows. Ensure English is installed; install Russian language data if Russian OCR is required. The app detects common Windows Tesseract locations automatically. You may also set `TESSERACT_CMD` to the full executable path.

## 2. Verify OCR

```bat
python -c "from src.document_ai.ocr import configure_tesseract,available_languages; print(configure_tesseract()); print(available_languages())"
```

Expected: Tesseract path plus `eng`; ideally `rus` too.

## 3. Start the app

Double-click `START_HOME_SERVER_WINDOWS.bat`, or run:

```bat
set GEOAI_HOST=0.0.0.0
set GEOAI_PORT=7871
python app.py
```

On the home PC open `http://127.0.0.1:7871`.

## 4. Test from the home LAN first

Run `ipconfig`, find the PC's IPv4 address, and from a phone on the same Wi-Fi open `http://PC_IP:7871`.

If Windows Firewall prompts for Python, allow access only on Private networks. Do not manually expose router ports to the internet.

## 5. Public HTTPS with Cloudflare Tunnel

Use Cloudflare Tunnel only on a network where you are authorized to run a public service.

Install `cloudflared` from Cloudflare's official distribution. Then, while the app is running:

```bat
cloudflared tunnel --url http://localhost:7871
```

For a quick test Cloudflare returns a temporary `https://...trycloudflare.com` URL. Anyone with the URL can reach the demo while the PC and tunnel are running.

For a stable branded URL, create a named Cloudflare Tunnel and map a hostname such as `geoai.yourdomain.com` to `http://localhost:7871`.

## 6. Security

- Never upload confidential/proprietary geological data to a public demo.
- Do not use a corporate network to bypass security controls.
- Do not use router port-forwarding when a managed outbound tunnel is available.
- Keep Windows, Python packages, Tesseract, and cloudflared updated.
- Stop the tunnel when the public demo is not needed.

## 7. Performance on an old workstation

Start with one concurrent request (the app is configured this way), one image at a time, and images around 1–3 MP. OCR deliberately evaluates multiple preprocessing/PSM variants, so it is more accurate but slower than a single Tesseract call. If the old PC is too slow, reduce input image resolution before upload or switch OCR to a smaller candidate set later.

## 8. Acceptance test

1. Rock Vision demo sample completes.
2. Searchable PDF uses active text extraction.
3. English photographed page produces readable OCR.
4. Russian page works if `rus` is installed.
5. Phone UI stacks into one column.
6. Phone camera/upload works over HTTPS.
7. `EXECUTED IN THIS RUN` is visible.
