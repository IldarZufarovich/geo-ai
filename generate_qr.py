from pathlib import Path
import argparse, qrcode
p=argparse.ArgumentParser(); p.add_argument('url'); p.add_argument('--output',default='assets/qr/live_demo_qr.png'); a=p.parse_args()
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); qrcode.make(a.url).save(out); print(out)
