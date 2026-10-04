# Model weights

The public CPU demo intentionally runs with deterministic lightweight fallbacks so it remains reliable without large downloads.

The research notebook trains a lightweight U-Net, CNN, and a simplified CNN–Transformer classifier. Export validated weights here when available, e.g. `geo_unet_v1.pt`. The live app should load weights lazily and fall back safely when a weight file is absent.

**Do not describe demo heuristics as field-validated AI.**
