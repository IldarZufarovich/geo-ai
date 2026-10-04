from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    app_name: str = "Multimodal Geological AI"
    model_version: str = "geo-ai-lightweight-1.3.0"
    random_seed: int = 42
    max_image_side: int = 1280
    max_pdf_pages: int = 5
    confidence_threshold: float = 0.60
    output_dir: Path = Path("outputs")
    examples_dir: Path = Path("assets/examples")
    fast_demo: bool = True

SETTINGS = Settings()
SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
