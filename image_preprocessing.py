from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from models import PageImage
from utils import ensure_dir


def preprocess_image(source_path: Path, target_path: Path) -> Path:
    with Image.open(source_path) as image:
        processed = ImageOps.grayscale(image)
        processed = ImageOps.autocontrast(processed)
        processed = processed.filter(ImageFilter.SHARPEN)
        processed.save(target_path, format="PNG")
    return target_path


def preprocess_pages(page_images: list[PageImage], output_dir: Path) -> list[PageImage]:
    ensure_dir(output_dir)
    processed_pages: list[PageImage] = []
    for page in page_images:
        processed_path = output_dir / page.image_path.name
        preprocess_image(page.image_path, processed_path)
        processed_pages.append(
            PageImage(
                source_file=page.source_file,
                page_number=page.page_number,
                image_path=processed_path,
            )
        )
    return processed_pages
