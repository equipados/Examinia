from __future__ import annotations

from pathlib import Path

from loguru import logger

from models import PageImage
from utils import ensure_dir


def discover_pdf_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        logger.warning(f"Carpeta de entrada no encontrada: {input_dir}")
        return []
    pdfs = sorted(path for path in input_dir.rglob("*.pdf") if path.is_file())
    logger.info(f"PDFs encontrados: {len(pdfs)}")
    return pdfs


def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 220,
    poppler_path: Path | None = None,
) -> list[PageImage]:
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise RuntimeError(
            "No se pudo importar pdf2image. Instala dependencias con: pip install -r requirements.txt"
        ) from exc

    ensure_dir(output_dir)
    logger.info(f"Convirtiendo PDF a imagenes: {pdf_path.name}")
    poppler_arg = str(poppler_path) if poppler_path else None
    try:
        pil_images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt="png",
            poppler_path=poppler_arg,
        )
    except Exception as exc:  # pragma: no cover - depends on local poppler binary
        suffix = f" (poppler_path={poppler_arg})" if poppler_arg else ""
        raise RuntimeError(
            f"Error al convertir {pdf_path.name}. Verifica que los binarios de Poppler esten disponibles{suffix}."
        ) from exc

    page_images: list[PageImage] = []
    for index, image in enumerate(pil_images, start=1):
        image_name = f"{pdf_path.stem}_p{index:03d}.png"
        image_path = output_dir / image_name
        image.save(image_path, format="PNG")
        page_images.append(
            PageImage(
                source_file=pdf_path.name,
                page_number=index,
                image_path=image_path,
            )
        )
    logger.info(f"Paginas convertidas ({pdf_path.name}): {len(page_images)}")
    return page_images
