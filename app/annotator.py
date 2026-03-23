"""Motor de anotación: dibuja correcciones sobre la portada del examen escaneado."""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

_FONT_DIR = Path(__file__).resolve().parent.parent / "static" / "fonts"
_CAVEAT_PATH = _FONT_DIR / "Caveat-Variable.ttf"

_RED = (204, 0, 0, 255)
_GREEN = (0, 140, 0, 255)
_ORANGE = (210, 120, 0, 255)
_DARK_RED = (150, 0, 0, 255)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(_CAVEAT_PATH), size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _pct_to_px(pct: float, total: int) -> int:
    return int(pct / 100.0 * total)


def _color_for_score(awarded: float, max_pts: float):
    if max_pts <= 0:
        return _ORANGE
    ratio = awarded / max_pts
    if ratio >= 0.99:
        return _GREEN
    if ratio >= 0.4:
        return _ORANGE
    return _RED


def _draw_nota(draw: ImageDraw.ImageDraw, img_w: int, img_h: int,
               nota_box: dict | None, total: float, max_total: float):
    """Escribe la nota total en el cuadro CALIFICACIÓN de la portada."""
    font = _load_font(42)
    score_text = f"{total:.2f}"

    if nota_box:
        # Posicionar dentro del cuadro CALIFICACIÓN detectado
        bx = _pct_to_px(nota_box["x_pct"], img_w)
        by = _pct_to_px(nota_box["y_pct"], img_h)
        bw = _pct_to_px(nota_box["w_pct"], img_w)
        bh = _pct_to_px(nota_box["h_pct"], img_h)
        # Centrar horizontalmente, en la mitad inferior del cuadro
        text_bbox = draw.textbbox((0, 0), score_text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        tx = bx + (bw - tw) // 2
        ty = by + bh // 2
        draw.text((tx, ty), score_text, fill=_RED, font=font)
    else:
        # Fallback: esquina superior derecha, sin solapar con el encabezado
        text_bbox = draw.textbbox((0, 0), score_text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        draw.text((img_w - tw - 30, 15), score_text, fill=_RED, font=font)


def _draw_exercise_score(draw: ImageDraw.ImageDraw, img_w: int, img_h: int,
                         q_bbox: dict | None, y_fallback: int,
                         question_id: str, awarded: float, max_pts: float,
                         parts: list[dict]) -> int:
    """Escribe la nota del ejercicio de forma compacta junto al número del ejercicio.

    Formato compacto: una línea por ejercicio, y debajo una línea por apartado si hay varios.
    Retorna la posición y para el siguiente ejercicio (fallback).
    """
    font_main = _load_font(28)
    font_part = _load_font(20)
    color = _color_for_score(awarded, max_pts)
    RIGHT_MARGIN = 15  # margen derecho mínimo

    # Texto principal: "Ej1: 3.00/3.00"
    main_text = f"{awarded:.2f}/{max_pts:.2f}"

    # Calcular posición
    if q_bbox:
        # Justo a la derecha del encabezado del ejercicio
        bx = _pct_to_px(q_bbox["x_pct"], img_w)
        by = _pct_to_px(q_bbox["y_pct"], img_h)
        bw = _pct_to_px(q_bbox["w_pct"], img_w)
        ann_x = bx + bw + 8
        ann_y = by
    else:
        ann_x = img_w - 200
        ann_y = y_fallback

    # Medir texto para no salirnos de la hoja
    main_bbox = draw.textbbox((0, 0), main_text, font=font_main)
    main_tw = main_bbox[2] - main_bbox[0]

    # Si se sale por la derecha, mover a la izquierda
    if ann_x + main_tw + RIGHT_MARGIN > img_w:
        ann_x = img_w - main_tw - RIGHT_MARGIN

    # Dibujar nota del ejercicio
    draw.text((ann_x, ann_y), main_text, fill=color, font=font_main)

    # Si hay varios apartados, listar debajo de forma compacta
    y = ann_y + 30
    if len(parts) > 1:
        for p in parts:
            p_color = _color_for_score(p["awarded"], p["max"])
            p_text = f"{p['label']}: {p['awarded']:.2f}/{p['max']:.2f}"
            # Medir y ajustar posición
            p_bbox = draw.textbbox((0, 0), p_text, font=font_part)
            p_tw = p_bbox[2] - p_bbox[0]
            p_x = ann_x + 10
            if p_x + p_tw + RIGHT_MARGIN > img_w:
                p_x = img_w - p_tw - RIGHT_MARGIN
            draw.text((p_x, y), p_text, fill=p_color, font=font_part)
            y += 22

    return max(y + 10, ann_y + 50)


def generate_annotated_pdf(submission, session, upload_dir: Path) -> bytes:
    """Genera un PDF anotando solo la portada del examen con las correcciones.

    La portada (hoja impresa) recibe:
    - Nota total en el cuadro CALIFICACIÓN existente
    - Nota por ejercicio junto a cada encabezado
    - Detalle por apartado (compacto, solo puntos)

    Las páginas manuscritas del alumno se incluyen sin modificar.
    """
    from pdf_processor import convert_pdf_to_images

    # 1. Localizar o regenerar imágenes
    temp_dir = upload_dir / ".tmp" / f"sub_{submission.id}"
    page_image_paths: list[tuple[int, Path]] = []

    if temp_dir.exists():
        pngs = sorted(temp_dir.glob("*.png"))
        for i, png in enumerate(pngs, start=1):
            page_image_paths.append((i, png))

    if not page_image_paths:
        pdf_path = Path(submission.pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")
        import os
        temp_dir.mkdir(parents=True, exist_ok=True)
        poppler_path = os.environ.get("POPPLER_PATH", "")
        imgs = convert_pdf_to_images(pdf_path, temp_dir, dpi=220, poppler_path=poppler_path or None)
        for pi in imgs:
            page_image_paths.append((pi.page_number, pi.image_path))

    if not page_image_paths:
        raise FileNotFoundError("No se encontraron páginas del examen")

    # 2. Cargar cover layout
    cover_layout = None
    if submission.cover_layout_json:
        try:
            cover_layout = json.loads(submission.cover_layout_json)
        except json.JSONDecodeError:
            pass

    cover_page_num = cover_layout.get("page_number", 1) if cover_layout else 1
    nota_box = cover_layout.get("nota_box") if cover_layout else None
    q_bboxes: dict[str, dict] = {}
    if cover_layout:
        for q in cover_layout.get("questions", []):
            if "bbox" in q:
                q_bboxes[q["question_id"]] = q["bbox"]

    # 3. Recopilar resultados por ejercicio
    question_data: list[dict] = []
    for qr in submission.question_results:
        parts_detail = []
        q_awarded = 0.0
        q_max = 0.0
        for pr in qr.part_results:
            if len(qr.part_results) <= 1 or pr.part_id == "single":
                label = f"Ej{qr.question_id}"
            else:
                label = f"{pr.part_id}"
            parts_detail.append({
                "label": label,
                "awarded": pr.awarded_points or 0.0,
                "max": pr.max_points or 0.0,
                "status": pr.status or "incorrecto",
            })
            q_awarded += pr.awarded_points or 0.0
            q_max += pr.max_points or 0.0
        question_data.append({
            "question_id": qr.question_id,
            "awarded": round(q_awarded, 2),
            "max": round(q_max, 2),
            "parts": parts_detail,
        })

    # 4. Anotar solo la portada
    total_points = submission.total_points or 0.0
    max_total = submission.max_total_points or 0.0
    annotated_images: list[Image.Image] = []

    for page_num, img_path in page_image_paths:
        img = Image.open(img_path).convert("RGBA")

        if page_num == cover_page_num:
            overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            img_w, img_h = img.size

            # Nota total en CALIFICACIÓN
            _draw_nota(draw, img_w, img_h, nota_box, total_points, max_total)

            # Nota por ejercicio
            y_fallback = 160
            for qd in question_data:
                q_bbox = q_bboxes.get(qd["question_id"])
                y_fallback = _draw_exercise_score(
                    draw, img_w, img_h,
                    q_bbox=q_bbox,
                    y_fallback=y_fallback,
                    question_id=qd["question_id"],
                    awarded=qd["awarded"],
                    max_pts=qd["max"],
                    parts=qd["parts"],
                )

            result = Image.alpha_composite(img, overlay)
            annotated_images.append(result.convert("RGB"))
        else:
            annotated_images.append(img.convert("RGB"))

    # 5. Ensamblar PDF
    buf = io.BytesIO()
    if len(annotated_images) == 1:
        annotated_images[0].save(buf, format="PDF", resolution=220)
    else:
        annotated_images[0].save(buf, format="PDF", resolution=220,
                                 save_all=True, append_images=annotated_images[1:])
    return buf.getvalue()
