"""Generador de informes de corrección en PDF para alumnos."""
from __future__ import annotations

import io

from fpdf import FPDF


_COURSE_DISPLAY = {"1o_bachillerato": "1º Bachillerato", "2o_bachillerato": "2º Bachillerato"}

# Mapa de caracteres Unicode comunes en matemáticas → equivalente latin1
_UNICODE_REPLACEMENTS = {
    "\u221a": "sqrt",     # √
    "\u00b2": "2",        # ² (superscript 2) — latin1 tiene este, pero por si acaso
    "\u00b3": "3",        # ³
    "\u2260": "!=",       # ≠
    "\u2264": "<=",       # ≤
    "\u2265": ">=",       # ≥
    "\u00b7": "*",        # ·
    "\u2022": "*",        # •
    "\u2013": "-",        # –
    "\u2014": "-",        # —
    "\u2018": "'",        # '
    "\u2019": "'",        # '
    "\u201c": '"',        # "
    "\u201d": '"',        # "
    "\u2026": "...",      # …
    "\u00d7": "x",        # ×
    "\u00f7": "/",        # ÷
    "\u03b1": "alpha",    # α
    "\u03b2": "beta",     # β
    "\u03c0": "pi",       # π
    "\u00b1": "+-",       # ±
    "\u221e": "inf",      # ∞
    "\u2192": "->",       # →
    "\u2248": "~=",       # ≈
}


def _sanitize_latin1(text: str) -> str:
    """Reemplaza caracteres no-latin1 por equivalentes legibles."""
    result = []
    for ch in text:
        if ch in _UNICODE_REPLACEMENTS:
            result.append(_UNICODE_REPLACEMENTS[ch])
        else:
            try:
                ch.encode("latin-1")
                result.append(ch)
            except UnicodeEncodeError:
                result.append("?")
    return "".join(result)


def generate_correction_pdf(
    student_name: str,
    course_level: str | None,
    session_name: str,
    session_date: str | None,
    questions: list[dict],
    total_points: float,
    max_total_points: float,
) -> bytes:
    """Genera un PDF con el informe de corrección.

    questions: [{question_id, parts: [{part_id, awarded, max, status, explanation}]}]
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Título
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Informe de corrección", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    # Datos del alumno
    course = _COURSE_DISPLAY.get(course_level or "", course_level or "")
    _header_field(pdf, "Alumno:", _sanitize_latin1(student_name))
    if course:
        _header_field(pdf, "Curso:", _sanitize_latin1(course))
    _header_field(pdf, "Prueba:", _sanitize_latin1(session_name))
    if session_date:
        _header_field(pdf, "Fecha:", session_date)
    pdf.ln(6)

    # Tabla de resultados
    col_widths = [35, 25, 130]  # Pregunta, Nota, Detalle
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    col_widths[2] = page_w - col_widths[0] - col_widths[1]

    # Cabecera
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(240, 240, 245)
    pdf.cell(col_widths[0], 8, "Pregunta", border=1, fill=True, align="C")
    pdf.cell(col_widths[1], 8, "Nota", border=1, fill=True, align="C")
    pdf.cell(col_widths[2], 8, "Detalle", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    # Filas
    for q in questions:
        parts = q.get("parts", [])
        for p in parts:
            label = _part_label(q["question_id"], p["part_id"], len(parts))
            nota = f"{p['awarded']:.2f} / {p['max']:.2f}"
            detail = ""
            if p.get("status") not in ("correcto",):
                detail = _sanitize_latin1((p.get("explanation") or "").strip())

            _table_row(pdf, col_widths, label, nota, detail)

    # Total
    pdf.set_font("Helvetica", "B", 11)
    pdf.ln(4)
    pdf.cell(0, 10, f"Total: {total_points:.2f} / {max_total_points:.2f}", new_x="LMARGIN", new_y="NEXT", align="R")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def _header_field(pdf: FPDF, label: str, value: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(22, 7, label)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")


def _part_label(question_id: str, part_id: str, num_parts: int) -> str:
    if num_parts <= 1 or part_id == "single":
        return str(question_id)
    return f"{question_id}.{part_id}"


def _table_row(pdf: FPDF, widths: list[float], label: str, nota: str, detail: str) -> None:
    """Dibuja una fila de la tabla con bordes alineados correctamente."""
    x_start = pdf.get_x()
    y_start = pdf.get_y()
    min_row_h = 7
    line_h = 5

    # Calcular altura real del detalle con dry_run
    detail_clean = detail.replace("\n", " ").strip() if detail else ""
    if detail_clean:
        pdf.set_font("Helvetica", "", 8)
        lines = pdf.multi_cell(widths[2], line_h, detail_clean, dry_run=True, output="LINES")
        n_lines = len(lines)
        row_h = max(min_row_h, n_lines * line_h)
        # Ajustar line_h para que n_lines * line_h == row_h exacto (evita huecos)
        actual_line_h = row_h / n_lines
    else:
        row_h = min_row_h
        actual_line_h = line_h

    # Salto de página si no cabe
    if y_start + row_h > pdf.h - pdf.b_margin:
        pdf.add_page()
        y_start = pdf.get_y()
        x_start = pdf.get_x()

    # Dibujar los 3 rectángulos con la misma altura (sin texto)
    pdf.rect(x_start, y_start, widths[0], row_h)
    pdf.rect(x_start + widths[0], y_start, widths[1], row_h)
    pdf.rect(x_start + widths[0] + widths[1], y_start, widths[2], row_h)

    # Columna 1: Pregunta (centrado verticalmente)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(x_start, y_start)
    pdf.cell(widths[0], row_h, label, border=0, align="C")

    # Columna 2: Nota (centrado verticalmente)
    pdf.set_xy(x_start + widths[0], y_start)
    pdf.cell(widths[1], row_h, nota, border=0, align="C")

    # Columna 3: Detalle
    pdf.set_xy(x_start + widths[0] + widths[1], y_start)
    if detail_clean:
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(widths[2], actual_line_h, detail_clean, border=0)

    # Mover cursor debajo de la fila
    pdf.set_xy(x_start, y_start + row_h)


def merge_pdfs(pdf_bytes_list: list[bytes]) -> bytes:
    """Une varios PDFs en uno solo."""
    from pypdf import PdfReader, PdfWriter
    writer = PdfWriter()
    for pdf_bytes in pdf_bytes_list:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def build_combined_pdf(submission, session) -> bytes:
    """Genera un PDF combinado: informe de corrección + examen escaneado."""
    from pathlib import Path
    report_pdf = build_pdf_from_submission(submission, session)
    scanned_path = submission.pdf_path
    if scanned_path and Path(scanned_path).exists():
        scanned_bytes = Path(scanned_path).read_bytes()
        return merge_pdfs([report_pdf, scanned_bytes])
    return report_pdf


def build_pdf_from_submission(submission, session) -> bytes:
    """Helper: genera PDF desde objetos ORM Submission + ExamSession."""
    student_name = (
        submission.student.display_name if submission.student else submission.student_name
    ) or "Alumno desconocido"

    questions = []
    for qr in submission.question_results:
        parts = []
        for pr in qr.part_results:
            parts.append({
                "part_id": pr.part_id,
                "awarded": pr.awarded_points or 0.0,
                "max": pr.max_points or 0.0,
                "status": pr.status or "incorrecto",
                "explanation": pr.explanation or "",
            })
        questions.append({"question_id": qr.question_id, "parts": parts})

    return generate_correction_pdf(
        student_name=student_name,
        course_level=submission.course_level or (session.course_level if session else None),
        session_name=session.name if session else "Examen",
        session_date=session.date if session else None,
        questions=questions,
        total_points=submission.total_points or 0.0,
        max_total_points=submission.max_total_points or 0.0,
    )
