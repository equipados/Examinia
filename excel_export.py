from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Font, PatternFill

from models import ExamGradeResult
from utils import ensure_dir

_REVISION_FILL = PatternFill(start_color="FFB347", end_color="FFB347", fill_type="solid")
_COURSE_LABELS = {"1o_bachillerato": "1º Bach.", "2o_bachillerato": "2º Bach."}


def _column_sort_key(column_id: str) -> tuple[int, str]:
    if "." in column_id:
        q, part = column_id.split(".", 1)
    else:
        q, part = column_id, ""
    try:
        q_num = int(q)
    except ValueError:
        q_num = 999
    return (q_num, part)


def export_results_to_excel(results: list[ExamGradeResult], output_file: Path) -> Path:
    ensure_dir(output_file.parent)
    part_columns = sorted(
        {
            part.column_id
            for result in results
            for question in result.questions
            for part in question.parts
        },
        key=_column_sort_key,
    )

    headers = ["Alumno", "Archivo", "Curso", "Modelo", *part_columns, "Total", "Revision manual", "Incidencias", "Informe"]
    rows: list[dict[str, object]] = []
    status_rows: list[dict[str, str]] = []

    for result in results:
        row: dict[str, object] = {
            "Alumno": result.student_name,
            "Archivo": result.source_file,
            "Curso": _COURSE_LABELS.get(result.course_level or "", result.course_level or ""),
            "Modelo": result.exam_model or "",
            "Total": result.total_points,
            "Incidencias": "; ".join(result.incidents),
            "Informe": result.report_path or "",
        }
        status_row: dict[str, str] = {}
        revision_parts: list[str] = []

        for column in part_columns:
            row[column] = None
        for question in result.questions:
            for part in question.parts:
                row[part.column_id] = part.awarded_points
                status_row[part.column_id] = part.status
                if part.status == "revision_manual":
                    revision_parts.append(part.column_id)

        row["Revision manual"] = ", ".join(revision_parts) if revision_parts else ""
        rows.append(row)
        status_rows.append(status_row)

    dataframe = pd.DataFrame(rows, columns=headers)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Resultados")
        workbook = writer.book
        worksheet = writer.sheets["Resultados"]

        bold_font = Font(bold=True)
        for cell in worksheet[1]:
            cell.font = bold_font

        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.freeze_panes = "A2"

        # Build header -> column_letter mapping
        col_letter: dict[str, str] = {}
        for cell in worksheet[1]:
            if cell.value is not None:
                col_letter[str(cell.value)] = cell.column_letter

        numeric_columns = set(part_columns + ["Total"])
        for col_cells in worksheet.columns:
            column_letter = col_cells[0].column_letter
            header_value = str(col_cells[0].value or "")
            max_length = len(header_value)
            for cell in col_cells[1:]:
                if cell.value is not None:
                    max_length = max(max_length, len(str(cell.value)))
                if header_value in numeric_columns and cell.row > 1:
                    cell.number_format = "0.00"
            worksheet.column_dimensions[column_letter].width = min(max_length + 2, 60)

        # Highlight revision_manual cells in orange
        for row_idx, status_row in enumerate(status_rows, start=2):
            for col_id, status in status_row.items():
                if status == "revision_manual" and col_id in col_letter:
                    worksheet[f"{col_letter[col_id]}{row_idx}"].fill = _REVISION_FILL
            # Also highlight the "Revision manual" summary cell if non-empty
            rev_col = col_letter.get("Revision manual")
            if rev_col:
                cell = worksheet[f"{rev_col}{row_idx}"]
                if cell.value:
                    cell.fill = _REVISION_FILL
                    cell.font = Font(bold=True)

        workbook.save(output_file)
    return output_file
