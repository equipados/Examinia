from __future__ import annotations

from pathlib import Path

from models import ExamGradeResult
from utils import deduplicate_name, ensure_dir, format_page_range, safe_filename


def _format_part_title(part_id: str) -> str:
    if part_id == "single":
        return "Apartado único"
    return f"Apartado {part_id})"


def build_markdown_report(result: ExamGradeResult) -> str:
    lines: list[str] = []
    lines.append("# Informe de corrección")
    lines.append("")
    lines.append(f"**Alumno:** {result.student_name}")
    lines.append(f"**Archivo origen:** {result.source_file}")
    _course_display = {"1o_bachillerato": "1º Bachillerato", "2o_bachillerato": "2º Bachillerato"}.get(
        result.course_level or "", result.course_level or "No detectado"
    )
    lines.append(f"**Curso:** {_course_display}")
    lines.append(f"**Modelo:** {result.exam_model or 'No detectado'}")
    lines.append(f"**Páginas:** {format_page_range(result.pages)}")
    lines.append("")

    for question in result.questions:
        lines.append(f"## Ejercicio {question.question_id} ({question.max_points:.2f} puntos)")
        lines.append("")
        for part in question.parts:
            _revision_flag = " [REQUIERE REVISION MANUAL]" if part.status == "revision_manual" else ""
            lines.append(
                f"### {_format_part_title(part.part_id)} — {part.awarded_points:.2f} / {part.max_points:.2f}{_revision_flag}"
            )
            lines.append(f"- Respuesta detectada: {part.detected_answer or 'No detectada'}")
            lines.append(f"- Interpretación matemática: {part.normalized_answer or 'No disponible'}")
            steps = "; ".join(part.steps_observed) if part.steps_observed else "No se identifican pasos claros."
            lines.append(f"- Observación del procedimiento: {steps}")
            lines.append(f"- Estado: {part.status}")
            lines.append(f"- Explicación: {part.explanation}")
            if part.incidents:
                lines.append(f"- Incidencias del apartado: {'; '.join(part.incidents)}")
            lines.append("")

    lines.append("## Incidencias")
    if result.incidents:
        for incident in result.incidents:
            lines.append(f"- {incident}")
    else:
        lines.append("- Sin incidencias relevantes.")
    lines.append("")
    lines.append("## Resultado final")
    lines.append(f"**Total: {result.total_points:.2f} / {result.max_total_points:.2f}**")
    lines.append("")
    return "\n".join(lines)


_COURSE_DISPLAY = {"1o_bachillerato": "1º Bachillerato", "2o_bachillerato": "2º Bachillerato"}


def build_report_from_db(submission, session_name: str | None = None, session_date: str | None = None) -> str:
    """Genera el informe markdown desde objetos ORM (Submission + QuestionResult + PartResult)."""
    student_name = (
        submission.student.display_name if submission.student else submission.student_name
    ) or "Alumno desconocido"
    course = _COURSE_DISPLAY.get(submission.course_level or "", submission.course_level or "No detectado")

    lines: list[str] = []
    lines.append("# Informe de corrección")
    lines.append("")
    lines.append(f"**Alumno:** {student_name}")
    lines.append(f"**Archivo origen:** {submission.source_filename}")
    lines.append(f"**Curso:** {course}")
    if session_name:
        lines.append(f"**Prueba:** {session_name}")
    if session_date:
        lines.append(f"**Fecha:** {session_date}")
    lines.append("")

    for qr in submission.question_results:
        q_max = sum(p.max_points or 0 for p in qr.part_results)
        lines.append(f"## Ejercicio {qr.question_id} ({q_max:.2f} puntos)")
        lines.append("")
        for pr in qr.part_results:
            revision_flag = " [REQUIERE REVISION MANUAL]" if pr.status == "revision_manual" else ""
            lines.append(
                f"### {_format_part_title(pr.part_id)} — {(pr.awarded_points or 0):.2f} / {(pr.max_points or 0):.2f}{revision_flag}"
            )
            lines.append(f"- Respuesta detectada: {pr.detected_answer or 'No detectada'}")
            lines.append(f"- Estado: {pr.status}")
            lines.append(f"- Explicación: {pr.explanation or ''}")
            if pr.incidents:
                import json
                try:
                    incs = json.loads(pr.incidents) if isinstance(pr.incidents, str) else pr.incidents
                except Exception:
                    incs = []
                if incs:
                    lines.append(f"- Incidencias del apartado: {'; '.join(incs)}")
            lines.append("")

    lines.append("## Resultado final")
    lines.append(f"**Total: {(submission.total_points or 0):.2f} / {(submission.max_total_points or 0):.2f}**")
    lines.append("")
    return "\n".join(lines)


def write_exam_report(result: ExamGradeResult, reports_dir: Path, seen_filenames: dict[str, int]) -> Path:
    ensure_dir(reports_dir)
    base = safe_filename(result.student_name, fallback="alumno_sin_nombre")
    unique = deduplicate_name(base, seen_filenames)
    report_path = reports_dir / f"{unique}.md"
    report_path.write_text(build_markdown_report(result), encoding="utf-8")
    return report_path
