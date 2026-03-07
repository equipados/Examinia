from __future__ import annotations

from pathlib import Path

from models import ExamGradeResult, PartGrade, QuestionGrade
from reporting import build_markdown_report, write_exam_report


def _sample_result() -> ExamGradeResult:
    return ExamGradeResult(
        exam_id="doc::exam_01",
        student_name="Ana Gómez",
        source_file="doc.pdf",
        exam_model="A",
        pages=[1, 2],
        questions=[
            QuestionGrade(
                question_id="1",
                max_points=1.0,
                awarded_points=0.5,
                parts=[
                    PartGrade(
                        question_id="1",
                        part_id="a",
                        column_id="1.a",
                        max_points=1.0,
                        awarded_points=0.5,
                        status="parcial",
                        detected_answer="x=3",
                        normalized_answer="x=3",
                        steps_observed=["plantea sistema"],
                        explanation="Explicación test",
                        incidents=[],
                    )
                ],
            )
        ],
        total_points=0.5,
        max_total_points=1.0,
        incidents=["Incidencia test"],
        report_path=None,
    )


def test_markdown_report_contains_total() -> None:
    report_text = build_markdown_report(_sample_result())
    assert "**Total: 0.50 / 1.00**" in report_text
    assert "Ejercicio 1" in report_text


def test_report_file_is_written_and_deduplicated(tmp_path: Path) -> None:
    seen: dict[str, int] = {}
    first = write_exam_report(_sample_result(), tmp_path, seen)
    second = write_exam_report(_sample_result(), tmp_path, seen)
    assert first.exists()
    assert second.exists()
    assert first.name != second.name
