from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook

from excel_export import export_results_to_excel
from models import ExamGradeResult, PartGrade, QuestionGrade


def _sample_result() -> ExamGradeResult:
    return ExamGradeResult(
        exam_id="doc::exam_01",
        student_name="Alumno Uno",
        source_file="doc.pdf",
        exam_model="A",
        pages=[1],
        questions=[
            QuestionGrade(
                question_id="1",
                max_points=1.0,
                awarded_points=0.75,
                parts=[
                    PartGrade(
                        question_id="1",
                        part_id="a",
                        column_id="1.a",
                        max_points=1.0,
                        awarded_points=0.75,
                        status="parcial",
                        detected_answer="x=1",
                        normalized_answer="x=1",
                        steps_observed=[],
                        explanation="Test",
                        incidents=[],
                    )
                ],
            )
        ],
        total_points=0.75,
        max_total_points=1.0,
        incidents=[],
        report_path="salidas/informes/alumno_uno.md",
    )


def test_excel_is_generated_with_expected_headers(tmp_path: Path) -> None:
    output_file = tmp_path / "resultados.xlsx"
    export_results_to_excel([_sample_result()], output_file)

    assert output_file.exists()
    workbook = load_workbook(output_file)
    sheet = workbook["Resultados"]
    headers = [cell.value for cell in sheet[1]]
    assert "Alumno" in headers
    assert "1.a" in headers
    assert "Total" in headers
    assert sheet["A1"].font.bold is True
