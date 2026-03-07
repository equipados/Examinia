from __future__ import annotations

from exam_parser import _sanitize_cover_page_answers, build_submission_from_pdf, normalize_submission_structure
from models import ExamSubmission, ExtractedPart, ExtractedQuestion, PageExtraction


def test_equitable_distribution_when_only_question_points() -> None:
    submission = ExamSubmission(
        exam_id="doc::exam_01",
        source_file="doc.pdf",
        pages=[1],
        student_name="Alumno Test",
        student_name_confidence=0.9,
        exam_model="A",
        questions=[
            ExtractedQuestion(
                question_id="1",
                max_points=2.0,
                parts=[
                    ExtractedPart(part_id="a", student_answer_raw="..."),
                    ExtractedPart(part_id="b", student_answer_raw="..."),
                ],
            )
        ],
        incidents=[],
    )

    normalized = normalize_submission_structure(submission)
    points = [part.max_points for part in normalized.questions[0].parts]
    assert points == [1.0, 1.0]
    assert any("Reparto equitativo" in item for item in normalized.incidents)


def test_build_submission_from_pdf_single_exam() -> None:
    cover = PageExtraction(
        source_file="doc.pdf",
        page_number=1,
        student_name="Ana Garcia",
        student_name_confidence=0.95,
        exam_model="B",
        start_header_detected=True,
        start_header_confidence=0.90,
        page_role="cover",
        handwritten_content_detected=False,
        handwritten_content_confidence=0.1,
        exam_boundary_hint="new_exam",
        boundary_confidence=0.9,
        questions=[
            ExtractedQuestion(
                question_id="1",
                statement="Calcula la derivada de f(x)=x^2",
                max_points=2.0,
                parts=[ExtractedPart(part_id="a", statement="Deriva f(x)=x^2", max_points=2.0, student_answer_raw="")],
            )
        ],
        incidents=[],
        extraction_confidence=0.9,
    )
    answer_page = PageExtraction(
        source_file="doc.pdf",
        page_number=2,
        student_name=None,
        student_name_confidence=0.0,
        exam_model=None,
        start_header_detected=False,
        start_header_confidence=0.0,
        page_role="answer",
        handwritten_content_detected=True,
        handwritten_content_confidence=0.9,
        exam_boundary_hint="continuation",
        boundary_confidence=0.9,
        questions=[
            ExtractedQuestion(
                question_id="1",
                parts=[ExtractedPart(part_id="a", student_answer_raw="2x")],
            )
        ],
        incidents=[],
        extraction_confidence=0.85,
    )

    submission = build_submission_from_pdf([cover, answer_page], "doc.pdf")

    assert submission.student_name == "Ana Garcia"
    assert submission.exam_model == "B"
    assert submission.exam_id == "doc.pdf::exam_01"
    assert submission.pages == [1, 2]
    assert len(submission.questions) == 2  # one from each page, merged later by normalize_submission_structure


def test_cover_page_without_handwriting_clears_answers() -> None:
    extraction = PageExtraction(
        source_file="doc.pdf",
        page_number=1,
        student_name="Alumno",
        student_name_confidence=0.9,
        exam_model="A",
        start_header_detected=True,
        start_header_confidence=0.95,
        page_role="cover",
        handwritten_content_detected=False,
        handwritten_content_confidence=0.1,
        exam_boundary_hint="new_exam",
        boundary_confidence=0.9,
        questions=[
            ExtractedQuestion(
                question_id="1",
                parts=[
                    ExtractedPart(
                        part_id="a",
                        student_answer_raw="x=2",
                        steps_detected=["paso 1"],
                        confidence=0.9,
                    )
                ],
            )
        ],
        incidents=[],
        extraction_confidence=0.9,
    )

    _sanitize_cover_page_answers(extraction)
    part = extraction.questions[0].parts[0]
    assert part.student_answer_raw == ""
    assert part.steps_detected == []
    assert any("caratula" in item.lower() for item in extraction.incidents)
