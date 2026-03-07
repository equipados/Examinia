from __future__ import annotations

from exam_segmenter import segment_exams
from models import PageExtraction


def _page(
    n: int,
    *,
    name: str = "Alumno Test",
    header: bool = False,
    header_conf: float = 0.0,
) -> PageExtraction:
    return PageExtraction(
        source_file="examen.pdf",
        page_number=n,
        student_name=name,
        student_name_confidence=0.95,
        exam_model="A",
        start_header_detected=header,
        start_header_confidence=header_conf,
        exam_boundary_hint="unknown",
        boundary_confidence=0.0,
        questions=[],
        incidents=[],
        extraction_confidence=0.9,
    )


def test_segment_uses_start_header_as_exam_boundary() -> None:
    pages = [
        _page(1, name="Ana", header=True, header_conf=0.98),
        _page(2, name="Ana", header=False),
        _page(3, name="Luis", header=True, header_conf=0.97),
        _page(4, name="Luis", header=False),
    ]
    submissions = segment_exams(
        page_extractions=pages,
        source_file="examen.pdf",
        boundary_confidence_threshold=0.72,
        header_confidence_threshold=0.80,
    )
    assert len(submissions) == 2
    assert submissions[0].pages == [1, 2]
    assert submissions[1].pages == [3, 4]


def test_low_confidence_header_does_not_force_split() -> None:
    pages = [
        _page(1, name="Ana", header=True, header_conf=0.98),
        _page(2, name="Ana", header=True, header_conf=0.40),
        _page(3, name="Ana", header=False),
    ]
    submissions = segment_exams(
        page_extractions=pages,
        source_file="examen.pdf",
        boundary_confidence_threshold=0.72,
        header_confidence_threshold=0.80,
    )
    assert len(submissions) == 1
    assert submissions[0].pages == [1, 2, 3]
