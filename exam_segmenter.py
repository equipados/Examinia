from __future__ import annotations

from collections import Counter

from loguru import logger

from models import ExamSubmission, PageExtraction


def _should_start_new_exam(
    previous: PageExtraction,
    current: PageExtraction,
    boundary_confidence_threshold: float,
    header_confidence_threshold: float,
) -> bool:
    # Strong rule from user domain: each exam starts with the printed header (logo left + student name).
    if current.start_header_detected and current.start_header_confidence >= header_confidence_threshold:
        return True

    # Fallback rule from LLM explicit boundary hint.
    if current.exam_boundary_hint == "new_exam" and current.boundary_confidence >= boundary_confidence_threshold:
        return True

    # Last fallback: high-confidence student name change.
    prev_name = (previous.student_name or "").strip().lower()
    curr_name = (current.student_name or "").strip().lower()
    if (
        prev_name
        and curr_name
        and prev_name != curr_name
        and previous.student_name_confidence >= boundary_confidence_threshold
        and current.student_name_confidence >= boundary_confidence_threshold
    ):
        return True
    return False


def _choose_student_name(pages: list[PageExtraction]) -> tuple[str | None, float]:
    candidates: list[tuple[str, float]] = []
    for page in pages:
        if page.student_name:
            candidates.append((page.student_name, page.student_name_confidence))
    if not candidates:
        return None, 0.0
    by_name: dict[str, list[float]] = {}
    for name, confidence in candidates:
        by_name.setdefault(name, []).append(confidence)
    best_name = max(by_name.items(), key=lambda item: (len(item[1]), max(item[1]), sum(item[1]) / len(item[1])))[0]
    best_conf = max(by_name[best_name])
    return best_name, best_conf


def _choose_exam_model(pages: list[PageExtraction]) -> str | None:
    models = [page.exam_model for page in pages if page.exam_model]
    if not models:
        return None
    return Counter(models).most_common(1)[0][0]


def _choose_course_level(pages: list[PageExtraction]) -> str | None:
    levels = [page.course_level for page in pages if page.course_level]
    if not levels:
        return None
    return Counter(levels).most_common(1)[0][0]


def segment_exams(
    page_extractions: list[PageExtraction],
    source_file: str,
    boundary_confidence_threshold: float = 0.70,
    header_confidence_threshold: float = 0.80,
) -> list[ExamSubmission]:
    if not page_extractions:
        return []
    pages = sorted(page_extractions, key=lambda p: p.page_number)
    page_groups: list[list[PageExtraction]] = [[pages[0]]]
    incidents_by_group: list[list[str]] = [[*pages[0].incidents]]

    for current in pages[1:]:
        previous = page_groups[-1][-1]
        start_new = _should_start_new_exam(
            previous=previous,
            current=current,
            boundary_confidence_threshold=boundary_confidence_threshold,
            header_confidence_threshold=header_confidence_threshold,
        )

        if current.start_header_detected and current.start_header_confidence < header_confidence_threshold:
            incidents_by_group[-1].append(
                (
                    f"Header de inicio detectado con baja confianza en pagina {current.page_number} "
                    f"({current.start_header_confidence:.2f})."
                )
            )
        if current.exam_boundary_hint == "new_exam" and current.boundary_confidence < boundary_confidence_threshold:
            incidents_by_group[-1].append(
                (
                    f"Segmentacion dudosa en pagina {current.page_number}: "
                    "marca de nuevo examen con baja confianza."
                )
            )

        if start_new:
            page_groups.append([current])
            incidents_by_group.append([*current.incidents])
        else:
            page_groups[-1].append(current)
            incidents_by_group[-1].extend(current.incidents)

    submissions: list[ExamSubmission] = []
    for index, group_pages in enumerate(page_groups, start=1):
        student_name, student_confidence = _choose_student_name(group_pages)
        exam_model = _choose_exam_model(group_pages)
        course_level = _choose_course_level(group_pages)
        merged_questions = []
        for page in group_pages:
            merged_questions.extend(page.questions)

        exam_id = f"{source_file}::exam_{index:02d}"
        submission = ExamSubmission(
            exam_id=exam_id,
            source_file=source_file,
            pages=[page.page_number for page in group_pages],
            student_name=student_name,
            student_name_confidence=student_confidence,
            exam_model=exam_model,
            course_level=course_level,
            questions=merged_questions,
            incidents=incidents_by_group[index - 1],
        )
        submissions.append(submission)
        logger.info(
            f"Examen segmentado: {exam_id} | alumno={submission.student_name or 'desconocido'} | paginas={submission.pages}"
        )
    return submissions
