from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

from loguru import logger
from tqdm import tqdm

from exam_segmenter import _choose_course_level, _choose_exam_model, _choose_student_name
from gemini_client import GeminiClient
from models import ExamSubmission, ExtractedPart, ExtractedQuestion, PageExtraction, PageImage
from utils import round_points


def _sanitize_cover_page_answers(extraction: PageExtraction) -> None:
    # Business rule: cover page usually has statements/points, not student handwritten answers.
    if extraction.page_role != "cover":
        return
    if extraction.handwritten_content_detected:
        return
    cleared = 0
    for question in extraction.questions:
        for part in question.parts:
            had_answer = bool((part.student_answer_raw or "").strip())
            had_steps = bool(part.steps_detected)
            if had_answer or had_steps:
                part.student_answer_raw = ""
                part.student_answer_normalized = None
                part.steps_detected = []
                part.confidence = min(part.confidence, 0.25)
                cleared += 1
    if cleared > 0:
        extraction.incidents.append(
            (
                f"Se limpiaron {cleared} respuestas en caratula sin manuscrito "
                f"(pagina {extraction.page_number})."
            )
        )


def analyze_pages_with_gemini(
    page_images: list[PageImage],
    gemini_client: GeminiClient,
    low_confidence_threshold: float = 0.72,
    reanalysis_threshold: float = 0.55,
    progress_callback=None,  # callable(current: int, total: int) opcional
    known_questions_context: list[dict] | None = None,
) -> list[PageExtraction]:
    extracted_pages: list[PageExtraction] = []
    questions_context: list[dict] = known_questions_context or []
    page_images_sorted = sorted(page_images, key=lambda p: p.page_number)
    _total_pages = len(page_images_sorted)

    for _page_idx, page in enumerate(tqdm(page_images_sorted, desc="Analizando paginas con Gemini", unit="pagina"), 1):
        use_cover_prompt = not questions_context

        if use_cover_prompt:
            extraction = gemini_client.extract_exam_questions(
                image_path=page.image_path,
                source_file=page.source_file,
                page_number=page.page_number,
            )
        else:
            extraction = gemini_client.extract_student_answers(
                image_path=page.image_path,
                source_file=page.source_file,
                page_number=page.page_number,
                questions_context=questions_context,
            )

        if extraction.extraction_confidence <= reanalysis_threshold:
            logger.warning(
                f"Baja confianza en extraccion ({page.source_file} pag. {page.page_number}). Reanalizando."
            )
            try:
                if use_cover_prompt:
                    second_pass = gemini_client.extract_exam_questions(
                        image_path=page.image_path,
                        source_file=page.source_file,
                        page_number=page.page_number,
                    )
                else:
                    second_pass = gemini_client.extract_student_answers(
                        image_path=page.image_path,
                        source_file=page.source_file,
                        page_number=page.page_number,
                        questions_context=questions_context,
                    )
                if second_pass.extraction_confidence > extraction.extraction_confidence:
                    extraction = second_pass
                    extraction.incidents.append("Se uso reanalisis por baja confianza inicial.")
            except Exception as exc:
                extraction.incidents.append(f"Fallo en reanalisis de pagina {page.page_number}: {exc}")

        if extraction.page_role == "cover" and extraction.questions:
            questions_context = [
                {
                    "question_id": q.question_id,
                    "statement": q.statement or "",
                    "parts": [{"part_id": p.part_id, "statement": p.statement or ""} for p in q.parts],
                }
                for q in extraction.questions
            ]

        _sanitize_cover_page_answers(extraction)

        if extraction.extraction_confidence < low_confidence_threshold:
            extraction.incidents.append(
                (
                    f"Confianza global baja en pagina {page.page_number} "
                    f"({extraction.extraction_confidence:.2f})."
                )
            )
        if extraction.student_name and extraction.student_name_confidence < low_confidence_threshold:
            extraction.incidents.append(
                (
                    f"Nombre detectado con baja confianza en pagina {page.page_number} "
                    f"({extraction.student_name_confidence:.2f})."
                )
            )
        if extraction.start_header_detected and extraction.start_header_confidence < low_confidence_threshold:
            extraction.incidents.append(
                (
                    f"Header de inicio detectado con baja confianza en pagina {page.page_number} "
                    f"({extraction.start_header_confidence:.2f})."
                )
            )
        if extraction.page_role == "cover" and extraction.handwritten_content_detected:
            extraction.incidents.append(
                (
                    f"Caratula con contenido manuscrito detectado en pagina {page.page_number}; "
                    "se mantiene para correccion."
                )
            )
        if extraction.handwritten_content_detected and extraction.handwritten_content_confidence < low_confidence_threshold:
            extraction.incidents.append(
                (
                    f"Manuscrito detectado con baja confianza en pagina {page.page_number} "
                    f"({extraction.handwritten_content_confidence:.2f})."
                )
            )

        extracted_pages.append(extraction)
        if progress_callback:
            try:
                progress_callback(_page_idx, _total_pages)
            except Exception:
                pass
    return extracted_pages


def _merge_part(existing: ExtractedPart, incoming: ExtractedPart, incidents: list[str], question_id: str) -> None:
    if incoming.max_points is not None:
        if existing.max_points is None:
            existing.max_points = incoming.max_points
        elif abs(existing.max_points - incoming.max_points) > 0.01:
            incidents.append(
                f"Conflicto de puntuacion detectado en ejercicio {question_id}.{existing.part_id}; se conserva la mayor."
            )
            existing.max_points = max(existing.max_points, incoming.max_points)

    if incoming.student_answer_raw.strip():
        if existing.student_answer_raw.strip():
            if incoming.student_answer_raw.strip() not in existing.student_answer_raw:
                existing.student_answer_raw = f"{existing.student_answer_raw}\n{incoming.student_answer_raw}".strip()
        else:
            existing.student_answer_raw = incoming.student_answer_raw.strip()

    if not existing.student_answer_normalized and incoming.student_answer_normalized:
        existing.student_answer_normalized = incoming.student_answer_normalized

    for step in incoming.steps_detected:
        if step and step not in existing.steps_detected:
            existing.steps_detected.append(step)
    existing.confidence = max(existing.confidence, incoming.confidence)


def _apply_points_distribution(question: ExtractedQuestion, incidents: list[str]) -> None:
    if not question.parts:
        question.parts = [
            ExtractedPart(
                part_id="single",
                statement=question.statement,
                max_points=question.max_points,
                student_answer_raw="",
                confidence=0.0,
            )
        ]
        if question.max_points is None:
            incidents.append(f"No se pudo determinar la puntuacion maxima de ejercicio {question.question_id}.")
        return

    # Tratar 0.0 como "no conocido" cuando la pregunta tiene puntuación total positiva,
    # ya que 0.0 es el valor por defecto del schema JSON, no un valor real.
    q_total = question.max_points or 0.0
    known_points = [
        part.max_points for part in question.parts
        if part.max_points is not None and (part.max_points > 0 or q_total == 0)
    ]
    missing_parts = [
        part for part in question.parts
        if part.max_points is None or (part.max_points == 0.0 and q_total > 0)
    ]

    if question.max_points is None and known_points:
        question.max_points = round_points(sum(known_points))

    if question.max_points is not None:
        total_known = sum(known_points)
        if missing_parts:
            if not known_points:
                n = len(question.parts)
                share = round_points(question.max_points / n)
                running = 0.0
                for pi, part in enumerate(question.parts):
                    if pi == n - 1:
                        part.max_points = round_points(question.max_points - running)
                    else:
                        part.max_points = share
                        running += share
                incidents.append(
                    (
                        f"Reparto equitativo aplicado en ejercicio {question.question_id}: "
                        f"{question.max_points:.2f} / {len(question.parts)}."
                    )
                )
            else:
                remaining = question.max_points - total_known
                if remaining < 0:
                    incidents.append(
                        (
                            f"Puntuaciones de apartados superan el maximo en ejercicio {question.question_id}; "
                            "ajuste conservador aplicado."
                        )
                    )
                    remaining = 0.0
                share = round_points(remaining / len(missing_parts)) if missing_parts else 0.0
                for part in missing_parts:
                    part.max_points = share
                incidents.append(
                    (
                        f"Reparto del remanente aplicado en ejercicio {question.question_id} "
                        f"para {len(missing_parts)} apartados."
                    )
                )
        elif abs(total_known - question.max_points) > 0.05:
            incidents.append(
                (
                    f"Inconsistencia de puntuacion en ejercicio {question.question_id}: "
                    f"partes={total_known:.2f}, total={question.max_points:.2f}."
                )
            )
    else:
        incidents.append(
            f"No se pudo determinar puntuacion del ejercicio {question.question_id}; requiere revision manual."
        )

    known = [p.max_points for p in question.parts if p.max_points is not None]
    question.max_points = round_points(sum(known)) if known else None


def normalize_submission_structure(submission: ExamSubmission) -> ExamSubmission:
    question_map: OrderedDict[str, ExtractedQuestion] = OrderedDict()
    incidents = list(submission.incidents)

    for question in submission.questions:
        qid = question.question_id
        if qid not in question_map:
            question_map[qid] = deepcopy(
                ExtractedQuestion(
                    question_id=qid,
                    statement=question.statement,
                    max_points=question.max_points,
                    parts=[],
                )
            )
        merged_question = question_map[qid]
        if not merged_question.statement and question.statement:
            merged_question.statement = question.statement

        if question.max_points is not None:
            if merged_question.max_points is None:
                merged_question.max_points = question.max_points
            elif abs(merged_question.max_points - question.max_points) > 0.01:
                incidents.append(
                    f"Conflicto de puntuacion detectado en ejercicio {qid}; se conserva la mayor."
                )
                merged_question.max_points = max(merged_question.max_points, question.max_points)

        part_map = {part.part_id: part for part in merged_question.parts}
        if not question.parts:
            question.parts = [ExtractedPart(part_id="single", statement=question.statement, student_answer_raw="")]
        for incoming_part in question.parts:
            if incoming_part.part_id not in part_map:
                copied = deepcopy(incoming_part)
                part_map[incoming_part.part_id] = copied
                merged_question.parts.append(copied)
            else:
                _merge_part(part_map[incoming_part.part_id], incoming_part, incidents, qid)

    normalized_questions = list(question_map.values())
    for question in normalized_questions:
        _apply_points_distribution(question, incidents)

    normalized = ExamSubmission(
        exam_id=submission.exam_id,
        source_file=submission.source_file,
        pages=submission.pages,
        student_name=submission.student_name,
        student_name_confidence=submission.student_name_confidence,
        exam_model=submission.exam_model,
        course_level=submission.course_level,
        questions=normalized_questions,
        incidents=incidents,
    )
    if not normalized.student_name:
        normalized.incidents.append("No se pudo detectar el nombre del alumno con suficiente certeza.")
    return normalized


def build_submission_from_pdf(
    page_extractions: list[PageExtraction],
    source_file: str,
) -> ExamSubmission:
    pages = sorted(page_extractions, key=lambda p: p.page_number)
    student_name, student_conf = _choose_student_name(pages)
    exam_model = _choose_exam_model(pages)
    course_level = _choose_course_level(pages)
    all_questions = [q for p in pages for q in p.questions]
    all_incidents = [i for p in pages for i in p.incidents]
    return ExamSubmission(
        exam_id=f"{source_file}::exam_01",
        source_file=source_file,
        pages=[p.page_number for p in pages],
        student_name=student_name,
        student_name_confidence=student_conf,
        exam_model=exam_model,
        course_level=course_level,
        questions=all_questions,
        incidents=all_incidents,
    )
