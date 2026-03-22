from __future__ import annotations

import json
from tokenize import TokenError
from pathlib import Path

from loguru import logger
from rapidfuzz import fuzz
from sympy import SympifyError, simplify
from sympy.parsing.sympy_parser import parse_expr

from gemini_client import GeminiClient
from models import (
    ExamGradeResult,
    ExamSubmission,
    GeminiAssessment,
    GeminiSolvedExercise,
    GradeDecision,
    PartGrade,
    QuestionGrade,
    SolutionTemplate,
)
from utils import normalize_identifier, part_column_id, round_points


class SolutionBank:
    def __init__(self, templates: list[SolutionTemplate]) -> None:
        self.templates = templates

    def find(self, exercise: str, part: str | None, exam_model: str | None = None) -> SolutionTemplate | None:
        exercise_n = normalize_identifier(exercise, "0")
        part_n = normalize_identifier(part, "") or None
        model_n = normalize_identifier(exam_model, "") or None

        candidates = [t for t in self.templates if t.exercise == exercise_n]
        if model_n:
            model_candidates = [t for t in candidates if normalize_identifier(t.exam_model, "") == model_n]
            if model_candidates:
                candidates = model_candidates
        exact = [t for t in candidates if t.part == part_n]
        if exact:
            return exact[0]
        generic = [t for t in candidates if t.part is None]
        if generic:
            return generic[0]
        return None


def _load_templates_from_file(path: Path) -> list[SolutionTemplate]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict) and "solutions" in data and isinstance(data["solutions"], list):
        raw_items = data["solutions"]
    elif isinstance(data, dict):
        raw_items = [data]
    else:
        raise ValueError(f"Formato JSON no soportado en {path}")
    return [SolutionTemplate.model_validate(item) for item in raw_items]


def load_solution_bank(solutions_dir: Path) -> SolutionBank:
    if not solutions_dir.exists():
        logger.warning(f"Carpeta de soluciones no encontrada: {solutions_dir}")
        return SolutionBank(templates=[])

    templates: list[SolutionTemplate] = []
    for json_file in sorted(solutions_dir.glob("*.json")):
        try:
            templates.extend(_load_templates_from_file(json_file))
            logger.info(f"Archivo de soluciones cargado: {json_file.name}")
        except Exception as exc:
            logger.error(f"No se pudo cargar {json_file.name}: {exc}")
    logger.info(f"Total de plantillas de solucion cargadas: {len(templates)}")
    return SolutionBank(templates=templates)


def _normalized_text(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = text.replace(" ", "")
    text = text.replace(",", ".")
    return text


def _try_symbolic_equivalence(expected: str, observed: str) -> bool:
    try:
        expected_expr = parse_expr(expected)
        observed_expr = parse_expr(observed)
    except (SympifyError, SyntaxError, TypeError, ValueError, TokenError):
        return False
    try:
        return bool(simplify(expected_expr - observed_expr) == 0)
    except Exception:
        return False


def are_answers_equivalent(expected: str, observed: str, equivalents: list[str]) -> bool:
    expected_candidates = [_normalized_text(expected), *[_normalized_text(eq) for eq in equivalents]]
    observed_n = _normalized_text(observed)
    if not observed_n:
        return False
    if observed_n in expected_candidates:
        return True
    if any(fuzz.ratio(candidate, observed_n) >= 96 for candidate in expected_candidates if candidate):
        return True
    for candidate in expected_candidates:
        if candidate and _try_symbolic_equivalence(candidate, observed_n):
            return True
    return False


def _is_rule_match(condition: str, assessment: GeminiAssessment) -> bool:
    text = condition.strip().lower()
    if "aritm" in text and assessment.detected_error_type == "arithmetic_error":
        return True
    if "signo" in text and assessment.detected_error_type == "sign_error":
        return True
    if "procedimiento correcto" in text and assessment.procedure_quality in {"correct", "mostly_correct"}:
        return True
    if "planteamiento correcto" in text and assessment.procedure_quality in {"correct", "mostly_correct", "partial"}:
        return True
    if "revision" in text and assessment.classification == "revision_manual":
        return True
    return False


def _decision_from_rules(
    template: SolutionTemplate,
    assessment: GeminiAssessment,
    max_points: float,
) -> GradeDecision | None:
    for rule in template.partial_credit_rules:
        if _is_rule_match(rule.condition, assessment):
            points = min(max_points, round_points(rule.points))
            status = "parcial" if 0 < points < max_points else ("correcto" if points == max_points else "incorrecto")
            return GradeDecision(awarded_points=points, status=status, rationale=rule.explanation)
    return None


def _merge_unique_strings(base: list[str], extra: list[str]) -> list[str]:
    merged = list(base)
    seen = {item.strip().lower() for item in merged}
    for value in extra:
        clean = value.strip()
        key = clean.lower()
        if clean and key not in seen:
            merged.append(clean)
            seen.add(key)
    return merged


def _build_effective_template_from_ai_solution(
    *,
    question_id: str,
    part_id: str,
    part_max: float,
    ai_solution: GeminiSolvedExercise,
) -> SolutionTemplate:
    return SolutionTemplate(
        exercise=question_id,
        part=part_id,
        topic=ai_solution.topic,
        expected_final_answer=ai_solution.solved_final_answer,
        accepted_equivalents=ai_solution.accepted_equivalents,
        expected_steps=ai_solution.expected_steps,
        common_errors=[],
        max_points=part_max,
        partial_credit_rules=[],
        comments=f"Plantilla generada por solver IA. {ai_solution.notes}".strip(),
    )


def _enrich_template_with_ai_solution(template: SolutionTemplate, ai_solution: GeminiSolvedExercise) -> SolutionTemplate:
    enriched = template.model_copy(deep=True)
    if not enriched.expected_final_answer and ai_solution.solved_final_answer:
        enriched.expected_final_answer = ai_solution.solved_final_answer
    enriched.accepted_equivalents = _merge_unique_strings(
        enriched.accepted_equivalents,
        ai_solution.accepted_equivalents,
    )
    enriched.expected_steps = _merge_unique_strings(
        enriched.expected_steps,
        ai_solution.expected_steps,
    )
    if not enriched.topic and ai_solution.topic:
        enriched.topic = ai_solution.topic
    return enriched


def decide_part_grade(
    *,
    max_points: float,
    answer_raw: str,
    template: SolutionTemplate,
    assessment: GeminiAssessment,
    low_confidence_threshold: float,
    strict_mode: bool,
) -> GradeDecision:
    normalized_answer = _normalized_text(answer_raw)
    if not normalized_answer:
        return GradeDecision(
            awarded_points=0.0,
            status="incorrecto",
            rationale="No aparece respuesta válida para este apartado.",
        )

    if assessment.classification == "revision_manual" or assessment.detected_error_type == "illegible":
        return GradeDecision(
            awarded_points=0.0,
            status="revision_manual",
            rationale="No se ha podido interpretar con suficiente fiabilidad la respuesta manuscrita.",
        )

    if assessment.confidence < low_confidence_threshold:
        if strict_mode:
            return GradeDecision(
                awarded_points=0.0,
                status="revision_manual",
                rationale="La evidencia extraida tiene baja confianza; se recomienda revision manual.",
            )
        conservative = round_points(max_points * 0.25)
        return GradeDecision(
            awarded_points=conservative,
            status="revision_manual",
            rationale="Confianza baja en la extraccion; se aplica criterio conservador y se recomienda revision manual.",
        )

    final_correct_local = are_answers_equivalent(
        template.expected_final_answer,
        answer_raw,
        template.accepted_equivalents,
    )
    final_correct = final_correct_local or (assessment.result_correct is True)
    good_procedure = assessment.procedure_quality in {"correct", "mostly_correct"}

    if final_correct and good_procedure:
        return GradeDecision(
            awarded_points=max_points,
            status="correcto",
            rationale="Resultado final correcto y procedimiento coherente.",
        )

    if final_correct and assessment.procedure_quality == "partial":
        points = round_points(max_points * 0.75)
        return GradeDecision(
            awarded_points=points,
            status="parcial",
            rationale="El resultado final es correcto, pero el procedimiento observado es incompleto.",
        )

    ruled = _decision_from_rules(template=template, assessment=assessment, max_points=max_points)
    if ruled is not None:
        return ruled

    if assessment.detected_error_type in {"arithmetic_error", "sign_error"} and good_procedure:
        points = round_points(max_points * 0.50)
        return GradeDecision(
            awarded_points=points,
            status="parcial",
            rationale="El planteamiento es correcto, pero hay un error de calculo/signo en pasos finales.",
        )

    if assessment.procedure_quality in {"mostly_correct", "partial"}:
        points = round_points(max_points * 0.35)
        return GradeDecision(
            awarded_points=points,
            status="parcial",
            rationale="Se aprecia parte del procedimiento correcto, pero no alcanza la solucion valida.",
        )

    return GradeDecision(
        awarded_points=0.0,
        status="incorrecto",
        rationale="No se observa un procedimiento matematico valido ni resultado correcto.",
    )


def grade_exam(
    submission: ExamSubmission,
    solution_bank: SolutionBank,
    gemini_client: GeminiClient,
    low_confidence_threshold: float,
    strict_mode: bool = False,
    allow_ai_solver: bool = True,
    ai_solver_min_confidence: float = 0.75,
    evaluation_criteria: str | None = None,
    correction_examples: dict[tuple[str, str], list[dict]] | None = None,
) -> ExamGradeResult:
    question_grades: list[QuestionGrade] = []
    incidents = list(submission.incidents)
    student_name = submission.student_name or f"alumno_provisional_{submission.exam_id.split('::')[-1]}"

    total_questions = len(submission.questions)
    for q_idx, question in enumerate(submission.questions, start=1):
        part_grades: list[PartGrade] = []
        total_parts = len(question.parts)
        logger.info(f"    Ejercicio {question.question_id} ({q_idx}/{total_questions}) — {total_parts} apartado(s)")
        for p_idx, part in enumerate(question.parts, start=1):
            label = f"{question.question_id}.{part.part_id}"
            base_template = solution_bank.find(question.question_id, part.part_id, exam_model=submission.exam_model)
            template = base_template.model_copy(deep=True) if base_template is not None else None
            part_max = round_points(
                part.max_points if part.max_points is not None else ((template.max_points if template else 0.0))
            )

            statement_text = (part.statement or "").strip() or (question.statement or "").strip()
            ai_solution: GeminiSolvedExercise | None = None
            if allow_ai_solver and statement_text:
                logger.debug(f"      [{label}] Resolviendo enunciado con IA ({p_idx}/{total_parts})...")
                try:
                    ai_solution = gemini_client.solve_math_question(
                        question_statement=statement_text,
                        question_id=question.question_id,
                        part_id=part.part_id,
                        exam_model=submission.exam_model,
                        course_level=submission.course_level,
                    )
                    if ai_solution.can_solve and ai_solution.confidence > 0.0:
                        if template is None:
                            template = _build_effective_template_from_ai_solution(
                                question_id=question.question_id,
                                part_id=part.part_id,
                                part_max=part_max,
                                ai_solution=ai_solution,
                            )
                            # Distinguir si vino de caché del profesor o fue generada por IA
                            if "validada por el profesor" not in (ai_solution.notes or ""):
                                incidents.append(
                                    f"Solución al ejercicio {question.question_id}.{part.part_id} generada automáticamente por IA (modo IA activo)."
                                )
                        else:
                            template = _enrich_template_with_ai_solution(template, ai_solution)
                        if ai_solution.confidence < ai_solver_min_confidence:
                            incidents.append(
                                (
                                    f"La IA resolvió el ejercicio {question.question_id}.{part.part_id} con baja confianza "
                                    f"({ai_solution.confidence:.2f}); resultado usado igualmente."
                                )
                            )
                    else:
                        incidents.append(
                            (
                                f"La IA no pudo resolver el ejercicio {question.question_id}.{part.part_id}; "
                                "la corrección se realizó sin solución de referencia."
                            )
                        )
                except Exception as exc:
                    incidents.append(
                        f"Fallo resolviendo enunciado con IA en {question.question_id}.{part.part_id}: {exc}"
                    )

            has_student_work = bool((part.student_answer_raw or "").strip()) or bool(part.steps_detected)

            if not has_student_work:
                part_grades.append(
                    PartGrade(
                        question_id=question.question_id,
                        part_id=part.part_id,
                        column_id=part_column_id(question.question_id, part.part_id),
                        max_points=part_max,
                        awarded_points=0.0,
                        status="incorrecto",
                        detected_answer=part.student_answer_raw,
                        normalized_answer=part.student_answer_normalized,
                        steps_observed=part.steps_detected,
                        explanation="No aparece respuesta válida para este apartado. No se concede puntuación.",
                        incidents=[],
                    )
                )
                continue

            if template is None:
                if part_max == 0.0:
                    incidents.append(
                        f"No hay plantilla de solucion para ejercicio {question.question_id}.{part.part_id}; revision manual."
                    )
                    part_grades.append(
                        PartGrade(
                            question_id=question.question_id,
                            part_id=part.part_id,
                            column_id=part_column_id(question.question_id, part.part_id),
                            max_points=part_max,
                            awarded_points=0.0,
                            status="revision_manual",
                            detected_answer=part.student_answer_raw,
                            normalized_answer=part.student_answer_normalized,
                            steps_observed=part.steps_detected,
                            explanation="Sin plantilla de solucion ni puntuacion conocida.",
                            incidents=["Sin plantilla de solucion"],
                        )
                    )
                    continue
                template = SolutionTemplate(
                    exercise=question.question_id,
                    part=part.part_id,
                    expected_final_answer="",
                    max_points=part_max,
                )

            part_max = round_points(part.max_points if part.max_points is not None else (template.max_points or 0.0))
            answer_for_assessment = (part.student_answer_raw or "").strip() or "; ".join(part.steps_detected)

            logger.debug(f"      [{label}] Evaluando respuesta con Gemini ({p_idx}/{total_parts})...")
            try:
                # Indicaciones de puntuación específicas del apartado (si existen)
                _part_scoring = template.scoring_instructions if template else None
                _part_corrections = (correction_examples or {}).get(
                    (question.question_id, part.part_id), []
                )
                assessment = gemini_client.assess_math_answer(
                    solution=template,
                    extracted_part=part,
                    question_statement=question.statement,
                    course_level=submission.course_level,
                    evaluation_criteria=evaluation_criteria,
                    scoring_instructions=_part_scoring,
                    correction_examples=_part_corrections[:5] if _part_corrections else None,
                )
            except Exception as exc:
                incidents.append(
                    f"Fallo de evaluacion Gemini en {question.question_id}.{part.part_id}: {exc}. Revision manual."
                )
                assessment = GeminiAssessment(
                    classification="revision_manual",
                    result_correct=None,
                    procedure_quality="not_enough_info",
                    detected_error_type="other",
                    confidence=0.0,
                    reasoning_summary="Error de API en evaluacion automatica.",
                )

            decision = decide_part_grade(
                max_points=part_max,
                answer_raw=answer_for_assessment,
                template=template,
                assessment=assessment,
                low_confidence_threshold=low_confidence_threshold,
                strict_mode=strict_mode,
            )

            logger.info(
                f"      [{label}] {decision.status.upper()} — {decision.awarded_points:.2f}/{part_max:.2f} pts"
            )
            logger.debug(f"      [{label}] Generando feedback para el alumno...")
            try:
                feedback = gemini_client.generate_feedback_explanation(
                    student_answer=answer_for_assessment,
                    expected_answer=template.expected_final_answer,
                    reasoning_summary=f"{assessment.reasoning_summary} | {decision.rationale}",
                    status=decision.status,
                    awarded_points=decision.awarded_points,
                    max_points=part_max,
                )
            except Exception:
                feedback = decision.rationale

            part_incidents: list[str] = []
            if assessment.confidence < low_confidence_threshold:
                part_incidents.append(
                    f"Confianza de evaluacion baja ({assessment.confidence:.2f}) en {question.question_id}.{part.part_id}."
                )
            if ai_solution is not None and ai_solution.can_solve and ai_solution.confidence < ai_solver_min_confidence:
                part_incidents.append(
                    (
                        "Solucion IA desde enunciado con baja confianza "
                        f"({ai_solution.confidence:.2f}); usada igualmente."
                    )
                )
            if decision.status == "revision_manual":
                part_incidents.append("Apartado recomendado para revision manual.")

            part_grades.append(
                PartGrade(
                    question_id=question.question_id,
                    part_id=part.part_id,
                    column_id=part_column_id(question.question_id, part.part_id),
                    max_points=part_max,
                    awarded_points=decision.awarded_points,
                    status=decision.status,
                    detected_answer=part.student_answer_raw,
                    normalized_answer=part.student_answer_normalized,
                    steps_observed=part.steps_detected,
                    explanation=feedback,
                    incidents=part_incidents,
                )
            )

        q_max = round_points(sum(part.max_points for part in part_grades))
        q_awarded = round_points(sum(part.awarded_points for part in part_grades))
        question_grades.append(
            QuestionGrade(
                question_id=question.question_id,
                max_points=q_max,
                awarded_points=q_awarded,
                parts=part_grades,
            )
        )

    total_points = round_points(sum(q.awarded_points for q in question_grades))
    max_total = round_points(sum(q.max_points for q in question_grades))

    return ExamGradeResult(
        exam_id=submission.exam_id,
        student_name=student_name,
        source_file=submission.source_file,
        exam_model=submission.exam_model,
        course_level=submission.course_level,
        pages=submission.pages,
        questions=question_grades,
        total_points=total_points,
        max_total_points=max_total,
        incidents=incidents,
        report_path=None,
    )
