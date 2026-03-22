from __future__ import annotations

from grading import SolutionBank, are_answers_equivalent, decide_part_grade, grade_exam
from models import (
    ExamSubmission,
    ExtractedPart,
    ExtractedQuestion,
    GeminiAssessment,
    GeminiSolvedExercise,
    PartialCreditRule,
    SolutionTemplate,
)


class FakeGeminiClient:
    def __init__(self, assessment: GeminiAssessment, solved: GeminiSolvedExercise | None = None) -> None:
        self._assessment = assessment
        self._solved = solved or GeminiSolvedExercise(
            can_solve=False,
            confidence=0.0,
            solved_final_answer="",
            accepted_equivalents=[],
            expected_steps=[],
            topic=None,
            notes="",
            incidents=[],
        )

    def solve_math_question(
        self,
        *,
        question_statement: str,
        question_id: str,
        part_id: str,
        exam_model: str | None = None,
        course_level: str | None = None,
    ) -> GeminiSolvedExercise:  # noqa: ARG002
        return self._solved

    def assess_math_answer(
        self,
        solution: SolutionTemplate,
        extracted_part: ExtractedPart,
        question_statement: str | None = None,
        course_level: str | None = None,
        evaluation_criteria: str | None = None,
        scoring_instructions: str | None = None,
        correction_examples: list[dict] | None = None,
    ) -> GeminiAssessment:  # noqa: ARG002
        return self._assessment

    def generate_feedback_explanation(
        self,
        student_answer: str,
        expected_answer: str,
        reasoning_summary: str,
        status: str,
        awarded_points: float,
        max_points: float,
    ) -> str:  # noqa: ARG002
        return "Explicacion test"


class FailIfAssessCalledGemini(FakeGeminiClient):
    def assess_math_answer(
        self,
        solution: SolutionTemplate,
        extracted_part: ExtractedPart,
        question_statement: str | None = None,
        course_level: str | None = None,
        evaluation_criteria: str | None = None,
        scoring_instructions: str | None = None,
        correction_examples: list[dict] | None = None,
    ) -> GeminiAssessment:  # noqa: ARG002
        raise AssertionError("assess_math_answer should not be called for blank answers")


def test_partial_rule_applies_for_arithmetic_error() -> None:
    template = SolutionTemplate(
        exercise="1",
        part="a",
        expected_final_answer="x=2",
        partial_credit_rules=[
            PartialCreditRule(
                condition="procedimiento correcto pero error aritmetico final",
                points=0.25,
                explanation="Regla aplicada",
            )
        ],
    )
    assessment = GeminiAssessment(
        classification="parcial",
        result_correct=False,
        procedure_quality="mostly_correct",
        detected_error_type="arithmetic_error",
        confidence=0.92,
        reasoning_summary="",
    )
    decision = decide_part_grade(
        max_points=0.5,
        answer_raw="x=3",
        template=template,
        assessment=assessment,
        low_confidence_threshold=0.7,
        strict_mode=False,
    )
    assert decision.awarded_points == 0.25
    assert decision.status == "parcial"


def test_grade_exam_total_matches_sum_of_parts() -> None:
    submission = ExamSubmission(
        exam_id="doc::exam_01",
        source_file="doc.pdf",
        pages=[1],
        student_name="Juan Perez",
        student_name_confidence=0.9,
        exam_model="A",
        questions=[
            ExtractedQuestion(
                question_id="1",
                max_points=1.0,
                parts=[ExtractedPart(part_id="a", max_points=1.0, student_answer_raw="2x+3")],
            )
        ],
        incidents=[],
    )
    template = SolutionTemplate(
        exercise="1",
        part="a",
        exam_model="A",
        expected_final_answer="2x+3",
        max_points=1.0,
    )
    bank = SolutionBank([template])
    fake_gemini = FakeGeminiClient(
        GeminiAssessment(
            classification="correcto",
            result_correct=True,
            procedure_quality="correct",
            detected_error_type="none",
            confidence=0.99,
            reasoning_summary="",
        )
    )

    result = grade_exam(
        submission=submission,
        solution_bank=bank,
        gemini_client=fake_gemini,  # type: ignore[arg-type]
        low_confidence_threshold=0.7,
        strict_mode=False,
    )
    assert result.total_points == 1.0
    assert result.max_total_points == 1.0


def test_ai_solver_can_create_effective_solution_when_template_missing() -> None:
    submission = ExamSubmission(
        exam_id="doc::exam_01",
        source_file="doc.pdf",
        pages=[1],
        student_name="Ana Perez",
        student_name_confidence=0.9,
        exam_model="A",
        questions=[
            ExtractedQuestion(
                question_id="2",
                max_points=1.0,
                statement="Deriva f(x)=x^2",
                parts=[
                    ExtractedPart(
                        part_id="a",
                        max_points=1.0,
                        statement="Deriva f(x)=x^2",
                        student_answer_raw="2x",
                    )
                ],
            )
        ],
        incidents=[],
    )
    bank = SolutionBank([])
    fake_gemini = FakeGeminiClient(
        assessment=GeminiAssessment(
            classification="correcto",
            result_correct=True,
            procedure_quality="correct",
            detected_error_type="none",
            confidence=0.99,
            reasoning_summary="",
        ),
        solved=GeminiSolvedExercise(
            can_solve=True,
            confidence=0.95,
            solved_final_answer="2x",
            accepted_equivalents=["2*x"],
            expected_steps=["aplica regla de potencia"],
            topic="derivadas",
            notes="",
            incidents=[],
        ),
    )

    result = grade_exam(
        submission=submission,
        solution_bank=bank,
        gemini_client=fake_gemini,  # type: ignore[arg-type]
        low_confidence_threshold=0.7,
        strict_mode=False,
        allow_ai_solver=True,
        ai_solver_min_confidence=0.75,
    )
    assert result.total_points == 1.0
    assert any("IA" in inc for inc in result.incidents)


def test_no_template_with_known_max_points_uses_minimal_template() -> None:
    submission = ExamSubmission(
        exam_id="doc::exam_01",
        source_file="doc.pdf",
        pages=[1],
        student_name="Maria Lopez",
        student_name_confidence=0.9,
        exam_model=None,
        questions=[
            ExtractedQuestion(
                question_id="3",
                max_points=1.5,
                statement="Calcula el limite cuando x tiende a 0 de sin(x)/x",
                parts=[
                    ExtractedPart(
                        part_id="single",
                        max_points=1.5,
                        statement="Calcula el limite",
                        student_answer_raw="1",
                    )
                ],
            )
        ],
        incidents=[],
    )
    bank = SolutionBank([])
    fake_gemini = FakeGeminiClient(
        assessment=GeminiAssessment(
            classification="correcto",
            result_correct=True,
            procedure_quality="correct",
            detected_error_type="none",
            confidence=0.99,
            reasoning_summary="",
        ),
        solved=GeminiSolvedExercise(
            can_solve=False,
            confidence=0.0,
            solved_final_answer="",
            accepted_equivalents=[],
            expected_steps=[],
            topic=None,
            notes="",
            incidents=[],
        ),
    )

    result = grade_exam(
        submission=submission,
        solution_bank=bank,
        gemini_client=fake_gemini,  # type: ignore[arg-type]
        low_confidence_threshold=0.7,
        strict_mode=False,
        allow_ai_solver=True,
        ai_solver_min_confidence=0.30,
    )
    assert result.questions[0].parts[0].max_points == 1.5
    assert any("IA" in inc for inc in result.incidents)
    assert result.total_points > 0.0


def test_symbolic_equivalence_does_not_crash_on_invalid_tokens() -> None:
    # Regression: parse_expr can raise tokenize.TokenError on malformed OCR strings.
    assert are_answers_equivalent("x=2\\", "x=2", []) is False


def test_blank_answer_skips_assessment_call() -> None:
    submission = ExamSubmission(
        exam_id="doc::exam_01",
        source_file="doc.pdf",
        pages=[1],
        student_name="Ana Perez",
        student_name_confidence=0.9,
        exam_model="A",
        questions=[
            ExtractedQuestion(
                question_id="1",
                max_points=1.0,
                parts=[ExtractedPart(part_id="a", max_points=1.0, student_answer_raw="")],
            )
        ],
        incidents=[],
    )
    template = SolutionTemplate(
        exercise="1",
        part="a",
        exam_model="A",
        expected_final_answer="2",
        max_points=1.0,
    )
    bank = SolutionBank([template])
    fake_gemini = FailIfAssessCalledGemini(
        assessment=GeminiAssessment(
            classification="correcto",
            result_correct=True,
            procedure_quality="correct",
            detected_error_type="none",
            confidence=0.99,
            reasoning_summary="",
        )
    )

    result = grade_exam(
        submission=submission,
        solution_bank=bank,
        gemini_client=fake_gemini,  # type: ignore[arg-type]
        low_confidence_threshold=0.7,
        strict_mode=False,
    )
    assert result.total_points == 0.0
    assert result.questions[0].parts[0].status == "incorrecto"
