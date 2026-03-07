from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils import normalize_identifier, parse_point_string, round_points


class IncidentSeverity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class Incident(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    severity: IncidentSeverity = IncidentSeverity.warning
    page: int | None = None
    question_id: str | None = None
    part_id: str | None = None


class PageImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_file: str
    page_number: int = Field(ge=1)
    image_path: Path


class ExtractedPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part_id: str = Field(default="single")
    statement: str | None = None
    max_points: float | None = None
    student_answer_raw: str = ""
    student_answer_normalized: str | None = None
    steps_detected: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("part_id", mode="before")
    @classmethod
    def normalize_part_id(cls, value: object) -> str:
        if value is None:
            return "single"
        normalized = normalize_identifier(str(value), default="single")
        return normalized

    @field_validator("max_points", mode="before")
    @classmethod
    def parse_points(cls, value: object) -> float | None:
        if value is None:
            return None
        parsed = parse_point_string(value) if not isinstance(value, (int, float)) else float(value)
        return round_points(parsed) if parsed is not None else None


class ExtractedQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    statement: str | None = None
    max_points: float | None = None
    parts: list[ExtractedPart] = Field(default_factory=list)

    @field_validator("question_id", mode="before")
    @classmethod
    def normalize_question_id(cls, value: object) -> str:
        if value is None:
            return "0"
        return normalize_identifier(str(value), default="0")

    @field_validator("max_points", mode="before")
    @classmethod
    def parse_points(cls, value: object) -> float | None:
        if value is None:
            return None
        parsed = parse_point_string(value) if not isinstance(value, (int, float)) else float(value)
        return round_points(parsed) if parsed is not None else None


class LLMPageExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_name: str | None = None
    student_name_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    exam_model: str | None = None
    course_level: str | None = None
    start_header_detected: bool = False
    start_header_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    page_role: Literal["cover", "answer", "mixed", "unknown"] = "unknown"
    handwritten_content_detected: bool = False
    handwritten_content_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    exam_boundary_hint: Literal["new_exam", "continuation", "unknown"] = "unknown"
    boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    questions: list[ExtractedQuestion] = Field(default_factory=list)
    incidents: list[str] = Field(default_factory=list)
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class PageExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_file: str
    page_number: int = Field(ge=1)
    student_name: str | None = None
    student_name_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    exam_model: str | None = None
    course_level: str | None = None
    start_header_detected: bool = False
    start_header_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    page_role: Literal["cover", "answer", "mixed", "unknown"] = "unknown"
    handwritten_content_detected: bool = False
    handwritten_content_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    exam_boundary_hint: Literal["new_exam", "continuation", "unknown"] = "unknown"
    boundary_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    questions: list[ExtractedQuestion] = Field(default_factory=list)
    incidents: list[str] = Field(default_factory=list)
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExamSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exam_id: str
    source_file: str
    pages: list[int] = Field(default_factory=list)
    student_name: str | None = None
    student_name_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    exam_model: str | None = None
    course_level: str | None = None
    questions: list[ExtractedQuestion] = Field(default_factory=list)
    incidents: list[str] = Field(default_factory=list)


class PartialCreditRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: str
    points: float = Field(ge=0.0)
    explanation: str


class SolutionTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exercise: str
    part: str | None = None
    exam_model: str | None = None
    topic: str | None = None
    expected_final_answer: str = ""
    accepted_equivalents: list[str] = Field(default_factory=list)
    expected_steps: list[str] = Field(default_factory=list)
    common_errors: list[str] = Field(default_factory=list)
    max_points: float | None = None
    partial_credit_rules: list[PartialCreditRule] = Field(default_factory=list)
    comments: str | None = None

    @field_validator("exercise", mode="before")
    @classmethod
    def normalize_exercise(cls, value: object) -> str:
        if value is None:
            return "0"
        return normalize_identifier(str(value), default="0")

    @field_validator("part", mode="before")
    @classmethod
    def normalize_part(cls, value: object) -> str | None:
        if value is None:
            return None
        normalized = normalize_identifier(str(value), default="")
        return normalized or None

    @field_validator("max_points", mode="before")
    @classmethod
    def parse_max_points(cls, value: object) -> float | None:
        if value is None:
            return None
        parsed = parse_point_string(value) if not isinstance(value, (int, float)) else float(value)
        return round_points(parsed) if parsed is not None else None


class GeminiAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["correcto", "parcial", "incorrecto", "revision_manual"] = "revision_manual"
    result_correct: bool | None = None
    procedure_quality: Literal["correct", "mostly_correct", "partial", "incorrect", "not_enough_info"] = (
        "not_enough_info"
    )
    detected_error_type: Literal[
        "none",
        "arithmetic_error",
        "sign_error",
        "conceptual_error",
        "illegible",
        "missing_response",
        "other",
    ] = "other"
    matched_expected_steps: list[str] = Field(default_factory=list)
    missing_steps: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning_summary: str = ""


class GeminiSolvedExercise(BaseModel):
    model_config = ConfigDict(extra="forbid")

    can_solve: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    solved_final_answer: str = ""
    accepted_equivalents: list[str] = Field(default_factory=list)
    expected_steps: list[str] = Field(default_factory=list)
    topic: str | None = None
    notes: str = ""
    incidents: list[str] = Field(default_factory=list)


class PartGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    part_id: str
    column_id: str
    max_points: float = Field(ge=0.0)
    awarded_points: float = Field(ge=0.0)
    status: Literal["correcto", "parcial", "incorrecto", "revision_manual"]
    detected_answer: str = ""
    normalized_answer: str | None = None
    steps_observed: list[str] = Field(default_factory=list)
    explanation: str
    incidents: list[str] = Field(default_factory=list)

    @field_validator("question_id", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> str:
        return normalize_identifier(str(value), default="0")

    @field_validator("part_id", mode="before")
    @classmethod
    def normalize_part(cls, value: object) -> str:
        return normalize_identifier(str(value), default="single")

    @field_validator("max_points", "awarded_points", mode="before")
    @classmethod
    def round_numeric(cls, value: object) -> float:
        return round_points(float(value))


class QuestionGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    max_points: float = Field(ge=0.0)
    awarded_points: float = Field(ge=0.0)
    parts: list[PartGrade] = Field(default_factory=list)

    @field_validator("question_id", mode="before")
    @classmethod
    def normalize_question(cls, value: object) -> str:
        return normalize_identifier(str(value), default="0")

    @field_validator("max_points", "awarded_points", mode="before")
    @classmethod
    def round_numeric(cls, value: object) -> float:
        return round_points(float(value))


class ExamGradeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exam_id: str
    student_name: str
    source_file: str
    exam_model: str | None = None
    course_level: str | None = None
    pages: list[int] = Field(default_factory=list)
    questions: list[QuestionGrade] = Field(default_factory=list)
    total_points: float = Field(ge=0.0)
    max_total_points: float = Field(ge=0.0)
    incidents: list[str] = Field(default_factory=list)
    report_path: str | None = None

    @field_validator("total_points", "max_total_points", mode="before")
    @classmethod
    def round_numeric(cls, value: object) -> float:
        return round_points(float(value))


class GradeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    awarded_points: float = Field(ge=0.0)
    status: Literal["correcto", "parcial", "incorrecto", "revision_manual"]
    rationale: str
