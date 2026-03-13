from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    hashed_pw: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ExamSession(Base):
    __tablename__ = "exam_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    course_level: Mapped[str | None] = mapped_column(Text)
    subject: Mapped[str | None] = mapped_column(Text)
    date: Mapped[str | None] = mapped_column(Text)  # ISO date string YYYY-MM-DD
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    status: Mapped[str] = mapped_column(Text, default="open")         # open | archived
    solution_mode: Mapped[str] = mapped_column(Text, default="ai")    # ai | teacher
    solver_provider: Mapped[str] = mapped_column(Text, default="gemini-pro")  # gemini-flash | gemini-pro | openai-gpt4o | openai-o4mini
    max_total_points: Mapped[float | None] = mapped_column(Float)     # puntuación máxima forzada (ej: 10)
    current_step: Mapped[str | None] = mapped_column(Text)            # último paso (resumen)
    session_log: Mapped[str | None] = mapped_column(Text)             # log acumulativo JSON [{t, msg}]

    submissions: Mapped[list[Submission]] = relationship("Submission", back_populates="session", cascade="all, delete-orphan")


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("exam_sessions.id"), nullable=False)
    source_filename: Mapped[str] = mapped_column(Text, nullable=False)
    pdf_path: Mapped[str] = mapped_column(Text, nullable=False)
    student_name: Mapped[str | None] = mapped_column(Text)
    exam_model: Mapped[str | None] = mapped_column(Text)
    course_level: Mapped[str | None] = mapped_column(Text)
    total_points: Mapped[float | None] = mapped_column(Float)
    max_total_points: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(Text, default="pending")  # pending|processing|done|error
    error_message: Mapped[str | None] = mapped_column(Text)
    incidents: Mapped[str | None] = mapped_column(Text)  # JSON list
    report_path: Mapped[str | None] = mapped_column(Text)
    processing_log: Mapped[str | None] = mapped_column(Text)  # JSON [{t, msg}]
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    processed_at: Mapped[datetime | None] = mapped_column(DateTime)

    student_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("students.id"), nullable=True)

    session: Mapped[ExamSession] = relationship("ExamSession", back_populates="submissions")
    student: Mapped["Student | None"] = relationship("Student", back_populates="submissions")
    question_results: Mapped[list[QuestionResult]] = relationship("QuestionResult", back_populates="submission", cascade="all, delete-orphan")


class QuestionResult(Base):
    __tablename__ = "question_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    submission_id: Mapped[int] = mapped_column(Integer, ForeignKey("submissions.id"), nullable=False)
    question_id: Mapped[str] = mapped_column(Text, nullable=False)
    max_points: Mapped[float | None] = mapped_column(Float)

    submission: Mapped[Submission] = relationship("Submission", back_populates="question_results")
    part_results: Mapped[list[PartResult]] = relationship("PartResult", back_populates="question", cascade="all, delete-orphan")


class PartResult(Base):
    __tablename__ = "part_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id_fk: Mapped[int] = mapped_column(Integer, ForeignKey("question_results.id"), nullable=False)
    part_id: Mapped[str] = mapped_column(Text, nullable=False)
    column_id: Mapped[str] = mapped_column(Text, nullable=False)
    awarded_points: Mapped[float | None] = mapped_column(Float)
    max_points: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str | None] = mapped_column(Text)
    explanation: Mapped[str | None] = mapped_column(Text)
    detected_answer: Mapped[str | None] = mapped_column(Text)
    incidents: Mapped[str | None] = mapped_column(Text)  # JSON list

    question: Mapped[QuestionResult] = relationship("QuestionResult", back_populates="part_results")


class TokenUsage(Base):
    """Registro de tokens consumidos por operación de IA."""
    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("exam_sessions.id"))
    submission_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("submissions.id"))
    # extract_solutions | extract_teacher_pdf | grade_submission
    operation: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    thinking_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    api_calls: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SessionHistory(Base):
    """Historial permanente de convocatorias — persiste aunque la convocatoria se borre."""
    __tablename__ = "session_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int | None] = mapped_column(Integer)          # referencia sin FK (puede estar borrada)
    session_name: Mapped[str] = mapped_column(Text, nullable=False)
    session_date: Mapped[str | None] = mapped_column(Text)
    subject: Mapped[str | None] = mapped_column(Text)
    course_level: Mapped[str | None] = mapped_column(Text)
    max_total_points: Mapped[float | None] = mapped_column(Float)
    total_submissions: Mapped[int] = mapped_column(Integer, default=0)
    graded_submissions: Mapped[int] = mapped_column(Integer, default=0)
    avg_score: Mapped[float | None] = mapped_column(Float)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost_eur: Mapped[float] = mapped_column(Float, default=0.0)
    snapshot_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime)    # null = convocatoria sigue activa


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(Text, nullable=False)
    course_level: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    submissions: Mapped[list["Submission"]] = relationship("Submission", back_populates="student")


class SessionSolution(Base):
    """Soluciones por convocatoria: se calculan una vez, el profesor las valida, y se reutilizan para corregir."""
    __tablename__ = "session_solutions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(Integer, ForeignKey("exam_sessions.id"), nullable=False)
    question_id: Mapped[str] = mapped_column(Text, nullable=False)
    part_id: Mapped[str] = mapped_column(Text, nullable=False)
    question_statement: Mapped[str | None] = mapped_column(Text)   # enunciado extraído
    part_statement: Mapped[str | None] = mapped_column(Text)       # enunciado del apartado
    solved_json: Mapped[str | None] = mapped_column(Text)          # GeminiSolvedExercise JSON (si la IA pudo)
    final_answer: Mapped[str | None] = mapped_column(Text)         # respuesta validada/corregida por el profesor
    max_points: Mapped[float | None] = mapped_column(Float)        # puntos de este apartado (prorrateados de la pregunta)
    evaluation_criteria: Mapped[str | None] = mapped_column(Text)  # criterios de evaluación del examen
    solution_image_path: Mapped[str | None] = mapped_column(Text)  # imagen subida por el profesor
    teacher_notes: Mapped[str | None] = mapped_column(Text)
    # ai_pending | ai_solved | ai_failed | validated | manual
    status: Mapped[str] = mapped_column(Text, default="ai_pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    validated_at: Mapped[datetime | None] = mapped_column(DateTime)
