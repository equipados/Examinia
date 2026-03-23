"""Solver de preguntas matemáticas usando la API de OpenAI."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from models import GeminiAssessment, GeminiSolvedExercise

_COURSE_LABELS = {
    "1o_bachillerato": "1o Bachillerato",
    "2o_bachillerato": "2o Bachillerato",
}
_CURRICULUM_1 = (
    "Curriculum 1o Bachillerato: funciones elementales y sus graficas, "
    "limites y continuidad, derivadas (definicion, reglas basicas, aplicaciones: tangente, monotonia, extremos), "
    "trigonometria (identidades, ecuaciones), numeros complejos, estadistica descriptiva, "
    "probabilidad (Laplace, Bayes, binomial, normal)."
)
_CURRICULUM_2 = (
    "Curriculum 2o Bachillerato: derivadas avanzadas (regla cadena/producto/cociente), "
    "integrales (por partes, sustitucion), limites (L'Hopital, infinitesimos), "
    "matrices y determinantes, sistemas lineales (Gauss, Cramer, Rouche-Frobenius), "
    "estadistica inferencial (normal, intervalos confianza, contraste hipotesis), "
    "geometria 3D (rectas, planos, distancias), conicas."
)

_SOLVE_SYSTEM = (
    "Eres un experto en Matematicas de Bachillerato (Espana). "
    "Resuelves ejercicios paso a paso y devuelves EXCLUSIVAMENTE un objeto JSON valido, sin texto adicional. "
    "IMPORTANTE: Todos los campos de texto (expected_steps, notes, topic, incidents) deben estar en español."
)

_JSON_SCHEMA = """{
  "can_solve": true,
  "confidence": 0.0,
  "solved_final_answer": "str",
  "accepted_equivalents": ["str"],
  "expected_steps": ["str"],
  "topic": "str o null",
  "notes": "str",
  "incidents": ["str"]
}"""


class OpenAISolver:
    """Solver de preguntas matemáticas vía OpenAI o cualquier API compatible (DeepSeek, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_retries: int = 3,
        rate_limit_seconds: float = 1.0,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
    ) -> None:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Falta {api_key_env} en variables de entorno o en archivo .env")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "No se pudo importar openai. Instala dependencias con: pip install openai"
            ) from exc
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model
        self.max_retries = max(1, int(max_retries))
        self.rate_limit_seconds = rate_limit_seconds
        self._last_request_ts = 0.0
        self._usage: dict[str, dict] = {}

    # ── Usage tracking (misma interfaz que GeminiClient) ──────────────

    def get_usage(self) -> dict[str, dict]:
        return dict(self._usage)

    def reset_usage(self) -> dict[str, dict]:
        snapshot = dict(self._usage)
        self._usage = {}
        return snapshot

    def _record_usage(self, response_usage) -> None:
        model = self.model
        if model not in self._usage:
            self._usage[model] = {"input": 0, "output": 0, "total": 0, "calls": 0}
        u = self._usage[model]
        u["input"] += getattr(response_usage, "prompt_tokens", 0)
        u["output"] += getattr(response_usage, "completion_tokens", 0)
        u["total"] += getattr(response_usage, "total_tokens", 0)
        u["calls"] += 1

    # ── Rate limiting ──────────────────────────────────────────────────

    def _wait_rate_limit(self) -> None:
        if self.rate_limit_seconds > 0:
            elapsed = time.monotonic() - self._last_request_ts
            if elapsed < self.rate_limit_seconds:
                time.sleep(self.rate_limit_seconds - elapsed)
        self._last_request_ts = time.monotonic()

    # ── Core call ─────────────────────────────────────────────────────

    def _call(self, system: str, user: str) -> tuple[str, object]:
        """Llama a la API de OpenAI y devuelve (text, usage)."""
        self._wait_rate_limit()
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    timeout=120,
                )
                return resp.choices[0].message.content or "", resp.usage
            except Exception as exc:
                err = str(exc)
                # Rate limit: esperar y reintentar
                if "rate_limit" in err.lower() or "429" in err:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"OpenAI rate limit. Esperando {wait}s...")
                    time.sleep(wait)
                    continue
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f"OpenAI falló tras {self.max_retries} intentos")

    def _call_text(self, system: str, user: str) -> tuple[str, object]:
        """Llama a la API de OpenAI sin forzar JSON y devuelve (text, usage)."""
        self._wait_rate_limit()
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.1,
                    timeout=120,
                )
                return resp.choices[0].message.content or "", resp.usage
            except Exception as exc:
                err = str(exc)
                if "rate_limit" in err.lower() or "429" in err:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"OpenAI rate limit. Esperando {wait}s...")
                    time.sleep(wait)
                    continue
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f"OpenAI falló tras {self.max_retries} intentos")

    # ── solve_math_question (misma firma que GeminiClient) ────────────

    def solve_math_question(
        self,
        *,
        question_statement: str,
        question_id: str,
        part_id: str,
        exam_model: str | None = None,
        course_level: str | None = None,
        image_paths: list[Path] | None = None,
        read_from_image: bool = False,
    ) -> GeminiSolvedExercise:
        course_label = _COURSE_LABELS.get(course_level or "", course_level or "Bachillerato")
        curriculum = _CURRICULUM_1 if course_level == "1o_bachillerato" else _CURRICULUM_2

        user_prompt = f"""Nivel: {course_label}
{curriculum}

Ejercicio {question_id}, apartado {part_id}.
Modelo de examen: {exam_model or 'no indicado'}.

Enunciado:
{question_statement}

Devuelve EXCLUSIVAMENTE este JSON (sin markdown, sin texto extra):
{_JSON_SCHEMA}

Instrucciones:
- solved_final_answer: respuesta exacta en texto ("x=2", "lim=1", "integral=x^3/3+C").
- accepted_equivalents: formas equivalentes de la misma respuesta.
- expected_steps: pasos clave que un alumno correcto mostraría.
- confidence: 0.0-1.0 (certeza de que la solución es correcta).
- Si no puedes resolver con fiabilidad, indica can_solve=false."""

        text, usage = self._call(_SOLVE_SYSTEM, user_prompt)
        self._record_usage(usage)

        # Parsear JSON
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            # Intentar extraer JSON de la respuesta
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                raw = json.loads(match.group())
            else:
                logger.warning(f"OpenAI devolvió respuesta no-JSON para {question_id}.{part_id}")
                return GeminiSolvedExercise(can_solve=False, notes="OpenAI no devolvió JSON válido")

        # Normalizar a GeminiSolvedExercise
        try:
            return GeminiSolvedExercise(
                can_solve=bool(raw.get("can_solve", True)),
                confidence=float(raw.get("confidence", 0.8)),
                solved_final_answer=str(raw.get("solved_final_answer", "")),
                accepted_equivalents=raw.get("accepted_equivalents") or [],
                expected_steps=raw.get("expected_steps") or [],
                topic=raw.get("topic"),
                notes=str(raw.get("notes", "")),
                incidents=raw.get("incidents") or [],
            )
        except (ValidationError, Exception) as exc:
            logger.warning(f"Error parseando respuesta OpenAI para {question_id}.{part_id}: {exc}")
            return GeminiSolvedExercise(can_solve=False, notes=f"Error de formato: {exc}")

    # ── assess_math_answer (misma firma que GeminiClient) ─────────────

    def assess_math_answer(
        self,
        solution,
        extracted_part,
        question_statement: str | None = None,
        course_level: str | None = None,
        evaluation_criteria: str | None = None,
        scoring_instructions: str | None = None,
        correction_examples: list[dict] | None = None,
    ) -> GeminiAssessment:
        _course_label = {
            "1o_bachillerato": "1o Bachillerato",
            "2o_bachillerato": "2o Bachillerato",
        }.get(course_level or "", "Bachillerato")

        solution_json = solution.model_dump_json(indent=2, exclude_none=True, exclude={"scoring_instructions"})
        student_json = json.dumps({
            "student_answer_raw": extracted_part.student_answer_raw,
            "student_answer_normalized": extracted_part.student_answer_normalized or "",
            "steps_detected": extracted_part.steps_detected,
            "ocr_confidence": extracted_part.confidence,
        }, ensure_ascii=False)

        scoring_block = (
            f"INDICACIONES DE PUNTUACION ESPECIFICAS PARA ESTE APARTADO (PRIORIDAD MAXIMA):\n"
            f"Las siguientes indicaciones detallan exactamente como se debe puntuar este apartado. "
            f"Aplica estas indicaciones de forma estricta para determinar la nota parcial:\n"
            f"{scoring_instructions}\n\n"
            if scoring_instructions else ""
        )
        from gemini_client import GeminiClient
        corrections_block = GeminiClient._render_correction_examples(correction_examples)
        criteria_block = (
            f"Criterios de evaluacion especificos del examen:\n{evaluation_criteria}\n"
            if evaluation_criteria else ""
        )

        system_prompt = (
            "Eres un corrector experto de Matematicas de Bachillerato (Espana). "
            "Evaluas respuestas de alumnos y devuelves EXCLUSIVAMENTE un objeto JSON valido. "
            "IMPORTANTE: Responde SIEMPRE en español."
        )

        user_prompt = f"""Evalua la respuesta de un alumno de Matematicas de {_course_label} (Espana).

Datos de solucion esperada:
{solution_json}

Contexto del enunciado:
{question_statement or "No disponible"}

Respuesta detectada del alumno:
{student_json}

{scoring_block}{corrections_block}{criteria_block}Criterios de evaluacion (estandar Bachillerato II):
- correcto: resultado final correcto Y procedimiento coherente.
- parcial: procedimiento correcto con error de calculo/signo, O resultado correcto sin procedimiento,
  O procedimiento mayormente correcto con resultado erroneo.
- incorrecto: procedimiento incorrecto o ausente Y resultado incorrecto.
- revision_manual: OCR ilegible o evidencia insuficiente.

IMPORTANTE sobre credito parcial:
- Un alumno con procedimiento correcto que comete un error aritmetico merece credito parcial.
- Solo usa incorrecto cuando el planteamiento es fundamentalmente erroneo o ausente.

Devuelve EXCLUSIVAMENTE este JSON (sin markdown, sin texto extra):
{{
  "classification": "correcto|parcial|incorrecto|revision_manual",
  "result_correct": true,
  "procedure_quality": "correct|mostly_correct|partial|incorrect|not_enough_info",
  "detected_error_type": "none|arithmetic_error|sign_error|conceptual_error|illegible|missing_response|other",
  "matched_expected_steps": ["str"],
  "missing_steps": ["str"],
  "confidence": 0.0,
  "reasoning_summary": "str"
}}"""

        text, usage = self._call(system_prompt, user_prompt)
        self._record_usage(usage)

        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                raw = json.loads(match.group())
            else:
                logger.warning("OpenAI devolvió respuesta no-JSON en assess_math_answer")
                return GeminiAssessment()

        _CLASSIFICATION_MAP = {
            "correcto": "correcto", "correct": "correcto", "ok": "correcto",
            "parcial": "parcial", "partial": "parcial",
            "incorrecto": "incorrecto", "incorrect": "incorrecto",
            "revision_manual": "revision_manual",
        }
        _PROCEDURE_MAP = {
            "correct": "correct", "correcto": "correct",
            "mostly_correct": "mostly_correct",
            "partial": "partial", "parcial": "partial",
            "incorrect": "incorrect", "incorrecto": "incorrect",
            "not_enough_info": "not_enough_info",
        }
        _ERROR_MAP = {
            "none": "none", "ninguno": "none",
            "arithmetic_error": "arithmetic_error",
            "sign_error": "sign_error",
            "conceptual_error": "conceptual_error",
            "illegible": "illegible",
            "missing_response": "missing_response",
            "other": "other",
        }

        try:
            classification = _CLASSIFICATION_MAP.get(
                str(raw.get("classification", "revision_manual")).lower(), "revision_manual"
            )
            procedure_quality = _PROCEDURE_MAP.get(
                str(raw.get("procedure_quality", "not_enough_info")).lower(), "not_enough_info"
            )
            detected_error_type = _ERROR_MAP.get(
                str(raw.get("detected_error_type", "other")).lower(), "other"
            )
            result_correct_raw = raw.get("result_correct")
            if isinstance(result_correct_raw, bool):
                result_correct = result_correct_raw
            elif result_correct_raw is None:
                result_correct = None
            else:
                result_correct = str(result_correct_raw).lower() in {"true", "1", "yes", "si"}

            confidence = float(raw.get("confidence", 0.0))
            if confidence > 1.0:
                confidence = confidence / 100.0
            confidence = max(0.0, min(1.0, confidence))

            return GeminiAssessment(
                classification=classification,
                result_correct=result_correct,
                procedure_quality=procedure_quality,
                detected_error_type=detected_error_type,
                matched_expected_steps=[str(s) for s in (raw.get("matched_expected_steps") or [])],
                missing_steps=[str(s) for s in (raw.get("missing_steps") or [])],
                confidence=confidence,
                reasoning_summary=str(raw.get("reasoning_summary", "")),
            )
        except (ValidationError, Exception) as exc:
            logger.warning(f"Error parseando assess_math_answer de OpenAI: {exc}")
            return GeminiAssessment()

    # ── generate_feedback_explanation (misma firma que GeminiClient) ───

    def generate_feedback_explanation(
        self,
        student_answer: str,
        expected_answer: str,
        reasoning_summary: str,
        status: str,
        awarded_points: float,
        max_points: float,
    ) -> str:
        system_prompt = (
            "Eres un corrector de examenes de Matematicas de Bachillerato (Espana). "
            "Redactas explicaciones breves y claras en español para los alumnos."
        )
        user_prompt = f"""Redacta una explicacion de correccion en espanol, clara y breve (2-4 frases), para un alumno de Bachillerato.

Datos:
- Estado final: {status}
- Puntuacion: {awarded_points:.2f}/{max_points:.2f}
- Respuesta detectada: {student_answer or "sin respuesta legible"}
- Solucion esperada: {expected_answer or "no especificada"}
- Resumen tecnico: {reasoning_summary or "sin resumen"}

Requisitos:
- No uses lenguaje ofensivo.
- Explica por que obtiene esa puntuacion.
- Si procede, sugiere revision manual."""

        text, usage = self._call_text(system_prompt, user_prompt)
        self._record_usage(usage)
        return text.strip()
