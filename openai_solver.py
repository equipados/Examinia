"""Solver de preguntas matemáticas usando la API de OpenAI."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

from models import GeminiSolvedExercise

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
    "Resuelves ejercicios paso a paso y devuelves EXCLUSIVAMENTE un objeto JSON valido, sin texto adicional."
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
    """Solver de preguntas matemáticas vía OpenAI (misma interfaz que GeminiClient.solve_math_question)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_retries: int = 3,
        rate_limit_seconds: float = 1.0,
    ) -> None:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta OPENAI_API_KEY en variables de entorno o en archivo .env")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "No se pudo importar openai. Instala dependencias con: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max(1, int(max_retries))
        self.rate_limit_seconds = rate_limit_seconds
        self._last_request_ts = 0.0
        self._usage: dict[str, dict] = {}

    # ── Usage tracking (misma interfaz que GeminiClient) ──────────────

    def get_usage(self) -> dict[str, dict]:
        return dict(self._usage)

    def reset_usage(self) -> None:
        self._usage = {}

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
                    temperature=0.1,
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
            import re
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
