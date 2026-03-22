from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from models import (
    ExtractedPart,
    GeminiAssessment,
    GeminiSolvedExercise,
    LLMPageExtraction,
    PageExtraction,
    SolutionTemplate,
    TeacherSolutionsExtraction,
)

T = TypeVar("T", bound=BaseModel)


class GeminiClient:
    def __init__(
        self,
        model: str,
        solver_model: str | None = None,
        max_retries: int = 3,
        rate_limit_seconds: float = 0.0,
        request_timeout_seconds: int = 120,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta GEMINI_API_KEY en variables de entorno o en archivo .env")

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "No se pudo importar google-genai. Instala dependencias con: pip install -r requirements.txt"
            ) from exc

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.solver_model = solver_model or model
        self.max_retries = max(1, int(max_retries))
        self.rate_limit_seconds = max(0.0, float(rate_limit_seconds))
        self.request_timeout_seconds = max(10, int(request_timeout_seconds))
        self._last_request_ts = 0.0
        self._native_schema_supported: bool | None = None
        # Acumulador de tokens por modelo: {model_name: {"input": N, "output": N, "calls": N}}
        self._usage: dict[str, dict[str, int]] = {}

    def _record_usage(self, model: str, response: object) -> None:
        """Extrae usage_metadata de la respuesta y lo acumula."""
        um = getattr(response, "usage_metadata", None)
        if um is None:
            return
        inp = int(getattr(um, "prompt_token_count", 0) or 0)
        out = int(getattr(um, "candidates_token_count", 0) or 0)
        think = int(getattr(um, "thoughts_token_count", 0) or 0)
        total = int(getattr(um, "total_token_count", 0) or 0)
        if total == 0:
            total = inp + out + think
        entry = self._usage.setdefault(model, {"input": 0, "output": 0, "thinking": 0, "total": 0, "calls": 0})
        entry["input"] += inp
        entry["output"] += out
        entry["thinking"] += think
        entry["total"] += total
        entry["calls"] += 1

    def get_usage(self) -> dict[str, dict[str, int]]:
        """Devuelve el acumulado de tokens desde la última llamada a reset_usage()."""
        return dict(self._usage)

    def reset_usage(self) -> dict[str, dict[str, int]]:
        """Devuelve el acumulado y lo resetea a cero."""
        snapshot = dict(self._usage)
        self._usage = {}
        return snapshot

    def _respect_rate_limit(self) -> None:
        if self.rate_limit_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_ts
        wait_time = self.rate_limit_seconds - elapsed
        if wait_time > 0:
            time.sleep(wait_time)

    def _build_contents(self, prompt: str, image_paths: list[Path] | None = None) -> list[object]:
        if not image_paths:
            return [prompt]
        images = []
        for p in image_paths:
            with Image.open(p) as img:
                images.append(img.copy())
        return [prompt] + images

    def _extract_text_fallback(self, response: object) -> str:
        text = getattr(response, "text", None)
        if text:
            return str(text)
        candidates = getattr(response, "candidates", None) or []
        chunks: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                piece = getattr(part, "text", None)
                if piece:
                    chunks.append(str(piece))
        return "\n".join(chunks).strip()

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(",", ".")
        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if not match:
            return default
        try:
            return float(match.group(1))
        except ValueError:
            return default

    @classmethod
    def _to_confidence(cls, value: Any, default: float = 0.0) -> float:
        conf = cls._to_float(value, default=default)
        if conf < 0:
            return 0.0
        if conf > 1:
            if conf <= 100:
                return round(conf / 100.0, 4)
            return 1.0
        return round(conf, 4)

    @staticmethod
    def _to_str(value: Any, default: str = "") -> str:
        if value is None:
            return default
        return str(value).strip()

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        raw = str(value).strip().lower()
        if raw in {"true", "1", "yes", "si", "s", "y"}:
            return True
        if raw in {"false", "0", "no", "n"}:
            return False
        return default

    @staticmethod
    def _to_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _pick_first(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return default

    @staticmethod
    def _normalize_boundary(value: Any) -> str:
        raw = str(value or "").strip().lower().replace(" ", "_")
        if raw in {"new_exam", "new", "newexam", "start", "first_page"}:
            return "new_exam"
        if raw in {"continuation", "continue", "same_exam", "next_page", "continued"}:
            return "continuation"
        return "unknown"

    @staticmethod
    def _normalize_page_role(value: Any) -> str:
        raw = str(value or "").strip().lower().replace(" ", "_")
        if raw in {"cover", "caratula", "header_page", "statement_page", "enunciados"}:
            return "cover"
        if raw in {"answer", "answers", "solution_page", "response_page", "respuesta"}:
            return "answer"
        if raw in {"mixed", "mixto"}:
            return "mixed"
        return "unknown"

    @staticmethod
    def _normalize_classification(value: Any) -> str:
        raw = str(value or "").strip().lower().replace(" ", "_")
        mapping = {
            "correcto": "correcto",
            "correct": "correcto",
            "ok": "correcto",
            "parcial": "parcial",
            "partial": "parcial",
            "incorrecto": "incorrecto",
            "incorrect": "incorrecto",
            "revision_manual": "revision_manual",
            "manual_review": "revision_manual",
            "review_manual": "revision_manual",
            "uncertain": "revision_manual",
        }
        return mapping.get(raw, "revision_manual")

    @staticmethod
    def _normalize_procedure_quality(value: Any) -> str:
        raw = str(value or "").strip().lower().replace(" ", "_")
        mapping = {
            "correct": "correct",
            "correcto": "correct",
            "mostly_correct": "mostly_correct",
            "mayormente_correcto": "mostly_correct",
            "partial": "partial",
            "parcial": "partial",
            "incorrect": "incorrect",
            "incorrecto": "incorrect",
            "not_enough_info": "not_enough_info",
            "insuficiente": "not_enough_info",
            "unknown": "not_enough_info",
        }
        return mapping.get(raw, "not_enough_info")

    @staticmethod
    def _normalize_error_type(value: Any) -> str:
        raw = str(value or "").strip().lower().replace(" ", "_")
        mapping = {
            "none": "none",
            "ninguno": "none",
            "arithmetic_error": "arithmetic_error",
            "error_aritmetico": "arithmetic_error",
            "sign_error": "sign_error",
            "error_de_signo": "sign_error",
            "conceptual_error": "conceptual_error",
            "error_conceptual": "conceptual_error",
            "illegible": "illegible",
            "ilegible": "illegible",
            "missing_response": "missing_response",
            "sin_respuesta": "missing_response",
            "other": "other",
        }
        return mapping.get(raw, "other")

    def _coerce_question(self, question: Any, index: int) -> dict[str, Any]:
        if not isinstance(question, dict):
            return {
                "question_id": str(index),
                "statement": "",
                "max_points": None,
                "parts": [],
            }

        question_id = self._pick_first(question, ["question_id", "id", "number", "exercise"], str(index))
        statement = self._pick_first(question, ["statement", "title", "prompt", "enunciado"], "")
        max_points = self._pick_first(question, ["max_points", "max_score", "points", "score", "value"], None)

        raw_parts = self._pick_first(question, ["parts", "subparts", "subquestions", "items"], [])
        parts: list[dict[str, Any]] = []
        if isinstance(raw_parts, list):
            for p_idx, part in enumerate(raw_parts, start=1):
                if not isinstance(part, dict):
                    continue
                part_id = self._pick_first(part, ["part_id", "id", "label", "letter"], str(p_idx))
                part_statement = self._pick_first(part, ["statement", "title", "prompt"], "")
                part_max = self._pick_first(part, ["max_points", "max_score", "points", "score", "value"], None)
                answer_raw = self._pick_first(
                    part,
                    ["student_answer_raw", "answer", "student_response", "raw_answer", "text"],
                    "",
                )
                answer_norm = self._pick_first(
                    part,
                    ["student_answer_normalized", "normalized_answer", "interpreted_answer"],
                    None,
                )
                steps = self._pick_first(part, ["steps_detected", "steps", "procedure_steps"], [])
                confidence = self._pick_first(part, ["confidence", "ocr_confidence", "extraction_confidence"], 0.0)
                parts.append(
                    {
                        "part_id": self._to_str(part_id, default=str(p_idx)),
                        "statement": self._to_str(part_statement, default=""),
                        "max_points": part_max,
                        "student_answer_raw": self._to_str(answer_raw, default=""),
                        "student_answer_normalized": self._to_str(answer_norm, default="") or None,
                        "steps_detected": self._to_str_list(steps),
                        "confidence": self._to_confidence(confidence, default=0.0),
                    }
                )

        if not parts:
            answer_raw = self._pick_first(
                question,
                ["student_answer_raw", "answer", "student_response", "raw_answer", "text"],
                "",
            )
            answer_norm = self._pick_first(
                question,
                ["student_answer_normalized", "normalized_answer", "interpreted_answer"],
                None,
            )
            steps = self._pick_first(question, ["steps_detected", "steps", "procedure_steps"], [])
            confidence = self._pick_first(question, ["confidence", "ocr_confidence", "extraction_confidence"], 0.0)
            parts.append(
                {
                    "part_id": "single",
                    "statement": self._to_str(statement, default=""),
                    "max_points": max_points,
                    "student_answer_raw": self._to_str(answer_raw, default=""),
                    "student_answer_normalized": self._to_str(answer_norm, default="") or None,
                    "steps_detected": self._to_str_list(steps),
                    "confidence": self._to_confidence(confidence, default=0.0),
                }
            )

        return {
            "question_id": self._to_str(question_id, default=str(index)),
            "statement": self._to_str(statement, default=""),
            "max_points": max_points,
            "parts": parts,
        }

    def _coerce_page_extraction_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        student_info = payload.get("student_info") if isinstance(payload.get("student_info"), dict) else {}
        exam_info = payload.get("exam_info") if isinstance(payload.get("exam_info"), dict) else {}
        page_metadata = payload.get("page_metadata") if isinstance(payload.get("page_metadata"), dict) else {}

        student_name = self._pick_first(payload, ["student_name", "student"], None)
        if not student_name:
            student_name = self._pick_first(student_info, ["name", "student_name"], None)

        student_conf = self._pick_first(payload, ["student_name_confidence"], None)
        if student_conf is None:
            student_conf = self._pick_first(student_info, ["confidence"], 0.0)

        exam_model = self._pick_first(payload, ["exam_model"], None)
        if not exam_model:
            exam_model = self._pick_first(exam_info, ["model", "exam_model"], None)

        course_level = self._pick_first(payload, ["course_level", "curso", "nivel_curso"], None)
        if not course_level:
            course_level = self._pick_first(exam_info, ["course_level", "curso"], None)

        header_detected = self._pick_first(
            payload,
            ["start_header_detected", "has_start_header", "header_detected", "is_exam_start_header"],
            None,
        )
        if header_detected is None:
            header_detected = self._pick_first(
                exam_info,
                ["start_header_detected", "has_start_header", "header_detected"],
                False,
            )
        header_confidence = self._pick_first(
            payload,
            ["start_header_confidence", "header_confidence"],
            None,
        )
        if header_confidence is None:
            header_confidence = self._pick_first(
                exam_info,
                ["start_header_confidence", "header_confidence", "confidence"],
                0.0,
            )

        page_role = self._pick_first(
            payload,
            ["page_role", "page_type", "role"],
            None,
        )
        if page_role is None:
            page_role = self._pick_first(
                page_metadata,
                ["page_role", "page_type", "role"],
                "unknown",
            )

        handwritten_detected = self._pick_first(
            payload,
            ["handwritten_content_detected", "has_handwriting", "handwritten_detected", "contains_handwriting"],
            None,
        )
        if handwritten_detected is None:
            handwritten_detected = self._pick_first(
                page_metadata,
                ["handwritten_content_detected", "has_handwriting", "handwritten_detected"],
                False,
            )
        handwritten_confidence = self._pick_first(
            payload,
            ["handwritten_content_confidence", "handwriting_confidence"],
            None,
        )
        if handwritten_confidence is None:
            handwritten_confidence = self._pick_first(
                page_metadata,
                ["handwritten_content_confidence", "handwriting_confidence", "confidence"],
                0.0,
            )

        boundary = self._pick_first(payload, ["exam_boundary_hint", "page_type"], None)
        if boundary is None:
            boundary = self._pick_first(exam_info, ["boundary_hint", "page_type"], None)

        boundary_conf = self._pick_first(payload, ["boundary_confidence"], None)
        if boundary_conf is None:
            boundary_conf = self._pick_first(exam_info, ["confidence"], 0.0)

        raw_questions = self._pick_first(payload, ["questions", "exercises"], [])
        if not isinstance(raw_questions, list):
            raw_questions = []
        questions = [self._coerce_question(question, idx) for idx, question in enumerate(raw_questions, start=1)]

        incidents = []
        for key in ("incidents", "warnings", "notes", "issues"):
            incidents.extend(self._to_str_list(payload.get(key)))

        extraction_conf = self._pick_first(payload, ["extraction_confidence", "confidence", "overall_confidence"], None)
        if extraction_conf is None:
            extraction_conf = self._pick_first(page_metadata, ["confidence"], 0.0)

        canonical: dict[str, Any] = {
            "student_name": self._to_str(student_name, default="") or None,
            "student_name_confidence": self._to_confidence(student_conf, default=0.0),
            "exam_model": self._to_str(exam_model, default="") or None,
            "course_level": self._to_str(course_level, default="") or None,
            "start_header_detected": self._to_bool(header_detected, default=False),
            "start_header_confidence": self._to_confidence(header_confidence, default=0.0),
            "page_role": self._normalize_page_role(page_role),
            "handwritten_content_detected": self._to_bool(handwritten_detected, default=False),
            "handwritten_content_confidence": self._to_confidence(handwritten_confidence, default=0.0),
            "exam_boundary_hint": self._normalize_boundary(boundary),
            "boundary_confidence": self._to_confidence(boundary_conf, default=0.0),
            "questions": questions,
            "incidents": incidents,
            "extraction_confidence": self._to_confidence(extraction_conf, default=0.0),
        }
        return canonical

    def _coerce_assessment_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        classification = self._pick_first(payload, ["classification", "status", "class"], "revision_manual")
        result_correct = self._pick_first(payload, ["result_correct", "final_answer_correct", "is_correct"], None)
        procedure_quality = self._pick_first(payload, ["procedure_quality", "procedure", "procedure_assessment"], None)
        error_type = self._pick_first(payload, ["detected_error_type", "error_type", "error"], None)
        matched_steps = self._pick_first(payload, ["matched_expected_steps", "matched_steps"], [])
        missing_steps = self._pick_first(payload, ["missing_steps", "unmatched_steps"], [])
        confidence = self._pick_first(payload, ["confidence", "score"], 0.0)
        reasoning = self._pick_first(payload, ["reasoning_summary", "analysis", "explanation", "justification"], "")

        canonical = {
            "classification": self._normalize_classification(classification),
            "result_correct": self._to_bool(result_correct, default=False) if result_correct is not None else None,
            "procedure_quality": self._normalize_procedure_quality(procedure_quality),
            "detected_error_type": self._normalize_error_type(error_type),
            "matched_expected_steps": self._to_str_list(matched_steps),
            "missing_steps": self._to_str_list(missing_steps),
            "confidence": self._to_confidence(confidence, default=0.0),
            "reasoning_summary": self._to_str(reasoning, default=""),
        }
        return canonical

    def _coerce_solved_exercise_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        can_solve = self._pick_first(payload, ["can_solve", "solvable", "is_solvable"], False)
        confidence = self._pick_first(payload, ["confidence", "solve_confidence", "score"], 0.0)
        solved_final = self._pick_first(
            payload,
            ["solved_final_answer", "final_answer", "answer", "result"],
            "",
        )
        accepted_equivalents = self._pick_first(
            payload,
            ["accepted_equivalents", "equivalents", "alternate_answers"],
            [],
        )
        expected_steps = self._pick_first(payload, ["expected_steps", "steps", "solution_steps"], [])
        topic = self._pick_first(payload, ["topic", "area"], None)
        notes = self._pick_first(payload, ["notes", "reasoning_summary", "analysis", "explanation"], "")
        incidents = []
        for key in ("incidents", "warnings", "issues"):
            incidents.extend(self._to_str_list(payload.get(key)))

        return {
            "can_solve": self._to_bool(can_solve, default=False),
            "confidence": self._to_confidence(confidence, default=0.0),
            "solved_final_answer": self._to_str(solved_final, default=""),
            "accepted_equivalents": self._to_str_list(accepted_equivalents),
            "expected_steps": self._to_str_list(expected_steps),
            "topic": self._to_str(topic, default="") or None,
            "notes": self._to_str(notes, default=""),
            "incidents": incidents,
        }

    def _coerce_payload_for_schema(self, payload: Any, schema: type[T]) -> Any:
        if not isinstance(payload, dict):
            return payload
        if schema is LLMPageExtraction:
            return self._coerce_page_extraction_payload(payload)
        if schema is GeminiAssessment:
            return self._coerce_assessment_payload(payload)
        if schema is GeminiSolvedExercise:
            return self._coerce_solved_exercise_payload(payload)
        return payload

    @staticmethod
    def _is_response_schema_unsupported_error(exc: Exception) -> bool:
        payload = str(exc).lower()
        return "invalid_argument" in payload and "response_schema" in payload and "additional_properties" in payload

    @staticmethod
    def _is_quota_exceeded(exc: Exception) -> bool:
        payload = str(exc)
        return "429" in payload or "RESOURCE_EXHAUSTED" in payload or "quota" in payload.lower()

    @staticmethod
    def _parse_retry_delay(exc: Exception) -> float:
        """Extrae el retryDelay sugerido por la API (segundos). Devuelve 0 si no lo encuentra."""
        import re as _re
        m = _re.search(r"retry[_ ]?[iI]n[' \"]*:?\s*['\"]?(\d+(?:\.\d+)?)", str(exc), _re.IGNORECASE)
        if m:
            return float(m.group(1))
        return 0.0

    @staticmethod
    def _parse_json_payload(text: str) -> dict | list:
        raw = text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            first_obj = raw.find("{")
            last_obj = raw.rfind("}")
            if first_obj != -1 and last_obj > first_obj:
                return json.loads(raw[first_obj : last_obj + 1])
            first_arr = raw.find("[")
            last_arr = raw.rfind("]")
            if first_arr != -1 and last_arr > first_arr:
                return json.loads(raw[first_arr : last_arr + 1])
            raise

    def _generate_structured_once(
        self,
        prompt: str,
        schema: type[T],
        image_paths: list[Path] | None = None,
        temperature: float = 0.1,
        use_native_schema: bool = True,
        model_override: str | None = None,
        disable_thinking: bool = False,
    ) -> T:
        self._respect_rate_limit()
        config_kwargs: dict[str, object] = {
            "temperature": temperature,
            "response_mime_type": "application/json",
        }
        if use_native_schema:
            config_kwargs["response_schema"] = schema
        if disable_thinking:
            try:
                config_kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        config = self._types.GenerateContentConfig(**config_kwargs)
        contents = self._build_contents(prompt=prompt, image_paths=image_paths)
        target_model = model_override or self.model
        response = self._client.models.generate_content(
            model=target_model,
            contents=contents,
            config=config,
        )
        self._last_request_ts = time.monotonic()
        self._record_usage(target_model, response)

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, schema):
                return parsed
            try:
                return schema.model_validate(parsed)
            except ValidationError:
                coerced = self._coerce_payload_for_schema(parsed, schema)
                return schema.model_validate(coerced)

        text = self._extract_text_fallback(response)
        if not text:
            raise ValueError("Gemini no devolvio contenido parseable")
        payload = self._parse_json_payload(text)
        try:
            return schema.model_validate(payload)
        except ValidationError:
            coerced = self._coerce_payload_for_schema(payload, schema)
            return schema.model_validate(coerced)

    def _generate_structured(
        self,
        prompt: str,
        schema: type[T],
        image_paths: list[Path] | None = None,
        temperature: float = 0.1,
        model_override: str | None = None,
        disable_thinking: bool = False,
    ) -> T:
        strict_prompt = prompt
        use_native_schema = self._native_schema_supported is not False
        retry_engine = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((ValidationError, json.JSONDecodeError, ValueError, RuntimeError)),
            reraise=True,
        )

        for attempt in retry_engine:
            with attempt:
                try:
                    result = self._generate_structured_once(
                        prompt=strict_prompt,
                        schema=schema,
                        image_paths=image_paths,
                        temperature=temperature,
                        use_native_schema=use_native_schema,
                        model_override=model_override,
                        disable_thinking=disable_thinking,
                    )
                    if use_native_schema and self._native_schema_supported is None:
                        self._native_schema_supported = True
                    return result
                except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                    logger.warning(f"Salida JSON invalida de Gemini (reintento): {exc}")
                    strict_prompt = (
                        f"{prompt}\n\n"
                        "RESTRICCION EXTRA: devuelve EXCLUSIVAMENTE JSON valido que cumpla el schema. "
                        "No incluyas markdown, comentarios ni texto fuera del JSON."
                    )
                    raise
                except Exception as exc:
                    if use_native_schema and self._is_response_schema_unsupported_error(exc):
                        logger.debug(
                            "Gemini rechazo response_schema nativo; se activa modo JSON puro + validacion Pydantic."
                        )
                        self._native_schema_supported = False
                        use_native_schema = False
                        raise RuntimeError(str(exc)) from exc
                    if self._is_quota_exceeded(exc):
                        delay = self._parse_retry_delay(exc)
                        if delay > 0:
                            logger.warning(f"Cuota API agotada (429); esperando {delay:.0f}s antes de reintentar...")
                            time.sleep(min(delay, 60))
                        else:
                            logger.warning("Cuota API agotada (429); sin delay sugerido, re-lanzando inmediatamente.")
                        raise RuntimeError(str(exc)) from exc
                    if use_native_schema and self._native_schema_supported is None:
                        self._native_schema_supported = True
                    logger.warning(f"Error en llamada Gemini (reintento): {exc}")
                    raise RuntimeError(str(exc)) from exc
        raise RuntimeError("No se pudo obtener salida estructurada valida de Gemini")

    def _generate_text(self, prompt: str, temperature: float = 0.3, model_override: str | None = None, disable_thinking: bool = False) -> str:
        self._respect_rate_limit()
        config_kwargs: dict[str, object] = {"temperature": temperature}
        if disable_thinking:
            try:
                config_kwargs["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        config = self._types.GenerateContentConfig(**config_kwargs)
        target_model = model_override or self.model
        response = self._client.models.generate_content(
            model=target_model,
            contents=[prompt],
            config=config,
        )
        self._last_request_ts = time.monotonic()
        self._record_usage(target_model, response)
        text = self._extract_text_fallback(response)
        if not text:
            raise RuntimeError("Gemini no devolvio texto")
        return text.strip()

    def analyze_exam_page(self, image_path: Path, prompt: str, schema: type[T]) -> T:
        return self._generate_structured(prompt=prompt, schema=schema, image_paths=[image_path], temperature=0.1)

    def extract_structured_exam_data(self, image_path: Path, source_file: str, page_number: int) -> PageExtraction:
        prompt = f"""
Eres un corrector experto de Matematicas II (2o Bachillerato) y analista OCR de manuscritos.
Analiza la imagen de la pagina y extrae una estructura JSON estricta.

Objetivo:
- Detectar nombre del alumno (si no es legible usa null).
- Detectar modelo de examen si aparece.
- Detectar si la pagina contiene encabezado de inicio de examen: logo a la izquierda + nombre del alumno.
- Clasificar rol de pagina: `cover` (caratula/enunciados), `answer` (resoluciones manuscritas), `mixed` o `unknown`.
- Detectar si hay contenido manuscrito real del alumno.
- Detectar si esta pagina parece inicio de nuevo examen (`new_exam`), continuacion (`continuation`) o dudoso (`unknown`).
- Extraer ejercicios y apartados con su respuesta manuscrita.
- Extraer pasos intermedios del procedimiento cuando aparezcan.
- Detectar puntuaciones: cada ejercicio puede llevar su puntuacion en negrita junto al numero,
  formato "(3p)", "(5p)", "(2,5 pts)", "(2 puntos)", etc. Extrae ese valor en question.max_points.
  Si los apartados tienen puntuacion individual, ponla en part.max_points; si no, deja 0.0.
- Marcar incidentes cuando haya baja legibilidad o ambiguedad.

Reglas:
- No inventes contenido no visible.
- Si algo no se puede leer, usa valores nulos/vacios e indica incidencia.
- Usa ids compactos para ejercicios y apartados (ej: "1", "2", "a", "b").
- `confidence` y `extraction_confidence` deben estar entre 0 y 1.
- Si `page_role=cover` y NO hay manuscrito del alumno, NO pongas respuestas de alumno:
  `student_answer_raw=""`, `steps_detected=[]`.
- Devuelve SOLO este formato de claves (sin claves extra):
  {{
    "student_name": "str|null",
    "student_name_confidence": 0.0,
    "exam_model": "str|null",
    "start_header_detected": true,
    "start_header_confidence": 0.0,
    "page_role": "cover|answer|mixed|unknown",
    "handwritten_content_detected": false,
    "handwritten_content_confidence": 0.0,
    "exam_boundary_hint": "new_exam|continuation|unknown",
    "boundary_confidence": 0.0,
    "questions": [
      {{
        "question_id": "str",
        "statement": "str|null",
        "max_points": 0.0,
        "parts": [
          {{
            "part_id": "str",
            "statement": "str|null",
            "max_points": 0.0,
            "student_answer_raw": "str",
            "student_answer_normalized": "str|null",
            "steps_detected": ["str"],
            "confidence": 0.0
          }}
        ]
      }}
    ],
    "incidents": ["str"],
    "extraction_confidence": 0.0
  }}

Metadatos de esta pagina:
- source_file: {source_file}
- page_number: {page_number}
"""
        payload = self.analyze_exam_page(image_path=image_path, prompt=prompt, schema=LLMPageExtraction)
        return PageExtraction(
            source_file=source_file,
            page_number=page_number,
            student_name=payload.student_name,
            student_name_confidence=payload.student_name_confidence,
            exam_model=payload.exam_model,
            start_header_detected=payload.start_header_detected,
            start_header_confidence=payload.start_header_confidence,
            page_role=payload.page_role,
            handwritten_content_detected=payload.handwritten_content_detected,
            handwritten_content_confidence=payload.handwritten_content_confidence,
            exam_boundary_hint=payload.exam_boundary_hint,
            boundary_confidence=payload.boundary_confidence,
            questions=payload.questions,
            incidents=payload.incidents,
            extraction_confidence=payload.extraction_confidence,
        )

    def extract_exam_questions(self, image_path: Path, source_file: str, page_number: int) -> PageExtraction:
        prompt = f"""Eres un lector experto de examenes impresos de Matematicas (Bachillerato, Espana).
Esta imagen es la HOJA IMPRESA del examen (texto mecanografiado, no manuscrito del alumno).

Tarea UNICA: extraer la estructura del examen impreso.

PUNTUACIONES (MUY IMPORTANTE):
- Cada ejercicio lleva su puntuacion maxima indicada en negrita justo despues del numero,
  en formato como "(3p)", "(3 puntos)", "(2,5 pts)", "(5p)", etc.
  Ejemplo: "1. (3p) Estudia la continuidad..." → question max_points = 3.0
  Ejemplo: "2. (5p) Calcula los limites..." → question max_points = 5.0
  Pon ese valor en el campo max_points del ejercicio (question-level).
- Si los apartados (a, b, c...) llevan puntuacion individual, ponla en el max_points del apartado.
- Si los apartados NO tienen puntuacion individual pero el ejercicio tiene puntuacion total,
  pon 0.0 en los apartados (el sistema la repartira automaticamente).

ENUNCIADOS:
- Para cada ejercicio: copia el ENUNCIADO COMPLETO tal como aparece impreso
  (incluyendo datos numericos, formulas, condiciones). No resumas.
- Para cada apartado (a, b, c...): extrae su enunciado completo.
- Si un ejercicio no tiene apartados, crea un unico apartado con part_id="single".

OTROS DATOS:
- Detecta nombre del alumno y modelo de examen si aparecen en el encabezado.
- Detecta el curso: "1o_bachillerato" si pone 1o Bachillerato o similar,
  "2o_bachillerato" si pone 2o Bachillerato, null si no se especifica.
- Si el alumno escribio alguna respuesta MANUSCRITA en esta hoja, incluyela en
  student_answer_raw solo si es claramente manuscrita y distinta del texto impreso.
- Si no hay respuesta manuscrita, student_answer_raw = "".
- Devuelve page_role = "cover".
- Nivel: Bachillerato (1o o 2o) Espana. Temas: derivadas, integrales, limites, matrices,
  sistemas lineales, estadistica, geometria analitica, secciones conicas.

Devuelve SOLO este formato de claves (sin claves extra):
  {{
    "student_name": "str|null",
    "student_name_confidence": 0.0,
    "exam_model": "str|null",
    "course_level": "1o_bachillerato|2o_bachillerato|null",
    "start_header_detected": true,
    "start_header_confidence": 0.0,
    "page_role": "cover",
    "handwritten_content_detected": false,
    "handwritten_content_confidence": 0.0,
    "exam_boundary_hint": "new_exam|continuation|unknown",
    "boundary_confidence": 0.0,
    "questions": [
      {{
        "question_id": "str",
        "statement": "str",
        "max_points": 0.0,
        "parts": [
          {{
            "part_id": "str",
            "statement": "str",
            "max_points": 0.0,
            "student_answer_raw": "",
            "student_answer_normalized": null,
            "steps_detected": [],
            "confidence": 0.0
          }}
        ]
      }}
    ],
    "incidents": ["str"],
    "extraction_confidence": 0.0
  }}

Metadatos de esta pagina:
- source_file: {source_file}
- page_number: {page_number}
"""
        payload = self.analyze_exam_page(image_path=image_path, prompt=prompt, schema=LLMPageExtraction)
        return PageExtraction(
            source_file=source_file,
            page_number=page_number,
            student_name=payload.student_name,
            student_name_confidence=payload.student_name_confidence,
            exam_model=payload.exam_model,
            course_level=payload.course_level,
            start_header_detected=payload.start_header_detected,
            start_header_confidence=payload.start_header_confidence,
            page_role=payload.page_role,
            handwritten_content_detected=payload.handwritten_content_detected,
            handwritten_content_confidence=payload.handwritten_content_confidence,
            exam_boundary_hint=payload.exam_boundary_hint,
            boundary_confidence=payload.boundary_confidence,
            questions=payload.questions,
            incidents=payload.incidents,
            extraction_confidence=payload.extraction_confidence,
        )

    def extract_student_answers(
        self,
        image_path: Path,
        source_file: str,
        page_number: int,
        questions_context: list[dict],
    ) -> PageExtraction:
        context_json = json.dumps(questions_context, ensure_ascii=False, indent=2)
        prompt = f"""Eres un experto en OCR de matematicas manuscritas para Bachillerato II (Espana).
Esta imagen muestra el TRABAJO MANUSCRITO de un alumno (calculos, desarrollo, respuestas).

Ejercicios esperados segun la hoja de enunciados:
{context_json}

Transcribe TODO el trabajo manuscrito visible.
Para cada ejercicio y apartado:
- student_answer_raw: la RESPUESTA FINAL del alumno. Usa notacion de texto legible:
  "x^2 + 3x - 2", "f'(x) = 2x+1", "integral(x^2 dx) = x^3/3 + C",
  "lim(x->0)(sin(x)/x) = 1", "[[1,2],[3,4]]" para matrices,
  "(x-2)^2 + (y+1)^2 = 9" para circunferencias, etc.
- student_answer_normalized: misma respuesta normalizada si puedes.
- steps_detected: lista de pasos intermedios. Cada uno como expresion corta:
  ["f'(x) = 2x+3", "f'(2) = 7", "y - 5 = 7(x - 2)"]
  Si un paso es ilegible: "[ilegible]".
- confidence: confianza en la transcripcion (0.0 a 1.0).

Temas posibles: derivadas (regla cadena, producto, cociente), integrales (por partes,
sustitucion), limites (L'Hopital), matrices/determinantes, sistemas Gauss/Cramer,
estadistica (distribucion normal, intervalos confianza), geometria analitica,
secciones conicas, funciones y continuidad.

NO inventes contenido. Zona ilegible: confidence baja + "[ilegible]" en steps.
Si no hay trabajo visible para un ejercicio, no lo incluyas en questions[].
Devuelve page_role = "answer".

Devuelve SOLO este formato de claves (sin claves extra):
  {{
    "student_name": null,
    "student_name_confidence": 0.0,
    "exam_model": null,
    "start_header_detected": false,
    "start_header_confidence": 0.0,
    "page_role": "answer",
    "handwritten_content_detected": true,
    "handwritten_content_confidence": 0.0,
    "exam_boundary_hint": "continuation",
    "boundary_confidence": 0.0,
    "questions": [
      {{
        "question_id": "str",
        "statement": null,
        "max_points": null,
        "parts": [
          {{
            "part_id": "str",
            "statement": null,
            "max_points": null,
            "student_answer_raw": "str",
            "student_answer_normalized": "str|null",
            "steps_detected": ["str"],
            "confidence": 0.0
          }}
        ]
      }}
    ],
    "incidents": ["str"],
    "extraction_confidence": 0.0
  }}

Metadatos de esta pagina:
- source_file: {source_file}
- page_number: {page_number}
"""
        payload = self.analyze_exam_page(image_path=image_path, prompt=prompt, schema=LLMPageExtraction)
        return PageExtraction(
            source_file=source_file,
            page_number=page_number,
            student_name=payload.student_name,
            student_name_confidence=payload.student_name_confidence,
            exam_model=payload.exam_model,
            start_header_detected=payload.start_header_detected,
            start_header_confidence=payload.start_header_confidence,
            page_role=payload.page_role,
            handwritten_content_detected=payload.handwritten_content_detected,
            handwritten_content_confidence=payload.handwritten_content_confidence,
            exam_boundary_hint=payload.exam_boundary_hint,
            boundary_confidence=payload.boundary_confidence,
            questions=payload.questions,
            incidents=payload.incidents,
            extraction_confidence=payload.extraction_confidence,
        )

    def assess_math_answer(
        self,
        solution: SolutionTemplate,
        extracted_part: ExtractedPart,
        question_statement: str | None = None,
        course_level: str | None = None,
        evaluation_criteria: str | None = None,
        scoring_instructions: str | None = None,
    ) -> GeminiAssessment:
        _course_label = {
            "1o_bachillerato": "1o Bachillerato",
            "2o_bachillerato": "2o Bachillerato",
        }.get(course_level or "", "Bachillerato")
        prompt = f"""
Evalua la respuesta de un alumno de Matematicas de {_course_label} (Espana) para un apartado concreto.
Devuelve JSON estricto segun schema.

Datos de solucion esperada:
{solution.model_dump_json(indent=2, exclude_none=True, exclude={"scoring_instructions"})}

Contexto del enunciado:
{question_statement or "No disponible"}

Respuesta detectada del alumno:
{{
  "student_answer_raw": {json.dumps(extracted_part.student_answer_raw, ensure_ascii=False)},
  "student_answer_normalized": {json.dumps(extracted_part.student_answer_normalized or "", ensure_ascii=False)},
  "steps_detected": {json.dumps(extracted_part.steps_detected, ensure_ascii=False)},
  "ocr_confidence": {extracted_part.confidence}
}}

{f"INDICACIONES DE PUNTUACION ESPECIFICAS PARA ESTE APARTADO (PRIORIDAD MAXIMA):{chr(10)}Las siguientes indicaciones detallan exactamente como se debe puntuar este apartado. Aplica estas indicaciones de forma estricta para determinar la nota parcial:{chr(10)}{scoring_instructions}{chr(10)}{chr(10)}" if scoring_instructions else ""}{f"Criterios de evaluacion especificos del examen:{chr(10)}{evaluation_criteria}{chr(10)}" if evaluation_criteria else ""}Criterios de evaluacion (estandar Bachillerato II):
- correcto: resultado final correcto Y procedimiento coherente.
- parcial: procedimiento correcto con error de calculo/signo en pasos finales (merecedor de credito parcial),
  O resultado correcto sin procedimiento visible, O procedimiento mayormente correcto con resultado erroneo.
- incorrecto: procedimiento incorrecto o ausente Y resultado incorrecto.
- revision_manual: OCR ilegible o evidencia insuficiente para decidir.

IMPORTANTE sobre credito parcial:
- Un alumno con procedimiento correcto que comete un error aritmetico merece credito parcial (NO incorrecto).
- Un alumno con planteamiento correcto pero desarrollo incompleto merece credito parcial.
- Solo usa incorrecto cuando el planteamiento matematico es fundamentalmente erroneo o ausente.

Puntos de analisis:
- Correccion del resultado final (comparar con expected_final_answer y accepted_equivalents).
- Validez del procedimiento (comparar steps con expected_steps).
- Tipo de error si aplica: signo, aritmetico, conceptual, ilegible.
- Si la evidencia OCR es de baja confianza y no se puede determinar, usa revision_manual.

Devuelve SOLO estas claves:
{{
  "classification": "correcto|parcial|incorrecto|revision_manual",
  "result_correct": true,
  "procedure_quality": "correct|mostly_correct|partial|incorrect|not_enough_info",
  "detected_error_type": "none|arithmetic_error|sign_error|conceptual_error|illegible|missing_response|other",
  "matched_expected_steps": ["str"],
  "missing_steps": ["str"],
  "confidence": 0.0,
  "reasoning_summary": "str"
}}
"""
        return self._generate_structured(prompt=prompt, schema=GeminiAssessment, temperature=0.1, disable_thinking=True)

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
        _course_label = {
            "1o_bachillerato": "1o Bachillerato",
            "2o_bachillerato": "2o Bachillerato",
        }.get(course_level or "", course_level or "Bachillerato (curso desconocido)")

        _curriculum_1 = """Curriculum 1o Bachillerato: funciones elementales y sus graficas,
limites y continuidad (definicion, calculo algebraico), derivadas (definicion, reglas basicas,
aplicaciones: tangente, monotonia, extremos), trigonometria (identidades, ecuaciones),
numeros complejos, estadistica descriptiva (media, varianza, desviacion tipica),
probabilidad (regla Laplace, Bayes, distribuciones binomial y normal)."""

        _curriculum_2 = """Curriculum 2o Bachillerato: derivadas avanzadas (regla cadena/producto/cociente),
integrales (por partes, sustitucion trigonometrica/racional), limites (L'Hopital, infinitesimos),
matrices y determinantes, sistemas lineales (Gauss, Cramer, Rouche-Frobenius),
estadistica inferencial (distribucion normal, intervalos de confianza, contraste de hipotesis),
geometria analitica 3D (rectas, planos, distancias), secciones conicas (circunferencia, elipse,
hiperbola, parabola), funciones y continuidad avanzada."""

        _curriculum = _curriculum_1 if course_level == "1o_bachillerato" else _curriculum_2

        if read_from_image and image_paths:
            prompt = f"""
Eres un experto en Matematicas de {_course_label} (Espana).
Se adjuntan las imagenes de las paginas de un examen impreso. El alumno ha escrito sus respuestas a mano — IGNORALAS completamente.
IMPORTANTE: Responde SIEMPRE en español. Todos los campos de texto (expected_steps, notes, topic, incidents) deben estar en español.

{_curriculum}

Tu tarea: localiza en las imagenes el ejercicio {question_id}, apartado {part_id}, lee su enunciado impreso/tipografiado y resuelvelo.

Instrucciones:
1. Lee el enunciado directamente de la imagen (puede haber formulas, graficas o tablas).
2. Resuelve paso a paso mostrando los pasos clave del procedimiento.
3. Indica la respuesta final exacta en solved_final_answer (usa notacion matematica de texto:
   "x=2", "f'(x)=2x+1", "integral = x^3/3 + C", "lim = 1", "[[1,2],[3,4]]" para matrices).
4. En accepted_equivalents incluye formas equivalentes habituales de la misma respuesta.
5. En expected_steps lista los pasos intermedios principales que un alumno correcto mostraria (en español).
6. confidence: certeza de que la solucion es correcta (0.0-1.0).
7. Si no encuentras el ejercicio o no puedes resolver con fiabilidad, indica can_solve=false.

Devuelve SOLO estas claves:
{{
  "can_solve": true,
  "confidence": 0.0,
  "solved_final_answer": "str",
  "accepted_equivalents": ["str"],
  "expected_steps": ["str"],
  "topic": "str|null",
  "notes": "str",
  "incidents": ["str"]
}}
"""
        else:
            prompt = f"""
Resuelve un apartado de Matematicas de {_course_label} (Espana) SOLO a partir del enunciado.
Si el enunciado no permite resolver con fiabilidad, indica can_solve=false.
No inventes datos no presentes en el enunciado.
IMPORTANTE: Responde SIEMPRE en español. Todos los campos de texto (expected_steps, notes, topic, incidents) deben estar en español.

{_curriculum}

Contexto:
- exercise: {question_id}
- part: {part_id}
- exam_model: {exam_model or "no_indicado"}
- course_level: {_course_label}

Enunciado:
{question_statement}

Instrucciones:
1. Resuelve paso a paso mostrando los pasos clave del procedimiento.
2. Indica la respuesta final exacta en solved_final_answer (usa notacion matematica de texto:
   "x=2", "f'(x)=2x+1", "integral = x^3/3 + C", "lim = 1", "[[1,2],[3,4]]" para matrices).
3. En accepted_equivalents incluye formas equivalentes habituales de la misma respuesta
   (p.ej. "2x" y "2*x", o "x^2-1" y "(x-1)(x+1)").
4. En expected_steps lista los pasos intermedios principales que un alumno correcto mostraria (en español).
5. confidence: certeza de que la solucion es correcta (0.0-1.0).

Devuelve SOLO estas claves:
{{
  "can_solve": true,
  "confidence": 0.0,
  "solved_final_answer": "str",
  "accepted_equivalents": ["str"],
  "expected_steps": ["str"],
  "topic": "str|null",
  "notes": "str",
  "incidents": ["str"]
}}
"""
        preferred_model = self.solver_model or self.model
        try:
            return self._generate_structured(
                prompt=prompt,
                schema=GeminiSolvedExercise,
                image_paths=image_paths,
                temperature=0.1,
                model_override=preferred_model,
            )
        except Exception as exc:
            if preferred_model != self.model:
                logger.warning(
                    (
                        f"Fallo en modelo solver {preferred_model} para {question_id}.{part_id}; "
                        f"fallback a {self.model}. Error: {exc}"
                    )
                )
                return self._generate_structured(
                    prompt=prompt,
                    schema=GeminiSolvedExercise,
                    image_paths=image_paths,
                    temperature=0.1,
                    model_override=self.model,
                )
            raise

    def generate_feedback_explanation(
        self,
        student_answer: str,
        expected_answer: str,
        reasoning_summary: str,
        status: str,
        awarded_points: float,
        max_points: float,
    ) -> str:
        prompt = f"""
Redacta una explicacion de correccion en espanol, clara y breve (2-4 frases), para un alumno de 2o Bachillerato.

Datos:
- Estado final: {status}
- Puntuacion: {awarded_points:.2f}/{max_points:.2f}
- Respuesta detectada: {student_answer or "sin respuesta legible"}
- Solucion esperada: {expected_answer or "no especificada"}
- Resumen tecnico: {reasoning_summary or "sin resumen"}

Requisitos:
- No uses lenguaje ofensivo.
- Explica por que obtiene esa puntuacion.
- Si procede, sugiere revision manual.
"""
        return self._generate_text(prompt=prompt, temperature=0.2, disable_thinking=True)

    def extract_teacher_solutions_from_pages(
        self,
        image_paths: list[Path],
    ) -> TeacherSolutionsExtraction:
        """Lee las soluciones correctas de las páginas del PDF del profesor."""
        prompt = """Eres un experto en matemáticas de Bachillerato (España).
Las imágenes adjuntas son páginas de un examen RESUELTO por el profesor.
La primera página puede ser una carátula (título, datos del alumno, etc.) — ignórala y no extraigas datos de ella.

Tu tarea es extraer:
1. Las respuestas correctas finales de cada ejercicio y apartado.
2. La puntuación máxima de cada pregunta (suele aparecer entre paréntesis junto al enunciado: "1. (3 puntos)", "2. (2,5p)", etc.).
3. Los criterios de evaluación GENERALES que aparezcan en el examen (rúbricas generales tipo "se valorará todo lo escrito", "cada error resta calificación", etc.).
4. Las INDICACIONES DE PUNTUACIÓN POR APARTADO: si el examen incluye una sección tipo "Indicaciones para la puntuación de cada apartado" con instrucciones específicas de cómo puntuar cada ejercicio/apartado (ej: "Análisis correcto de la continuidad: 0,25 puntos"), extráelas y asócialas al apartado correspondiente.

Instrucciones para las respuestas:
- Identifica cada ejercicio (1, 2, 3...) y sus apartados (a, b, c... o "single" si no hay).
- Para cada apartado extrae la respuesta final en texto matemático legible.
  Usa notación de texto: "x=2", "f'(x)=2x+1", "I=x^3/3+C", "lim=+inf", "[[1,2],[3,4]]".
- Incluye solo el resultado final, no el desarrollo.
- Si un apartado no tiene respuesta visible, omítelo.
- NO inventes respuestas. Si no ves la solución claramente, omite ese apartado.

Instrucciones para las puntuaciones:
- Hay DOS campos de puntuación: question_max_points (total de la pregunta) y part_max_points (puntos de este apartado concreto).
- question_max_points: puntuación TOTAL de la pregunta completa (ej: "1. (3 puntos)" → 3.0). Si la pregunta no tiene puntuación global visible, usa null. Incluye el mismo valor en TODOS los apartados de la misma pregunta.
- part_max_points: puntuación de ESTE APARTADO ESPECÍFICO. Si cada apartado tiene sus propios puntos indicados (ej: "a. (0,75 puntos)", "b. (0,5 puntos)"), extrae ese valor aquí. Si no hay puntos específicos por apartado, usa null.
- IMPORTANTE: Muchos exámenes ponen los puntos JUNTO A CADA APARTADO (ej: "a. (0,75 puntos)"). En ese caso, part_max_points debe tener el valor de ese apartado (0.75), y question_max_points debe ser la SUMA de todos los apartados de esa pregunta (ej: 0.75+0.75+0.5=2.0).
- Si solo hay puntos a nivel de pregunta (sin desglose por apartado), usa question_max_points y deja part_max_points como null.

Instrucciones para las indicaciones de puntuación por apartado:
- Busca secciones como "Indicaciones para la puntuación de cada apartado", "Rúbrica por apartado", "Criterios de puntuación por ejercicio" o tablas similares.
- Si existen, extrae las indicaciones COMPLETAS de cada apartado y ponlas en el campo "scoring_instructions" del apartado correspondiente.
- Incluye TODO el texto de las indicaciones para ese apartado, tal cual aparece (ej: "Análisis correcto de la continuidad de cada rama: 0,25 puntos\\nAnálisis correcto de la continuidad en x=0: 0,25 puntos\\nConclusión correcta: 0,25 puntos").
- Si un ejercicio no tiene apartados (es "single"), pon las indicaciones del ejercicio completo en scoring_instructions.
- Si no hay indicaciones específicas para un apartado, deja scoring_instructions vacío ("").

Devuelve JSON con esta estructura exacta:
{
  "solutions": [
    {"question_id": "1", "part_id": "a", "answer": "x=3", "question_max_points": 2.0, "part_max_points": 0.75, "scoring_instructions": "Análisis correcto: 0,5 puntos\\nConclusión: 0,25 puntos", "notes": ""},
    {"question_id": "1", "part_id": "b", "answer": "-3/2", "question_max_points": 2.0, "part_max_points": 0.75, "scoring_instructions": "", "notes": ""},
    {"question_id": "1", "part_id": "c", "answer": "y=2x+1", "question_max_points": 2.0, "part_max_points": 0.5, "scoring_instructions": "", "notes": ""},
    {"question_id": "2", "part_id": "single", "answer": "12", "question_max_points": 2.5, "part_max_points": null, "scoring_instructions": "Plantea bien la condición: 0,5 puntos\\nResuelve correctamente: 1 punto", "notes": ""}
  ],
  "evaluation_criteria": "Criterios generales de corrección: A. Se valorará todo lo escrito...",
  "notes": ""
}
"""
        return self._generate_structured(
            prompt=prompt,
            schema=TeacherSolutionsExtraction,
            image_paths=image_paths,
            temperature=0.1,
        )

    def extract_answer_from_solution_image(
        self,
        image_path: Path,
        question_statement: str | None = None,
        part_statement: str | None = None,
    ) -> str:
        """Extrae la respuesta final de una imagen de solución subida por el profesor."""
        context = ""
        if question_statement:
            context += f"\nEnunciado del ejercicio: {question_statement}"
        if part_statement:
            context += f"\nEnunciado del apartado: {part_statement}"
        prompt = f"""Eres un experto en matemáticas de Bachillerato II (España).
Esta imagen contiene la SOLUCIÓN CORRECTA de un ejercicio de examen escrita o impresa por el profesor.{context}

Tarea: extrae la respuesta final de esta solución como texto matemático legible.
- Usa notación de texto: "x^2 + 3x - 2", "f'(x) = 2x+1", "x = 3", "I = x^3/3 + C", etc.
- Incluye solo la respuesta final (resultado), no el desarrollo completo.
- Si hay varias partes/apartados, extrae la respuesta de cada una separada por " | ".
- Si la imagen no contiene una solución matemática clara, responde: "NO_LEGIBLE".
- Responde SOLO con la respuesta matemática, sin explicaciones adicionales.
"""
        self._respect_rate_limit()
        config = self._types.GenerateContentConfig(temperature=0.1)
        contents = self._build_contents(prompt, [image_path] if image_path else None)
        response = self._client.models.generate_content(
            model=self.solver_model,
            contents=contents,
            config=config,
        )
        self._last_request_ts = time.monotonic()
        text = self._extract_text_fallback(response).strip()
        return text if text else "NO_LEGIBLE"
