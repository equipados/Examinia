from __future__ import annotations

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

_executor = ThreadPoolExecutor(max_workers=2)
_solving_sessions: set[int] = set()  # guard against concurrent AI-solve runs per session
_teacher_sessions: set[int] = set()  # guard against concurrent teacher-extraction runs per session

_SOLVER_GEMINI_MODELS = {
    "gemini-flash": "gemini-2.5-flash",
    "gemini-pro":   "gemini-2.5-pro",
}


def _make_solver(solver_provider: str | None, cfg):
    """Crea el solver adecuado según solver_provider de la sesión."""
    provider = solver_provider or "gemini-pro"
    if provider.startswith("openai"):
        from openai_solver import OpenAISolver
        model_map = {
            "openai-gpt4o":  "gpt-4o",
            "openai-o4mini": "o4-mini",
        }
        model = model_map.get(provider, "gpt-4o")
        return OpenAISolver(model=model, max_retries=cfg.max_retries)
    elif provider.startswith("deepseek"):
        from openai_solver import OpenAISolver
        model_map = {
            "deepseek-v3": "deepseek-chat",
            "deepseek-r1": "deepseek-reasoner",
        }
        model = model_map.get(provider, "deepseek-chat")
        return OpenAISolver(
            model=model,
            max_retries=cfg.max_retries,
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
        )
    else:
        from gemini_client import GeminiClient
        solver_model = _SOLVER_GEMINI_MODELS.get(provider, cfg.gemini_solver_model)
        return GeminiClient(
            model=cfg.gemini_model,
            solver_model=solver_model,
            max_retries=cfg.max_retries,
            rate_limit_seconds=cfg.rate_limit_seconds,
            request_timeout_seconds=cfg.request_timeout_seconds,
        )


def _apply_session_max_points(submission, session_max: float) -> None:
    """Escala o rellena los max_points de cada parte para que el total sea session_max."""
    from utils import round_points

    all_parts = [part for q in submission.questions for part in q.parts]
    if not all_parts:
        return

    current_total = sum(p.max_points or 0.0 for p in all_parts)

    # Si hay max_points a nivel de pregunta (extraídos del enunciado) pero las partes tienen 0,
    # distribuir los puntos de pregunta entre sus partes antes de decidir si hacer reparto equitativo
    question_total = sum(q.max_points or 0.0 for q in submission.questions)
    if current_total == 0.0 and question_total > 0.01:
        for q in submission.questions:
            if q.max_points and q.max_points > 0:
                n = len(q.parts)
                part_running = 0.0
                for pi, p in enumerate(q.parts):
                    if pi == n - 1:
                        p.max_points = round_points(q.max_points - part_running)
                    else:
                        share = round_points(q.max_points / n)
                        p.max_points = share
                        part_running += share
        current_total = sum(p.max_points or 0.0 for p in all_parts)

    zero_parts = [p for p in all_parts if (p.max_points or 0.0) == 0.0]

    if current_total == 0.0 or zero_parts:
        # Sin puntuación en alguna(s) parte(s): reparto equitativo entre todas
        n_questions = len(submission.questions)
        pts_per_question = round_points(session_max / n_questions)
        running_total = 0.0
        for qi, q in enumerate(submission.questions):
            n_parts = len(q.parts)
            # Última pregunta recibe el resto para evitar errores de redondeo acumulados
            if qi == n_questions - 1:
                q_pts = round_points(session_max - running_total)
            else:
                q_pts = pts_per_question
            running_total += q_pts
            pts_per_part = round_points(q_pts / n_parts)
            part_running = 0.0
            for pi, p in enumerate(q.parts):
                if pi == n_parts - 1:
                    p.max_points = round_points(q_pts - part_running)
                else:
                    p.max_points = pts_per_part
                    part_running += pts_per_part
            q.max_points = round_points(sum(p.max_points for p in q.parts))
    elif abs(current_total - session_max) > 0.01:
        # Puntuación extraída pero con escala diferente: escalar proporcionalmente
        scale = session_max / current_total
        for p in all_parts:
            p.max_points = round_points((p.max_points or 0.0) * scale)
        for q in submission.questions:
            q.max_points = round_points(sum(p.max_points for p in q.parts))


def _save_token_usage(db, gemini_client, operation: str, session_id: int | None, submission_id: int | None) -> None:
    """Reemplaza (no acumula) el uso de tokens para esta operación+sesión/submission."""
    from app.db_models import TokenUsage
    usage = gemini_client.reset_usage()
    try:
        # Borrar filas anteriores de la misma operación para no acumular re-ejecuciones
        q = db.query(TokenUsage).filter(TokenUsage.operation == operation)
        if submission_id is not None:
            q = q.filter(TokenUsage.submission_id == submission_id)
        elif session_id is not None:
            q = q.filter(TokenUsage.session_id == session_id)
        q.delete()
        for model_name, counts in usage.items():
            if counts.get("total", 0) == 0 and counts.get("calls", 0) == 0:
                continue
            db.add(TokenUsage(
                session_id=session_id,
                submission_id=submission_id,
                operation=operation,
                model=model_name,
                input_tokens=counts.get("input", 0),
                output_tokens=counts.get("output", 0),
                thinking_tokens=counts.get("thinking", 0),
                total_tokens=counts.get("total", 0),
                api_calls=counts.get("calls", 0),
            ))
        db.commit()
    except Exception as e:
        logger.warning(f"No se pudo guardar token_usage: {e}")
        db.rollback()


def _make_web_log_sink(obj, log_attr: str, db, skip_prefix: str):
    """
    Devuelve un sink de loguru que escribe logs INFO+ del hilo actual en
    obj.<log_attr> (JSON array), saltándose mensajes ya gestionados por _step().
    """
    thread_id = threading.current_thread().ident

    def sink(message):
        record = message.record
        if record["thread"].id != thread_id:
            return
        if record["level"].no < 20:
            return
        msg_text = record["message"]
        if skip_prefix in msg_text:
            return
        if obj is None:
            return
        try:
            entries = json.loads(getattr(obj, log_attr) or "[]")
        except Exception:
            entries = []
        from datetime import datetime as _dt
        entries.append({"t": _dt.now().strftime("%H:%M:%S"), "msg": msg_text})
        setattr(obj, log_attr, json.dumps(entries[-100:], ensure_ascii=False))
        try:
            db.commit()
        except Exception:
            pass

    return sink


class _CachingGeminiWrapper:
    """Envuelve GeminiClient interceptando solve_math_question para reutilizar soluciones por convocatoria."""

    def __init__(self, real_client, session_id: int, db) -> None:
        self._client = real_client
        self._session_id = session_id
        self._db = db
        self._cache: dict[tuple[str, str], object] = self._load_cache()

    def _load_cache(self) -> dict:
        from app.db_models import SessionSolution
        from models import GeminiSolvedExercise
        cache = {}
        rows = self._db.query(SessionSolution).filter(
            SessionSolution.session_id == self._session_id
        ).all()
        for row in rows:
            try:
                # Las soluciones validadas por el profesor tienen máxima prioridad
                if row.status in ("validated", "manual") and row.final_answer:
                    solved = GeminiSolvedExercise(
                        can_solve=True,
                        confidence=1.0,
                        solved_final_answer=row.final_answer,
                        accepted_equivalents=[],
                        expected_steps=[],
                        topic=None,
                        notes="Solución validada por el profesor",
                        incidents=[],
                    )
                    cache[(row.question_id, row.part_id)] = solved
                elif row.solved_json and row.status == "ai_solved":
                    solved = GeminiSolvedExercise.model_validate_json(row.solved_json)
                    cache[(row.question_id, row.part_id)] = solved
            except Exception:
                pass
        logger.debug(f"Caché de soluciones cargada: {len(cache)} entradas para sesión {self._session_id}")
        return cache

    def solve_math_question(self, *, question_statement: str, question_id: str, part_id: str,
                            exam_model: str | None = None, course_level: str | None = None,
                            image_paths=None, read_from_image: bool = False):
        key = (question_id, part_id)
        if key in self._cache:
            logger.debug(f"      [{question_id}.{part_id}] Usando solución IA cacheada (sin llamada API)")
            return self._cache[key]

        # Usar _solver independiente si está configurado (puede ser OpenAI o Gemini)
        _active_solver = getattr(self, "_solver", None) or self._client
        logger.debug(f"      [{question_id}.{part_id}] Resolviendo con IA (primera vez en esta convocatoria)...")
        result = _active_solver.solve_math_question(
            question_statement=question_statement,
            question_id=question_id,
            part_id=part_id,
            exam_model=exam_model,
            course_level=course_level,
            image_paths=image_paths,
            read_from_image=read_from_image,
        )
        if result.can_solve:
            self._cache[key] = result
            self._save_to_db(question_id, part_id, result)
        return result

    def _save_to_db(self, question_id: str, part_id: str, solved) -> None:
        from app.db_models import SessionSolution
        existing = self._db.query(SessionSolution).filter(
            SessionSolution.session_id == self._session_id,
            SessionSolution.question_id == question_id,
            SessionSolution.part_id == part_id,
        ).first()
        if not existing:
            row = SessionSolution(
                session_id=self._session_id,
                question_id=question_id,
                part_id=part_id,
                solved_json=solved.model_dump_json(),
            )
            self._db.add(row)
            self._db.commit()

    # Delegate assess/feedback to solver if it supports them, else fall back to Gemini
    def assess_math_answer(self, *args, **kwargs):
        _active_solver = getattr(self, "_solver", None)
        if _active_solver is not None and _active_solver is not self._client and hasattr(_active_solver, "assess_math_answer"):
            return _active_solver.assess_math_answer(*args, **kwargs)
        return self._client.assess_math_answer(*args, **kwargs)

    def generate_feedback_explanation(self, *args, **kwargs):
        _active_solver = getattr(self, "_solver", None)
        if _active_solver is not None and _active_solver is not self._client and hasattr(_active_solver, "generate_feedback_explanation"):
            return _active_solver.generate_feedback_explanation(*args, **kwargs)
        return self._client.generate_feedback_explanation(*args, **kwargs)

    def extract_exam_questions(self, *args, **kwargs):
        return self._client.extract_exam_questions(*args, **kwargs)

    def extract_student_answers(self, *args, **kwargs):
        return self._client.extract_student_answers(*args, **kwargs)


def _run_pipeline(submission_id: int, db_path: str, upload_dir: str, config_overrides: dict) -> None:
    """Runs in a background thread. Creates its own DB session."""
    from app.database import init_db, get_db
    from app.db_models import Submission, QuestionResult, PartResult

    init_db(Path(db_path))
    db = next(get_db())

    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if submission is None:
        logger.error(f"Submission {submission_id} no encontrada en BD")
        return

    submission.status = "processing"
    submission.processing_log = "[]"
    db.commit()

    _sub_prefix = f"[Sub {submission_id}]"

    def _step(msg: str) -> None:
        from datetime import datetime as _dt
        try:
            entries = json.loads(submission.processing_log or "[]")
        except Exception:
            entries = []
        entries.append({"t": _dt.now().strftime("%H:%M:%S"), "msg": msg})
        submission.processing_log = json.dumps(entries[-100:], ensure_ascii=False)
        try:
            db.commit()
        except Exception:
            pass
        logger.info(f"{_sub_prefix} {msg}")

    _sink_id = logger.add(
        _make_web_log_sink(submission, "processing_log", db, skip_prefix=_sub_prefix),
        level="INFO",
        format="{message}",
    )

    _step(f"Iniciando corrección: {submission.source_filename}")

    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()

        from config import AppConfig
        from gemini_client import GeminiClient
        from pdf_processor import convert_pdf_to_images
        from image_preprocessing import preprocess_image
        from exam_parser import analyze_pages_with_gemini, build_submission_from_pdf, normalize_submission_structure
        from grading import SolutionBank, grade_exam
        from reporting import write_exam_report

        cfg = AppConfig(
            output_dir=Path(upload_dir) / "salidas",
            web_upload_dir=Path(upload_dir),
            **config_overrides,
        )

        pdf_path = Path(submission.pdf_path)
        temp_dir = Path(upload_dir) / ".tmp" / f"sub_{submission_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 1. PDF → images
        _step(f"Convirtiendo PDF a imágenes ({pdf_path.name})...")
        poppler_path = cfg.poppler_path
        page_images = convert_pdf_to_images(pdf_path, temp_dir, dpi=cfg.dpi, poppler_path=poppler_path)
        _step(f"{len(page_images)} página(s) convertidas. Analizando con {cfg.gemini_model}...")

        # 2. Optional preprocessing
        if cfg.preprocess_images:
            for pi in page_images:
                try:
                    preprocess_image(pi.image_path)
                except Exception:
                    pass

        # 3. Page extraction
        real_gemini = GeminiClient(
            model=cfg.gemini_model,
            solver_model=cfg.gemini_solver_model,
            max_retries=cfg.max_retries,
            rate_limit_seconds=cfg.rate_limit_seconds,
            request_timeout_seconds=cfg.request_timeout_seconds,
        )
        # Wrap with solution cache for this session
        gemini = _CachingGeminiWrapper(real_gemini, submission.session_id, db)

        page_extractions = analyze_pages_with_gemini(
            page_images=page_images,
            gemini_client=real_gemini,
            reanalysis_threshold=cfg.reanalysis_threshold,
            progress_callback=lambda cur, tot: _step(
                f"Página {cur}/{tot} analizada" + (" ✓" if cur == tot else "...")
            ),
        )

        # 4. Build submission
        exam_submission = build_submission_from_pdf(page_extractions, pdf_path.name)
        exam_submission = normalize_submission_structure(exam_submission)
        total_parts = sum(len(q.parts) for q in exam_submission.questions)
        _step(f"Alumno: {exam_submission.student_name or '?'} · Modelo: {exam_submission.exam_model or '?'} · {len(exam_submission.questions)} ejercicio(s), {total_parts} apartado(s)")
        # Log puntos extraídos del PDF del alumno (diagnóstico)
        pts_debug = " | ".join(
            f"Ej{q.question_id}={q.max_points}p ({len(q.parts)} apt)"
            for q in exam_submission.questions
        )
        _step(f"Puntos extraídos del PDF: {pts_debug or 'ninguno'}")

        # Cargar soluciones validadas de la convocatoria (necesarias para puntos y SolutionBank)
        from app.db_models import ExamSession, SessionSolution as _SS
        from models import SolutionTemplate as _ST
        session_obj = db.get(ExamSession, submission.session_id)
        sess_sols = db.query(_SS).filter(
            _SS.session_id == submission.session_id,
            _SS.status.in_(["validated", "manual"]),
            _SS.final_answer.isnot(None),
        ).all()

        # Aplicar puntuaciones: prioridad 1=PDF profesor, 2=PDF alumno, 3=reparto equitativo
        from utils import round_points as _rp
        sol_pts = {(s.question_id, s.part_id): s.max_points for s in sess_sols if s.max_points}

        if sol_pts:
            # Prioridad 1: puntos extraídos del PDF del profesor
            for q in exam_submission.questions:
                for p in q.parts:
                    key = (q.question_id, p.part_id)
                    if key in sol_pts:
                        p.max_points = sol_pts[key]
                q.max_points = _rp(sum(p.max_points or 0.0 for p in q.parts))
            total_extracted = sum(q.max_points or 0.0 for q in exam_submission.questions)
            _step(f"Puntuaciones del PDF del profesor aplicadas: {total_extracted} pts en total.")
            # Eliminar incidencias de reparto equitativo: ya no aplican al usar puntos del profesor
            exam_submission.incidents = [
                inc for inc in exam_submission.incidents
                if not inc.startswith("Reparto equitativo")
            ]
        else:
            # Prioridad 2: puntos ya extraídos del PDF del alumno (por el parser)
            current_parts_total = sum(p.max_points or 0.0 for q in exam_submission.questions for p in q.parts)
            if current_parts_total > 0.01:
                _step(f"Puntuaciones extraídas del PDF del alumno: {current_parts_total} pts.")
            elif session_obj and session_obj.max_total_points:
                # Prioridad 3: reparto equitativo si no hay puntos en ningún lado
                _apply_session_max_points(exam_submission, session_obj.max_total_points)
                _step(f"Puntuaciones repartidas equitativamente a {session_obj.max_total_points} pts.")

        # Escalar si max_total_points difiere del total actual
        if session_obj and session_obj.max_total_points:
            current = sum(q.max_points or 0.0 for q in exam_submission.questions)
            if current > 0.01 and abs(current - session_obj.max_total_points) > 0.05:
                _apply_session_max_points(exam_submission, session_obj.max_total_points)
                _step(f"Puntuaciones escaladas de {current:.2f} a {session_obj.max_total_points} pts.")

        # 5. Grade (uses caching wrapper for solver calls)
        # Usar el solver configurado en la convocatoria (Gemini o OpenAI)
        _solver_provider = session_obj.solver_provider if session_obj else None
        _solver = _make_solver(_solver_provider, cfg)
        gemini._solver = _solver  # solver independiente (puede ser OpenAI o Gemini)
        _solver_label = getattr(_solver, "model", _solver_provider or cfg.gemini_solver_model)
        _step(f"Corrigiendo respuestas con {cfg.gemini_model} (assess) + {_solver_label} (solver)...")

        bank = SolutionBank([
            _ST(
                exercise=s.question_id,
                part=s.part_id,
                expected_final_answer=s.final_answer,
                max_points=s.max_points if s.max_points else 0.0,
            )
            for s in sess_sols
        ])
        # Criterios de evaluación: instrucciones del profesor + criterios extraídos del PDF
        _eval_criteria = next((s.evaluation_criteria for s in sess_sols if s.evaluation_criteria), None)
        _grading_instr = session_obj.grading_instructions if session_obj else None
        if _grading_instr:
            parts = []
            parts.append(f"INSTRUCCIONES DEL PROFESOR:\n{_grading_instr}")
            if _eval_criteria:
                parts.append(f"CRITERIOS EXTRAÍDOS DEL EXAMEN:\n{_eval_criteria}")
            _eval_criteria = "\n\n".join(parts)
        reports_dir = Path(upload_dir) / "informes"
        reports_dir.mkdir(parents=True, exist_ok=True)

        result = grade_exam(
            submission=exam_submission,
            solution_bank=bank,
            gemini_client=gemini,
            low_confidence_threshold=cfg.low_confidence_threshold,
            strict_mode=cfg.strict_mode,
            allow_ai_solver=cfg.enable_ai_solver,
            ai_solver_min_confidence=cfg.ai_solver_min_confidence,
            evaluation_criteria=_eval_criteria,
        )

        # 6. Write report
        seen: dict[str, int] = {}
        report_path = write_exam_report(result, reports_dir, seen)
        result.report_path = str(report_path)

        # 7. Persist results
        for qr in submission.question_results:
            db.delete(qr)
        db.flush()

        for q in result.questions:
            qr = QuestionResult(
                submission_id=submission_id,
                question_id=q.question_id,
                max_points=q.max_points,
            )
            db.add(qr)
            db.flush()
            for p in q.parts:
                pr = PartResult(
                    question_id_fk=qr.id,
                    part_id=p.part_id,
                    column_id=p.column_id,
                    awarded_points=p.awarded_points,
                    max_points=p.max_points,
                    status=p.status,
                    explanation=p.explanation,
                    detected_answer=p.detected_answer,
                    incidents=json.dumps(p.incidents, ensure_ascii=False),
                )
                db.add(pr)

        submission.student_name = result.student_name
        submission.exam_model = result.exam_model
        submission.course_level = result.course_level
        submission.total_points = result.total_points
        submission.max_total_points = result.max_total_points
        submission.incidents = json.dumps(result.incidents, ensure_ascii=False)
        submission.report_path = result.report_path
        submission.status = "done"
        submission.processed_at = datetime.now(timezone.utc)
        _link_student(db, submission)
        db.commit()
        _save_token_usage(db, real_gemini, "grade_submission", submission.session_id, submission_id)
        # Guardar también tokens del solver externo (OpenAI/DeepSeek) si los usó
        if _solver is not real_gemini and hasattr(_solver, "reset_usage"):
            _save_token_usage(db, _solver, "grade_submission_solver", submission.session_id, submission_id)
        _step(f"✓ Completado: {result.student_name or '?'} — {result.total_points:.2f}/{result.max_total_points:.2f} pts")

        # Actualizar snapshot de historial tras cada corrección
        try:
            from app.db_models import ExamSession as _ES, SessionHistory as _SH
            from app.routers.sessions import _snapshot_session
            sess = db.get(_ES, submission.session_id)
            if sess:
                _snapshot_session(sess, db)
        except Exception:
            pass

        # Comprobar si toda la convocatoria ha terminado → enviar email
        try:
            from app.db_models import ExamSession as _ES2, User as _U
            sess2 = db.get(_ES2, submission.session_id)
            if sess2 and sess2.send_email_on_completion and sess2.created_by_user_id:
                remaining = db.query(Submission).filter(
                    Submission.session_id == submission.session_id,
                    Submission.status.notin_(["done", "error"]),
                ).count()
                if remaining == 0:
                    user = db.query(_U).filter(_U.id == sess2.created_by_user_id).first()
                    if user and user.email:
                        all_subs = db.query(Submission).filter(
                            Submission.session_id == submission.session_id,
                        ).all()
                        results = [
                            {
                                "student_name": s.student_name or s.source_filename,
                                "total": s.total_points or 0,
                                "max": s.max_total_points or 0,
                                "status": s.status,
                            }
                            for s in all_subs
                        ]
                        from app.email_service import send_session_completion_email
                        send_session_completion_email(user.email, sess2.name, results, teacher_name=user.display_name)
        except Exception:
            pass  # Email failure must never affect grading

    except Exception:
        tb = traceback.format_exc()
        logger.error(f"[Sub {submission_id}] ERROR:\n{tb}")
        try:
            _step(f"ERROR: {tb.splitlines()[-1][:120]}")
        except Exception:
            pass
        submission = db.query(Submission).filter(Submission.id == submission_id).first()
        if submission:
            submission.status = "error"
            submission.error_message = tb[-2000:]
            submission.processed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        try:
            logger.remove(_sink_id)
        except Exception:
            pass
        db.close()


def enqueue(submission_id: int, db_path: str, upload_dir: str, config_overrides: dict | None = None) -> None:
    _executor.submit(_run_pipeline, submission_id, db_path, upload_dir, config_overrides or {})


def solve_questions_for_session(session_id: int, db_path: str, config_overrides: dict | None = None) -> None:
    """Tarea background: extrae preguntas del primer PDF de la sesión y las resuelve con IA."""
    _executor.submit(_run_solve_questions, session_id, db_path, config_overrides or {})


def _run_solve_questions(session_id: int, db_path: str, config_overrides: dict) -> None:
    if session_id in _solving_sessions:
        logger.warning(f"[Sesión {session_id}] Resolución ya en curso, ignorando solicitud duplicada.")
        return
    _solving_sessions.add(session_id)

    from app.database import init_db, get_db
    from app.db_models import ExamSession, SessionSolution, Submission

    init_db(Path(db_path))
    db = next(get_db())

    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()

        from config import AppConfig
        from gemini_client import GeminiClient
        from pdf_processor import convert_pdf_to_images
        from image_preprocessing import preprocess_image
        from exam_parser import analyze_pages_with_gemini, build_submission_from_pdf, normalize_submission_structure

        cfg = AppConfig(**config_overrides)

        # Buscar el PDF de referencia o el primero disponible
        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        ref_sub = db.query(Submission).filter(
            Submission.session_id == session_id,
            Submission.pdf_path.isnot(None),
        ).order_by(Submission.created_at).first()

        def _step(msg: str) -> None:
            """Actualiza el log acumulativo en la BD para mostrarlo en la UI."""
            from datetime import datetime as _dt
            if session:
                session.current_step = msg
                # Acumular en session_log (JSON array, máx 50 entradas)
                try:
                    entries = json.loads(session.session_log or "[]")
                except Exception:
                    entries = []
                entries.append({"t": _dt.now().strftime("%H:%M:%S"), "msg": msg})
                session.session_log = json.dumps(entries[-100:], ensure_ascii=False)
                db.commit()
            logger.info(f"[Sesión {session_id}] {msg}")

        if ref_sub is None:
            _step("⚠ No hay PDFs disponibles para extraer preguntas.")
            return

        # Limpiar log anterior y activar sink de logs del servidor → web
        if session:
            session.session_log = "[]"
            db.commit()
        _sink_id = logger.add(
            _make_web_log_sink(session, "session_log", db, skip_prefix=f"[Sesión {session_id}]"),
            level="INFO",
            format="{message}",
        )

        _step(f"Convirtiendo PDF a imágenes: {ref_sub.source_filename}...")

        from pathlib import Path as P
        pdf_path = P(ref_sub.pdf_path)
        temp_dir = P(db_path).parent / ".tmp" / f"session_{session_id}_solve"
        temp_dir.mkdir(parents=True, exist_ok=True)

        page_images = convert_pdf_to_images(pdf_path, temp_dir, dpi=cfg.dpi, poppler_path=cfg.poppler_path)
        _step(f"PDF convertido: {len(page_images)} página(s). Analizando con IA...")

        if cfg.preprocess_images:
            for pi in page_images:
                try:
                    preprocess_image(pi.image_path)
                except Exception:
                    pass

        gemini = GeminiClient(
            model=cfg.gemini_model,
            solver_model=cfg.gemini_solver_model,
            max_retries=cfg.max_retries,
            rate_limit_seconds=cfg.rate_limit_seconds,
            request_timeout_seconds=cfg.request_timeout_seconds,
        )
        solver_provider = session.solver_provider if session else None
        solver = _make_solver(solver_provider, cfg)
        solver_label = solver_provider or "gemini-pro"

        _step(f"Analizando página 1/{len(page_images)} con Gemini...")
        page_extractions = analyze_pages_with_gemini(
            page_images=page_images,
            gemini_client=gemini,
            reanalysis_threshold=cfg.reanalysis_threshold,
            progress_callback=lambda cur, tot: _step(
                f"Página {cur}/{tot} analizada"
                + (" ✓" if cur == tot else " — analizando siguiente...")
            ),
        )

        exam_sub = build_submission_from_pdf(page_extractions, pdf_path.name)
        exam_sub = normalize_submission_structure(exam_sub)

        total_parts = sum(len(q.parts) for q in exam_sub.questions)
        _solver_model = getattr(solver, "model", solver_label)
        _step(f"Examen analizado: {len(exam_sub.questions)} ejercicio(s), {total_parts} apartado(s). Solver: {_solver_model}. Comenzando resolución...")

        course_level = exam_sub.course_level or (session.course_level if session else None)

        # Borrar soluciones anteriores (si se re-extrae)
        db.query(SessionSolution).filter(SessionSolution.session_id == session_id).delete()
        db.commit()

        # Crear filas pendientes para todas las partes primero (aparecen en la UI como "pendientes")
        all_rows: list[tuple] = []  # (row, question, part, full_statement)
        for question in exam_sub.questions:
            statement_text = (question.statement or "").strip()
            for part in question.parts:
                part_statement = (part.statement or "").strip()
                full_statement = part_statement or statement_text
                row = SessionSolution(
                    session_id=session_id,
                    question_id=question.question_id,
                    part_id=part.part_id,
                    question_statement=statement_text or None,
                    part_statement=part_statement or None,
                    status="ai_pending",
                )
                db.add(row)
                all_rows.append((row, question, part, full_statement))
        db.commit()

        # Resolver cada apartado
        solved_count = 0
        for row_id, question_id_str, part_id_str, full_statement in [
            (row.id, question.question_id, part.part_id, stmt)
            for row, question, part, stmt in all_rows
        ]:
            solved_count += 1
            pct = int(solved_count / total_parts * 100)
            _step(f"[{pct}%] Resolviendo ejercicio {question_id_str}.{part_id_str} ({solved_count}/{total_parts})...")

            # Re-fetch the row to avoid stale object issues after intermediate commits
            row = db.get(SessionSolution, row_id)
            if row is None:
                continue

            if not full_statement:
                row.status = "ai_failed"
                db.commit()
                continue

            try:
                solved = solver.solve_math_question(
                    question_statement=full_statement,
                    question_id=question_id_str,
                    part_id=part_id_str,
                    exam_model=exam_sub.exam_model,
                    course_level=course_level,
                )
                if not solved.can_solve or solved.confidence == 0.0:
                    retry_solver = gemini if solver_label.startswith("openai") else solver
                    retry_model = getattr(retry_solver, "solver_model", getattr(retry_solver, "model", "gemini"))
                    _step(f"  ↺ {question_id_str}.{part_id_str} no resuelto — reintentando con imagen ({retry_model})...")
                    page_imgs = [pi.image_path for pi in page_images]
                    solved = retry_solver.solve_math_question(
                        question_statement=full_statement,
                        question_id=question_id_str,
                        part_id=part_id_str,
                        exam_model=exam_sub.exam_model,
                        course_level=course_level,
                        image_paths=page_imgs,
                        read_from_image=True,
                    )
                row.solved_json = solved.model_dump_json()
                if solved.can_solve and solved.confidence > 0.0:
                    row.status = "ai_solved"
                    row.final_answer = solved.solved_final_answer
                    conf_pct = int(solved.confidence * 100)
                    answer_preview = (solved.solved_final_answer or "")[:60]
                    _step(f"  ✓ {question_id_str}.{part_id_str} resuelto (confianza {conf_pct}%): {answer_preview}")
                else:
                    row.status = "ai_failed"
                    _step(f"  ⚠ {question_id_str}.{part_id_str} no resuelto — se necesita solución manual")
                db.commit()
            except Exception as e:
                logger.error(f"Error resolviendo {question_id_str}.{part_id_str}: {e}")
                db.rollback()
                row = db.get(SessionSolution, row_id)
                if row:
                    row.status = "ai_failed"
                    db.commit()

        _save_token_usage(db, solver, "extract_solutions", session_id, None)
        if solver is not gemini:
            _save_token_usage(db, gemini, "extract_solutions_pages", session_id, None)
        _step(f"✓ Resolución completada: {total_parts} apartado(s) procesados. Revisa y valida las soluciones.")
    except Exception:
        import traceback
        logger.error(f"[Sesión {session_id}] ERROR en resolve_questions:\n{traceback.format_exc()}")
    finally:
        try:
            logger.remove(_sink_id)
        except Exception:
            pass
        _solving_sessions.discard(session_id)
        db.close()


def extract_teacher_solutions_for_session(
    session_id: int, teacher_pdf_path: str, db_path: str, config_overrides: dict | None = None
) -> None:
    """Tarea background: extrae preguntas y respuestas del PDF de soluciones del profesor."""
    _executor.submit(_run_extract_teacher_solutions, session_id, teacher_pdf_path, db_path, config_overrides or {})


def _run_extract_teacher_solutions(
    session_id: int, teacher_pdf_path: str, db_path: str, config_overrides: dict
) -> None:
    if session_id in _teacher_sessions:
        logger.warning(f"[Sesión {session_id}] Extracción del profesor ya en curso, ignorando solicitud duplicada.")
        return
    _teacher_sessions.add(session_id)

    from app.database import init_db, get_db
    from app.db_models import ExamSession, SessionSolution

    init_db(Path(db_path))
    db = next(get_db())

    try:
        from dotenv import load_dotenv
        load_dotenv()

        from config import AppConfig
        from gemini_client import GeminiClient
        from pdf_processor import convert_pdf_to_images

        cfg = AppConfig(**config_overrides)

        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()

        def _step(msg: str) -> None:
            from datetime import datetime as _dt
            if session:
                session.current_step = msg
                try:
                    entries = json.loads(session.session_log or "[]")
                except Exception:
                    entries = []
                entries.append({"t": _dt.now().strftime("%H:%M:%S"), "msg": msg})
                session.session_log = json.dumps(entries[-100:], ensure_ascii=False)
                db.commit()
            logger.info(f"[Sesión {session_id}] {msg}")

        # Limpiar log y marcar modo
        if session:
            session.session_log = "[]"
            session.solution_mode = "teacher"
            db.commit()

        # Activar sink de logs del servidor → web
        _sink_id = logger.add(
            _make_web_log_sink(session, "session_log", db, skip_prefix=f"[Sesión {session_id}]"),
            level="INFO",
            format="{message}",
        )

        pdf_path = Path(teacher_pdf_path)
        _step(f"Convirtiendo PDF del profesor a imágenes: {pdf_path.name}...")

        temp_dir = Path(db_path).parent / ".tmp" / f"session_{session_id}_teacher"
        temp_dir.mkdir(parents=True, exist_ok=True)

        page_images = convert_pdf_to_images(pdf_path, temp_dir, dpi=cfg.dpi, poppler_path=cfg.poppler_path)
        gemini = GeminiClient(
            model=cfg.gemini_model,
            solver_model=cfg.gemini_solver_model,
            max_retries=cfg.max_retries,
            rate_limit_seconds=cfg.rate_limit_seconds,
            request_timeout_seconds=cfg.request_timeout_seconds,
        )

        _step(f"PDF convertido: {len(page_images)} página(s). Leyendo respuestas del profesor con {gemini.model}...")
        image_paths = [pi.image_path for pi in page_images]
        extraction = gemini.extract_teacher_solutions_from_pages(image_paths)

        total_parts = len(extraction.solutions)
        _step(f"Respuestas leídas: {total_parts} apartado(s) encontrados. Guardando...")

        # Prorratear puntos por apartado: agrupar por question_id para saber cuántos apartados tiene cada pregunta
        from collections import defaultdict
        parts_per_question: dict[str, list] = defaultdict(list)
        for item in extraction.solutions:
            parts_per_question[item.question_id].append(item)

        # Calcular max_points por apartado (question_max_points / n_parts de esa pregunta)
        part_points: dict[tuple[str, str], float | None] = {}
        for qid, items in parts_per_question.items():
            q_pts = next((i.question_max_points for i in items if i.question_max_points is not None), None)
            if q_pts is not None:
                pts_each = round(q_pts / len(items), 4)
                for i in items:
                    part_points[(qid, i.part_id)] = pts_each
            else:
                for i in items:
                    part_points[(qid, i.part_id)] = None

        # Criterios de evaluación (comunes a todos los apartados de la convocatoria)
        criteria = extraction.evaluation_criteria or None
        if criteria:
            _step(f"Criterios de evaluación extraídos ({len(criteria)} caracteres).")

        # Borrar soluciones anteriores
        db.query(SessionSolution).filter(SessionSolution.session_id == session_id).delete()
        db.commit()

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        created = 0
        for item in extraction.solutions:
            has_answer = bool(item.answer)
            pts = part_points.get((item.question_id, item.part_id))
            row = SessionSolution(
                session_id=session_id,
                question_id=item.question_id,
                part_id=item.part_id,
                final_answer=item.answer or None,
                max_points=pts,
                evaluation_criteria=criteria,
                status="validated" if has_answer else "ai_failed",
                validated_at=now if has_answer else None,
            )
            db.add(row)
            created += 1
        db.commit()

        if criteria:
            pts_info = {qid: items[0].question_max_points for qid, items in parts_per_question.items() if items[0].question_max_points is not None}
            if pts_info:
                pts_str = ", ".join(f"Ej.{k}={v}p" for k, v in sorted(pts_info.items()))
                _step(f"Puntuaciones extraídas del examen: {pts_str}")

        _save_token_usage(db, gemini, "extract_teacher_pdf", session_id, None)
        _step(f"✓ Listo: {created} apartado(s) extraídos del PDF del profesor. Revisa y valida las respuestas.")
    except Exception:
        import traceback
        logger.error(f"[Sesión {session_id}] ERROR en extract_teacher_solutions:\n{traceback.format_exc()}")
    finally:
        try:
            logger.remove(_sink_id)
        except Exception:
            pass
        _teacher_sessions.discard(session_id)
        db.close()


def _link_student(db, submission) -> None:
    """Finds or creates a Student row and links submission.student_id to it."""
    from app.db_models import Student
    from utils import normalize_identifier
    name = submission.student_name
    if not name:
        return
    norm = normalize_identifier(name)
    if not norm:
        return
    course = submission.course_level
    candidates = db.query(Student).filter(Student.normalized_name == norm).all()
    if candidates:
        match = next((c for c in candidates if c.course_level == course), candidates[0])
        submission.student_id = match.id
        if course and not match.course_level:
            match.course_level = course
    else:
        s = Student(
            display_name=name.strip().title(),
            normalized_name=norm,
            course_level=course,
        )
        db.add(s)
        db.flush()
        submission.student_id = s.id


def recover_pending(db_path: str, upload_dir: str, config_overrides: dict | None = None) -> None:
    """Re-encola submissions que quedaron processing al reiniciar (solo si sus soluciones están validadas)."""
    from app.database import init_db, get_db
    from app.db_models import SessionSolution, Submission

    init_db(Path(db_path))
    db = next(get_db())
    stuck = db.query(Submission).filter(Submission.status.in_(["pending", "processing"])).all()
    if stuck:
        logger.info(f"Recuperando {len(stuck)} examen(es) pendiente(s) del arranque anterior...")
        for sub in stuck:
            sub.status = "pending"
        db.commit()

        # Solo encolar si las soluciones de la sesión están todas validadas
        processed_sessions: set[int] = set()
        for sub in stuck:
            if sub.session_id in processed_sessions:
                continue
            processed_sessions.add(sub.session_id)
            unvalidated = db.query(SessionSolution).filter(
                SessionSolution.session_id == sub.session_id,
                SessionSolution.status.notin_(["validated", "manual"]),
            ).first()
            if unvalidated is None:
                # Todas validadas (o no hay soluciones aún — encolar igualmente para no bloquear)
                session_subs = db.query(Submission).filter(
                    Submission.session_id == sub.session_id,
                    Submission.status == "pending",
                ).all()
                for s in session_subs:
                    enqueue(s.id, db_path, upload_dir, config_overrides)
            else:
                logger.info(
                    f"Sesión {sub.session_id}: soluciones pendientes de validación, "
                    "no se reencolan correcciones."
                )
    db.close()
