from __future__ import annotations

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

_executor = ThreadPoolExecutor(max_workers=2)
_solving_sessions: set[int] = set()  # guard against concurrent runs per session


def _save_token_usage(db, gemini_client, operation: str, session_id: int | None, submission_id: int | None) -> None:
    """Lee el uso acumulado en gemini_client y persiste una fila por modelo en token_usage."""
    from app.db_models import TokenUsage
    usage = gemini_client.reset_usage()
    for model_name, counts in usage.items():
        if counts.get("total", 0) == 0 and counts.get("calls", 0) == 0:
            continue
        row = TokenUsage(
            session_id=session_id,
            submission_id=submission_id,
            operation=operation,
            model=model_name,
            input_tokens=counts.get("input", 0),
            output_tokens=counts.get("output", 0),
            total_tokens=counts.get("total", 0),
            api_calls=counts.get("calls", 0),
        )
        db.add(row)
    try:
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
                            exam_model: str | None = None, course_level: str | None = None):
        key = (question_id, part_id)
        if key in self._cache:
            logger.debug(f"      [{question_id}.{part_id}] Usando solución IA cacheada (sin llamada API)")
            return self._cache[key]

        logger.debug(f"      [{question_id}.{part_id}] Resolviendo con IA (primera vez en esta convocatoria)...")
        result = self._client.solve_math_question(
            question_statement=question_statement,
            question_id=question_id,
            part_id=part_id,
            exam_model=exam_model,
            course_level=course_level,
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

    # Delegate all other methods to the real client
    def assess_math_answer(self, *args, **kwargs):
        return self._client.assess_math_answer(*args, **kwargs)

    def generate_feedback_explanation(self, *args, **kwargs):
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
        _step(f"{len(page_images)} página(s) convertidas. Analizando con Gemini...")

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

        # 5. Grade (uses caching wrapper for solver calls)
        _step("Corrigiendo respuestas con IA...")
        bank = SolutionBank([])
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
        db.commit()
        _save_token_usage(db, real_gemini, "grade_submission", submission.session_id, submission_id)
        _step(f"✓ Completado: {result.student_name or '?'} — {result.total_points:.2f}/{result.max_total_points:.2f} pts")

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
        _step(f"Examen analizado: {len(exam_sub.questions)} ejercicio(s), {total_parts} apartado(s). Comenzando resolución...")

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
                solved = gemini.solve_math_question(
                    question_statement=full_statement,
                    question_id=question_id_str,
                    part_id=part_id_str,
                    exam_model=exam_sub.exam_model,
                    course_level=course_level,
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

        _save_token_usage(db, gemini, "extract_solutions", session_id, None)
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
    if session_id in _solving_sessions:
        logger.warning(f"[Sesión {session_id}] Extracción ya en curso, ignorando solicitud duplicada.")
        return
    _solving_sessions.add(session_id)

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
        from image_preprocessing import preprocess_image
        from exam_parser import analyze_pages_with_gemini, build_submission_from_pdf, normalize_submission_structure

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

        _step(f"Analizando página 1/{len(page_images)} del PDF del profesor...")
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
        _step(f"Estructura detectada: {len(exam_sub.questions)} ejercicio(s), {total_parts} apartado(s). Guardando soluciones...")

        # Borrar soluciones anteriores
        db.query(SessionSolution).filter(SessionSolution.session_id == session_id).delete()
        db.commit()

        # Crear una solución por cada apartado encontrado en el PDF del profesor
        created = 0
        for question in exam_sub.questions:
            statement_text = (question.statement or "").strip()
            for part in question.parts:
                part_statement = (part.statement or "").strip()
                # La respuesta del "alumno" en el PDF del profesor es la respuesta correcta
                answer = (part.student_answer or "").strip()
                row = SessionSolution(
                    session_id=session_id,
                    question_id=question.question_id,
                    part_id=part.part_id,
                    question_statement=statement_text or None,
                    part_statement=part_statement or None,
                    final_answer=answer or None,
                    status="ai_solved" if answer else "ai_failed",
                )
                db.add(row)
                created += 1
        db.commit()

        _save_token_usage(db, gemini, "extract_teacher_pdf", session_id, None)
        _step(f"✓ Extracción completada: {created} apartado(s) encontrados. Revisa y valida las respuestas.")
    except Exception:
        import traceback
        logger.error(f"[Sesión {session_id}] ERROR en extract_teacher_solutions:\n{traceback.format_exc()}")
    finally:
        try:
            logger.remove(_sink_id)
        except Exception:
            pass
        _solving_sessions.discard(session_id)
        db.close()


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
