from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from app.templates import templates
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.db_models import ExamSession, SessionHistory, SessionSolution, Submission, QuestionResult, PartResult, TokenUsage, User

# Set from web.py at startup (same as submissions router)
_db_path: str = "corrector.db"
_config_overrides: dict = {}


def configure_solutions(db_path: str, config_overrides: dict) -> None:
    global _db_path, _config_overrides
    _db_path = db_path
    _config_overrides = config_overrides

router = APIRouter(prefix="/sessions")

_COURSE_LABELS = {"1o_bachillerato": "1º Bach.", "2o_bachillerato": "2º Bach."}


@router.get("/new", response_class=HTMLResponse)
def new_session_form(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    return templates.TemplateResponse("session_new.html", {"request": request, "user": current_user})


@router.post("")
def create_session(
    name: str = Form(...),
    course_level: str = Form(""),
    subject: str = Form(""),
    date: str = Form(""),
    notes: str = Form(""),
    max_total_points: str = Form(""),
    solver_provider: str = Form("gemini-pro"),
    grading_instructions: str = Form(""),
    send_email_on_completion: str = Form(""),
    weight: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    mtp = None
    try:
        mtp = float(max_total_points) if max_total_points.strip() else None
    except ValueError:
        pass
    w = 1.0
    try:
        w = float(weight) if weight.strip() else 1.0
    except ValueError:
        pass
    session = ExamSession(
        name=name,
        course_level=course_level or None,
        subject=subject or None,
        date=date or None,
        notes=notes or None,
        max_total_points=mtp,
        solver_provider=solver_provider if solver_provider in _VALID_SOLVER_PROVIDERS else "gemini-pro",
        grading_instructions=grading_instructions.strip() or None,
        send_email_on_completion=1 if send_email_on_completion else 0,
        created_by_user_id=current_user.id,
        weight=w,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return RedirectResponse(url=f"/sessions/{session.id}/upload", status_code=status.HTTP_302_FOUND)


_VALID_SOLVER_PROVIDERS = {"gemini-flash", "gemini-pro", "openai-gpt4o", "openai-o4mini", "deepseek-v3", "deepseek-r1"}


@router.post("/{session_id}/set-solver")
def set_solver(
    session_id: int,
    solver_provider: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session and solver_provider in _VALID_SOLVER_PROVIDERS:
        session.solver_provider = solver_provider
        db.commit()
    return RedirectResponse(url=f"/sessions/{session_id}", status_code=status.HTTP_302_FOUND)


@router.post("/{session_id}/set-max-points")
def set_max_points(
    session_id: int,
    max_total_points: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session:
        try:
            session.max_total_points = float(max_total_points) if max_total_points.strip() else None
        except ValueError:
            pass
        db.commit()
    return RedirectResponse(url=f"/sessions/{session_id}", status_code=status.HTTP_302_FOUND)


@router.post("/{session_id}/set-weight")
def set_weight(
    session_id: int,
    weight: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session:
        try:
            session.weight = float(weight) if weight.strip() else 1.0
        except ValueError:
            pass
        db.commit()
    return RedirectResponse(url=f"/sessions/{session_id}", status_code=status.HTTP_302_FOUND)


@router.post("/{session_id}/set-grading-instructions")
def set_grading_instructions(
    session_id: int,
    grading_instructions: str = Form(""),
    redirect_to: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session:
        session.grading_instructions = grading_instructions.strip() or None
        db.commit()
    redirect_url = redirect_to or f"/sessions/{session_id}"
    return RedirectResponse(url=redirect_url, status_code=status.HTTP_302_FOUND)


@router.get("/{session_id}", response_class=HTMLResponse)
def session_detail(
    session_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        return HTMLResponse("Convocatoria no encontrada", status_code=404)

    submissions = (
        db.query(Submission)
        .filter(Submission.session_id == session_id)
        .order_by(Submission.created_at.desc())
        .all()
    )

    # Collect unique column_ids for header
    all_columns: list[str] = []
    seen_cols: set[str] = set()
    for sub in submissions:
        for qr in sub.question_results:
            for pr in qr.part_results:
                if pr.column_id not in seen_cols:
                    seen_cols.add(pr.column_id)
                    all_columns.append(pr.column_id)

    all_columns.sort(key=_col_sort_key)

    pending = sum(1 for s in submissions if s.status in ("pending", "processing"))
    done_count = sum(1 for s in submissions if s.status == "done")
    error_count = sum(1 for s in submissions if s.status == "error")

    solutions = db.query(SessionSolution).filter(SessionSolution.session_id == session_id).all()
    solutions_total = len(solutions)
    solutions_validated = sum(1 for s in solutions if s.status in ("validated", "manual"))

    # Token usage aggregated by operation+model
    token_rows = db.query(TokenUsage).filter(TokenUsage.session_id == session_id).all()
    token_summary = _build_token_summary(token_rows)

    return templates.TemplateResponse(
        "session_detail.html",
        {
            "request": request,
            "session": session,
            "submissions": submissions,
            "all_columns": all_columns,
            "pending": pending,
            "done_count": done_count,
            "error_count": error_count,
            "solutions_total": solutions_total,
            "solutions_validated": solutions_validated,
            "solver_provider": session.solver_provider or "gemini-pro",
            "user": current_user,
            "course_labels": _COURSE_LABELS,
            "token_summary": token_summary,
        },
    )


@router.get("/{session_id}/progress")
def session_progress(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    subs = db.query(Submission).filter(Submission.session_id == session_id).all()
    return {
        "pending": sum(1 for s in subs if s.status == "pending"),
        "processing": sum(1 for s in subs if s.status == "processing"),
        "done": sum(1 for s in subs if s.status == "done"),
        "error": sum(1 for s in subs if s.status == "error"),
        "total": len(subs),
    }


@router.post("/{session_id}/start-grading")
def start_grading(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    """Encola todos los exámenes pendientes para corrección (solo tras validar soluciones)."""
    from app import scheduler
    from app.routers.submissions import _upload_dir

    pending_subs = db.query(Submission).filter(
        Submission.session_id == session_id,
        Submission.status == "pending",
    ).all()
    for sub in pending_subs:
        scheduler.enqueue(sub.id, _db_path, str(_upload_dir), _config_overrides)

    return RedirectResponse(url=f"/sessions/{session_id}", status_code=status.HTTP_302_FOUND)


@router.post("/{session_id}/regrade-all")
def regrade_all(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    """Resetea todas las submissions a pending y las re-encola para corrección."""
    from app import scheduler
    from app.routers.submissions import _upload_dir

    subs = db.query(Submission).filter(
        Submission.session_id == session_id,
        Submission.status.in_(["done", "error"]),
    ).all()

    for sub in subs:
        # Borrar resultados anteriores
        for qr in sub.question_results:
            db.query(PartResult).filter(PartResult.question_id_fk == qr.id).delete()
        db.query(QuestionResult).filter(QuestionResult.submission_id == sub.id).delete()
        sub.status = "pending"
        sub.error_message = None
        sub.total_points = None
        sub.max_total_points = None
        sub.incidents = None
        sub.processing_log = None
        sub.processed_at = None

    db.commit()

    for sub in subs:
        scheduler.enqueue(sub.id, _db_path, str(_upload_dir), _config_overrides)

    return RedirectResponse(url=f"/sessions/{session_id}", status_code=status.HTTP_302_FOUND)


@router.post("/{session_id}/archive")
def archive_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session:
        session.status = "archived"
        db.commit()
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


@router.get("/{session_id}/excel")
def download_excel(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    import io
    import json
    import traceback
    from pathlib import Path
    from excel_export import export_results_to_excel
    from models import ExamGradeResult, QuestionGrade, PartGrade

    try:
        session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
        if session is None:
            return HTMLResponse("Convocatoria no encontrada", status_code=404)

        results: list[ExamGradeResult] = []
        for sub in session.submissions:
            if sub.status != "done":
                continue
            questions = []
            for qr in sub.question_results:
                parts = []
                for pr in qr.part_results:
                    parts.append(PartGrade(
                        question_id=qr.question_id,
                        part_id=pr.part_id,
                        column_id=pr.column_id,
                        awarded_points=pr.awarded_points or 0.0,
                        max_points=pr.max_points or 0.0,
                        status=pr.status or "incorrecto",
                        explanation=pr.explanation or "",
                        detected_answer=pr.detected_answer,
                        incidents=json.loads(pr.incidents) if pr.incidents else [],
                    ))
                questions.append(QuestionGrade(
                    question_id=qr.question_id,
                    max_points=qr.max_points or 0.0,
                    awarded_points=sum(p.awarded_points for p in parts),
                    parts=parts,
                ))
            results.append(ExamGradeResult(
                exam_id=f"session_{session_id}::sub_{sub.id}",
                source_file=sub.source_filename,
                student_name=sub.student_name or "Desconocido",
                exam_model=sub.exam_model,
                course_level=sub.course_level,
                pages=[],
                total_points=sub.total_points or 0.0,
                max_total_points=sub.max_total_points or 0.0,
                questions=questions,
                incidents=json.loads(sub.incidents) if sub.incidents else [],
                report_path=sub.report_path,
            ))

        tmp = Path("salidas") / f"session_{session_id}_resultados.xlsx"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        export_results_to_excel(results, tmp, session_name=session.name)
        content = tmp.read_bytes()
    except Exception:
        tb = traceback.format_exc()
        return HTMLResponse(f"<pre>Error generando Excel:\n\n{tb}</pre>", status_code=500)
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=resultados_{session_id}.xlsx"},
    )


# Precios aproximados por millón de tokens (USD) — actualizar si cambian
from app.pricing import TOKEN_PRICES as _TOKEN_PRICES, OP_LABELS as _OP_LABELS, cost_eur as _cost_eur, provider_of as _provider_of

_USD_TO_EUR = 0.92


def _build_token_summary(rows) -> dict:
    """Agrupa filas de TokenUsage en un resumen por (operación, modelo) y totales."""
    by_op: dict[str, dict] = {}
    grand = {"input": 0, "output": 0, "total": 0, "calls": 0, "cost_usd": 0.0}
    for r in rows:
        key = f"{r.operation}||{r.model}"
        entry = by_op.setdefault(key, {
            "label": _OP_LABELS.get(r.operation, r.operation),
            "model": r.model,
            "input": 0, "output": 0, "total": 0, "calls": 0, "cost_usd": 0.0,
        })
        entry["input"] += r.input_tokens
        entry["output"] += r.output_tokens
        entry["total"] += r.total_tokens
        entry["calls"] += r.api_calls
        cost = _cost_eur(r.model, r.input_tokens, r.output_tokens)
        entry["cost_usd"] += cost
        grand["input"] += r.input_tokens
        grand["output"] += r.output_tokens
        grand["total"] += r.total_tokens
        grand["calls"] += r.api_calls
        grand["cost_usd"] += cost
    return {"by_op": by_op, "grand": grand}


def _natural_sort_tuple(s: str) -> list[int | str]:
    """Divide un string en partes texto/número para orden natural: 'ejercicio2' → ['ejercicio', 2]."""
    import re
    parts: list[int | str] = []
    for tok in re.split(r'(\d+)', s):
        if tok.isdigit():
            parts.append(int(tok))
        else:
            parts.append(tok.lower())
    return parts


def _natural_sol_key(sol) -> tuple[list[int | str], list[int | str]]:
    return (_natural_sort_tuple(sol.question_id), _natural_sort_tuple(sol.part_id))


def _col_sort_key(col: str) -> tuple[int, str]:
    if "." in col:
        q, part = col.split(".", 1)
    else:
        q, part = col, ""
    try:
        return (int(q), part)
    except ValueError:
        return (999, part)


# ── Rutas de gestión de soluciones ──────────────────────────────────────────

@router.get("/{session_id}/solutions", response_class=HTMLResponse)
def solutions_page(
    session_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        return HTMLResponse("Convocatoria no encontrada", status_code=404)

    solutions = (
        db.query(SessionSolution)
        .filter(SessionSolution.session_id == session_id)
        .all()
    )
    solutions.sort(key=_natural_sol_key)

    has_pending_pdf = db.query(Submission).filter(
        Submission.session_id == session_id,
        Submission.pdf_path.isnot(None),
    ).first() is not None

    # Estado global: cuántas están validadas
    validated = sum(1 for s in solutions if s.status in ("validated", "manual"))
    total_sols = len(solutions)
    solving_in_progress = any(s.status == "ai_pending" for s in solutions)

    # Criterios de evaluación LOMLOE (si hay currículo para este curso/asignatura)
    import json as _json
    from app.curriculum import get_criterios, has_curriculum
    _criterios = get_criterios(session.course_level, session.subject)
    _sol_criteria_map: dict[int, list[str]] = {}
    for s in solutions:
        if s.criteria_codes:
            try:
                _sol_criteria_map[s.id] = _json.loads(s.criteria_codes)
            except (ValueError, TypeError):
                _sol_criteria_map[s.id] = []

    return templates.TemplateResponse(
        "session_solutions.html",
        {
            "request": request,
            "session": session,
            "solutions": solutions,
            "validated": validated,
            "total": total_sols,
            "solving_in_progress": solving_in_progress,
            "has_pending_pdf": has_pending_pdf,
            "current_step": session.current_step,
            "session_log": session.session_log,
            "solution_mode": session.solution_mode or "ai",
            "user": current_user,
            "criterios": _criterios,
            "has_curriculum": has_curriculum(session.course_level, session.subject),
            "sol_criteria_map": _sol_criteria_map,
        },
    )


@router.post("/{session_id}/solutions/upload-teacher-pdf")
async def upload_teacher_pdf(
    session_id: int,
    teacher_pdf: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    """Recibe el PDF de soluciones del profesor, lo guarda y lanza extracción en background."""
    import shutil
    from pathlib import Path
    from app import scheduler

    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        return HTMLResponse("Convocatoria no encontrada", status_code=404)

    images_dir = Path(_db_path).parent / "solution_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(teacher_pdf.filename).suffix if teacher_pdf.filename else ".pdf"
    dest = images_dir / f"session_{session_id}_teacher{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(teacher_pdf.file, f)

    # Borrar soluciones anteriores para evitar duplicados al re-subir
    db.query(SessionSolution).filter(SessionSolution.session_id == session_id).delete()

    session.solution_mode = "teacher"
    session.session_log = "[]"
    session.current_step = "Iniciando extracción del PDF del profesor..."
    db.commit()

    scheduler.extract_teacher_solutions_for_session(session_id, str(dest), _db_path, _config_overrides)
    return RedirectResponse(url=f"/sessions/{session_id}/solutions", status_code=status.HTTP_302_FOUND)


@router.post("/{session_id}/solutions/validate-all")
def validate_all_solutions(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    """Valida en bloque todas las soluciones que tienen respuesta final."""
    from datetime import datetime, timezone
    solutions = db.query(SessionSolution).filter(
        SessionSolution.session_id == session_id,
        SessionSolution.status == "ai_solved",
        SessionSolution.final_answer.isnot(None),
    ).all()
    now = datetime.now(timezone.utc)
    for sol in solutions:
        sol.status = "validated"
        sol.validated_at = now
    db.commit()
    return RedirectResponse(url=f"/sessions/{session_id}/solutions", status_code=status.HTTP_302_FOUND)


@router.post("/{session_id}/solutions/extract")
def extract_and_solve(
    session_id: int,
    solver_provider: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    """Lanza la tarea background de extracción + resolución IA."""
    from app import scheduler
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session:
        session.current_step = "Iniciando extracción..."
        session.session_log = "[]"
        session.solution_mode = "ai"
        if solver_provider in _VALID_SOLVER_PROVIDERS:
            session.solver_provider = solver_provider
        db.commit()
    scheduler.solve_questions_for_session(session_id, _db_path, _config_overrides)
    return RedirectResponse(url=f"/sessions/{session_id}/solutions", status_code=status.HTTP_302_FOUND)


@router.get("/{session_id}/solutions/status")
def solutions_status(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    solutions = db.query(SessionSolution).filter(SessionSolution.session_id == session_id).all()
    return {
        "total": len(solutions),
        "ai_pending": sum(1 for s in solutions if s.status == "ai_pending"),
        "ai_solved": sum(1 for s in solutions if s.status == "ai_solved"),
        "ai_failed": sum(1 for s in solutions if s.status == "ai_failed"),
        "validated": sum(1 for s in solutions if s.status in ("validated", "manual")),
    }


@router.get("/{session_id}/log-poll")
def log_poll(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Devuelve el log de la sesión (extracción de soluciones) para polling AJAX sin recarga completa."""
    import json as _json
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        return {"log_entries": [], "current_step": "", "solving_in_progress": False}
    solutions = db.query(SessionSolution).filter(SessionSolution.session_id == session_id).all()
    total = len(solutions)
    completed = sum(1 for s in solutions if s.status not in ("ai_pending",))
    solving_in_progress = any(s.status == "ai_pending" for s in solutions)
    starting_up = bool(session.current_step) and total == 0
    try:
        log_entries = _json.loads(session.session_log or "[]")
    except Exception:
        log_entries = []
    return {
        "log_entries": log_entries,
        "current_step": session.current_step or "",
        "total": total,
        "completed": completed,
        "solving_in_progress": solving_in_progress or starting_up,
    }


@router.get("/{session_id}/grading-poll")
def grading_poll(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Devuelve el estado del grading en curso para polling AJAX sin recarga completa."""
    import json as _json
    subs = db.query(Submission).filter(Submission.session_id == session_id).all()
    total = len(subs)
    done_count = sum(1 for s in subs if s.status == "done")
    error_count = sum(1 for s in subs if s.status == "error")
    processing = [s for s in subs if s.status in ("processing", "pending")]
    still_active = bool(processing)
    result_subs = []
    for sub in processing[:6]:
        try:
            entries = _json.loads(sub.processing_log or "[]")
        except Exception:
            entries = []
        result_subs.append({
            "id": sub.id,
            "status": sub.status,
            "source_filename": sub.source_filename or "",
            "log_entries": entries[-5:],
        })
    return {
        "total": total,
        "done_count": done_count,
        "error_count": error_count,
        "processing_count": sum(1 for s in processing if s.status == "processing"),
        "pending_count": sum(1 for s in processing if s.status == "pending"),
        "pct": int((done_count + error_count) / total * 100) if total else 0,
        "still_active": still_active,
        "submissions": result_subs,
    }


@router.post("/{session_id}/solutions/{sol_id}/validate")
def validate_solution(
    session_id: int,
    sol_id: int,
    request: Request,
    final_answer: str = Form(...),
    max_points: str = Form(""),
    teacher_notes: str = Form(""),
    criteria_codes: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from datetime import datetime, timezone
    from fastapi.responses import JSONResponse
    import json as _json
    sol = db.query(SessionSolution).filter(
        SessionSolution.id == sol_id,
        SessionSolution.session_id == session_id,
    ).first()
    if sol:
        sol.final_answer = final_answer.strip()
        sol.teacher_notes = teacher_notes.strip() or None
        if max_points.strip():
            try:
                sol.max_points = float(max_points)
            except ValueError:
                pass
        # Criterios de evaluación (viene como string CSV o vacío)
        if criteria_codes.strip():
            codes = [c.strip() for c in criteria_codes.split(",") if c.strip()]
            sol.criteria_codes = _json.dumps(codes) if codes else None
        else:
            sol.criteria_codes = None
        sol.status = "validated"
        sol.validated_at = datetime.now(timezone.utc)
        db.commit()

    # Si es AJAX, devolver JSON en vez de redirect
    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        # Recalcular contadores
        all_sols = db.query(SessionSolution).filter(SessionSolution.session_id == session_id).all()
        validated = sum(1 for s in all_sols if s.status in ("validated", "manual"))
        total = len(all_sols)
        ai_failed = sum(1 for s in all_sols if s.status == "ai_failed")
        return JSONResponse({
            "ok": True,
            "sol_id": sol_id,
            "final_answer": sol.final_answer if sol else "",
            "max_points": sol.max_points if sol else None,
            "validated": validated,
            "total": total,
            "ai_failed": ai_failed,
        })
    return RedirectResponse(url=f"/sessions/{session_id}/solutions", status_code=status.HTTP_302_FOUND)


@router.post("/{session_id}/solutions/{sol_id}/upload-image")
async def upload_solution_image(
    session_id: int,
    sol_id: int,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    import shutil
    from datetime import datetime, timezone
    from fastapi import UploadFile  # noqa: F401
    from pathlib import Path
    from gemini_client import GeminiClient
    from dotenv import load_dotenv
    import os
    load_dotenv()

    sol = db.query(SessionSolution).filter(
        SessionSolution.id == sol_id,
        SessionSolution.session_id == session_id,
    ).first()
    if sol is None:
        return RedirectResponse(url=f"/sessions/{session_id}/solutions", status_code=status.HTTP_302_FOUND)

    # Guardar imagen
    images_dir = Path(_db_path).parent / "solution_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(image.filename).suffix if image.filename else ".jpg"
    dest = images_dir / f"sol_{sol_id}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    sol.solution_image_path = str(dest)

    # Extraer respuesta de la imagen con Gemini
    try:
        cfg_overrides = _config_overrides
        gemini = GeminiClient(
            model=cfg_overrides.get("gemini_solver_model", "gemini-2.5-pro"),
            solver_model=cfg_overrides.get("gemini_solver_model", "gemini-2.5-pro"),
        )
        extracted = gemini.extract_answer_from_solution_image(
            image_path=dest,
            question_statement=sol.question_statement,
            part_statement=sol.part_statement,
        )
        if extracted and extracted != "NO_LEGIBLE":
            sol.final_answer = extracted
            sol.status = "manual"
            sol.validated_at = datetime.now(timezone.utc)
        else:
            sol.status = "ai_failed"
    except Exception as e:
        from loguru import logger
        logger.error(f"Error extrayendo respuesta de imagen: {e}")
        sol.status = "ai_failed"

    db.commit()
    return RedirectResponse(url=f"/sessions/{session_id}/solutions", status_code=status.HTTP_302_FOUND)


@router.get("/{session_id}/informes-zip")
def download_informes_zip(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Descarga un ZIP con todos los informes PDF de la convocatoria."""
    import io
    import zipfile
    from app.pdf_report import build_pdf_from_submission

    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        return HTMLResponse("Convocatoria no encontrada", status_code=404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for sub in session.submissions:
            if sub.status != "done":
                continue
            student_name = (
                sub.student.display_name if sub.student else sub.student_name
            ) or "alumno"
            safe_name = student_name.replace(" ", "_").replace("/", "_")
            pdf_bytes = build_pdf_from_submission(sub, session)
            zf.writestr(f"informe_{safe_name}.pdf", pdf_bytes)

    buf.seek(0)
    safe_session = (session.name or "convocatoria").replace(" ", "_")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="informes_{safe_session}.zip"'},
    )


# ── Envío de informes a alumnos ───────────────────────────────────────────────

@router.get("/{session_id}/emails", response_class=HTMLResponse)
def session_emails_page(
    session_id: int,
    request: Request,
    sent: str = "",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    from app.email_service import smtp_configured
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        return HTMLResponse("Convocatoria no encontrada", status_code=404)

    submissions = (
        db.query(Submission)
        .filter(Submission.session_id == session_id)
        .order_by(Submission.student_name)
        .all()
    )

    sent_ids = set(sent.split(",")) if sent else set()

    return templates.TemplateResponse(
        "session_emails.html",
        {
            "request": request,
            "session": session,
            "submissions": submissions,
            "smtp_ok": smtp_configured(),
            "sent_ids": sent_ids,
            "user": current_user,
        },
    )


@router.post("/{session_id}/send-report/{submission_id}")
def send_student_report(
    session_id: int,
    submission_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    from app.email_service import send_student_report_email
    from app.pdf_report import build_combined_pdf

    sub = db.query(Submission).filter(
        Submission.id == submission_id,
        Submission.session_id == session_id,
    ).first()
    if not sub or sub.status != "done":
        return RedirectResponse(url=f"/sessions/{session_id}/emails", status_code=302)

    student = sub.student
    if not student or not student.email:
        return RedirectResponse(url=f"/sessions/{session_id}/emails", status_code=302)

    session = sub.session
    combined_pdf = build_combined_pdf(sub, session)

    send_student_report_email(
        to_email=student.email,
        student_name=student.display_name,
        session_name=session.name if session else "Examen",
        combined_pdf=combined_pdf,
    )

    # Redirect back with sent indicator
    return RedirectResponse(
        url=f"/sessions/{session_id}/emails?sent={submission_id}",
        status_code=302,
    )


@router.post("/{session_id}/send-selected-reports")
def send_selected_reports(
    session_id: int,
    submission_ids: list[int] = Form([]),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    from app.email_service import send_student_report_email
    from app.pdf_report import build_combined_pdf

    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        return RedirectResponse(url=f"/sessions/{session_id}/emails", status_code=302)

    if not submission_ids:
        return RedirectResponse(url=f"/sessions/{session_id}/emails", status_code=302)

    submissions = db.query(Submission).filter(
        Submission.id.in_(submission_ids),
        Submission.session_id == session_id,
        Submission.status == "done",
    ).all()

    sent_ids = []
    for sub in submissions:
        student = sub.student
        if not student or not student.email:
            continue
        combined_pdf = build_combined_pdf(sub, session)
        ok = send_student_report_email(
            to_email=student.email,
            student_name=student.display_name,
            session_name=session.name,
            combined_pdf=combined_pdf,
        )
        if ok:
            sent_ids.append(str(sub.id))

    return RedirectResponse(
        url=f"/sessions/{session_id}/emails?sent={','.join(sent_ids)}",
        status_code=302,
    )


# ── Historial / Borrado ──────────────────────────────────────────────────────

def _snapshot_session(session: ExamSession, db: Session, deleted: bool = False) -> None:
    """Guarda o actualiza el snapshot de estadísticas en session_history."""
    from datetime import datetime, timezone

    subs = db.query(Submission).filter(Submission.session_id == session.id).all()
    done = [s for s in subs if s.status == "done"]
    avg = round(sum(s.total_points for s in done if s.total_points is not None) / len(done), 3) if done else None

    token_rows = db.query(TokenUsage).filter(TokenUsage.session_id == session.id).all()
    total_tokens = sum(r.total_tokens for r in token_rows)
    total_cost = sum(_cost_eur(r.model, r.input_tokens, r.output_tokens) for r in token_rows)

    now = datetime.now(timezone.utc)
    existing = db.query(SessionHistory).filter(SessionHistory.session_id == session.id).first()
    if existing:
        existing.session_name = session.name
        existing.session_date = session.date
        existing.subject = session.subject
        existing.course_level = session.course_level
        existing.max_total_points = session.max_total_points
        existing.total_submissions = len(subs)
        existing.graded_submissions = len(done)
        existing.avg_score = avg
        existing.total_tokens = total_tokens
        existing.total_cost_eur = round(total_cost, 6)
        existing.snapshot_at = now
        if deleted:
            existing.deleted_at = now
    else:
        db.add(SessionHistory(
            session_id=session.id,
            session_name=session.name,
            session_date=session.date,
            subject=session.subject,
            course_level=session.course_level,
            max_total_points=session.max_total_points,
            total_submissions=len(subs),
            graded_submissions=len(done),
            avg_score=avg,
            total_tokens=total_tokens,
            total_cost_eur=round(total_cost, 6),
            deleted_at=now if deleted else None,
        ))
    db.commit()


@router.post("/{session_id}/delete")
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)

    _snapshot_session(session, db, deleted=True)

    db.query(TokenUsage).filter(TokenUsage.session_id == session_id).delete()
    db.query(SessionSolution).filter(SessionSolution.session_id == session_id).delete()
    db.delete(session)
    db.commit()

    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
