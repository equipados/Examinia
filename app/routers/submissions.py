from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from app.templates import templates
from sqlalchemy.orm import Session

from fastapi import Form
from app.auth import get_current_user
from app.database import get_db
from app.db_models import ExamSession, PartResult, QuestionResult, Submission, User

router = APIRouter(prefix="/submissions")

# Will be set from web.py at startup
_upload_dir: Path = Path("web_uploads")
_db_path: str = "corrector.db"
_config_overrides: dict = {}


def configure(upload_dir: Path, db_path: str, config_overrides: dict) -> None:
    global _upload_dir, _db_path, _config_overrides
    _upload_dir = upload_dir
    _db_path = db_path
    _config_overrides = config_overrides


@router.get("/sessions/{session_id}/upload", response_class=HTMLResponse)
def upload_form(
    session_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        return HTMLResponse("Convocatoria no encontrada", status_code=404)
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "session": session, "user": current_user},
    )


@router.post("/sessions/{session_id}/upload")
async def upload_pdfs(
    session_id: int,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    from app import scheduler

    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        return HTMLResponse("Convocatoria no encontrada", status_code=404)

    dest_dir = _upload_dir / f"session_{session_id}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            continue
        dest = dest_dir / upload.filename
        # Avoid overwriting: add suffix if exists
        counter = 1
        while dest.exists():
            stem = Path(upload.filename).stem
            dest = dest_dir / f"{stem}_{counter}.pdf"
            counter += 1

        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

        sub = Submission(
            session_id=session_id,
            source_filename=upload.filename,
            pdf_path=str(dest),
        )
        db.add(sub)
        db.flush()

    db.commit()

    # Decidir siguiente paso según el estado de las soluciones
    from app.db_models import SessionSolution
    validated_count = db.query(SessionSolution).filter(
        SessionSolution.session_id == session_id,
        SessionSolution.status.in_(["validated", "manual"]),
    ).count()
    unvalidated_count = db.query(SessionSolution).filter(
        SessionSolution.session_id == session_id,
        SessionSolution.status.notin_(["validated", "manual"]),
    ).count()

    is_teacher_mode = session.solution_mode == "teacher"

    if validated_count > 0 and unvalidated_count == 0:
        # Todas las soluciones están validadas → encolar corrección y volver a la convocatoria
        # Si no hay ninguna submission procesando en toda la BD, el pool puede estar atascado
        any_processing = db.query(Submission).filter(
            Submission.status == "processing",
        ).count()
        if any_processing == 0:
            scheduler.reset_executor()
        pending_subs = db.query(Submission).filter(
            Submission.session_id == session_id,
            Submission.status == "pending",
        ).all()
        for sub in pending_subs:
            scheduler.enqueue(sub.id, _db_path, str(_upload_dir), _config_overrides)
        return RedirectResponse(url=f"/sessions/{session_id}", status_code=status.HTTP_302_FOUND)

    if validated_count == 0 and unvalidated_count == 0 and not is_teacher_mode:
        # No hay soluciones y modo IA → extraer soluciones automáticamente
        scheduler.solve_questions_for_session(session_id, _db_path, _config_overrides)
        return RedirectResponse(url=f"/sessions/{session_id}/solutions", status_code=status.HTTP_302_FOUND)

    # Hay soluciones pendientes de validar, o modo profesor sin soluciones → ir a soluciones
    return RedirectResponse(url=f"/sessions/{session_id}", status_code=status.HTTP_302_FOUND)


@router.get("/{submission_id}", response_class=HTMLResponse)
def submission_detail(
    submission_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    sub = db.query(Submission).filter(Submission.id == submission_id).first()
    if sub is None:
        return HTMLResponse("Examen no encontrado", status_code=404)

    report_html: str | None = None
    if sub.report_path and Path(sub.report_path).exists():
        import markdown
        md_text = Path(sub.report_path).read_text(encoding="utf-8")
        report_html = markdown.markdown(md_text, extensions=["tables"])

    incidents = json.loads(sub.incidents) if sub.incidents else []

    return templates.TemplateResponse(
        "submission_detail.html",
        {
            "request": request,
            "sub": sub,
            "report_html": report_html,
            "incidents": incidents,
            "user": current_user,
        },
    )


@router.get("/{submission_id}/pdf")
def download_pdf(
    submission_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    sub = db.query(Submission).filter(Submission.id == submission_id).first()
    if sub is None or not Path(sub.pdf_path).exists():
        return HTMLResponse("PDF no encontrado", status_code=404)
    return FileResponse(sub.pdf_path, media_type="application/pdf", filename=sub.source_filename)


@router.get("/{submission_id}/report")
def download_report(
    submission_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    sub = db.query(Submission).filter(Submission.id == submission_id).first()
    if sub is None or not sub.report_path or not Path(sub.report_path).exists():
        return HTMLResponse("Informe no disponible", status_code=404)
    filename = Path(sub.report_path).name
    return FileResponse(sub.report_path, media_type="text/markdown", filename=filename)


@router.get("/{submission_id}/report-pdf")
def download_pdf_report(
    submission_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    from app.pdf_report import build_pdf_from_submission
    sub = db.query(Submission).filter(Submission.id == submission_id).first()
    if sub is None or sub.status != "done":
        return HTMLResponse("Informe no disponible", status_code=404)
    session = sub.session
    pdf_bytes = build_pdf_from_submission(sub, session)
    student_name = (sub.student.display_name if sub.student else sub.student_name) or "alumno"
    filename = f"informe_{student_name.replace(' ', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{submission_id}/annotated-pdf")
def download_annotated_pdf(
    submission_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    """Descarga el examen escaneado con las correcciones anotadas encima."""
    from app.annotator import generate_annotated_pdf
    sub = db.query(Submission).filter(Submission.id == submission_id).first()
    if sub is None or sub.status != "done":
        return HTMLResponse("Examen no disponible o no corregido", status_code=404)
    session = sub.session
    try:
        pdf_bytes = generate_annotated_pdf(sub, session, _upload_dir)
    except (FileNotFoundError, ValueError) as exc:
        return HTMLResponse(f"Error generando PDF anotado: {exc}", status_code=500)
    student_name = (sub.student.display_name if sub.student else sub.student_name) or "alumno"
    filename = f"corregido_{student_name.replace(' ', '_')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{submission_id}/reprocess")
def reprocess(
    submission_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    from app import scheduler
    sub = db.query(Submission).filter(Submission.id == submission_id).first()
    if sub and sub.status != "pending":
        sub.status = "pending"
        sub.error_message = None
        db.commit()
        scheduler.enqueue(sub.id, _db_path, str(_upload_dir), _config_overrides)
    return RedirectResponse(url=f"/submissions/{submission_id}", status_code=status.HTTP_302_FOUND)


@router.post("/{submission_id}/parts/{part_id}/edit")
def edit_part(
    submission_id: int,
    part_id: int,
    request: Request,
    awarded_points: str = Form(...),
    max_points: str = Form(""),
    explanation: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from fastapi.responses import JSONResponse

    pr = db.query(PartResult).filter(PartResult.id == part_id).first()
    if not pr:
        return JSONResponse({"ok": False, "error": "not found"}, status_code=404)

    # Verify the part belongs to this submission
    qr = db.query(QuestionResult).filter(QuestionResult.id == pr.question_id_fk).first()
    if not qr or qr.submission_id != submission_id:
        return JSONResponse({"ok": False, "error": "mismatch"}, status_code=400)

    try:
        pts = float(awarded_points)
    except ValueError:
        pts = pr.awarded_points or 0.0

    # Actualizar puntuación máxima si se proporcionó
    if max_points.strip():
        try:
            pr.max_points = float(max_points)
        except ValueError:
            pass

    new_explanation = explanation.strip() or pr.explanation

    # ── Capturar corrección del profesor como ejemplo de aprendizaje ──
    if pr.ai_awarded_points is not None:
        pts_changed = abs(pts - (pr.ai_awarded_points or 0)) > 0.01
        expl_changed = new_explanation and new_explanation != (pr.ai_explanation or "")
        if pts_changed or expl_changed:
            from app.db_models import CorrectionExample
            sub_for_ctx = db.query(Submission).filter(Submission.id == submission_id).first()
            session_obj = sub_for_ctx.session if sub_for_ctx else None
            # Buscar si ya existe un ejemplo para este apartado con esta respuesta
            existing_ex = db.query(CorrectionExample).filter(
                CorrectionExample.session_id == (sub_for_ctx.session_id if sub_for_ctx else 0),
                CorrectionExample.question_id == qr.question_id,
                CorrectionExample.part_id == pr.part_id,
                CorrectionExample.detected_answer == pr.detected_answer,
            ).first()
            if existing_ex:
                existing_ex.teacher_awarded_points = pts
                existing_ex.teacher_explanation = new_explanation or ""
            else:
                # Obtener enunciado del SessionSolution si existe
                from app.db_models import SessionSolution
                sol = db.query(SessionSolution).filter(
                    SessionSolution.session_id == (sub_for_ctx.session_id if sub_for_ctx else 0),
                    SessionSolution.question_id == qr.question_id,
                    SessionSolution.part_id == pr.part_id,
                ).first()
                db.add(CorrectionExample(
                    session_id=sub_for_ctx.session_id if sub_for_ctx else 0,
                    question_id=qr.question_id,
                    part_id=pr.part_id,
                    ai_awarded_points=pr.ai_awarded_points,
                    ai_explanation=pr.ai_explanation,
                    ai_classification=pr.status,
                    teacher_awarded_points=pts,
                    teacher_explanation=new_explanation or "",
                    max_points=pr.max_points,
                    subject=session_obj.subject if session_obj else None,
                    course_level=session_obj.course_level if session_obj else None,
                    question_statement=sol.question_statement if sol else None,
                    detected_answer=pr.detected_answer,
                ))

    pr.awarded_points = pts
    pr.explanation = new_explanation

    # Auto-update status
    max_pts = pr.max_points or 0.0
    if max_pts > 0 and pts >= max_pts:
        pr.status = "correcto"
    elif pts > 0:
        pr.status = "parcial"
    else:
        pr.status = "incorrecto"

    # Recalculate submission totals
    sub = db.query(Submission).filter(Submission.id == submission_id).first()
    if sub:
        total = 0.0
        max_total = 0.0
        for q in sub.question_results:
            for p in q.part_results:
                total += p.awarded_points or 0.0
                max_total += p.max_points or 0.0
        sub.total_points = round(total, 4)
        sub.max_total_points = round(max_total, 4)

    db.commit()

    # Regenerar informe markdown tras edición
    if sub and sub.report_path:
        try:
            from reporting import build_report_from_db
            md = build_report_from_db(sub, sub.session.name if sub.session else None, sub.session.date if sub.session else None)
            Path(sub.report_path).write_text(md, encoding="utf-8")
        except Exception:
            pass  # no bloquear la respuesta si falla la regeneración

    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JSONResponse({
            "ok": True,
            "part_id": part_id,
            "awarded_points": pts,
            "status": pr.status,
            "total_points": sub.total_points if sub else 0,
            "max_total_points": sub.max_total_points if sub else 0,
        })
    return RedirectResponse(url=f"/submissions/{submission_id}", status_code=status.HTTP_302_FOUND)
