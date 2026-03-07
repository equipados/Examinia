from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from app.templates import templates
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.db_models import ExamSession, Submission, User

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

    # Si no hay soluciones validadas todavía, lanzar extracción automática y redirigir a validación
    from app.db_models import SessionSolution
    solutions_exist = db.query(SessionSolution).filter(
        SessionSolution.session_id == session_id,
    ).first() is not None

    if not solutions_exist:
        scheduler.solve_questions_for_session(session_id, _db_path, _config_overrides)
        return RedirectResponse(url=f"/sessions/{session_id}/solutions", status_code=status.HTTP_302_FOUND)

    # Si ya hay soluciones validadas, encolar corrección directamente
    all_validated = db.query(SessionSolution).filter(
        SessionSolution.session_id == session_id,
        SessionSolution.status.notin_(["validated", "manual"]),
    ).first() is None

    if all_validated:
        subs = db.query(Submission).filter(
            Submission.session_id == session_id,
            Submission.status == "pending",
        ).all()
        for sub in subs:
            scheduler.enqueue(sub.id, _db_path, str(_upload_dir), _config_overrides)

    return RedirectResponse(url=f"/sessions/{session_id}/solutions", status_code=status.HTTP_302_FOUND)


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


@router.post("/{submission_id}/reprocess")
def reprocess(
    submission_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    from app import scheduler
    sub = db.query(Submission).filter(Submission.id == submission_id).first()
    if sub and sub.status not in ("pending", "processing"):
        sub.status = "pending"
        sub.error_message = None
        db.commit()
        scheduler.enqueue(sub.id, _db_path, str(_upload_dir), _config_overrides)
    return RedirectResponse(url=f"/submissions/{submission_id}", status_code=status.HTTP_302_FOUND)
