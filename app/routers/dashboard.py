from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from app.templates import templates
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.db_models import ExamSession, Submission, User

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    sessions = db.query(ExamSession).order_by(ExamSession.created_at.desc()).all()

    session_stats = []
    for s in sessions:
        subs = db.query(Submission).filter(Submission.session_id == s.id).all()
        total = len(subs)
        done = [sub for sub in subs if sub.status == "done"]
        pending = sum(1 for sub in subs if sub.status in ("pending", "processing"))
        errors = sum(1 for sub in subs if sub.status == "error")
        avg_score = (
            round(sum(sub.total_points for sub in done if sub.total_points is not None) / len(done), 2)
            if done else None
        )
        revision = sum(
            1 for sub in done
            if sub.total_points is not None and _has_revision_manual(sub)
        )
        session_stats.append({
            "session": s,
            "total": total,
            "done": len(done),
            "pending": pending,
            "errors": errors,
            "avg_score": avg_score,
            "revision_count": revision,
        })

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "session_stats": session_stats, "user": current_user},
    )


def _has_revision_manual(sub: Submission) -> bool:
    from app.db_models import PartResult, QuestionResult
    from app.database import _engine
    from sqlalchemy.orm import Session as S
    with S(_engine) as db:
        return db.query(PartResult).join(QuestionResult).filter(
            QuestionResult.submission_id == sub.id,
            PartResult.status == "revision_manual",
        ).first() is not None
