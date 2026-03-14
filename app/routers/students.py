from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.db_models import ExamSession, Student, Submission, User
from app.templates import templates
from utils import normalize_identifier

router = APIRouter(prefix="/students")


@router.get("", response_class=HTMLResponse)
def students_list(
    request: Request,
    course: str = "",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    # All unique course levels from students
    all_courses = [
        r[0]
        for r in db.query(Student.course_level).distinct().order_by(Student.course_level).all()
        if r[0]
    ]

    # Filter students
    q = db.query(Student)
    if course:
        q = q.filter(Student.course_level == course)
    students = q.order_by(Student.display_name).all()

    if not students:
        return templates.TemplateResponse(
            "students.html",
            {
                "request": request,
                "students": [],
                "sessions": [],
                "grid": {},
                "all_courses": all_courses,
                "selected_course": course,
                "session_max": {},
                "user": current_user,
            },
        )

    student_ids = [s.id for s in students]

    # Sessions that have at least one done submission linked to these students
    linked_session_ids = [
        r[0]
        for r in db.query(Submission.session_id)
        .filter(
            Submission.student_id.in_(student_ids),
            Submission.status == "done",
        )
        .distinct()
        .all()
    ]

    sessions = (
        db.query(ExamSession)
        .filter(ExamSession.id.in_(linked_session_ids))
        .order_by(ExamSession.date.desc().nullslast(), ExamSession.created_at.desc())
        .all()
    ) if linked_session_ids else []

    # Build done-submission map: (student_id, session_id) -> Submission
    subs = (
        db.query(Submission)
        .filter(
            Submission.student_id.in_(student_ids),
            Submission.status == "done",
        )
        .all()
    ) if student_ids else []

    grid: dict[tuple[int, int], Submission] = {}
    for sub in subs:
        key = (sub.student_id, sub.session_id)
        # Keep the most recent if duplicate
        if key not in grid or (sub.processed_at and (grid[key].processed_at is None or sub.processed_at > grid[key].processed_at)):
            grid[key] = sub

    # Per-session denominator for % calculation
    session_max: dict[int, float] = {}
    for sess in sessions:
        if sess.max_total_points:
            session_max[sess.id] = sess.max_total_points
        else:
            pts = [
                sub.total_points
                for sub in subs
                if sub.session_id == sess.id and sub.total_points is not None
            ]
            session_max[sess.id] = max(pts) if pts else 0.0

    return templates.TemplateResponse(
        "students.html",
        {
            "request": request,
            "students": students,
            "sessions": sessions,
            "grid": grid,
            "all_courses": all_courses,
            "selected_course": course,
            "session_max": session_max,
            "user": current_user,
        },
    )


@router.get("/{student_id}", response_class=HTMLResponse)
def student_detail(
    student_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    student = db.query(Student).filter(Student.id == student_id).first()
    if student is None:
        return HTMLResponse("Alumno no encontrado", status_code=404)

    subs = (
        db.query(Submission)
        .filter(Submission.student_id == student_id, Submission.status == "done")
        .order_by(Submission.processed_at.desc())
        .all()
    )

    # Enrich with session info
    session_ids = list({s.session_id for s in subs})
    sessions_map: dict[int, ExamSession] = {}
    if session_ids:
        for sess in db.query(ExamSession).filter(ExamSession.id.in_(session_ids)).all():
            sessions_map[sess.id] = sess

    history = []
    total_pct_sum = 0.0
    pct_count = 0
    for sub in subs:
        sess = sessions_map.get(sub.session_id)
        denom = (sess.max_total_points if sess and sess.max_total_points else sub.max_total_points) or 0.0
        pct: float | None = None
        if denom and sub.total_points is not None:
            pct = round(sub.total_points / denom * 100, 1)
            total_pct_sum += pct
            pct_count += 1
        history.append({
            "submission": sub,
            "session": sess,
            "pct": pct,
            "denom": denom,
        })

    avg_pct = round(total_pct_sum / pct_count, 1) if pct_count else None

    return templates.TemplateResponse(
        "student_detail.html",
        {
            "request": request,
            "student": student,
            "history": history,
            "avg_pct": avg_pct,
            "user": current_user,
        },
    )


@router.post("/{student_id}/rename")
def rename_student(
    student_id: int,
    display_name: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        return RedirectResponse(url="/students", status_code=status.HTTP_302_FOUND)

    new_name = display_name.strip()
    new_norm = normalize_identifier(new_name)

    # Buscar si ya existe otro alumno con el mismo nombre normalizado
    existing = (
        db.query(Student)
        .filter(Student.normalized_name == new_norm, Student.id != student_id)
        .first()
    )

    if existing:
        # Merge: mover todas las submissions del alumno actual al existente
        db.query(Submission).filter(Submission.student_id == student_id).update(
            {Submission.student_id: existing.id}, synchronize_session="fetch"
        )
        # Conservar notas si el original no tenía
        if student.notes and not existing.notes:
            existing.notes = student.notes
        if student.course_level and not existing.course_level:
            existing.course_level = student.course_level
        db.delete(student)
        db.commit()
        return RedirectResponse(url=f"/students/{existing.id}", status_code=status.HTTP_302_FOUND)

    # Sin duplicado: simplemente renombrar
    student.display_name = new_name
    student.normalized_name = new_norm
    db.commit()
    return RedirectResponse(url=f"/students/{student_id}", status_code=status.HTTP_302_FOUND)
