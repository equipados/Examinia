"""Rutas para vistas de competencias LOMLOE y áreas de mejora por alumno."""
from __future__ import annotations

import json
from collections import defaultdict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.db_models import (
    ExamSession,
    PartResult,
    QuestionResult,
    SessionSolution,
    Student,
    Submission,
    User,
)
from app.curriculum import (
    get_criterios,
    get_competencias_clave,
    get_competencias_especificas,
    get_criterios_for_cc,
    has_curriculum,
)
from app.templates import templates

router = APIRouter(prefix="/students", tags=["competencias"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _student_sessions_data(student_id: int, db: Session):
    """Carga datos comunes del alumno: student, submissions con sesiones, media ponderada."""
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        return None, [], {}, None

    subs = (
        db.query(Submission)
        .filter(Submission.student_id == student_id, Submission.status == "done")
        .order_by(Submission.processed_at.asc())
        .all()
    )

    session_ids = list({s.session_id for s in subs})
    sessions_map: dict[int, ExamSession] = {}
    if session_ids:
        for sess in db.query(ExamSession).filter(ExamSession.id.in_(session_ids)).all():
            sessions_map[sess.id] = sess

    # Media ponderada
    weighted_sum = 0.0
    weight_total = 0.0
    for sub in subs:
        sess = sessions_map.get(sub.session_id)
        denom = (sess.max_total_points if sess and sess.max_total_points else sub.max_total_points) or 0.0
        w = (sess.weight if sess and sess.weight is not None else 1.0)
        if denom and sub.total_points is not None:
            pct = sub.total_points / denom * 100
            weighted_sum += pct * w
            weight_total += w
    avg_pct = round(weighted_sum / weight_total, 1) if weight_total > 0 else None

    return student, subs, sessions_map, avg_pct


def _build_criteria_grades(
    subs: list[Submission],
    sessions_map: dict[int, ExamSession],
    db: Session,
) -> tuple[list[ExamSession], dict[str, dict[int, float]]]:
    """Calcula notas por criterio por sesión.

    Returns:
        ordered_sessions: lista de sesiones ordenadas cronológicamente
        grades: {criterio_code: {session_id: porcentaje}} — solo para criterios que tienen datos
    """
    # Sesiones ordenadas por fecha
    seen_sess_ids = list(dict.fromkeys(s.session_id for s in subs))
    ordered_sessions = [sessions_map[sid] for sid in seen_sess_ids if sid in sessions_map]

    # Para cada submission, cargar PartResults y mapearlos a criterios via SessionSolution
    # grades[criterio_code][session_id] = porcentaje (0-100)
    # Acumulamos awarded y max por (criterio, session) para promediar
    awarded_acc: dict[tuple[str, int], float] = defaultdict(float)  # (crit, sess_id) → sum(awarded)
    max_acc: dict[tuple[str, int], float] = defaultdict(float)      # (crit, sess_id) → sum(max)

    for sub in subs:
        sess_id = sub.session_id
        # Cargar PartResults de esta submission
        parts = (
            db.query(PartResult, QuestionResult.question_id)
            .join(QuestionResult, PartResult.question_id_fk == QuestionResult.id)
            .filter(QuestionResult.submission_id == sub.id)
            .all()
        )
        # Cargar criteria_codes de SessionSolutions de esta sesión
        sol_criteria: dict[tuple[str, str], list[str]] = {}
        sols = db.query(SessionSolution).filter(
            SessionSolution.session_id == sess_id,
            SessionSolution.criteria_codes.isnot(None),
        ).all()
        for sol in sols:
            try:
                codes = json.loads(sol.criteria_codes)
                if codes:
                    sol_criteria[(sol.question_id, sol.part_id)] = codes
            except (ValueError, TypeError):
                pass

        # Mapear cada PartResult a sus criterios
        for pr, q_id in parts:
            key = (q_id, pr.part_id)
            crits = sol_criteria.get(key, [])
            if not crits:
                continue
            awarded = pr.awarded_points or 0.0
            mx = pr.max_points or 0.0
            if mx <= 0:
                continue
            for crit_code in crits:
                awarded_acc[(crit_code, sess_id)] += awarded
                max_acc[(crit_code, sess_id)] += mx

    # Convertir a porcentajes
    grades: dict[str, dict[int, float]] = defaultdict(dict)
    for (crit_code, sess_id), mx in max_acc.items():
        if mx > 0:
            grades[crit_code][sess_id] = round(awarded_acc[(crit_code, sess_id)] / mx * 100, 1)

    return ordered_sessions, dict(grades)


# ---------------------------------------------------------------------------
# Ruta 1: Notas por criterios de evaluación
# ---------------------------------------------------------------------------

@router.get("/{student_id}/criterios", response_class=HTMLResponse)
def student_criterios(
    student_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    student, subs, sessions_map, avg_pct = _student_sessions_data(student_id, db)
    if not student:
        return HTMLResponse("Alumno no encontrado", status_code=404)

    # Detectar curso/asignatura del alumno (de sus sesiones)
    course_level = student.course_level
    subject = None
    for sess in sessions_map.values():
        if sess.subject:
            subject = sess.subject
        if sess.course_level and not course_level:
            course_level = sess.course_level

    criterios = get_criterios(course_level, subject)
    ce_list = get_competencias_especificas(course_level, subject)
    ce_map = {ce.code: ce.name for ce in ce_list}

    if not criterios:
        return templates.TemplateResponse("student_criterios.html", {
            "request": request, "student": student, "avg_pct": avg_pct,
            "user": current_user, "has_data": False, "active_tab": "criterios",
        })

    ordered_sessions, grades = _build_criteria_grades(subs, sessions_map, db)

    # Construir filas: cada criterio con sus notas por sesión y media
    rows = []
    for crit in criterios:
        sess_grades = grades.get(crit.code, {})
        per_session = []
        vals = []
        for sess in ordered_sessions:
            g = sess_grades.get(sess.id)
            per_session.append(g)
            if g is not None:
                vals.append(g)
        avg = round(sum(vals) / len(vals), 1) if vals else None
        rows.append({
            "code": crit.code,
            "ce_code": crit.ce_code,
            "ce_name": ce_map.get(crit.ce_code, ""),
            "description": crit.description,
            "saberes": ", ".join(crit.saberes),
            "per_session": per_session,
            "avg": avg,
        })

    return templates.TemplateResponse("student_criterios.html", {
        "request": request,
        "student": student,
        "avg_pct": avg_pct,
        "user": current_user,
        "has_data": True,
        "active_tab": "criterios",
        "sessions": ordered_sessions,
        "rows": rows,
        "course_level": course_level,
        "subject": subject,
    })


# ---------------------------------------------------------------------------
# Ruta 2: Notas por competencias clave
# ---------------------------------------------------------------------------

@router.get("/{student_id}/competencias-clave", response_class=HTMLResponse)
def student_competencias_clave(
    student_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    student, subs, sessions_map, avg_pct = _student_sessions_data(student_id, db)
    if not student:
        return HTMLResponse("Alumno no encontrado", status_code=404)

    course_level = student.course_level
    subject = None
    for sess in sessions_map.values():
        if sess.subject:
            subject = sess.subject
        if sess.course_level and not course_level:
            course_level = sess.course_level

    cc_list = get_competencias_clave(course_level, subject)
    if not cc_list:
        return templates.TemplateResponse("student_competencias.html", {
            "request": request, "student": student, "avg_pct": avg_pct,
            "user": current_user, "has_data": False, "active_tab": "competencias",
        })

    ordered_sessions, crit_grades = _build_criteria_grades(subs, sessions_map, db)

    # Para cada CC, promediar las notas de criterios que mapean a esa CC
    rows = []
    for cc in cc_list:
        linked_crits = get_criterios_for_cc(cc.code, course_level, subject)
        if not linked_crits:
            rows.append({
                "code": cc.code, "name": cc.name,
                "per_session": [None] * len(ordered_sessions), "avg": None,
            })
            continue

        per_session = []
        all_vals = []
        for sess in ordered_sessions:
            # Promediar notas de criterios vinculados a esta CC en esta sesión
            vals = []
            for crit_code in linked_crits:
                g = crit_grades.get(crit_code, {}).get(sess.id)
                if g is not None:
                    vals.append(g)
            if vals:
                avg_sess = round(sum(vals) / len(vals), 1)
                per_session.append(avg_sess)
                all_vals.append(avg_sess)
            else:
                per_session.append(None)

        avg = round(sum(all_vals) / len(all_vals), 1) if all_vals else None
        rows.append({
            "code": cc.code,
            "name": cc.name,
            "per_session": per_session,
            "avg": avg,
        })

    return templates.TemplateResponse("student_competencias.html", {
        "request": request,
        "student": student,
        "avg_pct": avg_pct,
        "user": current_user,
        "has_data": True,
        "active_tab": "competencias",
        "sessions": ordered_sessions,
        "rows": rows,
    })


# ---------------------------------------------------------------------------
# Ruta 3: Áreas de mejora (feedback de la IA)
# ---------------------------------------------------------------------------

@router.get("/{student_id}/mejoras", response_class=HTMLResponse)
def student_mejoras(
    student_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    student, subs, sessions_map, avg_pct = _student_sessions_data(student_id, db)
    if not student:
        return HTMLResponse("Alumno no encontrado", status_code=404)

    # Recopilar feedback de PartResults donde status != "correcto"
    exam_feedbacks = []
    for sub in subs:
        sess = sessions_map.get(sub.session_id)
        denom = (sess.max_total_points if sess and sess.max_total_points else sub.max_total_points) or 0.0

        parts_feedback = []
        qrs = (
            db.query(QuestionResult)
            .filter(QuestionResult.submission_id == sub.id)
            .all()
        )
        for qr in qrs:
            for pr in qr.part_results:
                if pr.status and pr.status != "correcto":
                    parts_feedback.append({
                        "question_id": qr.question_id,
                        "part_id": pr.part_id,
                        "awarded": pr.awarded_points or 0.0,
                        "max": pr.max_points or 0.0,
                        "status": pr.status,
                        "explanation": pr.explanation or pr.status,
                    })

        if parts_feedback:
            exam_feedbacks.append({
                "session_name": sess.name if sess else "Convocatoria eliminada",
                "session_date": (sess.date if sess else None) or "",
                "total_points": sub.total_points,
                "max_total_points": denom,
                "parts": parts_feedback,
            })

    return templates.TemplateResponse("student_mejoras.html", {
        "request": request,
        "student": student,
        "avg_pct": avg_pct,
        "user": current_user,
        "active_tab": "mejoras",
        "exam_feedbacks": exam_feedbacks,
    })
