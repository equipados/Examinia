from __future__ import annotations

import random
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from app.templates import templates
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.db_models import ExamSession, Submission, User

router = APIRouter()

_QUOTES = [
    ("La educación es el arma más poderosa que puedes usar para cambiar el mundo.", "Nelson Mandela"),
    ("El que enseña aprende dos veces.", "Joseph Joubert"),
    ("Enseñar es aprender dos veces.", "Robert Frost"),
    ("La mente no es un vaso por llenar, sino una lámpara por encender.", "Plutarco"),
    ("Dime y lo olvido, enséñame y lo recuerdo, involúcrame y lo aprendo.", "Benjamin Franklin"),
    ("La educación no es preparación para la vida; la educación es la vida misma.", "John Dewey"),
    ("Un buen maestro puede inspirar esperanza, encender la imaginación y sembrar el amor por aprender.", "Brad Henry"),
    ("El arte supremo del maestro es despertar la curiosidad en la expresión creativa y el conocimiento.", "Albert Einstein"),
    ("Los profesores afectan la eternidad; nunca pueden decir dónde termina su influencia.", "Henry Adams"),
    ("Lo que la escultura es a un bloque de mármol, la educación es al alma.", "Joseph Addison"),
    ("La educación es el movimiento de la oscuridad a la luz.", "Allan Bloom"),
    ("El objetivo de la educación es la virtud y el deseo de convertirse en un buen ciudadano.", "Platón"),
    ("Aprender sin pensar es inútil; pensar sin aprender es peligroso.", "Confucio"),
    ("El conocimiento es poder.", "Francis Bacon"),
    ("La raíz de la educación es amarga, pero su fruto es dulce.", "Aristóteles"),
    ("Nunca consideres el estudio como una obligación, sino como una oportunidad.", "Albert Einstein"),
    ("Es la marca de una mente educada poder considerar un pensamiento sin aceptarlo.", "Aristóteles"),
    ("La creatividad es la inteligencia divirtiéndose.", "Albert Einstein"),
    ("No es que sea muy listo, es que me quedo con los problemas más tiempo.", "Albert Einstein"),
    ("Cada día sabemos más y entendemos menos.", "Albert Einstein"),
    ("Solo sé que no sé nada.", "Sócrates"),
    ("El verdadero signo de la inteligencia no es el conocimiento, sino la imaginación.", "Albert Einstein"),
    ("Las matemáticas son el alfabeto con el que Dios ha escrito el universo.", "Galileo Galilei"),
    ("La paciencia es la madre de la ciencia.", "Proverbio español"),
    ("Quien se atreve a enseñar, nunca debe dejar de aprender.", "John Cotton Dana"),
    ("Un profesor trabaja para la eternidad: nadie puede predecir dónde acabará su influencia.", "Henry Adams"),
    ("El éxito es la suma de pequeños esfuerzos repetidos día tras día.", "Robert Collier"),
    ("No busques ser una persona de éxito, sino una persona de valor.", "Albert Einstein"),
    ("La mejor forma de predecir el futuro es creándolo.", "Peter Drucker"),
    ("Lo que con mucho trabajo se adquiere, más se ama.", "Aristóteles"),
]


def _greeting(user: User) -> str:
    """Saludo según la hora del día + nombre del usuario."""
    hour = datetime.now().hour
    if hour < 14:
        saludo = "Buenos días"
    elif hour < 21:
        saludo = "Buenas tardes"
    else:
        saludo = "Buenas noches"
    name = user.full_name.split()[0] if user.full_name else user.username
    return f"{saludo}, {name}"


def _random_quote() -> tuple[str, str]:
    return random.choice(_QUOTES)


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

    quote_text, quote_author = _random_quote()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "session_stats": session_stats,
            "user": current_user,
            "greeting": _greeting(current_user),
            "quote_text": quote_text,
            "quote_author": quote_author,
        },
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
