from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from app.templates import templates
from sqlalchemy.orm import Session

from app.auth import authenticate_user, create_access_token, get_current_user
from app.database import get_db
from app.db_models import User

router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@router.post("/login", response_model=None)
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
) -> RedirectResponse | HTMLResponse:
    user = authenticate_user(username, password, db)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Usuario o contraseña incorrectos"},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    token = create_access_token(user.username)
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=8 * 3600,
    )
    return response


@router.post("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token")
    return response


@router.get("/profile", response_class=HTMLResponse)
def profile_page(
    request: Request,
    saved: str = "",
    test: str = "",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    from app.email_service import smtp_configured
    return templates.TemplateResponse("profile.html", {
        "request": request,
        "user": current_user,
        "saved": bool(saved),
        "test": test,
        "smtp_configured": smtp_configured(),
    })


@router.post("/profile/update")
def update_profile(
    full_name: str = Form(""),
    email: str = Form(""),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    current_user.full_name = full_name.strip() or None
    current_user.email = email.strip() or None
    db.commit()
    return RedirectResponse(url="/profile?saved=1", status_code=status.HTTP_302_FOUND)


@router.post("/profile/test-email")
def test_email(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RedirectResponse:
    from app.email_service import smtp_configured, send_session_completion_email
    if not current_user.email:
        return RedirectResponse(url="/profile?test=no_email", status_code=status.HTTP_302_FOUND)
    if not smtp_configured():
        return RedirectResponse(url="/profile?test=no_smtp", status_code=status.HTTP_302_FOUND)
    ok = send_session_completion_email(
        to_email=current_user.email,
        session_name="Prueba de conexión",
        results=[
            {"student_name": "Alumno de ejemplo", "total": 7.5, "max": 10.0, "status": "done"},
            {"student_name": "Otro alumno", "total": 5.0, "max": 10.0, "status": "done"},
        ],
        teacher_name=current_user.display_name,
    )
    return RedirectResponse(
        url=f"/profile?test={'ok' if ok else 'error'}",
        status_code=status.HTTP_302_FOUND,
    )
