from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from app.templates import templates
from sqlalchemy.orm import Session

from app.auth import authenticate_user, create_access_token
from app.database import get_db

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
