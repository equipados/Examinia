from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from app.auth import get_current_user
from app.db_models import User
from app.templates import templates

router = APIRouter()


@router.get("/help", response_class=HTMLResponse)
def help_page(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> HTMLResponse:
    return templates.TemplateResponse("help.html", {"request": request, "user": current_user})
