from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

APP_VERSION = "1.8.1"

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_path = Path(os.environ.get("WEB_DB_PATH", "corrector.db"))
    upload_dir = Path(os.environ.get("WEB_UPLOAD_DIR", "web_uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    from app.database import init_db
    init_db(db_path)

    from app.routers import submissions as sub_router
    poppler_path = os.environ.get("POPPLER_PATH")
    config_overrides: dict = {}
    if poppler_path:
        config_overrides["poppler_path"] = Path(poppler_path)
    sub_router.configure(
        upload_dir=upload_dir,
        db_path=str(db_path),
        config_overrides=config_overrides,
    )

    from app.routers.sessions import configure_solutions
    configure_solutions(db_path=str(db_path), config_overrides=config_overrides)

    # Re-encolar submissions que quedaron pendientes antes de un reinicio
    from app import scheduler
    scheduler.recover_pending(str(db_path), str(upload_dir), config_overrides)

    yield


app = FastAPI(title="Examinia", lifespan=lifespan)

# Static files
_static_dir = Path("static")
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Import shared templates and inject app version as global
from app.templates import templates as _shared_templates, _set_app_version  # noqa: F401
_set_app_version(APP_VERSION)


# Register routers
from app.routers import auth, dashboard, sessions, submissions as sub_module, reports
from app.routers.students import router as students_router
from app.routers.competencias import router as competencias_router

app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(sessions.router)
app.include_router(reports.router)
app.include_router(students_router)
app.include_router(competencias_router)
# Upload routes live under /submissions prefix but need session_id param
app.add_api_route(
    "/sessions/{session_id}/upload",
    sub_module.upload_form,
    methods=["GET"],
    response_class=__import__("fastapi.responses", fromlist=["HTMLResponse"]).HTMLResponse,
)
app.add_api_route(
    "/sessions/{session_id}/upload",
    sub_module.upload_pdfs,
    methods=["POST"],
)
app.include_router(sub_module.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=False)
