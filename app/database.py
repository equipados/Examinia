from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

_engine = None
_SessionLocal = None


class Base(DeclarativeBase):
    pass


def init_db(db_path: Path = Path("corrector.db")) -> None:
    global _engine, _SessionLocal
    _engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    # Activar WAL mode para permitir lecturas concurrentes mientras se escribe
    @event.listens_for(_engine, "connect")
    def set_wal_mode(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        dbapi_conn.execute("PRAGMA busy_timeout=10000")

    _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    from app import db_models  # noqa: F401 — registers ORM classes
    Base.metadata.create_all(bind=_engine)

    # Migration: add student_id FK column to submissions if not present
    from sqlalchemy.exc import OperationalError
    with _engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE submissions ADD COLUMN student_id INTEGER REFERENCES students(id)"))
            conn.commit()
        except OperationalError:
            pass  # Column already exists

    # Migration: add thinking_tokens column to token_usage if not present
    with _engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE token_usage ADD COLUMN thinking_tokens INTEGER DEFAULT 0"))
            conn.commit()
        except Exception:
            pass  # Column already exists

    _backfill_students(_engine)


def _backfill_students(engine) -> None:
    """Link existing done submissions to Student rows based on student_name."""
    from app.db_models import Student, Submission
    from utils import normalize_identifier
    with Session(engine) as db:
        pending = db.query(Submission).filter(
            Submission.status == "done",
            Submission.student_name.isnot(None),
            Submission.student_id.is_(None),
        ).all()
        for sub in pending:
            name = sub.student_name
            norm = normalize_identifier(name)
            if not norm:
                continue
            course = sub.course_level
            candidates = db.query(Student).filter(Student.normalized_name == norm).all()
            if candidates:
                match = next((c for c in candidates if c.course_level == course), candidates[0])
                sub.student_id = match.id
                if course and not match.course_level:
                    match.course_level = course
            else:
                s = Student(
                    display_name=name.strip().title(),
                    normalized_name=norm,
                    course_level=course,
                )
                db.add(s)
                db.flush()
                sub.student_id = s.id
        db.commit()


def get_db() -> Generator[Session, None, None]:
    assert _SessionLocal is not None, "DB not initialized — call init_db() first"
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
