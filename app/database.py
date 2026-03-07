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


def get_db() -> Generator[Session, None, None]:
    assert _SessionLocal is not None, "DB not initialized — call init_db() first"
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()
