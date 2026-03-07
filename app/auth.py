from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Cookie, Depends, HTTPException, status
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.database import get_db
from app.db_models import User

SECRET_KEY = os.environ.get("WEB_SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(username: str, expires_hours: int = ACCESS_TOKEN_EXPIRE_HOURS) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(username: str, password: str, db: Session) -> User | None:
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.hashed_pw):
        return user
    return None


def get_current_user(
    access_token: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_302_FOUND,
        headers={"Location": "/login"},
    )
    if not access_token:
        raise credentials_exception
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise credentials_exception
    return user


def create_user(username: str, password: str) -> None:
    """CLI helper: python -c "from app.auth import create_user; create_user('admin','pw')" """
    from app.database import get_db, init_db
    init_db()
    db = next(get_db())
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        print(f"Usuario '{username}' ya existe.")
        return
    user = User(username=username, hashed_pw=hash_password(password))
    db.add(user)
    db.commit()
    print(f"Usuario '{username}' creado correctamente.")
