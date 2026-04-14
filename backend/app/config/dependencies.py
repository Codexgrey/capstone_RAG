"""
config/dependencies.py — Shared Auth Utilities
================================================
Contains:
  - Password hashing / verification  (bcrypt via passlib)
  - JWT token creation / decoding    (reads from settings.py)
  - get_current_user()               (FastAPI dependency — protects any route)

Reads JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_MINUTES from settings.py.
Never calls os.getenv() directly.

Usage in any router:
    from app.config.dependencies import get_current_user
    from app.models.db_models import User

    @router.get("/protected")
    def protected_route(current_user: User = Depends(get_current_user)):
        return {"user": current_user.username}
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.config.database import get_db
from app.models.db_models import User


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    # Hash a plain password using bcrypt.
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    # Check plain password against its bcrypt hash.
    return pwd_context.verify(plain, hashed)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    # Make a signed JWT with user ID.
    # Returns token for frontend auth.
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    FastAPI dependency — decodes JWT and returns the authenticated User.

    Add to any route that requires login:
        current_user: User = Depends(get_current_user)

    Raises HTTP 401 if:
      - Token is missing from the request
      - Token signature is invalid
      - Token has expired
      - User no longer exists in the database
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token. Please log in again.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception

    return user