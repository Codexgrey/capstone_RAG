"""
routers/auth.py: Authentication Routes
Endpoints:
    POST /auth/register: create a new user account
    POST /auth/login: verify credentials, return JWT token
  The frontend stores access_token and sends it in every request header:
    Authorization: Bearer eyJhbGci...
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config.database import get_db
from app.models.db_models import User, Log
from app.config.dependencies import hash_password, verify_password, create_access_token
from app.models.schemas import UserRegisterRequest, UserResponse, UserLoginRequest, TokenResponse

import uuid
from datetime import datetime

router = APIRouter(prefix="/auth", tags=["Authentication"])


# POST /auth/register
@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user"
)
def register(payload: UserRegisterRequest, db: Session = Depends(get_db)):
    """
    Creates a new user account.

    Checks:
      - Email is not already registered
      - Username is not already taken

    Password is hashed with bcrypt before storing — never stored in plain text.
    """
    # Check email uniqueness
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists."
        )

    # Check username uniqueness
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This username is already taken."
        )

    # Create user
    new_user = User(
        id=uuid.uuid4(),
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
        created_at=datetime.utcnow()
    )
    db.add(new_user)

    # Log the action
    db.add(Log(
        user_id=new_user.id,
        action="user_registered",
        detail=f"New user registered: {new_user.username}",
        timestamp=datetime.utcnow()
    ))

    db.commit()
    db.refresh(new_user)

    return new_user


# POST /auth/login
@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive a JWT token"
)
def login(payload: UserLoginRequest, db: Session = Depends(get_db)):
    """
    Verifies credentials and returns a JWT access token.

    The frontend stores this token and sends it in the
    Authorization header on every subsequent request:
        Authorization: Bearer <token>

    The token payload contains:
        { "sub": "<user_id>" }

    This is decoded by get_current_user() in dependencies.py
    to identify WHO is making each request.
    """
    # Find user by email
    user = db.query(User).filter(User.email == payload.email).first()

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password."
        )

    # Generate JWT: sub = user's UUID (this is how every request is tied to a user)
    token = create_access_token(data={"sub": str(user.id)})

    # Log the action
    db.add(Log(
        user_id=user.id,
        action="user_login",
        detail=f"User logged in: {user.username}",
        timestamp=datetime.utcnow()
    ))
    db.commit()

    # I used TokenResponse + UserResponse instead of returning a plain dict,
    # because this technique ensures FastAPI validates the response correctly
    # against the schema and includes all required fields (id, username, email, created_at).
    return TokenResponse(
        access_token=token,
        token_type = "bearer", 
        user=UserResponse.model_validate(user)
    )

    # {
    #     "access_token": token,
    #     "token_type": "bearer",
    #     "user": {
    #         "id": str(user.id),
    #         "username": user.username
    #     }
    # }