"""
config/database.py — PostgreSQL connection and auto-migration helpers.

On startup, _ensure_enum_values() adds any missing values to the
retrieval_method_enum PostgreSQL enum type. This means teammates never
need to run manual ALTER TYPE commands after a code update.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("POSTGRE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. "
        "Make sure your .env file exists and contains POSTGRE_URL."
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def get_db():
    """Yield a database session for a single request, then close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_enum_values():
    """
    Add any missing values to retrieval_method_enum without dropping the type.
    Safe to run on every startup — no-ops if values already exist.
    PostgreSQL requires each ADD VALUE in its own transaction.
    """
    required = ["vector", "keyword", "hybrid", "none"]
    with engine.connect() as conn:
        for value in required:
            try:
                conn.execute(
                    text(f"ALTER TYPE retrieval_method_enum ADD VALUE IF NOT EXISTS '{value}'")
                )
                conn.commit()
            except Exception:
                conn.rollback()   # value already present — ignore


def init_db():
    """Create all tables and ensure enum values are up to date."""
    from app.models import db_models  # noqa: F401 — registers models with Base
    Base.metadata.create_all(bind=engine)
    try:
        _ensure_enum_values()
    except Exception as e:
        # Non-fatal: enum may not exist yet on first run (create_all makes it)
        print(f"  ℹ️  Enum migration note: {e}")
