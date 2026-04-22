import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DATABASE_URL = os.getenv("POSTGRE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. "
        "Make sure your .env file exists and contains DATABASE_URL."
    )

# pool_pre_ping=True checks the connection is alive before using it.
# This prevents errors if PostgreSQL restarted while the app was running.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# autocommit=False - we manually commit transactions (safer)
# autoflush=False  - we control when changes are written to DB
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False
)

Base = declarative_base()

def get_db():
    #Yields a database session for a single request, then closes it.
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# creates all tables on startup
def init_db():             
    from app.models.db_models import User, Document, DocumentChunk, \
                                     ChatSession, ChatMessage, Log  # noqa
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created / verified.")