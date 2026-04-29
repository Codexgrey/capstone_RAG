from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.config.database import get_db, init_db
from app.config.settings import settings
from app.api import auth 
from app.api import upload
from app.api import query



app = FastAPI(
    title = "Capstone RAG API",
    description = "Backend for the RAG system",
    version = "1.0.0"
)

# Add the CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.on_event("startup")
def startup():
    settings.validate()
    init_db()
    _create_admin_if_missing()
    print("🚀 RAG Backend running")

def _create_admin_if_missing():
    """
    Auto-creates the admin account on first startup.
    Credentials:
        email:    admin@admin.com
        username: admin
        password: admin1234
    """
    from app.config.database import SessionLocal
    from app.models.db_models import User
    from app.config.dependencies import hash_password
    import uuid
    from datetime import datetime
 
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == "admin@admin.com").first()
        if not existing:
            admin = User(
                id            = uuid.uuid4(),
                username      = "admin",
                email         = "admin@admin.com",
                password_hash = hash_password("admin1234"),
                created_at    = datetime.utcnow()
            )
            db.add(admin)
            db.commit()
            print("✅ Admin account created: admin@admin.com / admin1234")
        else:
            print("✅ Admin account already exists")
    except Exception as e:
        print(f"⚠️  Admin creation error: {e}")
    finally:
        db.close()
 


# endpoints for register and login
app.include_router(auth.router, prefix="/api") 
app.include_router(upload.router, prefix="/api")
# endpoints for query and chat sessions
app.include_router(query.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Backend Running"}

@app.get("/db_test")
def db_test(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT version();"))
    version = result.fetchone()[0]

    return {"Postgres Version": version}