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
    title = "Capstone RAG APO",
    description = "Backend for the Rag system",
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
    allow_method = ["*"],
    allow_headers = ["*"],
)

@app.on_event("startup")
def startup():
    settings.validate()
    init_db()
    print("RAG Backend running")

# endpoints for register and login
app.include_router(auth.router, prefix="/api")
# endpoint for document upload and status 
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