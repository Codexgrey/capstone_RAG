from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.config.database import get_db, init_db
from app.config.settings import settings
from app.api import auth
from app.api import upload

app = FastAPI()

@app.on_event("startup")
def startup():
    settings.validate()
    init_db()
    print("RAG Backend running")

# endpoints for register and login
app.include_router(auth.router, prefix="/api")
# endpoint for document upload and status 
app.include_router(upload.router, prefix="/api") 

@app.get("/")
def root():
    return {"message": "Backend Running"}

@app.get("/db_test")
def db_test(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT version();"))
    version = result.fetchone()[0]

    return {"Postgres Version": version}