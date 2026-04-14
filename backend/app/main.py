from fastapi import FastAPI, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from database import get_db

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Backend Running"}

@app.get("/db_test")
def db_test(db: Session = Depends(get_db)):
    result = db.execute(text("SELECT version();"))
    version = result.fetchone()[0]

    return {"Postgres Version": version}