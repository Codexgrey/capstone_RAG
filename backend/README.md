# Capstone RAG Backend
This is the backend for Retrieval-Augmented Generation system for intelligent document Q & A.
Will try to keep it updated as i update the files for better usability

## Setup

### Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```
### Instal Dependencies
pip install -r requirements.txt

### Environment Variables
Create a .env file in backend/:

DATABASE_URL=postgresql://postgres:<password>@localhost:5432/ragdb
# JWT secret key (must NOT be left as default)
# Generate a secure random key:
# In Ubuntu terminal (prefered) type:  openssl rand -hex 32
JWT_SECRET=<paste-generated-key>

# Token settings
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

## Database

### Create the database in PostgreSQL
```bash
sudo -u postgres psql
``` 
inside psql:
```Sql
    CREATE DATABASE ragdb;
```

## Run App
```bash
uvicorn app.main:app --reload
```

## Test
http://localhost:8000/ -> {"message":"Backend Running"}
http://localhost:8000/db_test -> shows PostgreSQL version