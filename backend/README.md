# Backend — FastAPI RAG Orchestration

Khalid's FastAPI backend. Handles document ingestion, query routing,
LLM answer generation, and chat history persistence.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

Create `backend/.env`:
```env
POSTGRE_URL=postgresql://user:password@localhost:5432/ragdb
JWT_SECRET=<run: openssl rand -hex 32>
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60
STORAGE_PATH=./storage/documents
CHROMA_PERSIST_DIR=./chroma_storage
LLM_BACKEND=groq
GROQ_API_KEY=your_groq_api_key_here
RAG_PROJECT_ROOT=/absolute/path/to/capstone_RAG
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Default Admin

On first startup, the backend auto-creates:
- Email: `admin@admin.com`
- Password: `admin1234`

## Key Endpoints

| Method | Path                        | Description                  |
|--------|-----------------------------|------------------------------|
| POST   | `/api/auth/register`        | Register new user            |
| POST   | `/api/auth/login`           | Login, get JWT token         |
| POST   | `/api/upload`               | Upload + ingest document     |
| GET    | `/api/documents`            | List user's documents        |
| POST   | `/api/query`                | Ask a question               |
| POST   | `/api/chat/sessions`        | Create chat session          |
| GET    | `/api/chat/sessions`        | List chat sessions           |
| GET    | `/api/chat/sessions/{id}`   | Get session messages         |

## Retrieval Routing

`POST /api/query` body:
```json
{
  "question": "What is RAG?",
  "retrieval_method": "vector",   // "vector" | "keyword" | "hybrid"
  "top_k": 5,
  "session_id": null
}
```

The backend routes to the correct module adapter in `app/retrieval/`.

## Structure

```
backend/app/
├── api/          # HTTP endpoints (upload, query, auth)
├── config/       # Settings, database, dependencies
├── generation/   # LLM prompt builder, client, formatter
├── ingestion/    # Parser, chunker, ChromaDB indexer
├── models/       # SQLAlchemy models + Pydantic schemas
├── retrieval/    # Adapters bridging to retrieval modules
└── services/     # RAG orchestration (rag_service.py)
```
