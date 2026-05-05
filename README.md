# Capstone RAG System

A research-oriented Retrieval-Augmented Generation (RAG) system comparing three retrieval
approaches: **Vector** (FAISS + sentence-transformers), **Keyword** (BM25), and **Hybrid**
(FAISS + BM25 + Reciprocal Rank Fusion).

## Architecture

```
User → Frontend (React/TypeScript/Vite)
          ↓
     Backend (FastAPI)
          ↓
   ┌────────┬──────────┬────────┐
   │ Vector │ Keyword  │ Hybrid │
   |(FAISS) │ (BM25)   │ (RRF)  │
   └────────┴──────────┴────────┘
          ↓
     LLM Generation (Groq)
          ↓
     Answer + Citations
```

## Repository Structure

```
capstone_RAG/
├── frontend/            # React + TypeScript UI (Patricia)
├── backend/             # FastAPI orchestration + ingestion (Khalid)
├── vector_retrieval/    # FAISS semantic retrieval (Collins)
├── keyword_retrieval/   # BM25 keyword retrieval (Olivier)
├── hybrid_retrieval/    # FAISS + BM25 + RRF hybrid (Nathan)
└── shared_data/         # Schemas, contracts, evaluation
    ├── schemas/         # JSON schemas (chunk, request, response, answer)
    └── api_contracts/   # Integration contracts
```

## Quick Start (Full System)

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL running locally
- Groq API key (free at https://console.groq.com)

### 1. Environment Setup

**Backend** — create `backend/.env`:
```env
POSTGRE_URL=        postgresql://user:password@localhost:5432/ragdb
JWT_SECRET=         <run: openssl rand -hex 32>
JWT_ALGORITHM=      HS256
JWT_EXPIRE_MINUTES= 60
STORAGE_PATH=       ./storage/documents
CHROMA_PERSIST_DIR= ./chroma_storage
LLM_BACKEND=        groq
GROQ_API_KEY=       your_groq_api_key_here
RAG_PROJECT_ROOT=   /absolute/path/to/capstone_RAG
```

**Frontend** — `frontend/.env` (already in repo):
```env
VITE_API_URL=http://localhost:8000
```

### 2. Install Dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install ( for first time frontend setup )

# Retrieval modules (install each)
cd vector_retrieval   && pip install -r requirements.txt && cd ..
cd keyword_retrieval  && pip install -r requirements.txt && cd ..
cd hybrid_retrieval   && pip install -r requirements.txt && cd ..
```

### 3. Start the System

Open **three terminals**:

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

**Terminal 3 — (optional) watch logs**

### 4. Use the System

1. Open http://localhost:5173
2. Login: `admin@admin.com` / `admin1234`
3. Click the **Controls** button (hamburger) to open the upload panel
4. Upload a PDF, DOCX, TXT, or MD file
5. Select a retrieval method in the chat input: **Vector**, **Keyword**, or **Hybrid**
6. Ask a question

## Shared Contract (Integration Reference)

All three retrieval modules implement the same interface. See `shared_data/api_contracts/`.

**Retrieval request:**
```json
{ "query": "...", "top_k": 5, "method": "vector|keyword|hybrid" }
```

**Retrieval response:**
```json
{ "query": "...", "method": "...", "results": [...], "latency_ms": 42.9 }
```

Each module exposes:
- `ingest(file_paths, chunk_size, chunk_overlap) → dict`
- `retrieve(query, top_k) → dict`

## Team

| Name    | Role              | Module             |
|---------|-------------------|--------------------|
| Patricia | Frontend         | `frontend/`        |
| Khalid  | Backend          | `backend/`         |
| Collins | Vector Retrieval | `vector_retrieval/`|
| Olivier | Keyword Retrieval| `keyword_retrieval/`|
| Nathan  | Hybrid Retrieval | `hybrid_retrieval/`|

**Supervisor:** Dr. Fuat Uyguroğlu — ENGI401, Cyprus International University
