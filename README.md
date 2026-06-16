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
POSTGRE_URL=        postgresql://<user:password>@localhost:5432/<your_local_rag_dbname>
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
7. Use the **TriviaQA Benchmark** row (Groundedness / Precision) to run evaluations — the
   active button turns red ("Close Groundedness" / "Close Precision") to indicate it can be
   clicked again to close the panel

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

## Evaluation (TriviaQA Benchmarks)

The system includes two complementary TriviaQA-based benchmark tracks (Joshi et al., ACL 2017),
both runnable from the **TriviaQA Benchmark** button row in the UI.

### Groundedness (EM/F1) — `EvalBenchmark`

A static 5,000-question bank scored with official Exact Match (EM) and token-level F1 metrics
against the system's own 9-document corpus.

- `GET  /api/evaluate/triviaqa/stats` — bank statistics (domain/answer-type breakdown)
- `GET  /api/evaluate/triviaqa/questions` — paginated question list (no answers)
- `POST /api/evaluate/triviaqa/score` — score one predicted answer against ground truth
- `POST /api/evaluate/triviaqa/run` — run the live RAG pipeline on a subset and score results

### Close Precision — `PrecisionBenchmark`

An isolated per-question Precision@k / Recall@k / Answer-in-Context@k / Top-1 / MRR benchmark,
streaming live from `trivia_qa/rc/validation` on HuggingFace. For each question, that question's
own evidence is chunked and ingested in isolation, retrieved, scored, then deleted before the
next question starts — so questions never share an index and gold relevance is unambiguous.

- `GET  /api/evaluate/triviaqa-precision/status?retrieval_method=vector|keyword|hybrid` —
  resume index + cumulative metrics for a method
- `POST /api/evaluate/triviaqa-precision/reset?retrieval_method=vector|keyword|hybrid` —
  discard checkpoint, restart from question 0
- `POST /api/evaluate/triviaqa-precision/run` — run the next batch
  (`retrieval_method`, `top_k`, `batch_size`, `chunk_size`, `chunk_overlap`)

Resumable via per-method checkpoint files
(`backend/app/evaluation/triviaqa_precision_checkpoint_{method}.json`).

**Hybrid isolation note:** Hybrid has no index of its own — it fuses results from the Vector
and Keyword indexes via RRF. For Hybrid runs, each question's evidence is ingested into
**both** the Vector and Keyword indexes (and removed from both during cleanup), with the
hybrid adapter's in-memory FAISS/BM25 caches invalidated on each step so retrieval always
reflects the current isolated index — never the production corpus or stale cached state.

Both benchmark runs do **not** appear in chat history — they execute with `persist=False`
so evaluation traffic stays separate from real conversations.

### Index maintenance — `delete()`

Vector and Keyword adapters (and their backend bridges) now expose `delete(document_id)`,
removing that document's chunks and rebuilding the index/model from the remaining chunks.
This underlies the Close Precision benchmark's per-question isolation and is also available
for general document removal.

## Chat History Management

Each user's conversations are stored in PostgreSQL and persist across logins:

- `GET    /api/chat/sessions` — list sessions
- `GET    /api/chat/sessions/{id}` — get messages in a session
- `PATCH  /api/chat/sessions/{id}` — rename a session
- `DELETE /api/chat/sessions/{id}` — delete one session
- `DELETE /api/chat/sessions` — delete **all** sessions for the current user

## Team

| Name    | Role              | Module             |
|---------|-------------------|--------------------|
| Patricia | Frontend         | `frontend/`        |
| Khalid  | Backend          | `backend/`         |
| Collins | Vector Retrieval | `vector_retrieval/`|
| Olivier | Keyword Retrieval| `keyword_retrieval/`|
| Nathan  | Hybrid Retrieval | `hybrid_retrieval/`|

**Supervisor:** Dr. Fuat Uyguroğlu — ENGI401, Cyprus International University
