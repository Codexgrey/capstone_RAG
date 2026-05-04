# Capstone RAG Backend

Backend for a Retrieval-Augmented Generation system for intelligent document Q&A.
Supports three retrieval methods: **vector (FAISS)**, **keyword (BM25)**, and **hybrid (FAISS + BM25 + RRF)**.

---

## Project Structure

```
RAGAI/
 backend/                  ← Khalid — FastAPI backend (you are here)
 vector_retrieval/         ← Collins — FAISS vector retrieval
 keyword_retrieval/        ← Olivier — BM25 keyword retrieval
 hybrid_retrieval/         ← Nathan — FAISS + BM25 + RRF hybrid retrieval
 frontend/                 ← Patricia — React/TypeScript UI
```

---

## Setup

### 1. Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate      # Linux / macOS / WSL
venv\Scripts\activate         # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> If you are on WSL or Linux and get a `faiss` error, install it manually:
> ```bash
> pip install faiss-cpu
> ```

### 3. Environment Variables

Create a `.env` file inside the `backend/` folder.
Copy `.env.example` and fill in your values:

```bash
cp .env.example .env
```

**Full `.env` reference:**

```dotenv
# Database
POSTGRE_URL=postgresql://postgres:<your_password>@localhost:5432/ragdb

# JWT
# Generate a secure key:  openssl rand -hex 32  (run in Ubuntu/WSL terminal)
JWT_SECRET=<paste-generated-key-here>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Storage
STORAGE_PATH=./storage/documents
CHROMA_PERSIST_DIR=./chroma_storage

# LLM Backend 
LLM_BACKEND=groq
GROQ_API_KEY=<your-groq-api-key>

# Project Root (REQUIRED — set this to your machine's path)
# This must point to the folder that contains backend/, vector_retrieval/,
# keyword_retrieval/, and hybrid_retrieval/ as direct subfolders.
#
# Windows:
RAG_PROJECT_ROOT=C:\Users\<yourname>\OneDrive\Desktop\RAGAI
#
# WSL / Linux / macOS:
# RAG_PROJECT_ROOT=/home/<yourname>/RAGAI
```

> **Every teammate must set `RAG_PROJECT_ROOT` to their own local path.**
> Without it, Olivier's and Nathan's retrieval modules will not be found.

---

### 4. Database

Create the database in PostgreSQL:

```bash
sudo -u postgres psql
```

Inside psql:

```sql
CREATE DATABASE ragdb;
\q
```

The tables are created automatically when the backend starts for the first time.

#### One-time enum migration (run once after DB is created)

If you already have the database from an earlier version that used `"clara"` as a retrieval method, run this inside psql after connecting to `ragdb`:

```sql
\c ragdb
ALTER TYPE retrieval_method_enum ADD VALUE 'hybrid';
```

---

### 5. LLM Backend — Groq (default)

1. Get a free API key at [https://console.groq.com](https://console.groq.com)
2. Add to your `.env`:
   ```dotenv
   GROQ_API_KEY=your_key_here
   LLM_BACKEND=groq
   ```
3. Install the package:
   ```bash
   pip install groq
   ```

If no LLM backend is configured, the system runs in placeholder mode (hardcoded replies).
OpenAI and Anthropic are supported — add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to `.env` when ready.

---

### 6. Run the Backend

```bash
uvicorn app.main:app --reload
```

**Verify it's running:**
- `http://localhost:8000/` → `{"message": "Backend Running"}`
- `http://localhost:8000/db_test` → shows PostgreSQL version

---

## Retrieval Methods

The backend supports three retrieval methods selectable per query.
Each method has a full fallback chain so queries always return results.

| Method | Primary | Fallback 1 | Fallback 2 |
|--------|---------|------------|------------|
| `vector` | Collins — FAISS similarity | ChromaDB | PostgreSQL raw |
| `keyword` | Olivier — BM25 scoring | ChromaDB | PostgreSQL raw |
| `hybrid` | Nathan — FAISS + BM25 + RRF fusion | ChromaDB | PostgreSQL raw |

When you upload a document, **all three indexes are updated automatically** — no extra steps needed.

### How the Module Loading Works

Each team's retrieval module lives in a separate folder (`vector_retrieval/`, `keyword_retrieval/`, `hybrid_retrieval/`) and shares internal package names (`src`, `utils`, `retrieval`, etc.).

`backend/app/retrieval/module_loader.py` isolates each module's imports so they never conflict with each other. It:
1. Snapshots and removes conflicting `sys.modules` entries before loading
2. Sets a clean `sys.path` containing only that module's root
3. Restores everything after loading

This means the backend works without any renaming of folders inside each team's module.

---

## Frontend Setup

In a separate terminal, navigate to the `frontend/` folder:

```bash
cd ../frontend
npm install -D tailwindcss@3 postcss autoprefixer daisyui@4
npm install lucide-react react-hot-toast
npx tailwindcss init -p
npm run dev
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/auth/register` | Register a new user |
| `POST` | `/api/auth/login` | Login and get JWT token |
| `POST` | `/api/upload` | Upload a PDF/TXT document |
| `POST` | `/api/query` | Ask a question (`retrieval_method`: `vector` / `keyword` / `hybrid`) |
| `GET` | `/api/chat/sessions` | List chat sessions |
| `GET` | `/api/chat/sessions/{id}` | Get messages for a session |
| `DELETE` | `/api/chat/sessions/{id}` | Delete a session |

---

## Troubleshooting

**`No module named 'src.indexing.bm25_store'` or similar**
→ Make sure `RAG_PROJECT_ROOT` is set correctly in your `.env`.
→ Make sure you have empty `__init__.py` files in every subfolder of `keyword_retrieval/src/`:
```bash
# From the repo root (Linux/macOS/WSL)
touch keyword_retrieval/src/__init__.py
touch keyword_retrieval/src/{indexing,retrieval,preprocessing,utils,models,evaluation,generation}/__init__.py
```

**`No module named 'faiss'`**
→ Run: `pip install faiss-cpu`

**`invalid input value for enum retrieval_method_enum: "hybrid"`**
→ Run the enum migration in psql (see step 4 above).

**Olivier or Nathan index not found on first query**
→ Upload at least one document first. All three indexes are built on upload.

**Backend can't find `keyword_retrieval/` or `hybrid_retrieval/`**
→ `RAG_PROJECT_ROOT` is pointing to the wrong folder or is not set.
  It must point to the folder containing `backend/`, `keyword_retrieval/`, etc. as **direct subfolders**.