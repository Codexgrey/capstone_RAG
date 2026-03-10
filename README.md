# Capstone - RAG System
CLaRa — using a shared backend interface, a React + TypeScript
frontend, and a Python backend for retrieval, orchestration, 
and LLM-based answer generation.


## Project Goal

Build a document Q&A system that:
- supports document upload and querying,
- compares multiple retrieval strategies fairly,
- returns grounded answers with citations,
- and provides an evaluation area for benchmarking retrieval 
  quality.


## Core Components

- `frontend/` — React + TypeScript user interface
- `backend/` — Python backend and integrated RAG pipeline
- `vector/` — vector retrieval research and implementation
- `keyword/` — keyword/BM25 retrieval research & implementation
- `clara/` — CLaRa retrieval research and implementation
- `shared/` — shared schemas, prompts, evaluation assets, 
   API contracts, and docs


## High-Level Flow

User → Frontend → Backend API → Retrieval Method → 
Top-k Chunks → LLM Generation → Answer + Citations


## Tech Stack

### Frontend
- React
- TypeScript
- Vite
- Axios
- Tailwind / Material UI

### Backend
- Python
- FastAPI


### RAG Stack
- SentenceTransformers
- FAISS or ChromaDB
- BM25
- OpenAI API or another LLM


## Notes

This repository is structured to support both independent team
development and final system integration.