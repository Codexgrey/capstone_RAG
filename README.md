# Capstone - RAG System
Retrieval-Augmented Generation system for intelligent document Q & A. 
We're implementing 3 different approaches using a shared backend interface. 
React + TypeScript for frontend, Python at the backend for retrieval, orchestration, 
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
- `vector_retrieval/` — vector retrieval research and implementation
- `keyword_retrieval/` — keyword/BM25 retrieval research & implementation
- `clara_retrieval/` — CLaRa retrieval research and implementation
- `shared_data/` — shared schemas, prompts, evaluation assets, 
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