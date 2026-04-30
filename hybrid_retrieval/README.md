📚 Hybrid RAG Retrieval Project

This project implements a hybrid Retrieval-Augmented Generation (RAG) system that combines:

🔎 Dense retrieval (FAISS embeddings)
🔤 Sparse retrieval (BM25 keyword search)
⚡ Hybrid fusion (Reciprocal Rank Fusion - RRF)
📄 PDF text extraction + OCR for scanned documents
✂️ Chunking for retrieval optimization
🤖 LLM-based answer generation (Groq API)

⚙️ Setup Instructions
1. Clone the project
git clone repository
cd capstone_RAG/hybrid_retrieval

2. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
3. Install dependencies
pip install -r requirements.txt
🧩 External Dependencies
4. Install Poppler (PDF processing)

Required for PDF image rendering.

Download:
https://github.com/oschwartz10612/poppler-windows/releases/

Then add to PATH:

C:\poppler\Library\bin
5. Install Tesseract OCR

Required for scanned PDF text extraction.

Download:
https://github.com/UB-Mannheim/tesseract/wiki

Then add to PATH:

C:\Program Files\Tesseract-OCR
How to Run the Project
python -m src.main
Pipeline Overview

The system works in 4 main stages:

1. Document Processing
Load PDFs from /content
Extract text using:
native PDF text extraction
OCR (for scanned pages via Tesseract)
Normalize and clean text
2. Chunking
Documents are split into smaller chunks
Each chunk is stored with metadata:
document title
chunk id
position
word count
3. Indexing

Two retrieval indexes are built:

- Sparse Index (BM25)
Keyword-based search
Good for exact terms and legal/policy queries
- Dense Index (FAISS)
Embedding-based semantic search
Captures meaning, not just keywords
4. Retrieval + Generation

The system supports 3 retrieval modes:

- Vector Retrieval (FAISS)

Semantic similarity search using embeddings

- Keyword Retrieval (BM25)

Lexical search using token overlap

- Hybrid Retrieval (RRF)

Combines both rankings using Reciprocal Rank Fusion

Final Step: Generation

Retrieved chunks are passed to a Groq LLM, which:

synthesizes an answer
uses retrieved context
adds citations from source chunks
📊 Output Example
Vector results (semantic)
BM25 results (keyword-based)
Hybrid results (fused ranking)
Final generated answer with citations
