# 📚 Hybrid RAG Retrieval Project

This project implements a **hybrid document retrieval system** combining:
- Traditional text extraction
- OCR for scanned PDFs
- Chunking for retrieval preparation

## Setup Instructions

### 1. Install Python dependencies
pip install -r requirements.txt

---

### 2. Install Poppler
Download:
https://github.com/oschwartz10612/poppler-windows/releases/

Add to PATH:
C:\poppler\Library\bin

---

### 3. Install Tesseract OCR
Download:
https://github.com/UB-Mannheim/tesseract/wiki

Add to PATH:
C:\Program Files\Tesseract-OCR

---

## 🚀 How to Run the Project

### 1. Activate virtual environment (Windows)
```bash
.venv\Scripts\activate

#Go to project folder
cd hybrid_retrieval

Run the pipeline
python -m src.main

Expected Output
Documents ingestion starts
OCR is triggered automatically for scanned PDFs
Chunking is performed
Total number of chunks is displayed

## 📊 Pipeline Overview

PDF → Poppler → Images → Tesseract OCR → Text → Chunking → Retrieval Dataset