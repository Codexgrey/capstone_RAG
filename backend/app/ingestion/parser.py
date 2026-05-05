"""
ingestion/parser.py — Document Text Extraction

Extracts plain text from uploaded files.
Supported formats: .txt, .md, .pdf, .docx

PDF extraction strategy:
  1. pypdf  — fast, works for text-based PDFs
  2. OCR    — fallback for scanned/image PDFs (requires tesseract + poppler)
"""

import os
from pathlib import Path


def extract_text(filepath: str, file_type: str) -> str:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    ft = file_type.lower().strip(".")

    if ft in ("txt", "md"):
        return _extract_txt(path)
    elif ft == "pdf":
        return _extract_pdf(path)
    elif ft == "docx":
        return _extract_docx(path)
    else:
        raise ValueError(
            f"Unsupported file type: '{ft}'. Supported: txt, md, pdf, docx"
        )


def get_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower().strip(".")
    if not suffix:
        raise ValueError(f"Cannot determine file type: {filename}")
    supported = {"pdf", "txt", "md", "docx"}
    if suffix not in supported:
        raise ValueError(
            f"Unsupported file type: '.{suffix}'. Supported: {', '.join(sorted(supported))}"
        )
    return suffix


# ── Extractors ────────────────────────────────────────────────────────────────

def _extract_txt(path: Path) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf(path: Path) -> str:
    """pypdf first; OCR fallback for scanned PDFs."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Run: pip install pypdf")

    reader = PdfReader(str(path))
    pages  = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            pages.append(f"[PAGE {i}]\n{text.strip()[:2000]}")

    if not pages:
        print(f"  ⚠️  pypdf found no text — trying OCR for {path.name}")
        pages = _extract_pdf_ocr(path)

    if not pages:
        raise ValueError(
            f"No text extracted from: {path.name}. "
            "PDF may be password-protected or image-only."
        )

    return "\n\n".join(pages)


def _extract_pdf_ocr(path: Path) -> list:
    """OCR fallback using pdf2image + pytesseract. Cross-platform."""
    try:
        from pdf2image import convert_from_path
        import pytesseract

        # Let pytesseract find tesseract automatically (works on Linux/Mac).
        # On Windows, set TESSERACT_CMD env var to the tesseract.exe path.
        tesseract_cmd = os.environ.get("TESSERACT_CMD", "")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # On Windows, set POPPLER_PATH env var to the poppler bin directory.
        poppler_path = os.environ.get("POPPLER_PATH", None)

        kwargs = {"dpi": 150}
        if poppler_path:
            kwargs["poppler_path"] = poppler_path

        images = convert_from_path(str(path), **kwargs)

        pages = []
        for i, image in enumerate(images, start=1):
            text = pytesseract.image_to_string(image)
            if text.strip():
                pages.append(f"[PAGE {i}]\n{text.strip()[:2000]}")

        if not pages:
            print(f"  ⚠️  OCR found no text in {path.name}")
        return pages

    except ImportError:
        print("  ⚠️  OCR unavailable — install: pip install pdf2image pytesseract")
        return []
    except Exception as e:
        print(f"  ⚠️  OCR failed: {e}")
        return []


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Run: pip install python-docx")

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)
