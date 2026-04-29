"""
ingestion/parser.py — Document Text Extraction
================================================
Extracts plain text from uploaded files.

Supported formats:
    .txt  → read directly
    .pdf  → extract with pypdf
    .docx → extract with python-docx (added for future use)
    .md   → read directly

Used by:
    app/api/upload.py (Step 7) after saving the file to disk.
    Output is passed directly to chunker.py (Step 6).

Usage:
    from app.ingestion.parser import extract_text
    text = extract_text("/storage/documents/user123/report.pdf", "pdf")
"""

from pathlib import Path


def extract_text(filepath: str, file_type: str) -> str:
    """
    Extract plain text from a document file.

    Args:
        filepath:  Absolute path to the saved file on disk.
        file_type: File extension without dot — "pdf", "txt", "md", "docx"

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist at the given path.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    file_type = file_type.lower().strip(".")

    if file_type in ("txt", "md"):
        return _extract_txt(path)

    elif file_type == "pdf":
        return _extract_pdf(path)

    elif file_type == "docx":
        return _extract_docx(path)

    else:
        raise ValueError(
            f"Unsupported file type: '{file_type}'. "
            f"Supported: txt, md, pdf, docx"
        )


# ── Extractors ─────────────────────────────────────────────────────────────────

def _extract_txt(path: Path) -> str:
    """Read plain text or markdown file."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback for files with different encoding
        try:
            return path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF — tries pypdf first, falls back to OCR for scanned PDFs."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Run: pip install pypdf")

    reader = PdfReader(str(path))
    pages  = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            pages.append(f"[PAGE {page_num}]\n{page_text[:1000]}")

    # If pypdf got nothing — PDF is likely scanned (image-based)
    if not pages:
        print(f"  ⚠️  pypdf found no text — trying OCR for {path.name}")
        pages = _extract_pdf_ocr(path)

    if not pages:
        raise ValueError(
            f"No text could be extracted from: {path.name}. "
            f"The PDF may be password protected or corrupted."
        )

    return "\n\n".join(pages)


def _extract_pdf_ocr(path: Path) -> list:
    try:
        from pdf2image import convert_from_path
        import pytesseract
        import os

        # Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        # Poppler path — your exact install location
        POPPLER_PATH = r"C:\Program Files\Release-25.12.0-0\poppler-25.12.0\Library\bin"

        images = convert_from_path(
            str(path),
            dpi=150,
            poppler_path=POPPLER_PATH
        )

        pages = []
        for i, image in enumerate(images, start=1):
            text = pytesseract.image_to_string(image)
            if text.strip():
                pages.append(f"[PAGE {i}]\n{text[:1000]}")

        if not pages:
            print(f"  ⚠️  OCR found no text in {path.name}")

        return pages

    except ImportError:
        print("  ⚠️  OCR not available — install: pip install pdf2image pytesseract")
        return []
    except Exception as e:
        print(f"  ⚠️  OCR failed: {e}")
        return []
def _extract_docx(path: Path) -> str:
    """Extract text from Word document using python-docx."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX parsing. "
            "Run: pip install python-docx"
        )

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def get_file_type(filename: str) -> str:
    """
    Extract the file extension from a filename.

    Args:
        filename: e.g. "report.pdf" or "notes.txt"

    Returns:
        Lowercase extension without dot: "pdf", "txt", "docx", "md"

    Raises:
        ValueError: If the file has no extension or unsupported type.
    """
    suffix = Path(filename).suffix.lower().strip(".")

    if not suffix:
        raise ValueError(f"Cannot determine file type from filename: {filename}")

    supported = {"pdf", "txt", "md", "docx"}
    if suffix not in supported:
        raise ValueError(
            f"Unsupported file type: '.{suffix}'. "
            f"Supported: {', '.join(sorted(supported))}"
        )

    return suffix