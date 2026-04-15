"""
ingestion/parser.py — Document Text Extraction
Extracts plain text from uploaded files.
Supported formats:
    .txt: read directly
    .pdf: extract with pypdf
    .docx: extract with python-docx (added for future use)
    .md: read directly

Used by:
    app/api/upload.py (Step 7) after saving the file to disk.
    Output is passed directly to chunker.py (Step 6).

Usage:
    from app.ingestion.parser import extract_text
    text = extract_text("/storage/documents/user123/report.pdf", "pdf")
"""
from pathlib import Path


def extract_text(filepath: str, file_type: str) -> str:
    # Extract plain text from a document file.
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


# Extractors 
def _extract_txt(path: Path) -> str:
    # Read plain text or markdown file.
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback for files with different encoding
        try:
            return path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf(path: Path) -> str:
    # Extract text from PDF using pypdf page by page to save memory.
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Run: pip install pypdf")

    reader = PdfReader(str(path))
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            # Only keep first 1000 chars per page to save memory
            pages.append(f"[PAGE {page_num}]\n{page_text[:1000]}")

        # Limit to first 10 pages for now
        if page_num >= 10:
            break

    if not pages:
        raise ValueError(f"No text extracted from: {path.name}")

    return "\n\n".join(pages)

def _extract_docx(path: Path) -> str:
    # Extract text from Word document using python-docx.
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
    # Extract the file extension from a filename.
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