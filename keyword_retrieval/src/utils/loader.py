"""
loader
================

Supported formats
-----------------
  .txt / .md     built-in open
  .pdf           pymupdf  (fitz)
  .docx / .doc   python-docx
  .html / .htm   beautifulsoup4
  http/https URL requests + beautifulsoup4

When a document is loaded:
  - Local files  → copied in their ORIGINAL format into STORAGE_DIR

Storage folder is configured in config.py:
  C:\\Users\\DC\\Desktop\\keyword_RAG_01\\tests\\

Public API used by main.py
---------------------------
  load_document(source)     → (text, source_label)
  open_file_dialog()        → selected file path or None
  ensure_storage_dir()      → Path to storage folder
  SUPPORTED_EXTENSIONS      → list of supported file extensions
  _FORMAT_LOADERS           → dict mapping extension → loader function
"""

import re
import shutil
from pathlib import Path

from config import STORAGE_DIR as _STORAGE_DIR


# =============================================================================
# STORAGE FOLDER  (configured in config.py)
# =============================================================================

STORAGE_FOLDER = Path(_STORAGE_DIR)

# Supported file extensions (used by main.py to filter the storage list)
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"]

# File type filter shown in the Windows file picker dialog
FILETYPES_FOR_DIALOG = [
    ("All supported files", "*.pdf *.docx *.doc *.txt *.md *.html *.htm"),
    ("PDF files",           "*.pdf"),
    ("Word documents",      "*.docx *.doc"),
    ("Text / Markdown",     "*.txt *.md"),
    ("HTML files",          "*.html *.htm"),
    ("All files",           "*.*"),
]


def ensure_storage_dir() -> Path:
    """Create the storage folder if it does not exist yet. Returns the Path."""
    STORAGE_FOLDER.mkdir(parents=True, exist_ok=True)
    return STORAGE_FOLDER


# =============================================================================
# FORMAT-SPECIFIC TEXT EXTRACTORS
# Each function takes a Path and returns a plain text string.
# =============================================================================

def _load_txt(path: Path) -> str:
    """Read a plain text or markdown file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF using pymupdf (fitz)."""
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF loading.\n"
            "Run:  pip install pymupdf"
        )
    doc   = fitz.open(str(path))
#    ----------------------------Extract text from each page
    pages = [page.get_text() for page in doc] 
    doc.close()
    # ---------------------------"Combine pages"
    text  = "\n\n".join(pages) 
    if not text.strip():
        raise ValueError(
            f"No extractable text found in '{path.name}'.\n"
            "The PDF may be scanned — OCR is not supported in this version."
        )
    return text


def _load_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX loading.\n"
            "Run:  pip install python-docx"
        )
    try:
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        if path.suffix.lower() == ".doc":
            raise ValueError(
                f"'{path.name}' is a legacy .doc file.\n"
                "Convert it to .docx first: File → Save As in Word."
            ) from e
        raise


def _load_html(path: Path) -> str:
    """Extract visible text from an HTML file using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "beautifulsoup4 is required for HTML loading.\n"
            "Run:  pip install beautifulsoup4"
        )
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _load_url(url: str) -> str:
    """Fetch a web page and extract its visible text."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "requests and beautifulsoup4 are required for URL loading.\n"
            "Run:  pip install requests beautifulsoup4"
        )
    headers  = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Loader/1.0)"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    if not text.strip():
        raise ValueError(f"No readable text could be extracted from: {url}")
    return text


# Dispatch table: extension → loader function
_FORMAT_LOADERS = {
    ".txt":  _load_txt,
    ".md":   _load_txt,
    ".pdf":  _load_pdf,
    ".docx": _load_docx,
    ".doc":  _load_docx,
    ".html": _load_html,
    ".htm":  _load_html,
}


# =============================================================================
# NATIVE WINDOWS FILE PICKER DIALOG
# =============================================================================

def open_file_dialog() -> str | None:
    """
    Open a native Windows 'Open File' dialog.

    Returns the selected file path as a string, or None if the user
    cancelled without selecting a file.

    Requires tkinter — bundled with standard Python on Windows.
    If missing, reinstall Python and tick 'tcl/tk and IDLE'.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise ImportError(
            "tkinter is required for the file dialog.\n"
            "It ships with standard Python on Windows.\n"
            "If missing, reinstall Python and tick 'tcl/tk and IDLE'."
        )

    root = tk.Tk()
    root.withdraw()                    # hide the blank Tk window
    root.attributes("-topmost", True)  # bring dialog to the front

    file_path = filedialog.askopenfilename(
        title      = "Select a document to load",
        initialdir = str(Path.home()),
        filetypes  = FILETYPES_FOR_DIALOG,
    )

    root.destroy()
    return file_path if file_path else None


# =============================================================================
# STORAGE HELPERS
# =============================================================================

def _copy_to_storage(source_path: Path) -> Path:
    """
    Copy a local file into STORAGE_FOLDER keeping its original extension.
    If a file with the same name already exists, adds a counter suffix.
    Returns the path of the saved copy.
    """
    store   = ensure_storage_dir()
    dest    = store / source_path.name
    counter = 1
    while dest.exists():
        dest = store / f"{source_path.stem}_{counter}{source_path.suffix}"
        counter += 1
    shutil.copy2(source_path, dest)
    return dest


def _save_url_as_txt(text: str, url: str) -> Path:
    """
    Save extracted web page text as a .txt file in STORAGE_FOLDER.
    Filename is derived from the URL.
    Returns the saved file path.
    """
    from urllib.parse import urlparse

    store  = ensure_storage_dir()
    parsed = urlparse(url)
    base   = (parsed.netloc + parsed.path).replace("/", "_").replace(".", "_").strip("_") or "web_page"
    base   = re.sub(r'[<>:"/\\|?*]', "_", base)

    dest    = store / f"{base}.txt"
    counter = 1
    while dest.exists():
        dest = store / f"{base}_{counter}.txt"
        counter += 1

    dest.write_text(text, encoding="utf-8")
    return dest


# =============================================================================
# PUBLIC API
# =============================================================================

def load_document(source: str) -> tuple[str, str]:
    """
    Load a document from a file path or web URL.

    - Local files are copied to STORAGE_FOLDER in their ORIGINAL format.
    - Web pages are saved as .txt (no original file to preserve).

    Parameters
    ----------
    source : str
        Absolute or relative file path, OR an http/https URL.

    Returns
    -------
    (text, source_label)
        text         — extracted plain text for the pipeline
        source_label — original file path or URL string

    Raises
    ------
    FileNotFoundError  — file path does not exist
    ValueError         — unsupported format or no text could be extracted
    ImportError        — a required library is not installed
    """
    # ── Web URL ──────────────────────────────────────────────────────────
    if source.startswith("http://") or source.startswith("https://"):
        print(f"\n  [Loader] Fetching: {source}")
        text  = _load_url(source)
        saved = _save_url_as_txt(text, source)
        print(f"  [Loader] {len(text):,} chars | {len(text.split()):,} words")
        print(f"  [Loader] Saved (web → .txt): {saved.name}")
        return text, source

    # ── Local file ───────────────────────────────────────────────────────
    path = Path(source).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext not in _FORMAT_LOADERS:
        raise ValueError(
            f"Unsupported file type: '{ext}'\n"
            f"Supported: {', '.join(_FORMAT_LOADERS)}  or an http/https URL."
        )

    print(f"\n  [Loader] Reading {ext.upper()}: {path.name}")
    text  = _FORMAT_LOADERS[ext](path)
    saved = _copy_to_storage(path)

    print(f"  [Loader] {len(text):,} chars | {len(text.split()):,} words")
    print(f"  [Loader] Stored original {ext.upper()}: {saved.name}")

    return text, str(path)