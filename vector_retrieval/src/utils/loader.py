"""
src/utils/loader.py
Helper utility — loads documents from disk.

Supports .txt, .md, .pdf, and .docx file types.
Returns document text and file-level metadata for each loaded file.
"""

import os
import glob
from datetime import datetime, timezone
from pypdf import PdfReader
from docx import Document as DocxDocument


# ---------------------------------------------------------------------------
# folder scanner
# ---------------------------------------------------------------------------

def get_files_from_folder(folder_path: str, extensions=('.txt', '.pdf', '.docx', '.md')) -> list:
    """
    Scan a folder and return a sorted list of all files matching the given extensions.

    Args:
        folder_path: Path to the folder to scan.
        extensions:  Tuple of file extensions to include.

    Returns:
        Sorted list of matching file paths.
    """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
    return sorted(files)


# ---------------------------------------------------------------------------
# core loader
# ---------------------------------------------------------------------------

def load_document(path: str) -> tuple:
    """
    Load a document from disk and return its text content plus file-level metadata.

    Supports: .txt, .md, .pdf, .docx

    Args:
        path: File path to the document.

    Returns:
        Tuple of (text: str, file_metadata: dict).
        Metadata keys: file_name, file_type, file_size_kb, uploaded_at

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file extension is not supported.
        RuntimeError:      If the file cannot be read.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'Document not found: {path}')

    ext          = os.path.splitext(path)[1].lower()
    file_name    = os.path.basename(path)
    file_size_kb = round(os.path.getsize(path) / 1024, 2)
    uploaded_at  = datetime.now(timezone.utc).isoformat()

    try:
        if ext in ('.txt', '.md'):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

        elif ext == '.pdf':
            reader = PdfReader(path)
            pages  = [page.extract_text() or '' for page in reader.pages]
            text   = '\n'.join(pages)

        elif ext == '.docx':
            doc  = DocxDocument(path)
            text = '\n'.join(para.text for para in doc.paragraphs if para.text.strip())

        else:
            raise ValueError(
                f'Unsupported file type: "{ext}". Supported: .txt, .md, .pdf, .docx'
            )

    except (ValueError, FileNotFoundError):
        raise
    except Exception as e:
        raise RuntimeError(f'Failed to read document at {path}: {e}')

    file_metadata = {
        'file_name':    file_name,
        'file_type':    ext.lstrip('.'),
        'file_size_kb': file_size_kb,
        'uploaded_at':  uploaded_at,
    }

    return text, file_metadata
