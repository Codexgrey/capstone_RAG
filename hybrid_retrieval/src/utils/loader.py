import os
from datetime import datetime, timezone

from pypdf import PdfReader
from docx import Document as DocxDocument

# OCR imports
from pdf2image import convert_from_path
import pytesseract


def load_document(path: str):
    """
    Loads a document from disk with OCR fallback for scanned PDFs.
    Returns: (text: str, file_metadata: dict)
    Supports: .txt, .md, .pdf, .docx
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f'Document not found: {path}')

    ext = os.path.splitext(path)[1].lower()
    file_name = os.path.basename(path)
    file_size_kb = round(os.path.getsize(path) / 1024, 2)
    uploaded_at = datetime.now(timezone.utc).isoformat()

    text = ""

    # =========================
    # TXT / MD
    # =========================
    if ext in ('.txt', '.md'):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

    # =========================
    # PDF (TEXT + OCR fallback)
    # =========================
    elif ext == '.pdf':
        reader = PdfReader(path)

        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

        text = '\n'.join(pages_text).strip()

        # 🔥 OCR FALLBACK if PDF is scanned
        if len(text) < 50:
            print(f"[OCR ACTIVATED] for {file_name}")

            images = convert_from_path(path)
            ocr_text = []

            for img in images:
                ocr_text.append(pytesseract.image_to_string(img))

            text = '\n'.join(ocr_text).strip()

    # =========================
    # DOCX
    # =========================
    elif ext == '.docx':
        doc = DocxDocument(path)
        text = '\n'.join(
            para.text for para in doc.paragraphs if para.text.strip()
        ).strip()

    else:
        raise ValueError(
            f'Unsupported file type: "{ext}". Supported: .txt, .md, .pdf, .docx'
        )

    file_metadata = {
        'file_name': file_name,
        'file_type': ext.lstrip('.'),
        'file_size_kb': file_size_kb,
        'uploaded_at': uploaded_at
    }

    return text, file_metadata