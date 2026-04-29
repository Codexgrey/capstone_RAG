import os
from datetime import datetime, timezone
from pypdf import PdfReader
from docx import Document as DocxDocument

def load_document(path: str):
    """
    Loads a document from disk.
    Returns: (text: str, file_metadata: dict)
    Supports: .txt, .md, .pdf, .docx
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'Document not found: {path}')

    #  Extraction des informations du fichier
    ext = os.path.splitext(path)[1].lower()
    file_name = os.path.basename(path)
    file_size_kb = round(os.path.getsize(path) / 1024, 2)
    uploaded_at = datetime.now(timezone.utc).isoformat()

    # Lecture du contenu pour certains types de fichiers (ici : r = ouvre le fichier en mode lecture | f = lit tout le contenu du fichier )
    if ext in ('.txt', '.md'):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

    elif ext == '.pdf':
        reader = PdfReader(path)
        pages = [page.extract_text() or '' for page in reader.pages]
        text = '\n'.join(pages)

    elif ext == '.docx':
        doc = DocxDocument(path)
        text = '\n'.join(para.text for para in doc.paragraphs if para.text.strip())

    else:
        raise ValueError(f'Unsupported file type: "{ext}". Supported: .txt, .md, .pdf, .docx')

    file_metadata = {
        'file_name': file_name,
        'file_type': ext.lstrip('.'),
        'file_size_kb': file_size_kb,
        'uploaded_at': uploaded_at
    }

    return text, file_metadata