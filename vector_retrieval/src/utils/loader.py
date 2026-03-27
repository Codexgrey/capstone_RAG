"""
src/utils/loader.py
Helper utility — loads a raw text document from disk.

Test document setup (mirrors notebook Section 2 and Section 3):
    document_title  = 'Vector Retrieval RAG Notes'
    document_source = 'tests/sample.txt'
    document_id     = 'doc-001'

    The loader reads sample.txt, prints the same metadata header that
    appears in the notebook, and returns the full text for the pipeline.
"""

import os

# ---------------------------------------------------------------------------
# Test document constants  (Section 2 — Test Document Setup)
# ---------------------------------------------------------------------------
TEST_DOCUMENT_TITLE  = 'Vector Retrieval RAG Notes'
TEST_DOCUMENT_SOURCE = os.path.join('tests', 'sample.txt')
TEST_DOCUMENT_ID     = 'doc-001'


def load_document(path: str) -> str:
    """
    Read a UTF-8 text file and return its full contents as a string.

    Args:
        path: File path to the document.

    Returns:
        Full text content of the document.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        RuntimeError:      If the file cannot be read.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f'Document not found at path: {path}')
    except Exception as e:
        raise RuntimeError(f'Failed to read document at {path}: {e}')


def load_test_document() -> tuple:
    """
    Load the standard test document used across the notebook and main pipeline.

    Mirrors notebook Section 2 (test document metadata printout) and
    Section 3 (loader call + character-count + preview printout).

    Returns:
        Tuple of (document_text, document_title, document_source, document_id).
    """
    # Section 2 — print document metadata (mirrors notebook cell output)
    print('Sample document ready.')
    print('Title    :', TEST_DOCUMENT_TITLE)
    print('Source   :', TEST_DOCUMENT_SOURCE)
    print('Document ID:', TEST_DOCUMENT_ID)
    print()

    # Section 3 — load and preview (mirrors notebook cell output)
    document_text = load_document(TEST_DOCUMENT_SOURCE)
    print('Document length (characters):', len(document_text))
    print()
    print(document_text[:100])
    print()

    return document_text, TEST_DOCUMENT_TITLE, TEST_DOCUMENT_SOURCE, TEST_DOCUMENT_ID
