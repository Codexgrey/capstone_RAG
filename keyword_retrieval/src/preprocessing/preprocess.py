"""
preprocessing
==============
Text cleaning, language detection, and tokenisation.

"""

import re

import nltk
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer
from nltk.tokenize import word_tokenize
from langdetect    import detect

# Download required NLTK data on first run (silent if already present)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# =============================================================================
# LANGUAGE MAP
# Maps langdetect codes → NLTK stopword corpus names.
# =============================================================================

LANGUAGE_MAP = {
    "en": "english",
    "fr": "french",
    "tr": "turkish",
    "de": "german",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "ar": "arabic",
    "nl": "dutch",
    "fi": "finnish",
    "no": "norwegian",
    "sv": "swedish",
    "ru": "russian",
}

# Shared stemmer instance
_stemmer = PorterStemmer()

# Minimal fallback stopwords used when NLTK corpus is not available.
# Deliberately short — only the most common function words.

_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "this", "that", "these", "those", "it", "its", "i", "you",
    "he", "she", "we", "they", "not", "no", "so", "if", "then", "than",
    "also", "into", "about", "after", "before", "between", "during",
    "each", "more", "other", "such", "which", "who", "whom", "when",
    "where", "while", "how", "what", "our", "their", "his", "her",
    "my", "your", "up", "out", "over", "under", "again", "further",
    "just", "both", "all", "any", "some", "most", "own", "same",
    "too", "very", "s", "t", "don", "ll", "ve", "re", "m", "d",
}


# =============================================================================
# STEP 2 — CLEAN TEXT
# =============================================================================

def clean_text(text: str) -> str:
   
    # Remove invisible control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)

    # Collapse multiple spaces/tabs into one space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =============================================================================
# STEP 3 — DETECT LANGUAGE
# =============================================================================

def detect_language(text: str) -> tuple[str, str]:
    """
    Automatically detect the document language.

    Samples the first 2 000 characters for speed.
    Falls back to English if detection fails or language is not in LANGUAGE_MAP.

    """
    try:
        lang_code = detect(text[:2000])
    except Exception:
        lang_code = "en"

    nltk_lang = LANGUAGE_MAP.get(lang_code, "english")
    return lang_code, nltk_lang


# =============================================================================
# STEP 5 — TOKENISE CHUNK
# =============================================================================

def tokenize_chunk(text: str, nltk_lang: str = "english") -> list[str]:
    """
    Convert a chunk of text into a clean list of stemmed tokens.

    """
    # Step 1 — lowercase
    text = text.lower()

    # Step 2 — tokenise
    # word_tokenize requires NLTK punkt_tab corpus.
   
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Extract sequences of alphabetic characters only
        tokens = re.findall(r"[a-z]+", text)

    # Step 3 — keep only alphabetic tokens (removes punctuation, numbers)
    tokens = [t for t in tokens if t.isalpha()]

    # Step 4 — remove stopwords
    # Try NLTK corpus first; fall back to the minimal built-in set.
    try:
        stop_words = set(stopwords.words(nltk_lang))
    except (OSError, LookupError):
        stop_words = _FALLBACK_STOPWORDS

    tokens = [t for t in tokens if t not in stop_words]

    # Step 5 — stem each token to its root form
    tokens = [_stemmer.stem(t) for t in tokens]

    return tokens