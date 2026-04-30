import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from langdetect import detect

# Download NLTK resources
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)

# Language Map 
LANG_MAP = {
    'en': 'english',
    'fr': 'french',
    'tr': 'turkish',
    'de': 'german',
    'es': 'spanish',
    'it': 'italian',
    'pt': 'portuguese',
    'ar': 'arabic',
}

# Using SnowballStemmer instead of PorterStemmer
# for better multilingual support (English, French, German, Spanish, etc.)
STEMMERS = {
    "english": SnowballStemmer("english"),
    "french": SnowballStemmer("french"),
    "german": SnowballStemmer("german"),
    "spanish": SnowballStemmer("spanish"),
    "italian": SnowballStemmer("italian"),
    "portuguese": SnowballStemmer("portuguese"),
}

# Language Detection 
def detect_language(text: str) -> tuple:
    """
    Detects the language of the text.
    Returns (lang_code, nltk_corpus_name).
    Falls back to English if unrecognised.
    """
    try:
        code = detect(text[:2000])
    except Exception:
        code = 'en'

    nltk_name = LANG_MAP.get(code, 'english')
    return code, nltk_name


# Text Cleaning 
def clean_text(text: str) -> str:
    """
    Strips formatting noise and normalises whitespace.
    """
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    text = re.sub(r'[ \t]+', ' ',   text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


# Tokenizer
def tokenize_chunk(text: str, nltk_lang: str = 'english') -> list:
    """
    Lowercase → tokenise → remove stopwords → stem.
    Returns a list of stemmed tokens.
    """
    text = text.lower()
    tokens = word_tokenize(text)

    try:
        stop_words = set(stopwords.words(nltk_lang))
    except OSError:
        stop_words = set(stopwords.words('english'))

    # Use correct stemmer based on language
    stemmer = STEMMERS.get(nltk_lang, STEMMERS["english"])

    tokens = [
        stemmer.stem(t)
        for t in tokens
        if t.isalpha() and t not in stop_words
    ]

    return tokens