import os
import glob
import numpy as np
from groq import Groq
from dotenv import load_dotenv

from utils.loader import load_document
from utils.chunker import chunk_text_with_metadata
from utils.prompts import build_prompt
from models.embedding_model import load_embedding_model
from indexing.vector_store import build_and_save_index, load_index
from indexing.bm25_indexer import build_inverted_index, build_bm25
from retrieval.vector_retriever import retrieve as vector_retrieve
from retrieval.bm25_retriever import normalise_query, retrieve_bm25
from preprocessing.preprocess import detect_language, tokenize_chunk

from retrieval.hybrid_retriever import reciprocal_rank_fusion

# ─── Configuration ─────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY      = os.getenv('GROQ_API_KEY', 'your_groq_api_key_here')
GENERATOR_MODEL   = 'llama-3.1-8b-instant'
CONTENT_FOLDER    = './content/School Regulations/'
INDEX_SAVE_PATH   = 'faiss_index.bin'
CHUNKS_SAVE_PATH  = 'chunk_records.npy'
CHUNK_SIZE        = 150
CHUNK_OVERLAP     = 30
TOP_K             = 3

groq_client = Groq(api_key=GROQ_API_KEY)

# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — INGESTION
# ═══════════════════════════════════════════════════════════════════
def get_files_from_folder(folder_path, extensions=('.txt', '.pdf', '.docx', '.md')):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
    return sorted(files)

DOCUMENT_PATHS = get_files_from_folder(CONTENT_FOLDER)
print('=' * 70)
print('PHASE 1 — INGESTION')
print('=' * 70)
print(f'{len(DOCUMENT_PATHS)} document(s) trouvé(s) :')
for path in DOCUMENT_PATHS:
    print(f'  - {path}')

all_chunk_records = []
ingestion_log     = []

print('\nIngestion en cours...')
print('-' * 70)

for doc_index, path in enumerate(DOCUMENT_PATHS):
    doc_id    = f'doc-{doc_index + 1:03d}'
    doc_title = (os.path.splitext(os.path.basename(path))[0]
                 .replace('_', ' ')
                 .replace('-', ' ')
                 .title())
    try:
        text, file_metadata = load_document(path)
        chunks = chunk_text_with_metadata(
            text,
            chunk_size     = CHUNK_SIZE,
            overlap        = CHUNK_OVERLAP,
            document_title = doc_title,
            source         = file_metadata['file_name'],
            document_id    = doc_id,
            file_metadata  = file_metadata
        )
        all_chunk_records.extend(chunks)
        ingestion_log.append({
            'document_id': doc_id,
            'file_name'  : path,
            'status'     : 'OK'
        })
        print(f'[{doc_id}] {doc_title}')
        print(f'  Chunks : {len(chunks)} | Status : OK\n')

    except Exception as e:
        ingestion_log.append({
            'document_id': doc_id,
            'file_name'  : path,
            'status'     : f'FAILED: {e}'
        })
        print(f'[{doc_id}] FAILED: {path}\n  Erreur : {e}\n')

print('-' * 70)
print(f'Documents traités : {len(DOCUMENT_PATHS)}')
print(f'Total chunks      : {len(all_chunk_records)}')

# ═══════════════════════════════════════════════════════════════════
# PHASE 2 — INDEXING
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('PHASE 2 — INDEXING')
print('=' * 70)

# ─── 2A. Vector Index (FAISS) ──────────────────────────────────────
print('\n[ 2A ] Vector Index (FAISS)')
print('-' * 70)

if os.path.exists(INDEX_SAVE_PATH) and os.path.exists(CHUNKS_SAVE_PATH):
    print('Index existant détecté — chargement depuis le disque...')
    embedding_model          = load_embedding_model()
    index, all_chunk_records = load_index(INDEX_SAVE_PATH, CHUNKS_SAVE_PATH)
else:
    print('Aucun index trouvé — création depuis zéro...')
    embedding_model = load_embedding_model()

    print('\nGénération des embeddings...')
    chunk_texts = [chunk['text'] for chunk in all_chunk_records]
    embeddings  = embedding_model.encode(
        chunk_texts,
        convert_to_numpy  = True,
        show_progress_bar = True
    )
    embeddings = np.array(embeddings, dtype='float32')
    print(f'Embeddings shape : {embeddings.shape}')

    print('\nConstruction du FAISS index...')
    index = build_and_save_index(
        embeddings,
        all_chunk_records,
        INDEX_SAVE_PATH,
        CHUNKS_SAVE_PATH
    )

# ─── 2B. Keyword Index (BM25) ──────────────────────────────────────
print('\n[ 2B ] Keyword Index (BM25)')
print('-' * 70)

# Detect language from first document
full_text        = ' '.join([c['text'] for c in all_chunk_records])
lang_code, nltk_lang = detect_language(full_text)
print(f'Detected language : {lang_code} → {nltk_lang}')

# Tokenize all chunks
print('Tokenizing chunks...')
tokenized_chunks = [
    tokenize_chunk(chunk['text'], nltk_lang)
    for chunk in all_chunk_records
]
print(f'Tokenized {len(tokenized_chunks)} chunks.')

# Build inverted index + BM25
inverted_index = build_inverted_index(all_chunk_records, tokenized_chunks)
bm25           = build_bm25(tokenized_chunks)

print(f'Inverted index built : {len(inverted_index)} unique terms')
print(f'BM25 store built     : {len(tokenized_chunks)} chunks indexed')

# ═══════════════════════════════════════════════════════════════════
# PHASE 3 — RETRIEVAL
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('PHASE 3 — RETRIEVAL')
print('=' * 70)

query = input('\nEntre ta question : ')
print()

# ─── 3A. Vector Retrieval ──────────────────────────────────────────
print('[ 3A ] Vector Retrieval (FAISS)')
print('-' * 70)

vector_results = vector_retrieve(
    query,
    embedding_model,
    index,
    all_chunk_records,
    top_k = TOP_K
)

for item in vector_results:
    print(f"Rank {item['rank']} | {item['chunk_id']} | Similarity: {item['similarity']:.4f}")
    print(f"  {item['text'][:100]}...")
    print()

# ─── 3B. Keyword Retrieval (BM25) ──────────────────────────────────
print('[ 3B ] Keyword Retrieval (BM25)')
print('-' * 70)

normalised_query = normalise_query(query, groq_client, GENERATOR_MODEL)
print(f'Original query   : {query}')
print(f'Normalised query : {normalised_query}')
print()

bm25_results = retrieve_bm25(
    normalised_query,
    bm25,
    all_chunk_records,
    inverted_index,
    nltk_lang = nltk_lang,
    top_k     = TOP_K
)

for item in bm25_results:
    print(f"Rank {item['rank']} | {item['chunk_id']} | BM25 Score: {item['bm25_score']:.4f}")
    print(f"  Matched terms : {item['matched_terms']}")
    print(f"  {item['text'][:100]}...")
    print()

# ═══════════════════════════════════════════════════════════════════
# PHASE 4 — GENERATION
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('PHASE 4 — GENERATION')
print('=' * 70)

# ─── 4A. Vector Response ───────────────────────────────────────────
print('\n[ 4A ] Response from Vector Retrieval')
print('-' * 70)

vector_prompt    = build_prompt(query, vector_results)
vector_response  = groq_client.chat.completions.create(
    model       = GENERATOR_MODEL,
    messages    = [{'role': 'user', 'content': vector_prompt}],
    max_tokens  = 500,
    temperature = 0.1,
)
vector_answer = vector_response.choices[0].message.content
print(vector_answer)

# ─── 4B. Keyword Response ──────────────────────────────────────────
print('\n[ 4B ] Response from Keyword Retrieval')
print('-' * 70)

bm25_prompt    = build_prompt(query, bm25_results)
bm25_response  = groq_client.chat.completions.create(
    model       = GENERATOR_MODEL,
    messages    = [{'role': 'user', 'content': bm25_prompt}],
    max_tokens  = 500,
    temperature = 0.1,
)
bm25_answer = bm25_response.choices[0].message.content
print(bm25_answer)


# ═══════════════════════════════════════════════════════════════════
# PHASE 5 — HYBRID RETRIEVAL (RRF Fusion)
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('PHASE 5 — HYBRID RETRIEVAL (Reciprocal Rank Fusion)')
print('=' * 70)

hybrid_results = reciprocal_rank_fusion(
    bm25_results   = bm25_results,
    vector_results = vector_results,
    k              = 60,
    top_k          = TOP_K
)

print(f'\nFusion complete — Top-{TOP_K} hybrid chunks :')
print('-' * 70)
for item in hybrid_results:
    print(f"Rank {item['rank']} | {item['chunk_id']} | RRF Score: {item['rrf_score']:.6f}")
    print(f"  Source     : {item['retrieval']}")
    if item['similarity'] is not None:
        print(f"  Similarity : {item['similarity']:.4f}")
    if item['bm25_score'] is not None:
        print(f"  BM25 Score : {item['bm25_score']:.4f}")
    print(f"  Text       : {item['text'][:100]}...")
    print()

# ─── Hybrid Generation ─────────────────────────────────────────────
print('[ 5A ] Response from Hybrid Retrieval')
print('-' * 70)

hybrid_prompt   = build_prompt(query, hybrid_results)
hybrid_response = groq_client.chat.completions.create(
    model       = GENERATOR_MODEL,
    messages    = [{'role': 'user', 'content': hybrid_prompt}],
    max_tokens  = 500,
    temperature = 0.1,
)
hybrid_answer = hybrid_response.choices[0].message.content
print(hybrid_answer)

# ─── Final Comparison ──────────────────────────────────────────────
print()
print('=' * 70)
print('FINAL COMPARISON')
print('=' * 70)
print(f'Query : {query}')
print()
print('Vector Retrieval chunks :')
for item in vector_results:
    print(f"  Rank {item['rank']} | {item['chunk_id']} | Similarity : {item['similarity']:.4f}")
print()
print('Keyword Retrieval chunks :')
for item in bm25_results:
    print(f"  Rank {item['rank']} | {item['chunk_id']} | BM25 : {item['bm25_score']:.4f} | Terms : {item['matched_terms']}")
print()
print('Hybrid Retrieval chunks (RRF) :')
for item in hybrid_results:
    print(f"  Rank {item['rank']} | {item['chunk_id']} | RRF : {item['rrf_score']:.6f} | Source : {item['retrieval']}")
print('=' * 70)


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'Query              : {query}')
print(f'Normalised query   : {normalised_query}')
print(f'Total chunks       : {len(all_chunk_records)}')
print()
print('Vector Retrieval :')
for item in vector_results:
    print(f"  Rank {item['rank']} | {item['chunk_id']} | Similarity : {item['similarity']:.4f}")
print()
print('Keyword Retrieval :')
for item in bm25_results:
    print(f"  Rank {item['rank']} | {item['chunk_id']} | BM25 Score : {item['bm25_score']:.4f} | Terms : {item['matched_terms']}")
print('=' * 70)

