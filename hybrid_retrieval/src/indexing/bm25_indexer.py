import collections
from rank_bm25 import BM25Okapi


# Build Inverted Index 
def build_inverted_index(chunk_records: list, tokenized_chunks: list) -> dict:
    index = {}

    for chunk_idx, (chunk, tokens) in enumerate(zip(chunk_records, tokenized_chunks)):
        chunk_id = chunk['chunk_id']

        term_positions = collections.defaultdict(list)

        for pos, token in enumerate(tokens):
            token = token.lower().strip()

            if len(token) <= 2:
                continue

            term_positions[token].append(pos)

        for term, positions in term_positions.items():
            if term not in index:
                index[term] = {'doc_freq': 0, 'postings': []}

            index[term]['doc_freq'] += 1

            index[term]['postings'].append({
                'chunk_id': chunk_id,
                'chunk_idx': chunk_idx,
                'tf': len(positions) / len(tokens),  # normalized TF
                'positions': positions,
            })

    return index


# Build BM25 Store 
def build_bm25(tokenized_chunks: list) -> BM25Okapi:
    """
        Builds a BM25Okapi object from tokenized chunks.
        Used for fast relevance-scored retrieval.
    """
    if not tokenized_chunks:
        raise ValueError('tokenized_chunks is empty. Cannot build BM25.')

    return BM25Okapi(tokenized_chunks)


# Inspect Term 
def inspect_term(term: str, inverted_index: dict, stemmer) -> None:
    stemmed = stemmer.stem(term.lower())
    entry = inverted_index.get(stemmed)

    if not entry:
        print(f'Term "{term}" (stemmed: "{stemmed}") not found in index.')
        return

    print(f'Term     : "{term}" (stemmed → "{stemmed}")')
    print(f'doc_freq : {entry["doc_freq"]} chunk(s)')

    for p in entry['postings']:
        print(
            f'  chunk_id={p["chunk_id"]} | '
            f'tf={p["tf"]:.4f} | '
            f'positions={p["positions"][:5]}'
        )