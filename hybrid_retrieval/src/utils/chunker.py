def chunk_text_with_metadata(
    text,
    chunk_size=150,
    overlap=30, # Le premier chunk va de 0 à 150 mots. Ensuite, on avance de 120 mots à chaque fois, ce qui fait que chaque chunk partage 30 mots avec le précédent pour garder le contexte.
    document_title='Untitled',
    source='unknown',
    document_id='doc-000',
    file_metadata=None
):
    if chunk_size <= 0:
        raise ValueError('chunk_size must be greater than 0.')
    if overlap < 0:
        raise ValueError('overlap cannot be negative.')
    if overlap >= chunk_size:
        raise ValueError('overlap must be smaller than chunk_size.')

    if file_metadata is None:
        file_metadata = {}

    words = text.split()
    chunk_records = []
    step = chunk_size - overlap

    for start_idx in range(0, len(words), step):
        end_idx = start_idx + chunk_size
        chunk_words = words[start_idx:end_idx]

        if not chunk_words:
            continue

        chunk_text = ' '.join(chunk_words)
        chunk_index = len(chunk_records)

        chunk_records.append({
            'document_id': document_id,
            'document_title': document_title,
            'source': source,
            'chunk_id': f'{document_id}-chunk-{chunk_index + 1}',
            'chunk_index': chunk_index,
            'start_word_index': start_idx,
            'end_word_index': min(end_idx, len(words)),
            'word_count': len(chunk_words),
            'overlap': overlap,
            'text': chunk_text,
            'metadata': {
                'file_name': file_metadata.get('file_name', source),
                'file_type': file_metadata.get('file_type', 'unknown'),
                'file_size_kb': file_metadata.get('file_size_kb', 0),
                'uploaded_at': file_metadata.get('uploaded_at', '')
            }
        })

        if end_idx >= len(words):
            break

    return chunk_records



# Chaque chunk = 150 mots  Mais : les 30 derniers mots du chunk 1 deviennent les 30 premiers mots du chunk 2