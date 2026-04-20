"""
src/utils/response_printer.py
Helper utility — formats and prints the final retrieval report to stdout.
Preserves the Answer / Evidence Used / Citations structure of the generated response.
"""

import textwrap
from typing import List, Dict, Any


def print_report(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    generated_output: str,
    index_path: str = ''
) -> None:
    """
    Print the full retrieval report: query info, top-k chunks, and generated response.

    Args:
        query:             The original user query.
        retrieved_results: List of result dicts from the retriever.
        generated_output:  Raw text output from the generator.
        index_path:        Path of the FAISS index used (optional, for display).
    """
    print(
        f'\nFINAL RETRIEVAL REPORT\n'
        f'{"=" * 110}\n'
        f'Query  : {query}\n'
        f'Top-K  : {len(retrieved_results)}\n'
        + (f'Index  : {index_path}\n' if index_path else '')
        + f'\nTOP RETRIEVED CHUNKS\n'
        f'{"-" * 110}'
    )

    for item in retrieved_results:
        metadata = item.get('metadata', {})
        print(
            f"Result {item['rank']}\n"
            f"  Document Title : {item['document_title']}\n"
            f"  Source         : {item['source']}\n"
            f"  File Type      : {metadata.get('file_type', 'unknown')}\n"
            f"  File Size      : {metadata.get('file_size_kb', 'N/A')} KB\n"
            f"  Chunk ID       : {item['chunk_id']}\n"
            f"  Chunk Index    : {item['chunk_index']}\n"
            f"  Word Count     : {item['word_count']}\n"
            f"  Distance       : {item['distance']:.4f}\n"
            f"  Similarity     : {item['similarity']:.4f}\n"
            f"  Citation       : {item['citation']}\n"
            f"  Text           :\n"
            f"{textwrap.fill(item['text'], width=120, initial_indent='    ', subsequent_indent='    ')}\n"
        )
        print()

    print(
        f'\nGENERATED RESPONSE\n'
        f'{"-" * 110}'
    )

    for line in generated_output.splitlines():
        stripped = line.strip()
        if not stripped:
            print()
        elif stripped.startswith('- '):
            print(textwrap.fill(stripped, width=120, initial_indent='  ', subsequent_indent='    '))
        else:
            print(textwrap.fill(stripped, width=120))
