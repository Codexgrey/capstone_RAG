"""
main
========
Keyword Retrieval RAG — main entry point.

This file controls the full pipeline step by step.

Pipeline order:
    1. Load document
    2. Clean text
    3. Detect language
    4. Chunk text
    5. Tokenise chunks
    6. Build inverted index + BM25
    7. Normalise query
    8. Retrieve top-K chunks
    9. Build prompt
   10. Generate answer
"""

import textwrap
from pathlib import Path
from config import (
    GROQ_API_KEY,
    QUERY_MODEL_NAME,
    GENERATOR_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
)


# =============================================================================
# PIPELINE STATE
# =============================================================================

state = {
    # ── Current document (updated on each load) ──────────────────
    "text":              None,   # raw text of the most recently loaded doc
    "source":            None,   # file path or URL of the most recently loaded doc
    "cleaned_text":      None,   # cleaned text of the most recently loaded doc
    "lang_code":         None,   # detected language code, e.g. 'en', 'fr'
    "nltk_lang":         None,   # NLTK stopword set name, e.g. 'english'

    # ── Accumulated across ALL loaded documents ───────────────────
    # These grow with every document loaded — never reset unless
    # the user explicitly resets the pipeline.
    "all_chunk_records":    [],  # chunks from every document loaded so far
    "all_tokenized_chunks": [],  # token lists from every document loaded
    "loaded_documents":     [],  # source labels of all docs loaded so far

    # ── Index and BM25 (rebuilt after each document load) ─────────
    # Always covers ALL documents in the session.
    "chunk_records":     None,   # kept in sync with all_chunk_records
    "tokenized_chunks":  None,   # kept in sync with all_tokenized_chunks
    "inverted_index":    None,   # word → chunk lookup across ALL documents
    "bm25":              None,   # BM25 model across ALL documents

    # ── Query and answer ──────────────────────────────────────────
    "query":             None,   # original user question
    "normalised_query":  None,   # keywords extracted from question
    "retrieved_results": None,   # top-K most relevant chunks
    "prompt":            None,   # final prompt sent to the LLM
    "answer":            None,   # generated answer from the LLM
}


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

LINE = "─" * 52

def show_header(title):
    print(f"\n{'=' * 52}")
    print(f"  {title}")
    print(f"{'=' * 52}\n")

def show_success(message):
    print(f"  ✓  {message}")

def show_error(message):
    print(f"  ✗  {message}")

def show_info(message):
    print(f"  →  {message}")

def wait_for_enter():
    input("\n  Press Enter to go back to the menu...")

def check_required(*keys):
    """Check that required pipeline steps have been completed."""
    missing = [k for k in keys if state[k] is None]
    if missing:
        show_error(f"You need to complete these steps first: {', '.join(missing)}")
        return False
    return True


# =============================================================================
# AUTO-PIPELINE — runs Steps 2 to 6 automatically after every document load
# Chunks accumulate across ALL documents so queries search everything.
# =============================================================================

def _auto_run_steps_2_to_6():
    """
    Automatically run Steps 2 to 6 right after a document is loaded.

    MULTI-DOCUMENT SUPPORT:
    Chunks from every document are accumulated in:
        state["all_chunk_records"]     — grows with each document
        state["all_tokenized_chunks"]  — grows with each document

    The inverted index and BM25 model are rebuilt each time to cover
    ALL documents loaded so far — not just the most recent one.
    This means queries search across every document in the session.
    """
    # ── Step 2 — Clean text ───────────────────────────────────────
    print("  [Step 2]  Cleaning text...", end=" ", flush=True)
    try:
        from preprocessing.preprocess import clean_text
        cleaned = clean_text(state["text"])
        state["cleaned_text"] = cleaned
        print(f"done  ({len(cleaned):,} chars)")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    # ── Step 3 — Detect language ──────────────────────────────────
    print("  [Step 3]  Detecting language...", end=" ", flush=True)
    try:
        from preprocessing.preprocess import detect_language
        lang_code, nltk_lang = detect_language(state["cleaned_text"])
        state["lang_code"]   = lang_code
        state["nltk_lang"]   = nltk_lang
        print(f"done  ({lang_code} → {nltk_lang})")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    # ── Step 4 — Chunk this document and accumulate ───────────────
    print("  [Step 4]  Chunking text...", end=" ", flush=True)
    try:
        from utils.chunker import chunk_text_with_metadata

        source    = state["source"] or "document"
        doc_title = Path(source).stem if not source.startswith("http") else "web_document"
        doc_num   = len(state["loaded_documents"]) + 1
        doc_id    = f"doc-{doc_num:03d}"

        new_chunks = chunk_text_with_metadata(
            state["cleaned_text"],
            chunk_size    = DEFAULT_CHUNK_SIZE,
            overlap       = DEFAULT_CHUNK_OVERLAP,
            document_title= doc_title,
            source        = source,
            document_id   = doc_id,
            lang_code     = state["lang_code"],
        )

        # Accumulate — add new chunks to the global list
        state["all_chunk_records"].extend(new_chunks)
        state["loaded_documents"].append(source)
        state["chunk_records"] = state["all_chunk_records"]

        total = len(state["all_chunk_records"])
        ndocs = len(state["loaded_documents"])
        print(f"done  ({len(new_chunks)} new chunks | {total} total across {ndocs} doc(s))")

    except Exception as e:
        print(f"FAILED: {e}")
        return

    # ── Step 5 — Tokenise new chunks and accumulate ───────────────
    print("  [Step 5]  Tokenising chunks...", end=" ", flush=True)
    try:
        from preprocessing.preprocess import tokenize_chunk

        new_tokenized = [
            tokenize_chunk(c["text"], state["nltk_lang"])
            for c in new_chunks
        ]
        state["all_tokenized_chunks"].extend(new_tokenized)
        state["tokenized_chunks"] = state["all_tokenized_chunks"]

        print(f"done  ({len(state['all_tokenized_chunks'])} total chunks tokenised)")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    # ── Step 6 — Rebuild index + BM25 across ALL documents ────────
    print("  [Step 6]  Rebuilding index + BM25 (all documents)...", end=" ", flush=True)
    try:
        from indexing.indexer    import build_inverted_index
        from indexing.bm25_store import build_bm25

        inv_idx = build_inverted_index(
            state["all_chunk_records"],
            state["all_tokenized_chunks"],
        )
        bm25 = build_bm25(state["all_tokenized_chunks"])

        state["inverted_index"] = inv_idx
        state["bm25"]           = bm25

        print(f"done  ({len(inv_idx):,} unique terms across all documents)")
    except Exception as e:
        print(f"FAILED: {e}")
        return

    ndocs  = len(state["loaded_documents"])
    ntotal = len(state["all_chunk_records"])
    print()
    show_success(f"{ndocs} document(s) indexed | {ntotal} total chunks in index.")
    print("  Load another document or go to Step 7 to ask a question.")


# =============================================================================
# STEP 1 — LOAD DOCUMENT
# =============================================================================

def _do_load(source, load_document_fn):
    """
    Internal helper: load a document from a path or URL,
    save it to storage, update pipeline state, show a preview,
    then automatically run Steps 2 to 6.
    """
    try:
        text, label = load_document_fn(source)

        state["text"]   = text
        state["source"] = label

        name = Path(label).name if not label.startswith("http") else label
        show_success(f"Document loaded:  {name}")
        show_success(f"Size:  {len(text):,} characters  |  {len(text.split()):,} words")

        print(f"\n  --- Text preview (first 300 characters) ---\n")
        print(textwrap.fill(
            text[:300], width=66,
            initial_indent="    ",
            subsequent_indent="    "
        ))
        if len(text) > 300:
            print("    ...")

        # Auto-run Steps 2 → 6 immediately after loading
        print(f"\n  {LINE}")
        print("  Running Steps 2 → 6 automatically...\n")
        _auto_run_steps_2_to_6()

    except FileNotFoundError as e:
        show_error(f"File not found: {e}")
    except ValueError as e:
        show_error(f"Could not read file: {e}")
    except Exception as e:
        show_error(f"Unexpected error: {e}")

    wait_for_enter()


def step_load_document():
    """
    STEP 1 — Load Document
    ----------------------
    Lets the user choose how to load a document.
    Supported formats: PDF, DOCX, TXT, MD, HTML, or a web URL.
    The original file is copied to the tests/ storage folder.
    After loading, Steps 2 to 6 run automatically.
    """
    from utils.loader import (
        load_document,
        open_file_dialog,
        ensure_storage_dir,
        SUPPORTED_EXTENSIONS,
        _FORMAT_LOADERS,
    )

    while True:
        show_header("Step 1 — Load Document")

        print("  How do you want to load your document?\n")
        print("  [1]  Browse my laptop  (opens file picker)")
        print("  [2]  Enter a web URL   (https://...)")
        print("  [3]  Type / paste a file path manually")
        print("  [4]  Choose from already loaded documents")
        print("  [0]  Back to main menu")
        print()

        choice = input("  Your choice: ").strip()

        # ── [0] Back ──────────────────────────────────────────────
        if choice == "0":
            return

        # ── [1] File picker dialog ────────────────────────────────
        elif choice == "1":
            while True:
                print()
                show_info("Opening file picker — check your taskbar if it does not appear.")
                print()
                selected = open_file_dialog()
                if not selected:
                    print("\n  No file was selected.\n")
                    print("  [1]  Try again")
                    print("  [0]  Back to loader menu")
                    print()
                    retry = input("  Your choice: ").strip()
                    if retry == "1":
                        continue
                    else:
                        break
                else:
                    _do_load(selected, load_document)
                    return

        # ── [2] Web URL ───────────────────────────────────────────
        elif choice == "2":
            while True:
                print()
                url = input("  Paste the URL  (or type 0 to go back): ").strip()
                if url == "0":
                    break
                if not url.startswith("http://") and not url.startswith("https://"):
                    show_error("URL must start with http:// or https://  — please try again.")
                    continue
                _do_load(url, load_document)
                return

        # ── [3] Manual file path ──────────────────────────────────
        elif choice == "3":
            print()
            path_str = input("  Enter the file path  (or type 0 to go back): ").strip().strip('"').strip("'")
            if path_str == "0":
                continue
            _do_load(path_str, load_document)
            return

        # ── [4] Pick from storage folder ──────────────────────────
        elif choice == "4":
            store = ensure_storage_dir()
            stored_files = sorted(
                f for f in store.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS + [".txt"]
            )
            if not stored_files:
                print(f"\n  Storage folder is empty: {store}")
                print("  Load a document first using options 1, 2, or 3.\n")
                wait_for_enter()
                continue

            while True:
                print(f"\n  Documents in storage folder:\n  {store}\n")
                for i, f in enumerate(stored_files, start=1):
                    size_kb = f.stat().st_size / 1024
                    print(f"    [{i}]  {f.name}  ({size_kb:.1f} KB)")
                print(f"\n    [0]  Back to loader menu")
                print()
                pick = input("  Your choice: ").strip()

                if pick == "0":
                    break

                try:
                    idx = int(pick) - 1
                    if not (0 <= idx < len(stored_files)):
                        raise ValueError
                except ValueError:
                    show_error("Please enter a valid number from the list.")
                    continue

                chosen = stored_files[idx]

                # Read directly — do NOT call load_document() to avoid
                # copying the file again into storage.
                try:
                    ext = chosen.suffix.lower()
                    if ext in _FORMAT_LOADERS:
                        text = _FORMAT_LOADERS[ext](chosen)
                    else:
                        text = chosen.read_text(encoding="utf-8", errors="replace")

                    state["text"]   = text
                    state["source"] = str(chosen)

                    show_success(f"Document loaded:  {chosen.name}")
                    show_success(f"Size:  {len(text):,} characters  |  {len(text.split()):,} words")
                    print(f"\n  Preview:\n")
                    print(textwrap.fill(
                        text[:300], width=66,
                        initial_indent="    ",
                        subsequent_indent="    "
                    ))
                    if len(text) > 300:
                        print("    ...")

                    # Auto-run Steps 2 → 6
                    print(f"\n  {LINE}")
                    print("  Running Steps 2 → 6 automatically...\n")
                    _auto_run_steps_2_to_6()

                except Exception as e:
                    show_error(f"Could not read file: {e}")

                wait_for_enter()
                return

        else:
            show_error("Invalid choice — please type 1, 2, 3, 4, or 0.")


# =============================================================================
# STEP 2 — CLEAN TEXT
# =============================================================================

def step_clean_text():
    """
    STEP 2 — Clean Text
    -------------------
    Strips control characters, collapses whitespace, and removes
    formatting artefacts so the text is clean for chunking.
    Requires Step 1 to be completed first.
    """
    show_header("Step 2 — Clean Text")
    if not check_required("text"):
        wait_for_enter()
        return
    from preprocessing.preprocess import clean_text
    cleaned = clean_text(state["text"])
    state["cleaned_text"] = cleaned
    show_success(f"Before cleaning:  {len(state['text']):,} characters")
    show_success(f"After  cleaning:  {len(cleaned):,} characters")
    wait_for_enter()


# =============================================================================
# STEP 3 — DETECT LANGUAGE
# =============================================================================

def step_detect_language():
    """
    STEP 3 — Detect Language
    ------------------------
    Automatically detects the document language and selects the
    matching NLTK stopword list.
    Requires Step 2 to be completed first.
    """
    show_header("Step 3 — Detect Language")
    if not check_required("cleaned_text"):
        wait_for_enter()
        return
    from preprocessing.preprocess import detect_language
    lang_code, nltk_lang = detect_language(state["cleaned_text"])
    state["lang_code"]   = lang_code
    state["nltk_lang"]   = nltk_lang
    show_success(f"Detected language:   {lang_code}")
    show_success(f"NLTK stopword set:   {nltk_lang}")
    wait_for_enter()


# =============================================================================
# STEP 4 — CHUNK TEXT
# =============================================================================

def step_chunk_text():
    """
    STEP 4 — Chunk Text
    -------------------
    Splits the cleaned text into overlapping word blocks.
    Uses defaults from config.py — no prompts needed.
    Requires Steps 2 and 3 to be completed first.
    """
    show_header("Step 4 — Chunk Text")
    if not check_required("cleaned_text", "lang_code"):
        wait_for_enter()
        return
    from utils.chunker import chunk_text_with_metadata
    show_info(f"Using chunk size: {DEFAULT_CHUNK_SIZE} words  |  overlap: {DEFAULT_CHUNK_OVERLAP} words")
    source    = state["source"] or "document"
    doc_title = Path(source).stem if not source.startswith("http") else "web_document"
    doc_num   = len(state["loaded_documents"]) + 1
    doc_id    = f"doc-{doc_num:03d}"
    new_chunks = chunk_text_with_metadata(
        state["cleaned_text"],
        chunk_size    = DEFAULT_CHUNK_SIZE,
        overlap       = DEFAULT_CHUNK_OVERLAP,
        document_title= doc_title,
        source        = source,
        document_id   = doc_id,
        lang_code     = state["lang_code"],
    )
    # Accumulate across all documents
    state["all_chunk_records"].extend(new_chunks)
    state["loaded_documents"].append(source)
    state["chunk_records"] = state["all_chunk_records"]
    total = len(state["all_chunk_records"])
    ndocs = len(state["loaded_documents"])
    show_success(f"{len(new_chunks)} new chunks | {total} total across {ndocs} doc(s)")
    print(f"\n  --- First 3 chunks ---\n")
    for chunk in new_chunks[:3]:
        print(f"  {chunk['chunk_id']}  |  {chunk['word_count']} words  "
              f"|  span: {chunk['start_word_index']} → {chunk['end_word_index']}")
        print(textwrap.fill(
            chunk['text'][:140], width=66,
            initial_indent="    ",
            subsequent_indent="    "
        ))
        print(f"  {LINE}")
    wait_for_enter()


# =============================================================================
# STEP 5 — TOKENISE CHUNKS
# =============================================================================

def step_tokenise():
    """
    STEP 5 — Tokenise Chunks
    ------------------------
    Converts each chunk into a list of clean stemmed tokens.
    Requires Steps 4 and 3 to be completed first.
    """
    show_header("Step 5 — Tokenise Chunks")
    if not check_required("chunk_records", "nltk_lang"):
        wait_for_enter()
        return
    from preprocessing.preprocess import tokenize_chunk
    # Tokenise only the new chunks (those not yet in all_tokenized_chunks)
    already = len(state["all_tokenized_chunks"])
    new_chunks_to_tokenize = state["all_chunk_records"][already:]
    new_tokenized = [
        tokenize_chunk(c["text"], state["nltk_lang"])
        for c in new_chunks_to_tokenize
    ]
    state["all_tokenized_chunks"].extend(new_tokenized)
    state["tokenized_chunks"] = state["all_tokenized_chunks"]
    total = len(state["all_tokenized_chunks"])
    show_success(f"{len(new_tokenized)} new chunks tokenised | {total} total")
    if state["all_tokenized_chunks"]:
        show_success(f"Sample tokens from chunk 0:  {state['all_tokenized_chunks'][0][:12]}")
    wait_for_enter()


# =============================================================================
# STEP 6 — BUILD INVERTED INDEX + BM25
# =============================================================================

def step_build_index():
    """
    STEP 6 — Build Inverted Index + BM25  ⭐
    -----------------------------------------
    The inverted index maps every word to the chunks that contain it.
    BM25 scores chunks by relevance to the query.
    Requires Steps 4 and 5 to be completed first.
    """
    show_header("Step 6 — Build Inverted Index + BM25  ⭐")
    if not check_required("chunk_records", "tokenized_chunks"):
        wait_for_enter()
        return
    from indexing.indexer    import build_inverted_index
    from indexing.bm25_store import build_bm25
    inv_idx = build_inverted_index(
        state["all_chunk_records"],
        state["all_tokenized_chunks"]
    )
    bm25 = build_bm25(state["all_tokenized_chunks"])
    state["inverted_index"] = inv_idx
    state["bm25"]           = bm25
    ndocs = len(state["loaded_documents"])
    total = len(state["all_chunk_records"])
    show_success(f"Inverted index built  —  {len(inv_idx):,} unique terms across {ndocs} doc(s)")
    show_success(f"BM25 model built      —  {total} total chunks indexed")
    print(f"\n  --- Sample index entries ---\n")
    for term in list(inv_idx.keys())[:6]:
        entry = inv_idx[term]
        print(f"    '{term}'  →  {entry['doc_freq']} chunk(s)  |  {len(entry['postings'])} posting(s)")
    wait_for_enter()


# =============================================================================
# STEP 7 — NORMALISE QUERY
# =============================================================================

def step_normalise_query():
    """
    STEP 7 — Normalise Query
    ------------------------
    Takes the user question and extracts key search terms using the
    Groq LLM. Falls back to raw question if API call fails.
    """
    show_header("Step 7 — Normalise Query")

    print("  Type your question about the documents.\n")
    raw_query = input("  Your question: ").strip()
    if not raw_query:
        show_error("Question cannot be empty.")
        wait_for_enter()
        return

    state["query"] = raw_query
    api_key = GROQ_API_KEY

    if not api_key:
        state["normalised_query"] = raw_query
        show_success("No API key in config — using raw question.")
    else:
        show_info("Sending query to LLM for keyword extraction...")
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            system_prompt = (
                "You are a keyword extraction assistant for a search engine. "
                "Given a user question, extract only the most important "
                "content-bearing keywords. Remove filler words, articles, "
                "and conversational phrases. "
                "Return ONLY a space-separated list of keywords. "
                "No punctuation. No explanation."
            )
            response = client.chat.completions.create(
                model    = QUERY_MODEL_NAME,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Question: {raw_query}"},
                ],
                max_tokens  = 100,
                temperature = 0.0,
            )
            normalised = response.choices[0].message.content.strip()
            state["normalised_query"] = normalised
            show_success(f"Original question:  {raw_query}")
            show_success(f"Extracted keywords: {normalised}")
        except Exception as e:
            show_error(f"LLM call failed: {e}")
            show_info("Falling back to raw question.")
            state["normalised_query"] = raw_query

    wait_for_enter()


# =============================================================================
# STEP 8 — RETRIEVE TOP-K CHUNKS
# =============================================================================

def step_retrieve():
    """
    STEP 8 — Retrieve Top-K Chunks
    --------------------------------
    Scores every chunk across ALL loaded documents using BM25
    and returns the top-K most relevant results.
    Requires Steps 6 and 7 to be completed first.
    """
    show_header("Step 8 — Retrieve Top-K Chunks")
    if not check_required("normalised_query", "bm25", "chunk_records", "inverted_index", "nltk_lang"):
        wait_for_enter()
        return

    from retrieval.retriever import retrieve

    top_k = DEFAULT_TOP_K
    ndocs = len(state["loaded_documents"])
    show_info(f"Searching across {ndocs} document(s) | {len(state['all_chunk_records'])} total chunks")
    show_info(f"Retrieving top {top_k} chunks")

    results = retrieve(
        state["normalised_query"],
        state["bm25"],
        state["chunk_records"],
        state["inverted_index"],
        nltk_lang = state["nltk_lang"],
        top_k     = top_k,
    )
    state["retrieved_results"] = results
    show_success(f"{len(results)} chunks retrieved for: '{state['query']}'")
    print()

    for item in results:
        print(f"  Rank {item['rank']}  |  {item['chunk_id']}  |  BM25: {item['bm25_score']:.4f}")
        print(f"  Document : {item['document_title']}")
        print(f"  Matched  : {item['matched_terms']}")
        print(textwrap.fill(
            item['text'][:160], width=66,
            initial_indent="    ",
            subsequent_indent="    "
        ))
        print(f"  {LINE}")

    wait_for_enter()


# =============================================================================
# STEP 9 — BUILD PROMPT
# =============================================================================

def step_build_prompt():
    """
    STEP 9 — Build Prompt
    ----------------------
    Assembles the retrieved chunks into a structured prompt for the LLM.
    Requires Steps 7 and 8 to be completed first.
    """
    show_header("Step 9 — Build Prompt")
    if not check_required("query", "retrieved_results"):
        wait_for_enter()
        return
    from utils.prompts import build_prompt
    prompt = build_prompt(state["query"], state["retrieved_results"])
    state["prompt"] = prompt
    show_success(f"Prompt built  —  {len(prompt):,} characters")
    print(f"\n  --- Prompt preview (first 400 characters) ---\n")
    print(textwrap.fill(
        prompt[:400], width=66,
        initial_indent="    ",
        subsequent_indent="    "
    ))
    if len(prompt) > 400:
        print("    ...")
    wait_for_enter()


# =============================================================================
# STEP 10 — GENERATE ANSWER
# =============================================================================

def step_generate_answer():
    """
    STEP 10 — Generate Answer
    --------------------------
    Sends the prompt to the Groq LLM and prints the grounded answer
    with evidence and citations from the retrieved chunks.
    Requires Step 9 to be completed first.
    """
    show_header("Step 10 — Generate Answer")
    if not check_required("prompt"):
        wait_for_enter()
        return
    try:
        from generation.generator import generate_answer
        show_info(f"Sending prompt to {GENERATOR_MODEL} — please wait...")
        answer = generate_answer(state["prompt"])
        state["answer"] = answer

        print(f"\n{'=' * 52}")
        print("  ANSWER")
        print(f"{'=' * 52}\n")

        for line in answer.splitlines():
            stripped = line.strip()
            if not stripped:
                print()
            elif stripped.startswith("- "):
                print(textwrap.fill(
                    stripped, width=66,
                    initial_indent="  ",
                    subsequent_indent="    ",
                ))
            else:
                print(textwrap.fill(
                    stripped, width=66,
                    initial_indent="  ",
                ))
        print()
        show_success("Answer generated.")

    except (ValueError, ImportError, RuntimeError) as e:
        show_error(str(e))

    wait_for_enter()


# =============================================================================
# EXTRAS — Full pipeline, state viewer, reset
# =============================================================================

def run_full_pipeline():
    """
    Run the full pipeline from start to finish.

    Smart behaviour:
    - Step 1  (Load document)  always runs — you pick the document.
    - Steps 2-6 are SKIPPED if they already ran automatically after
      loading (which they always do). No need to repeat them.
    - Step 7  (Query) pauses and waits — if left empty the pipeline
      stops cleanly instead of crashing through Steps 8-10.
    - Steps 8-10 run automatically after a valid query is entered.
    """
    show_header("Full Pipeline — Steps 1 to 10")
    print("  This will run every step in order.\n")
    confirm = input("  Are you sure? Type  yes  to continue: ").strip().lower()
    if confirm != "yes":
        show_info("Cancelled — returning to menu.")
        wait_for_enter()
        return

    # ── Step 1 — always load a document ──────────────────────────
    step_load_document()

    # ── Steps 2-6 — skip if auto-pipeline already ran them ───────
    if state["inverted_index"] is not None:
        show_info("Steps 2-6 already completed automatically after loading.")
        show_info(
            f"{len(state['loaded_documents'])} doc(s) | "
            f"{len(state['all_chunk_records'])} chunks | "
            f"{len(state['inverted_index']):,} terms in index."
        )
    else:
        # Auto-pipeline did not run — run steps manually
        for step_fn in [step_clean_text, step_detect_language,
                        step_chunk_text, step_tokenise, step_build_index]:
            step_fn()

    # ── Step 7 — query (stop if empty) ───────────────────────────
    step_normalise_query()
    if state["normalised_query"] is None:
        show_error("No query entered — pipeline stopped. Run Step 7 when ready.")
        return

    # ── Steps 8-10 — run automatically ───────────────────────────
    step_retrieve()
    step_build_prompt()
    step_generate_answer()


def show_pipeline_state():
    """Show which steps have been completed and which are pending."""
    show_header("Pipeline State")

    # Show loaded documents list
    docs = state["loaded_documents"]
    if docs:
        print(f"  Documents in index: {len(docs)}")
        for i, d in enumerate(docs, 1):
            name = Path(d).name if not d.startswith("http") else d
            print(f"    [{i}]  {name}")
        print(f"  Total chunks     : {len(state['all_chunk_records'])}")
        print()

    step_labels = [
        ("text",                "Step 1  — Document loaded"),
        ("cleaned_text",        "Step 2  — Text cleaned"),
        ("lang_code",           "Step 3  — Language detected"),
        ("all_chunk_records",   "Step 4  — Chunks created"),
        ("all_tokenized_chunks","Step 5  — Tokens ready"),
        ("inverted_index",      "Step 6  — Index built"),
        ("normalised_query",    "Step 7  — Query ready"),
        ("retrieved_results",   "Step 8  — Chunks retrieved"),
        ("prompt",              "Step 9  — Prompt built"),
        ("answer",              "Step 10 — Answer generated"),
    ]

    for key, label in step_labels:
        val = state[key]
        if val is None or val == []:
            symbol = "○"
            detail = "not done yet"
        else:
            symbol = "✓"
            if isinstance(val, str):
                detail = f"{len(val):,} characters"
            elif isinstance(val, list):
                detail = f"{len(val)} items"
            elif isinstance(val, dict):
                detail = f"{len(val):,} keys"
            else:
                detail = "ready"
        print(f"  [{symbol}]  {label:<35}  {detail}")

    wait_for_enter()


def reset_pipeline():
    """Clear all pipeline state and start fresh."""
    show_header("Reset Pipeline")
    print("  This will clear all pipeline data including all loaded documents.\n")
    confirm = input("  Are you sure? Type  yes  to reset: ").strip().lower()
    if confirm == "yes":
        for key in state:
            state[key] = None
        # Re-initialise list fields so .extend() works after reset
        state["all_chunk_records"]    = []
        state["all_tokenized_chunks"] = []
        state["loaded_documents"]     = []
        show_success("Pipeline reset — all documents and chunks cleared.")
    else:
        show_info("Reset cancelled.")
    wait_for_enter()


# =============================================================================
# MAIN MENU
# =============================================================================

MENU_OPTIONS = [
    ("1",  "Step 1  — Load document",           step_load_document),
    ("2",  "Step 2  — Clean text",               step_clean_text),
    ("3",  "Step 3  — Detect language",          step_detect_language),
    ("4",  "Step 4  — Chunk text",               step_chunk_text),
    ("5",  "Step 5  — Tokenise chunks",          step_tokenise),
    ("6",  "Step 6  — Build index + BM25  ⭐",   step_build_index),
    ("7",  "Step 7  — Normalise query",          step_normalise_query),
    ("8",  "Step 8  — Retrieve chunks",          step_retrieve),
    ("9",  "Step 9  — Build prompt",             step_build_prompt),
    ("10", "Step 10 — Generate answer",          step_generate_answer),
    ("─",  None,                                 None),
    ("f",  "Run full pipeline  (Steps 1 → 10)",  run_full_pipeline),
    ("s",  "Show pipeline state",                show_pipeline_state),
    ("r",  "Reset pipeline",                     reset_pipeline),
    ("0",  "Exit",                               None),
]


def print_menu():
    print(f"\n{'=' * 52}")
    print("  Keyword Retrieval RAG — Menu")
    print(f"{'=' * 52}\n")
    for key, label, _ in MENU_OPTIONS:
        if key == "─":
            print(f"  {LINE}")
        else:
            print(f"  [{key:>2}]  {label}")
    print()


def main():
    option_map = {
        key: fn
        for key, label, fn in MENU_OPTIONS
        if key != "─"
    }
    while True:
        print_menu()
        choice = input("  Your choice: ").strip().lower()
        if choice == "0":
            print("\n  Goodbye!\n")
            break
        if choice not in option_map:
            show_error("Invalid choice — please type a number or letter from the menu.")
            continue
        fn = option_map[choice]
        if fn is not None:
            fn()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()