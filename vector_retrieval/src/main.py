"""
main.py
========
Keyword Retrieval RAG — main entry point.

Run from your terminal:
    cd C:\\Users\\DC\\Desktop\\keyword_RAG_01\\src
    python main.py
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
    "text":              None,
    "source":            None,
    "cleaned_text":      None,
    "lang_code":         None,
    "nltk_lang":         None,
    "all_chunk_records":    [],
    "all_tokenized_chunks": [],
    "loaded_documents":     [],
    "chunk_records":     None,
    "tokenized_chunks":  None,
    "inverted_index":    None,
    "bm25":              None,
    "query":             None,
    "normalised_query":  None,
    "retrieved_results": None,
    "prompt":            None,
    "answer":            None,
}


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

LINE  = "─" * 52
DLINE = "=" * 52

def show_main_header():
    print(f"\n{DLINE}")
    print("  Keyword Retrieval RAG — Main Menu")
    print(f"{DLINE}")

def show_sub_header(title):
    print(f"\n{DLINE}")
    print(f"  {title}")
    print(f"{DLINE}\n")

def show_success(msg): print(f"  ✓  {msg}")
def show_error(msg):   print(f"  ✗  {msg}")
def show_info(msg):    print(f"  →  {msg}")

def wait_for_enter():
    input(f"\n  Press Enter to go back to the menu...\n{LINE}\n")

def check_required(*keys):
    missing = [k for k in keys if state[k] is None]
    if missing:
        show_error(f"Complete these steps first: {', '.join(missing)}")
        return False
    return True


# =============================================================================
# AUTO PIPELINE
# =============================================================================

def _auto_run_steps_2_to_6():
    print("  [Step 2]  Cleaning text...", end=" ", flush=True)
    try:
        from preprocessing.preprocess import clean_text
        cleaned = clean_text(state["text"])
        state["cleaned_text"] = cleaned
        print(f"done  ({len(cleaned):,} chars)")
    except Exception as e:
        print(f"FAILED: {e}"); return

    print("  [Step 3]  Detecting language...", end=" ", flush=True)
    try:
        from preprocessing.preprocess import detect_language
        lang_code, nltk_lang = detect_language(state["cleaned_text"])
        state["lang_code"]   = lang_code
        state["nltk_lang"]   = nltk_lang
        print(f"done  ({lang_code} -> {nltk_lang})")
    except Exception as e:
        print(f"FAILED: {e}"); return

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
        state["all_chunk_records"].extend(new_chunks)
        state["loaded_documents"].append(source)
        state["chunk_records"] = state["all_chunk_records"]
        total = len(state["all_chunk_records"])
        ndocs = len(state["loaded_documents"])
        print(f"done  ({len(new_chunks)} new | {total} total across {ndocs} doc(s))")
    except Exception as e:
        print(f"FAILED: {e}"); return

    print("  [Step 5]  Tokenising chunks...", end=" ", flush=True)
    try:
        from preprocessing.preprocess import tokenize_chunk
        new_tokenized = [
            tokenize_chunk(c["text"], state["nltk_lang"])
            for c in new_chunks
        ]
        state["all_tokenized_chunks"].extend(new_tokenized)
        state["tokenized_chunks"] = state["all_tokenized_chunks"]
        print(f"done  ({len(state['all_tokenized_chunks'])} total)")
    except Exception as e:
        print(f"FAILED: {e}"); return

    print("  [Step 6]  Building index + BM25...", end=" ", flush=True)
    try:
        from indexing.indexer    import build_inverted_index, _save_pickle
        from indexing.bm25_store import build_bm25, save_bm25
        inv_idx = build_inverted_index(
            state["all_chunk_records"],
            state["all_tokenized_chunks"],
        )
        bm25 = build_bm25(state["all_tokenized_chunks"])
        state["inverted_index"] = inv_idx
        state["bm25"]           = bm25
        _save_pickle(inv_idx,                    "keyword_index.pkl")
        save_bm25(bm25,                          "keyword_bm25.pkl")
        _save_pickle(state["all_chunk_records"], "keyword_chunks.pkl")
        print(f"done  ({len(inv_idx):,} terms)  — saved to disk")
    except Exception as e:
        print(f"FAILED: {e}"); return

    ndocs  = len(state["loaded_documents"])
    ntotal = len(state["all_chunk_records"])
    print()
    show_success(f"{ndocs} document(s) indexed | {ntotal} total chunks | saved to disk.")
    print("  Load another document or choose option [3] to ask a question.")


# =============================================================================
# STEP 1 — LOAD DOCUMENT
# =============================================================================

def _do_load(source, load_document_fn):
    try:
        text, label = load_document_fn(source)
        state["text"]   = text
        state["source"] = label
        name = Path(label).name if not label.startswith("http") else label
        show_success(f"Document loaded:  {name}")
        show_success(f"Size: {len(text):,} chars | {len(text.split()):,} words")
        print(f"\n  --- Text preview (first 300 characters) ---\n")
        print(textwrap.fill(text[:300], width=66,
              initial_indent="    ", subsequent_indent="    "))
        if len(text) > 300:
            print("    ...")
        print(f"\n  {LINE}")
        print("  Running Steps 2 to 6 automatically...\n")
        _auto_run_steps_2_to_6()
    except FileNotFoundError as e: show_error(f"File not found: {e}")
    except ValueError as e:        show_error(f"Could not read file: {e}")
    except Exception as e:         show_error(f"Unexpected error: {e}")
    wait_for_enter()


def step_load_document():
    from utils.loader import (
        load_document, open_file_dialog,
        ensure_storage_dir, SUPPORTED_EXTENSIONS, _FORMAT_LOADERS,
    )
    while True:
        show_sub_header("Step 1 — Load Document")
        print("  How do you want to load your document?\n")
        print("  [1]  Browse my laptop  (opens file picker)")
        print("  [2]  Enter a web URL   (https://...)")
        print("  [3]  Type / paste a file path manually")
        print("  [4]  Choose from already stored documents")
        print("  [0]  Back to main menu")
        print()
        choice = input("  Your choice: ").strip()

        if choice == "0":
            return
        elif choice == "1":
            while True:
                print()
                show_info("Opening file picker...")
                selected = open_file_dialog()
                if not selected:
                    print("\n  No file selected.\n  [1] Try again  [0] Back")
                    if input("\n  > ").strip() != "1": break
                else:
                    _do_load(selected, load_document); return
        elif choice == "2":
            while True:
                print()
                url = input("  URL  (or 0 to go back): ").strip()
                if url == "0": break
                if not url.startswith("http://") and not url.startswith("https://"):
                    show_error("URL must start with http:// or https://"); continue
                _do_load(url, load_document); return
        elif choice == "3":
            print()
            path_str = input("  File path  (or 0 to go back): ").strip().strip('"').strip("'")
            if path_str != "0":
                _do_load(path_str, load_document); return
        elif choice == "4":
            store = ensure_storage_dir()
            stored_files = sorted(
                f for f in store.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS + [".txt"]
            )
            if not stored_files:
                print(f"\n  Storage folder is empty: {store}")
                wait_for_enter(); continue
            while True:
                print(f"\n  Storage folder: {store}\n")
                for i, f in enumerate(stored_files, 1):
                    print(f"    [{i}]  {f.name}  ({f.stat().st_size/1024:.1f} KB)")
                print(f"\n    [0]  Back\n")
                pick = input("  Your choice: ").strip()
                if pick == "0": break
                try:
                    idx = int(pick) - 1
                    if not (0 <= idx < len(stored_files)): raise ValueError
                except ValueError:
                    show_error("Invalid selection."); continue
                chosen = stored_files[idx]
                try:
                    ext  = chosen.suffix.lower()
                    text = _FORMAT_LOADERS[ext](chosen) if ext in _FORMAT_LOADERS \
                           else chosen.read_text(encoding="utf-8", errors="replace")
                    state["text"]   = text
                    state["source"] = str(chosen)
                    show_success(f"Loaded: {chosen.name}")
                    show_success(f"Size: {len(text):,} chars | {len(text.split()):,} words")
                    print(f"\n  Preview:\n")
                    print(textwrap.fill(text[:300], width=66,
                          initial_indent="    ", subsequent_indent="    "))
                    if len(text) > 300: print("    ...")
                    print(f"\n  {LINE}")
                    print("  Running Steps 2 to 6 automatically...\n")
                    _auto_run_steps_2_to_6()
                except Exception as e:
                    show_error(f"Could not read file: {e}")
                wait_for_enter(); return
        else:
            show_error("Invalid choice.")


# =============================================================================
# INDIVIDUAL STEPS WITH RICH OUTPUT
# =============================================================================

def step_clean_text():
    show_sub_header("Step 2 — Clean Text")
    if not check_required("text"):
        wait_for_enter(); return
    from preprocessing.preprocess import clean_text

    print("  BEFORE cleaning (first 400 chars):\n")
    print(textwrap.fill(state["text"][:400], width=66,
          initial_indent="    ", subsequent_indent="    "))
    print()
    cleaned = clean_text(state["text"])
    state["cleaned_text"] = cleaned
    print("  AFTER cleaning (first 400 chars):\n")
    print(textwrap.fill(cleaned[:400], width=66,
          initial_indent="    ", subsequent_indent="    "))
    print()
    show_success(f"Before: {len(state['text']):,} chars")
    show_success(f"After : {len(cleaned):,} chars")
    show_info("Removed: control characters, extra spaces, triple blank lines.")
    wait_for_enter()


def step_detect_language():
    show_sub_header("Step 3 — Detect Language")
    if not check_required("cleaned_text"):
        wait_for_enter(); return
    from preprocessing.preprocess import detect_language

    print("  Text sample used for detection (first 500 chars):\n")
    print(textwrap.fill(state["cleaned_text"][:500], width=66,
          initial_indent="    ", subsequent_indent="    "))
    print()
    lang_code, nltk_lang = detect_language(state["cleaned_text"])
    state["lang_code"]   = lang_code
    state["nltk_lang"]   = nltk_lang
    show_success(f"Detected language code : {lang_code}")
    show_success(f"NLTK stopword set      : {nltk_lang}")
    print()
    show_info("How it works: langdetect analyses character n-gram frequencies")
    show_info("in the first 2000 characters and matches them to 55 language profiles.")
    show_info(f"'{lang_code}' maps to the '{nltk_lang}' NLTK stopword corpus.")
    wait_for_enter()


def step_chunk_text():
    show_sub_header("Step 4 — Chunk Text")
    if not check_required("cleaned_text", "lang_code"):
        wait_for_enter(); return
    from utils.chunker import chunk_text_with_metadata

    show_info(f"Chunk size: {DEFAULT_CHUNK_SIZE} words | Overlap: {DEFAULT_CHUNK_OVERLAP} words")
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
    state["all_chunk_records"].extend(new_chunks)
    state["loaded_documents"].append(source)
    state["chunk_records"] = state["all_chunk_records"]
    show_success(f"{len(new_chunks)} new chunks | {len(state['all_chunk_records'])} total")

    while True:
        print()
        print("  What would you like to see?\n")
        print("  [1]  Top 5 chunks  (first 400 characters each)")
        print("  [2]  All chunks    (full text)")
        print("  [0]  Continue")
        print()
        pick = input("  Your choice: ").strip()
        if pick == "0":
            break
        elif pick == "1":
            print()
            for chunk in new_chunks[:5]:
                print(f"  {chunk['chunk_id']}  |  {chunk['word_count']} words  "
                      f"|  span: {chunk['start_word_index']}to{chunk['end_word_index']}")
                print(textwrap.fill(chunk['text'][:400], width=66,
                      initial_indent="    ", subsequent_indent="    "))
                print(f"  {LINE}")
        elif pick == "2":
            print()
            for chunk in new_chunks:
                print(f"  {chunk['chunk_id']}  |  {chunk['word_count']} words  "
                      f"|  span: {chunk['start_word_index']}to{chunk['end_word_index']}")
                print(textwrap.fill(chunk['text'], width=66,
                      initial_indent="    ", subsequent_indent="    "))
                print(f"  {LINE}")
        else:
            show_error("Invalid choice.")
    wait_for_enter()


def step_tokenise():
    show_sub_header("Step 5 — Tokenise Chunks")
    if not check_required("chunk_records", "nltk_lang"):
        wait_for_enter(); return
    from preprocessing.preprocess import tokenize_chunk

    already = len(state["all_tokenized_chunks"])
    new_chunks_to_tokenize = state["all_chunk_records"][already:]
    new_tokenized = [
        tokenize_chunk(c["text"], state["nltk_lang"])
        for c in new_chunks_to_tokenize
    ]
    state["all_tokenized_chunks"].extend(new_tokenized)
    state["tokenized_chunks"] = state["all_tokenized_chunks"]

    show_success(f"{len(new_tokenized)} new chunks tokenised | {len(state['all_tokenized_chunks'])} total")
    print()
    show_info("Pipeline: lowercase -> split -> remove stopwords -> stem")
    print()
    chunks_to_show = state["all_chunk_records"][:5]
    tokens_to_show = state["all_tokenized_chunks"][:5]
    for chunk, tokens in zip(chunks_to_show, tokens_to_show):
        print(f"  {chunk['chunk_id']}")
        print(f"  Original (first 100 chars) : {chunk['text'][:100]}")
        print(f"  Tokens   (first 15)        : {tokens[:15]}")
        print(f"  {LINE}")
    wait_for_enter()


def step_build_index():
    show_sub_header("Step 6 — Build Inverted Index + BM25")
    if not check_required("chunk_records", "nltk_lang"):
        wait_for_enter(); return
    from indexing.indexer    import build_inverted_index, _save_pickle
    from indexing.bm25_store import build_bm25, save_bm25

    already = len(state["all_tokenized_chunks"])
    new_to_tok = state["all_chunk_records"][already:]
    if new_to_tok:
        from preprocessing.preprocess import tokenize_chunk
        new_tok = [tokenize_chunk(c["text"], state["nltk_lang"]) for c in new_to_tok]
        state["all_tokenized_chunks"].extend(new_tok)
        state["tokenized_chunks"] = state["all_tokenized_chunks"]

    print("  Building inverted index...\n")
    show_info("Maps every word to the chunks that contain it.")
    show_info("Structure: { term: { doc_freq, postings: [chunk_id, tf, positions] } }")
    print()
    inv_idx = build_inverted_index(
        state["all_chunk_records"],
        state["all_tokenized_chunks"]
    )
    print("  Building BM25 model...\n")
    show_info("Scores chunks using: term frequency + IDF + length normalisation.")
    show_info("Parameters: k1=1.5 (TF saturation), b=0.75 (length normalisation)")
    print()
    bm25 = build_bm25(state["all_tokenized_chunks"])
    state["inverted_index"] = inv_idx
    state["bm25"]           = bm25

    _save_pickle(inv_idx,                    "keyword_index.pkl")
    save_bm25(bm25,                          "keyword_bm25.pkl")
    _save_pickle(state["all_chunk_records"], "keyword_chunks.pkl")

    ndocs = len(state["loaded_documents"])
    total = len(state["all_chunk_records"])
    show_success(f"Index: {len(inv_idx):,} unique terms across {ndocs} doc(s)")
    show_success(f"BM25 : {total} total chunks indexed")
    show_success("Saved: keyword_index.pkl | keyword_bm25.pkl | keyword_chunks.pkl")
    print()
    print("  --- Top 8 terms by document frequency ---\n")
    top_terms = sorted(inv_idx.items(), key=lambda x: x[1]["doc_freq"], reverse=True)[:8]
    print(f"  {'Term':<20}  {'doc_freq':>10}  {'total_tf':>10}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}")
    for term, entry in top_terms:
        total_tf = sum(p["tf"] for p in entry["postings"])
        print(f"  {term:<20}  {entry['doc_freq']:>10}  {total_tf:>10}")
    wait_for_enter()


def step_normalise_query():
    show_sub_header("Step 7 — Normalise Query")
    print("  Type your question about the documents.\n")
    raw_query = input("  Your question: ").strip()
    if not raw_query:
        show_error("Question cannot be empty.")
        wait_for_enter(); return
    state["query"] = raw_query
    api_key = GROQ_API_KEY
    if not api_key:
        state["normalised_query"] = raw_query
        show_success("No API key — using raw question.")
    else:
        show_info("Sending query to LLM for keyword extraction...")
        print(f"\n  Original question  : {raw_query}")
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
                model   = QUERY_MODEL_NAME,
                messages= [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Question: {raw_query}"},
                ],
                max_tokens=100, temperature=0.0,
            )
            normalised = response.choices[0].message.content.strip()
            state["normalised_query"] = normalised
            print(f"  Extracted keywords : {normalised}")
            print()
            show_info("How it works: the LLM removes filler words like 'what', 'how',")
            show_info("'tell me about' and keeps only content-bearing search terms.")
        except Exception as e:
            show_error(f"LLM failed: {e} — using raw question.")
            state["normalised_query"] = raw_query
    wait_for_enter()


def step_retrieve():
    show_sub_header("Step 8 — Retrieve Top-K Chunks")
    if not check_required("normalised_query", "bm25", "chunk_records", "inverted_index", "nltk_lang"):
        wait_for_enter(); return
    from retrieval.retriever import retrieve
    top_k  = DEFAULT_TOP_K
    ndocs  = len(state["loaded_documents"])
    ntotal = len(state["all_chunk_records"])
    show_info(f"Searching across {ndocs} document(s) | {ntotal} total chunks")
    show_info(f"Retrieving top {top_k} chunks")
    print()
    results = retrieve(
        state["normalised_query"], state["bm25"],
        state["chunk_records"], state["inverted_index"],
        nltk_lang=state["nltk_lang"], top_k=top_k,
    )
    state["retrieved_results"] = results
    show_success(f"{len(results)} chunks retrieved for: '{state['query']}'")
    print()
    for item in results:
        print(f"  Rank {item['rank']}  |  {item['chunk_id']}  |  BM25: {item['bm25_score']:.4f}")
        print(f"  Document : {item['document_title']}")
        print(f"  Matched  : {item['matched_terms']}")
        print(textwrap.fill(item['text'][:160], width=66,
              initial_indent="    ", subsequent_indent="    "))
        print(f"  {LINE}")
    wait_for_enter()


def step_build_prompt():
    show_sub_header("Step 9 — Build Prompt")
    if not check_required("query", "retrieved_results"):
        wait_for_enter(); return
    from utils.prompts import build_prompt
    prompt = build_prompt(state["query"], state["retrieved_results"])
    state["prompt"] = prompt
    show_success(f"Prompt built — {len(prompt):,} characters")
    print(f"\n  --- Prompt preview (first 400 characters) ---\n")
    print(textwrap.fill(prompt[:400], width=66,
          initial_indent="    ", subsequent_indent="    "))
    if len(prompt) > 400: print("    ...")
    wait_for_enter()


def step_generate_answer():
    show_sub_header("Step 10 — Generate Answer")
    if not check_required("prompt"):
        wait_for_enter(); return
    try:
        from generation.generator import generate_answer
        show_info(f"Sending prompt to {GENERATOR_MODEL} — please wait...")
        answer = generate_answer(state["prompt"])
        state["answer"] = answer
        print(f"\n{DLINE}\n  ANSWER\n{DLINE}\n")
        for line in answer.splitlines():
            stripped = line.strip()
            if not stripped: print()
            elif stripped.startswith("- "):
                print(textwrap.fill(stripped, width=66,
                      initial_indent="  ", subsequent_indent="    "))
            else:
                print(textwrap.fill(stripped, width=66, initial_indent="  "))
        print()
        show_success("Answer generated.")
    except (ValueError, ImportError, RuntimeError) as e:
        show_error(str(e))
    wait_for_enter()


# =============================================================================
# OPTION 1 — FULL PROCESS
# =============================================================================

def full_process():
    show_sub_header("Full Process — Steps 1 to 10")
    print("  This runs the complete pipeline from document to answer.\n")
    confirm = input("  Type  yes  to continue: ").strip().lower()
    if confirm != "yes":
        show_info("Cancelled.")
        wait_for_enter(); return

    step_load_document()

    if state["inverted_index"] is not None:
        show_info("Steps 2-6 already completed automatically after loading.")
        show_info(f"{len(state['loaded_documents'])} doc(s) | "
                  f"{len(state['all_chunk_records'])} chunks | "
                  f"{len(state['inverted_index']):,} index terms.")
    else:
        for fn in [step_clean_text, step_detect_language,
                   step_chunk_text, step_tokenise, step_build_index]:
            fn()

    step_normalise_query()
    if state["normalised_query"] is None:
        show_error("No query entered — use option [3] to ask later.")
        return

    step_retrieve()
    step_build_prompt()
    step_generate_answer()


# =============================================================================
# OPTION 2 — STEP BY STEP MODE
# =============================================================================

def step_by_step_mode():
    while True:
        show_sub_header("Keyword Retrieval RAG — Step by Step")
        print("  [ 1]  Step 1  — Load document")
        print("  [ 2]  Step 2  — Clean text")
        print("  [ 3]  Step 3  — Detect language")
        print("  [ 4]  Step 4  — Chunk text")
        print("  [ 5]  Step 5  — Tokenise chunks")
        print("  [ 6]  Step 6  — Build index + BM25")
        print("  [ 7]  Step 7  — Normalise query")
        print("  [ 8]  Step 8  — Retrieve chunks")
        print("  [ 9]  Step 9  — Build prompt")
        print("  [10]  Step 10 — Generate answer")
        print(f"\n  {LINE}")
        print("  [ 0]  Back to main menu")
        print()
        choice = input("  Your choice: ").strip()

        steps = {
            "1": step_load_document, "2": step_clean_text,
            "3": step_detect_language, "4": step_chunk_text,
            "5": step_tokenise, "6": step_build_index,
            "7": step_normalise_query, "8": step_retrieve,
            "9": step_build_prompt, "10": step_generate_answer,
        }

        if choice == "0": return
        elif choice in steps: steps[choice]()
        else: show_error("Invalid choice.")


# =============================================================================
# OPTION 3 — ASK YOUR QUESTION
# =============================================================================

def ask_question():
    show_sub_header("Ask Your Question")

    if state["bm25"] is None or state["inverted_index"] is None:
        show_error("No documents indexed yet.")
        show_info("Use option [1] or [2] to load documents first,")
        show_info("or option [6] to restore a previous session from disk.")
        wait_for_enter(); return

    ndocs  = len(state["loaded_documents"])
    ntotal = len(state["all_chunk_records"])
    show_info(f"Searching across {ndocs} document(s) | {ntotal} total chunks")
    print()
    for i, d in enumerate(state["loaded_documents"], 1):
        name = Path(d).name if not d.startswith("http") else d
        print(f"  [{i}]  {name}")
    print()

    raw_query = input("  Your question: ").strip()
    if not raw_query:
        show_error("Question cannot be empty.")
        wait_for_enter(); return

    state["query"] = raw_query
    api_key = GROQ_API_KEY

    if api_key:
        show_info("Extracting keywords...")
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            system_prompt = (
                "You are a keyword extraction assistant for a search engine. "
                "Extract only the most important content-bearing keywords. "
                "Return ONLY a space-separated list of keywords. No punctuation."
            )
            resp = client.chat.completions.create(
                model   = QUERY_MODEL_NAME,
                messages= [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Question: {raw_query}"},
                ],
                max_tokens=100, temperature=0.0,
            )
            normalised = resp.choices[0].message.content.strip()
            state["normalised_query"] = normalised
            show_info(f"Keywords : {normalised}")
        except Exception as e:
            show_error(f"LLM failed: {e} — using raw question.")
            state["normalised_query"] = raw_query
    else:
        state["normalised_query"] = raw_query

    from retrieval.retriever import retrieve
    show_info(f"Searching {ntotal} chunks across {ndocs} document(s)...")
    results = retrieve(
        state["normalised_query"], state["bm25"],
        state["chunk_records"], state["inverted_index"],
        nltk_lang=state["nltk_lang"] or "english",
        top_k=DEFAULT_TOP_K,
    )
    state["retrieved_results"] = results

    if not results:
        show_error("No matching chunks found. Try different keywords.")
        wait_for_enter(); return

    from utils.prompts import build_prompt
    prompt = build_prompt(raw_query, results)
    state["prompt"] = prompt

    try:
        from generation.generator import generate_answer
        show_info(f"Generating answer with {GENERATOR_MODEL}...")
        answer = generate_answer(prompt)
        state["answer"] = answer
        print(f"\n{DLINE}\n  ANSWER\n{DLINE}\n")
        for line in answer.splitlines():
            stripped = line.strip()
            if not stripped: print()
            elif stripped.startswith("- "):
                print(textwrap.fill(stripped, width=66,
                      initial_indent="  ", subsequent_indent="    "))
            else:
                print(textwrap.fill(stripped, width=66, initial_indent="  "))
        print()
        show_success("Done.")
    except Exception as e:
        show_error(f"Generation failed: {e}")

    wait_for_enter()


# =============================================================================
# OPTION 4 — CHECK SYSTEM STATUS
# =============================================================================

def check_system_status():
    show_sub_header("Check System Status")

    steps = [
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

    for key, label in steps:
        val = state[key]
        if val is None or val == []:
            symbol, detail = "o", "not done yet"
        else:
            symbol = "v"
            if isinstance(val, str):    detail = f"{len(val):,} characters"
            elif isinstance(val, list): detail = f"{len(val)} items"
            elif isinstance(val, dict): detail = f"{len(val):,} keys"
            else:                       detail = "ready"
        print(f"  [{symbol}]  {label:<35}  {detail}")

    print()
    if state["inverted_index"]:
        print(f"  Index summary:")
        print(f"    Documents : {len(state['loaded_documents'])}")
        print(f"    Chunks    : {len(state['all_chunk_records'])}")
        print(f"    Terms     : {len(state['inverted_index']):,}")
        print(f"    Disk      : keyword_index.pkl | keyword_bm25.pkl | keyword_chunks.pkl")
    else:
        show_info("No index built yet.")

    wait_for_enter()


# =============================================================================
# OPTION 5 — VIEW STORED DOCUMENTS
# =============================================================================

def view_stored_documents():
    show_sub_header("View Stored Documents")

    if not state["all_chunk_records"]:
        show_error("No documents loaded yet.")
        wait_for_enter(); return

    inv_idx = state["inverted_index"]
    chunks  = state["all_chunk_records"]
    docs    = state["loaded_documents"]

    while True:
        print(f"  What would you like to view?\n")
        print(f"  [1]  Loaded files    ({len(docs)} document(s))")
        print(f"  [2]  All chunks      ({len(chunks)} total)")
        if inv_idx:
            print(f"  [3]  Index terms    ({len(inv_idx):,} terms)")
        else:
            print(f"  [3]  Index terms    (not built yet)")
        print(f"  [4]  Look up a term  (search the inverted index)")
        print(f"  [0]  Back to main menu")
        print()
        pick = input("  Your choice: ").strip()

        if pick == "0":
            break
        elif pick == "1":
            print()
            for i, d in enumerate(docs, 1):
                name = Path(d).name if not d.startswith("http") else d
                doc_chunks = [c for c in chunks if c["source"] == d]
                print(f"  [{i}]  {name}")
                print(f"       Doc ID : {doc_chunks[0]['document_id'] if doc_chunks else 'n/a'}")
                print(f"       Chunks : {len(doc_chunks)}")
                print()
        elif pick == "2":
            print()
            for chunk in chunks:
                print(f"  {chunk['chunk_id']:<25}  "
                      f"{chunk['word_count']} words  |  "
                      f"span: {chunk['start_word_index']}to{chunk['end_word_index']}  |  "
                      f"doc: {chunk['document_title']}")
            print()
        elif pick == "3":
            if not inv_idx:
                show_error("Index not built yet."); continue
            top_terms = sorted(
                inv_idx.items(), key=lambda x: x[1]["doc_freq"], reverse=True
            )[:25]
            print()
            print(f"  {'Term':<22}  {'doc_freq':>10}  {'total_tf':>10}")
            print(f"  {'─'*22}  {'─'*10}  {'─'*10}")
            for term, entry in top_terms:
                total_tf = sum(p["tf"] for p in entry["postings"])
                print(f"  {term:<22}  {entry['doc_freq']:>10}  {total_tf:>10}")
            print(f"\n  (top 25 by document frequency — {len(inv_idx):,} total terms)")
            print()
        elif pick == "4":
            if not inv_idx:
                show_error("Index not built yet."); continue
            print()
            from nltk.stem import PorterStemmer
            lookup = input("  Enter a term to look up: ").strip()
            if lookup:
                stemmed = PorterStemmer().stem(lookup.lower())
                entry   = inv_idx.get(stemmed)
                if not entry:
                    print(f"\n  '{lookup}' (stemmed: '{stemmed}') — not found in index.")
                else:
                    print(f"\n  Term     : '{lookup}'  (stemmed -> '{stemmed}')")
                    print(f"  doc_freq : {entry['doc_freq']} chunk(s)")
                    for p in entry["postings"]:
                        print(f"    chunk_id={p['chunk_id']}  "
                              f"tf={p['tf']}  "
                              f"positions={p['positions'][:6]}")
            print()
        else:
            show_error("Invalid choice.")


# =============================================================================
# OPTION 6 — RESTORE PREVIOUS WORK
# =============================================================================

def restore_previous_work():
    show_sub_header("Restore Previous Work")

    files   = ["keyword_index.pkl", "keyword_bm25.pkl", "keyword_chunks.pkl"]
    missing = [f for f in files if not Path(f).exists()]

    if missing:
        show_error(f"Missing files: {', '.join(missing)}")
        show_info("No previous session found.")
        show_info("Load documents using option [1] or [2] first.")
        wait_for_enter(); return

    try:
        from indexing.bm25_store import load_bm25
        bm25, inv_idx, chunks = load_bm25(
            "keyword_bm25.pkl",
            "keyword_index.pkl",
            "keyword_chunks.pkl",
        )
        state["bm25"]              = bm25
        state["inverted_index"]    = inv_idx
        state["all_chunk_records"] = chunks
        state["chunk_records"]     = chunks

        seen = []
        for c in chunks:
            if c["source"] not in seen:
                seen.append(c["source"])
        state["loaded_documents"] = seen

        if chunks:
            state["lang_code"] = chunks[0].get("lang_code", "en")
            from preprocessing.preprocess import detect_language
            _, nltk_lang = detect_language(chunks[0]["text"])
            state["nltk_lang"] = nltk_lang

        ndocs  = len(state["loaded_documents"])
        ntotal = len(chunks)
        nterms = len(inv_idx)

        show_success("Previous session restored successfully.")
        show_success(f"Documents : {ndocs}")
        show_success(f"Chunks    : {ntotal}")
        show_success(f"Terms     : {nterms:,}")
        print()
        print("  Documents in index:")
        for i, d in enumerate(state["loaded_documents"], 1):
            name = Path(d).name if not d.startswith("http") else d
            print(f"    [{i}]  {name}")
        print()
        show_info("You can now use option [3] to ask a question.")

    except Exception as e:
        show_error(f"Restore failed: {e}")

    wait_for_enter()


# =============================================================================
# OPTION 7 — RESET EVERYTHING
# =============================================================================

def reset_everything():
    show_sub_header("Reset Everything")
    print("  This will clear ALL pipeline data including all loaded documents.\n")
    confirm = input("  Type  yes  to confirm: ").strip().lower()
    if confirm == "yes":
        for key in state:
            state[key] = None
        state["all_chunk_records"]    = []
        state["all_tokenized_chunks"] = []
        state["loaded_documents"]     = []
        show_success("Everything has been reset. You can start fresh.")
    else:
        show_info("Reset cancelled.")
    wait_for_enter()


# =============================================================================
# MAIN MENU
# =============================================================================

def main():
    while True:
        show_main_header()
        print()
        print("  [1]  Start full process    -> Run all steps from beginning to final answer")
        print("  [2]  Step by step mode     -> Run each step one by one manually")
        print("  [3]  Ask your question     -> Search documents and generate an answer")
        print("  [4]  Check system status   -> See completed steps, chunks, and index status")
        print("  [5]  View stored documents -> Show loaded files, chunks, and indexed terms")
        print("  [6]  Restore previous work -> Reload saved chunks and index from disk")
        print("  [7]  Reset everything      -> Clear all pipeline data and start fresh")
        print(f"\n  {LINE}")
        print("  [0]  Exit Program")
        print(f"  {LINE}\n")
        choice = input("  Your choice: ").strip()

        menu = {
            "1": full_process,
            "2": step_by_step_mode,
            "3": ask_question,
            "4": check_system_status,
            "5": view_stored_documents,
            "6": restore_previous_work,
            "7": reset_everything,
        }

        if choice == "0":
            print("\n  Goodbye!\n")
            break
        elif choice in menu:
            menu[choice]()
        else:
            show_error("Invalid choice — type a number from 0 to 7.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
