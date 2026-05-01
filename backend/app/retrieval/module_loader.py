"""
app/retrieval/module_loader.py

Why this exists:
    Collins's vector_adapter.py adds vector_retrieval/src/ to sys.path at
    import time (regular package with __init__.py). Python caches finder
    results in sys.path_importer_cache — so even after we modify sys.path,
    old cache entries still point to Collins's utils/loader/etc.

Fix:
    1. Snapshot + evict conflicting sys.modules entries
    2. Strip all other team paths from sys.path
    3. Call importlib.invalidate_caches() to clear the path finder cache
    4. exec_module — imports now resolve from the correct root
    5. Restore everything in finally
"""

import os
import sys
import importlib
import importlib.util

_TEAM_ROOTS = ("vector_retrieval", "keyword_retrieval", "hybrid_retrieval")

_CONFLICT_PREFIXES = (
    "src", "retrieval", "models", "indexing",
    "utils", "preprocessing", "evaluation", "generation",
)


def load_adapter(adapter_path: str, module_root: str):
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    # 1. Snapshot conflicting sys.modules entries
    snapshot = {
        k: v for k, v in sys.modules.items()
        if any(k == p or k.startswith(p + ".") for p in _CONFLICT_PREFIXES)
    }

    saved_path = sys.path[:]
    saved_finder_cache = sys.path_importer_cache.copy()

    # 2. Build clean sys.path — only this module's roots + non-team paths
    clean_path = [module_root, os.path.join(module_root, "src")] + [
        p for p in saved_path
        if not any(root in p for root in _TEAM_ROOTS)
    ]

    try:
        # 3. Evict cached modules + apply clean path
        for k in snapshot:
            sys.modules.pop(k, None)
        sys.path[:] = clean_path

        # 4. Clear the path-based importer cache so Python re-scans sys.path
        sys.path_importer_cache.clear()
        importlib.invalidate_caches()

        # 5. Load the adapter
        unique_name = os.path.splitext(os.path.basename(adapter_path))[0]
        spec = importlib.util.spec_from_file_location(unique_name, adapter_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    finally:
        # 6. Evict this module's newly cached entries, restore everything
        for k in list(sys.modules):
            if any(k == p or k.startswith(p + ".") for p in _CONFLICT_PREFIXES):
                sys.modules.pop(k, None)
        sys.modules.update(snapshot)
        sys.path[:] = saved_path
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(saved_finder_cache)
        importlib.invalidate_caches()