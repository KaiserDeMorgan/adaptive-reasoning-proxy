"""Shared test setup.

Runs before any test module is imported, so environment defaults here apply
to the proxy modules at import time.
"""

import os
import tempfile

# Keep the regex classifier in tests (no network for embeddings).
os.environ.setdefault("EMBEDDINGS_ENABLED", "0")

# Point the query log at a throwaway DB so tests never touch the real one.
_tmp_db = os.path.join(tempfile.gettempdir(), "arp_test_query_log.db")
os.environ.setdefault("QUERY_LOG_DB", _tmp_db)
