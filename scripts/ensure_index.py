"""Build the index if it is missing, then exit. Safe to run on every boot.

Render's filesystem is ephemeral: a deploy, a restart or a free-tier spin-down
can leave the container with no ``chroma_db/``. Without this, the UI greets the
first visitor with "Index not built" and tells them to run a command they
cannot run, because there is no shell on a Render web service.

So the start command is::

    python scripts/ensure_index.py && streamlit run src/app.py

Idempotent by design. If the collection already has chunks it prints the count
and exits in under a second, so a warm container does not re-fetch five pages on
every restart. Only a genuinely empty index triggers a network ingest.

If the network is unavailable and there is no index, this fails loudly rather
than starting a UI that cannot answer anything -- a confusing 500 on first click
is worse than a build that stops with a readable message.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from src.config import CONFIG

    try:
        from src.ingest.store import open_collection

        count = open_collection().count()
    except Exception as exc:  # noqa: BLE001 - no index yet is expected
        print(f"ensure_index: could not open collection ({type(exc).__name__}); ingesting")
        count = 0

    if count > 0:
        print(f"ensure_index: index present ({count} chunks) at {CONFIG.chroma_dir}")
        return 0

    print("ensure_index: no index found, running ingest (this fetches 5 pages)")
    try:
        from src.ingest.pipeline import run_ingest

        run_ingest()
    except Exception as exc:  # noqa: BLE001
        print(
            f"ensure_index: ingest FAILED: {type(exc).__name__}: {exc}\n"
            "The index is required for the app to answer anything. Check that the "
            "build has outbound network access to groww.in.",
            file=sys.stderr,
        )
        return 1

    from src.ingest.store import open_collection

    print(f"ensure_index: index ready ({open_collection().count()} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
