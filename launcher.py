#!/usr/bin/env python
"""Sale Deed AI - single entry point.

    python launcher.py

Starts PostgreSQL if needed, applies migrations, verifies the fine-tuned model,
starts the inference server, and opens the desktop application. Stops everything
cleanly on exit.

    --check      run the checks and exit, changing nothing
    --headless   start the services without the desktop window
    --no-ai      browsing and export only, no inference server

This file stays a thin shim so the package can be imported and tested without
executing anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# The packages live in `src/`, not beside this file. This is the single place
# that has to know that: everything under `src/` then imports normally.
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from launcher.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
