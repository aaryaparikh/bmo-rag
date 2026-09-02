"""Repository-local launcher for the observability monitor."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

app = import_module("bmo_rag.cli").app


if __name__ == "__main__":
    sys.argv.insert(1, "monitor")
    app()
