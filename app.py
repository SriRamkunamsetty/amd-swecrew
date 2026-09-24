"""Harness entrypoint: /app/app.py inside the container."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from swecrew.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
