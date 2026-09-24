"""The grading-harness contract shared by every challenge.

From the official challenge document:
  * the harness runs ``python3 /app/app.py --input-... <path>`` once per test item,
  * the answer is written to ``/app/output/<input stem>_output.json``,
  * answers are compared after normalization (uppercase, whitespace removed,
    ``- . · _`` removed), pass/fail.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path(os.getenv("HARNESS_OUTPUT_DIR", "/app/output"))
_STRIP = re.compile(r"[\s\-._·]+")


def grader_normalize(text: str) -> str:
    """Replicates the official normalization so we can score ourselves locally."""
    return _STRIP.sub("", str(text)).upper()


def output_path_for(input_path: str | os.PathLike[str], output_dir: str | os.PathLike[str] | None = None) -> Path:
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    return out_dir / f"{Path(input_path).stem}_output.json"


def write_json_atomic(path: str | os.PathLike[str], payload: dict[str, Any]) -> Path:
    """Write-then-rename so the harness never reads a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_input_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Accept JSON objects, JSON strings, or plain text; always return a dict."""
    raw = Path(path).read_text(encoding="utf-8-sig").strip()
    if not raw:
        return {}
    if raw[0] in "{[\"":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"text": raw}
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            return {"text": data}
        return {"items": data}
    return {"text": raw}
