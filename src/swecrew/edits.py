"""Search/replace block edits.

The coder LLM never emits a raw unified diff (models are unreliable at line numbers and hunk
headers); it emits ``{"path", "search", "replace"}`` blocks. ``search == ""`` creates a new
file. Applying is exact-match only -- no fuzzy matching -- so a bad edit fails loudly instead
of silently corrupting a file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from swecrew.models import FileEdit


class EditError(ValueError):
    pass


@dataclass
class AppliedEdit:
    path: str
    created: bool


def apply_edit(root: Path, edit: FileEdit) -> AppliedEdit:
    target = (root / edit.path).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise EditError(f"edit path escapes the repository: {edit.path!r}")

    if edit.search == "":
        if target.exists() and target.read_text(encoding="utf-8", errors="replace").strip():
            raise EditError(f"{edit.path}: file already exists and is non-empty; use search/replace instead")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit.replace, encoding="utf-8")
        return AppliedEdit(edit.path, created=True)

    if not target.exists():
        raise EditError(f"{edit.path}: file does not exist")
    original = target.read_text(encoding="utf-8", errors="replace")
    count = original.count(edit.search)
    if count == 0:
        raise EditError(f"{edit.path}: search text not found (check whitespace/indentation)")
    if count > 1:
        raise EditError(f"{edit.path}: search text is not unique ({count} occurrences); make it more specific")
    target.write_text(original.replace(edit.search, edit.replace, 1), encoding="utf-8")
    return AppliedEdit(edit.path, created=False)


def apply_edits(root: Path, edits: list[FileEdit]) -> list[AppliedEdit]:
    """All-or-nothing: if any edit fails, none of them are left applied."""
    applied: list[AppliedEdit] = []
    backups: dict[str, str | None] = {}
    try:
        for edit in edits:
            target = root / edit.path
            if edit.path not in backups:
                backups[edit.path] = target.read_text(encoding="utf-8", errors="replace") if target.exists() else None
            applied.append(apply_edit(root, edit))
        return applied
    except EditError:
        for path, original in backups.items():
            target = root / path
            if original is None:
                target.unlink(missing_ok=True)
            else:
                target.write_text(original, encoding="utf-8")
        raise
