import pytest

from swecrew.edits import EditError, apply_edit, apply_edits
from swecrew.models import FileEdit


def test_create_new_file(tmp_path):
    edit = apply_edit(tmp_path, FileEdit("pkg/new.py", "", "def f():\n    return 1\n"))
    assert edit.created
    assert (tmp_path / "pkg" / "new.py").read_text() == "def f():\n    return 1\n"


def test_create_refuses_existing_nonempty_file(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    with pytest.raises(EditError, match="already exists"):
        apply_edit(tmp_path, FileEdit("a.py", "", "y = 2\n"))


def test_replace_unique_text(tmp_path):
    (tmp_path / "a.py").write_text("def add(a, b):\n    return a - b\n")
    apply_edit(tmp_path, FileEdit("a.py", "return a - b", "return a + b"))
    assert (tmp_path / "a.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_replace_missing_text_raises(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    with pytest.raises(EditError, match="not found"):
        apply_edit(tmp_path, FileEdit("a.py", "y = 2", "y = 3"))


def test_replace_ambiguous_text_raises(tmp_path):
    (tmp_path / "a.py").write_text("pass\npass\n")
    with pytest.raises(EditError, match="not unique"):
        apply_edit(tmp_path, FileEdit("a.py", "pass", "return"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(EditError, match="does not exist"):
        apply_edit(tmp_path, FileEdit("nope.py", "x", "y"))


def test_edit_cannot_escape_repo_root(tmp_path):
    with pytest.raises(EditError, match="escapes"):
        apply_edit(tmp_path, FileEdit("../outside.py", "", "evil = True\n"))


def test_apply_edits_all_or_nothing_rollback(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    with pytest.raises(EditError):
        apply_edits(tmp_path, [FileEdit("a.py", "x = 1", "x = 100"), FileEdit("b.py", "MISSING", "y = 200")])
    assert (tmp_path / "a.py").read_text() == "x = 1\n"  # rolled back
    assert (tmp_path / "b.py").read_text() == "y = 2\n"


def test_apply_edits_success(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    applied = apply_edits(tmp_path, [FileEdit("a.py", "x = 1", "x = 2"), FileEdit("c.py", "", "z = 3\n")])
    assert [a.path for a in applied] == ["a.py", "c.py"]
    assert (tmp_path / "a.py").read_text() == "x = 2\n"
    assert (tmp_path / "c.py").read_text() == "z = 3\n"
