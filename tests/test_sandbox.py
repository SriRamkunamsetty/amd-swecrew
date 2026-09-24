import sys

from swecrew.models import FileEdit
from swecrew.sandbox import Workspace
from swecrew.task import Task

BUGGY = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
TEST = (
    "from pkg.calc import add, mul\n\n"
    "def test_add():\n    assert add(2, 3) == 5\n\n"
    "def test_mul():\n    assert mul(2, 3) == 6\n"
)


def _make_repo(root):
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "calc.py").write_text(BUGGY)
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "test_calc.py").write_text(TEST)


def _task(repo_dir) -> Task:
    return Task(problem_statement="add() returns a - b instead of a + b", repo_path=repo_dir)


def test_workspace_creates_git_baseline_from_plain_directory(tmp_path):
    repo = tmp_path / "src"
    repo.mkdir()
    _make_repo(repo)
    ws = Workspace.create(_task(repo), work_dir=str(tmp_path / "ws1"))
    try:
        assert (ws.path / ".git").exists()
        assert ws.diff() == ""  # nothing changed yet
    finally:
        ws.cleanup()


def test_apply_diff_and_reset_roundtrip(tmp_path):
    repo = tmp_path / "src"
    repo.mkdir()
    _make_repo(repo)
    ws = Workspace.create(_task(repo), work_dir=str(tmp_path / "ws2"))
    try:
        changed = ws.apply([FileEdit("pkg/calc.py", "return a - b", "return a + b")])
        assert changed == ["pkg/calc.py"]
        diff = ws.diff()
        assert "-    return a - b" in diff and "+    return a + b" in diff
        assert ws.changed_files() == ["pkg/calc.py"]
        ws.reset()
        assert ws.diff() == ""
        assert "a - b" in ws.read_file("pkg/calc.py")
    finally:
        ws.cleanup()


def test_run_tests_reports_fail_to_pass_and_pass_to_pass(tmp_path):
    repo = tmp_path / "src"
    repo.mkdir()
    _make_repo(repo)
    ws = Workspace.create(_task(repo), work_dir=str(tmp_path / "ws3"))
    try:
        f2p, p2p = ["tests/test_calc.py::test_add"], ["tests/test_calc.py::test_mul"]
        before = ws.run_tests(None, f2p, p2p, timeout=30, python=sys.executable)
        assert before.ran and before.fail_to_pass_passed == 0 and before.pass_to_pass_passed == 1
        assert not before.all_green

        ws.apply([FileEdit("pkg/calc.py", "return a - b", "return a + b")])
        after = ws.run_tests(None, f2p, p2p, timeout=30, python=sys.executable)
        assert after.fail_to_pass_passed == 1 and after.pass_to_pass_passed == 1
        assert after.all_green
        assert after.score > before.score
    finally:
        ws.cleanup()


def test_run_tests_detects_regression_from_a_bad_edit(tmp_path):
    repo = tmp_path / "src"
    repo.mkdir()
    _make_repo(repo)
    ws = Workspace.create(_task(repo), work_dir=str(tmp_path / "ws4"))
    try:
        f2p, p2p = ["tests/test_calc.py::test_add"], ["tests/test_calc.py::test_mul"]
        # "fixes" add() but breaks mul() -- a net-negative patch
        ws.apply([
            FileEdit("pkg/calc.py", "return a - b", "return a + b"),
            FileEdit("pkg/calc.py", "return a * b", "return a - b"),
        ])
        outcome = ws.run_tests(None, f2p, p2p, timeout=30, python=sys.executable)
        assert outcome.fail_to_pass_passed == 1 and outcome.pass_to_pass_passed == 0
        assert outcome.score < 0
    finally:
        ws.cleanup()


def test_run_tests_without_node_ids_uses_pytest_summary(tmp_path):
    repo = tmp_path / "src"
    repo.mkdir()
    _make_repo(repo)
    ws = Workspace.create(_task(repo), work_dir=str(tmp_path / "ws5"))
    try:
        outcome = ws.run_tests(None, [], [], timeout=30, python=sys.executable)
        assert outcome.ran and outcome.fail_to_pass_total == 2 and outcome.fail_to_pass_passed == 1
    finally:
        ws.cleanup()
