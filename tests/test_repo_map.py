from swecrew.repo_map import build_repo_map, extract_traceback_paths, rank_by_relevance


def _make_repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "calculator.py").write_text(
        "class Calculator:\n    def add(self, a, b):\n        return a - b\n\n\ndef helper():\n    pass\n"
    )
    (tmp_path / "pkg" / "utils.py").write_text("def unrelated():\n    return 42\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calculator.py").write_text(
        "from pkg.calculator import Calculator\n\ndef test_add():\n    assert Calculator().add(2, 3) == 5\n"
    )
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("function ignored() {}\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")


def test_build_repo_map_extracts_python_symbols_and_ignores_vendored_dirs(tmp_path):
    _make_repo(tmp_path)
    repo_map = build_repo_map(tmp_path)
    paths = {f.path for f in repo_map.files}
    assert "pkg/calculator.py" in paths and "tests/test_calculator.py" in paths
    assert not any("node_modules" in p or p.startswith(".git") for p in paths)
    calc = repo_map.by_path("pkg/calculator.py")
    assert {"Calculator", "add", "helper"} <= set(calc.symbols)


def test_rank_by_relevance_prefers_mentioned_file_and_deprioritizes_tests(tmp_path):
    _make_repo(tmp_path)
    repo_map = build_repo_map(tmp_path)
    rank_by_relevance(repo_map, "Calculator.add returns a - b instead of a + b in calculator.py")
    top = repo_map.top(3)
    assert top[0].path == "pkg/calculator.py"
    assert repo_map.by_path("pkg/utils.py").score < repo_map.by_path("pkg/calculator.py").score
    # a test file with the same symbol mentioned should score lower than the implementation file
    test_score = repo_map.by_path("tests/test_calculator.py").score
    assert test_score < top[0].score


def test_rank_by_relevance_uses_hints(tmp_path):
    _make_repo(tmp_path)
    repo_map = build_repo_map(tmp_path)
    rank_by_relevance(repo_map, "something is broken", hints=["utils.py"])
    assert repo_map.top(1)[0].path == "pkg/utils.py"


def test_extract_traceback_paths():
    tb = 'Traceback (most recent call last):\n  File "/app/pkg/calculator.py", line 3, in add\n    return a - b\n'
    assert extract_traceback_paths(tb) == ["calculator.py"]
