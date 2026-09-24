import json

from swecrew import cli
from swecrew.orchestrator import SweCrew
from swecrew.prompts import build_context
from swecrew.repo_map import build_repo_map, rank_by_relevance
from swecrew.sandbox import Workspace
from swecrew.task import Task


def make_repo(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "pkg").mkdir()
    (root / "pkg" / "calc.py").write_text("def add(a, b):\n    return a - b\n")


def test_task_load_json_aliases(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"issue": "fix add()", "repo_path": "repo", "fail_to_pass_tests": "t1, t2"}))
    (tmp_path / "repo").mkdir()
    task = Task.load(p)
    assert task.problem_statement == "fix add()"
    assert task.repo_path == tmp_path / "repo"
    assert task.fail_to_pass == ["t1", "t2"]


def test_task_load_plain_text_and_sibling_files(tmp_path):
    (tmp_path / "repo").mkdir()
    (tmp_path / "issue.txt").write_text("add() is backwards")
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"repo": "repo"}))
    task = Task.load(p)
    assert task.problem_statement == "add() is backwards"
    assert task.repo_path == tmp_path / "repo"


def test_task_load_bare_text_file(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("Please fix the crash in parser.py")
    task = Task.load(p)
    assert "parser.py" in task.problem_statement


def test_cli_always_writes_output_even_on_crash(tmp_path, monkeypatch):
    make_repo(tmp_path / "repo")
    inp = tmp_path / "task_01.json"
    inp.write_text(json.dumps({"problem_statement": "fix add()", "repo": str(tmp_path / "repo")}))

    async def boom(self, task, budget_s=None):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(SweCrew, "fix", boom)
    assert cli.main(["--input-file", str(inp), "--output-dir", str(tmp_path / "out"), "--no-llm"]) == 0
    data = json.loads((tmp_path / "out" / "task_01_output.json").read_text())
    assert data["patch"] == "" and "simulated crash" in data["error"]


def test_cli_without_llm_returns_empty_but_valid_result(tmp_path):
    make_repo(tmp_path / "repo")
    inp = tmp_path / "task_02.json"
    inp.write_text(json.dumps({"problem_statement": "fix add()", "repo": str(tmp_path / "repo")}))
    assert cli.main(["--input-file", str(inp), "--output-dir", str(tmp_path / "out"), "--no-llm",
                     "--budget", "20"]) == 0
    data = json.loads((tmp_path / "out" / "task_02_output.json").read_text())
    assert data["patch"] == "" and data["confidence"] == 0.0 and "error" not in data


def test_cli_missing_task_writes_error_without_touching_a_repo(tmp_path):
    inp = tmp_path / "task_03.json"
    inp.write_text(json.dumps({"problem_statement": ""}))
    assert cli.main(["--input-file", str(inp), "--output-dir", str(tmp_path / "out"), "--no-llm"]) == 0
    data = json.loads((tmp_path / "out" / "task_03_output.json").read_text())
    assert "no problem_statement" in data["error"]


def test_build_context_includes_issue_and_localized_files(tmp_path):
    repo, ws_root = tmp_path / "repo", tmp_path / "ws"
    make_repo(repo)
    task = Task(problem_statement="add() is backwards", repo_path=repo)
    ws = Workspace.create(task, work_dir=str(ws_root))
    try:
        context = build_context(task, ws, ["pkg/calc.py"], settings_max_chars=10_000)
        assert "add() is backwards" in context
        assert "return a - b" in context and "pkg/calc.py" in context
    finally:
        ws.cleanup()


def test_build_context_respects_char_budget(tmp_path):
    repo, ws_root = tmp_path / "repo", tmp_path / "ws"
    repo.mkdir()
    (repo / "big.py").write_text("x = 1\n" * 5000)
    task = Task(problem_statement="issue", repo_path=repo)
    ws = Workspace.create(task, work_dir=str(ws_root))
    try:
        context = build_context(task, ws, ["big.py"], settings_max_chars=500)
        assert len(context) < 2000
    finally:
        ws.cleanup()


def test_repo_map_and_ranking_are_wired_for_a_real_layout(tmp_path):
    make_repo(tmp_path)
    repo_map = build_repo_map(tmp_path)
    rank_by_relevance(repo_map, "add() in calc.py returns the wrong value")
    assert repo_map.top(1)[0].path == "pkg/calc.py"
