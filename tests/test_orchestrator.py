"""End-to-end orchestrator tests with a scripted fake LLM (no network, no real model)."""

import json

import httpx
import pytest

from academy_core import LLMClient, LLMSettings
from swecrew.config import Settings
from swecrew.models import FileEdit
from swecrew.orchestrator import SweCrew
from swecrew.task import Task

BUGGY = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
TEST = (
    "from pkg.calc import add, mul\n\n"
    "def test_add():\n    assert add(2, 3) == 5\n\n"
    "def test_mul():\n    assert mul(2, 3) == 6\n"
)


def make_repo(root):
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "calc.py").write_text(BUGGY)
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "test_calc.py").write_text(TEST)


def _edits_json(*edits: FileEdit, summary: str) -> str:
    return json.dumps({"edits": [{"path": e.path, "search": e.search, "replace": e.replace} for e in edits],
                       "summary": summary})


def scripted_llm(routes: list[tuple[str, str]]) -> LLMClient:
    """``routes``: [(substring to match in the last user message, canned JSON content), ...],
    checked in order; the first match wins."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "fake-coder"}]})
        body = json.loads(request.content)
        last_user = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
        for needle, content in routes:
            if needle in last_user:
                return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"edits": [], "summary": ""}'}}]})

    return LLMClient(LLMSettings(base_url="http://fake/v1", max_retries=0), transport=httpx.MockTransport(handler))


def base_settings(**kw) -> Settings:
    base = dict(use_llm=True, num_candidates=2, coder_temperatures=[0.0, 0.4], max_debug_rounds=1,
               install_deps=False, time_budget_s=60, llm_coder_timeout_s=10, llm_localize_timeout_s=10,
               llm_debug_timeout_s=10, test_timeout_s=20)
    return Settings(**(base | kw))


LOCALIZE_JSON = json.dumps({"files": ["pkg/calc.py"], "reasoning": "the bug is in calc.py"})
WRONG_FIX = _edits_json(FileEdit("pkg/calc.py", "return a - b", "return a - b - 1"), summary="off by one, wrong")
RIGHT_FIX = _edits_json(FileEdit("pkg/calc.py", "return a - b", "return a + b"), summary="use addition")
REGRESSING_FIX = _edits_json(
    FileEdit("pkg/calc.py", "return a - b", "return a + b"),
    FileEdit("pkg/calc.py", "return a * b", "return a - b"),
    summary="fixes add, accidentally breaks mul",
)


async def test_first_candidate_wins_when_it_passes_everything(tmp_path):
    make_repo(tmp_path)
    llm = scripted_llm([("Candidate files", LOCALIZE_JSON), ("attempt #2", WRONG_FIX), ("", RIGHT_FIX)])
    task = Task(problem_statement="add() returns a - b instead of a + b", repo_path=tmp_path,
               fail_to_pass=["tests/test_calc.py::test_add"], pass_to_pass=["tests/test_calc.py::test_mul"])
    async with SweCrew(base_settings(), llm=llm) as crew:
        result = await crew.fix(task)
    assert result.confidence == pytest.approx(0.95)
    assert "return a + b" in result.patch
    assert result.tests["fail_to_pass_passed"] == 1 and result.tests["pass_to_pass_passed"] == 1
    assert result.llm_used and result.candidates_tried >= 2


async def test_debug_round_repairs_a_failing_first_attempt(tmp_path):
    make_repo(tmp_path)
    # both initial coder attempts are wrong; the debug/repair call gets it right.
    llm = scripted_llm([("Candidate files", LOCALIZE_JSON), ("A previous fix attempt", RIGHT_FIX),
                        ("", WRONG_FIX)])
    task = Task(problem_statement="add() returns a - b instead of a + b", repo_path=tmp_path,
               fail_to_pass=["tests/test_calc.py::test_add"], pass_to_pass=["tests/test_calc.py::test_mul"])
    async with SweCrew(base_settings(), llm=llm) as crew:
        result = await crew.fix(task)
    assert result.confidence == pytest.approx(0.95)
    assert "return a + b" in result.patch


async def test_regressing_candidate_is_never_selected_over_doing_nothing(tmp_path):
    make_repo(tmp_path)
    llm = scripted_llm([("Candidate files", LOCALIZE_JSON), ("", REGRESSING_FIX)])
    task = Task(problem_statement="add() returns a - b instead of a + b", repo_path=tmp_path,
               fail_to_pass=["tests/test_calc.py::test_add"], pass_to_pass=["tests/test_calc.py::test_mul"])
    async with SweCrew(base_settings(max_debug_rounds=0), llm=llm) as crew:
        result = await crew.fix(task)
    assert result.patch == "" and result.confidence == 0.0


async def test_no_llm_returns_empty_result_without_crashing(tmp_path):
    make_repo(tmp_path)
    task = Task(problem_statement="add() is broken", repo_path=tmp_path)
    async with SweCrew(base_settings(use_llm=False), llm=LLMClient(LLMSettings(enabled=False))) as crew:
        result = await crew.fix(task)
    assert result.patch == "" and result.confidence == 0.0 and not result.llm_used


async def test_malformed_repo_path_does_not_crash(tmp_path):
    task = Task(problem_statement="fix it", repo_path=tmp_path / "does-not-exist")
    async with SweCrew(base_settings(use_llm=False), llm=LLMClient(LLMSettings(enabled=False))) as crew:
        result = await crew.fix(task, budget_s=10)
    assert result.patch == "" and "workspace" in result.summary.lower()
