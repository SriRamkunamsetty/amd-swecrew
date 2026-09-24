"""Task loading: the official input shape for this challenge is not published yet, so this
accepts every plausible layout and normalizes to one ``Task``.

Recognized JSON keys (aliases chosen to match SWE-bench-style datasets, the most common
convention for "repo + issue -> patch"):
    repo | repo_path | repo_dir           local path to the repository (may already be mounted)
    repo_url | repository                 a git URL to clone instead
    problem_statement | issue | task | description | prompt      the bug/feature description
    base_commit | commit                  optional commit to check out before editing
    test_cmd | test_command                optional shell command that runs the test suite
    fail_to_pass | fail_to_pass_tests      optional list of test node ids that should start failing
    pass_to_pass | pass_to_pass_tests      optional list of test node ids that must keep passing
    time_budget | time_budget_s            optional per-task override

If ``repo`` is a path to a mounted directory next to the input file, sibling files named
``problem_statement.txt`` / ``issue.txt`` are used when the JSON has no description.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_KEYS = ("repo", "repo_path", "repo_dir", "repository_path")
URL_KEYS = ("repo_url", "repository", "git_url", "clone_url")
TEXT_KEYS = ("problem_statement", "issue", "task", "description", "prompt", "bug_report")
COMMIT_KEYS = ("base_commit", "commit", "sha")
CMD_KEYS = ("test_cmd", "test_command", "test_script")
F2P_KEYS = ("fail_to_pass", "fail_to_pass_tests", "failing_tests")
P2P_KEYS = ("pass_to_pass", "pass_to_pass_tests", "regression_tests")
BUDGET_KEYS = ("time_budget", "time_budget_s", "budget_s")


def _first(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if payload.get(key):
            return payload[key]
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.replace(",", "\n").splitlines() if v.strip()]
    return [str(v) for v in value]


@dataclass
class Task:
    problem_statement: str
    repo_path: Path | None = None
    repo_url: str | None = None
    base_commit: str | None = None
    test_cmd: str | None = None
    fail_to_pass: list[str] = field(default_factory=list)
    pass_to_pass: list[str] = field(default_factory=list)
    time_budget_s: float | None = None
    task_id: str = "task"

    @classmethod
    def from_payload(cls, payload: dict[str, Any], input_path: str | Path | None = None) -> Task:
        repo_value = _first(payload, REPO_KEYS)
        repo_path = None
        if repo_value:
            candidate = Path(str(repo_value))
            if not candidate.is_absolute() and input_path is not None:
                sibling = Path(input_path).parent / candidate
                candidate = sibling if sibling.exists() else candidate
            repo_path = candidate
        elif input_path is not None:
            # common harness layout: {input}_repo/ or a "repo" sibling directory
            base = Path(input_path)
            for guess in (base.parent / "repo", base.parent / f"{base.stem}_repo", base.with_suffix("")):
                if guess.is_dir():
                    repo_path = guess
                    break

        text = _first(payload, TEXT_KEYS)
        if not text and input_path is not None:
            for name in ("problem_statement.txt", "issue.txt", "task.txt"):
                sibling = Path(input_path).parent / name
                if sibling.exists():
                    text = sibling.read_text(encoding="utf-8", errors="replace")
                    break
        budget = _first(payload, BUDGET_KEYS)

        return cls(
            problem_statement=str(text or "").strip(),
            repo_path=repo_path,
            repo_url=_first(payload, URL_KEYS),
            base_commit=_first(payload, COMMIT_KEYS),
            test_cmd=_first(payload, CMD_KEYS),
            fail_to_pass=_as_list(_first(payload, F2P_KEYS)),
            pass_to_pass=_as_list(_first(payload, P2P_KEYS)),
            time_budget_s=float(budget) if budget else None,
            task_id=str(payload.get("id") or payload.get("instance_id") or "task"),
        )

    @classmethod
    def load(cls, path: str | Path) -> Task:
        raw = Path(path).read_text(encoding="utf-8-sig").strip()
        if raw and raw[0] in "{[":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"problem_statement": raw}
        else:
            data = {"problem_statement": raw}
        if not isinstance(data, dict):
            data = {"problem_statement": str(data)}
        return cls.from_payload(data, input_path=path)
