"""One shared context builder, reused verbatim across localize/coder/debug calls.

Why this matters beyond prompt hygiene: all coder attempts and the debugger share the *same*
long prefix (issue + localized file contents). Served through vLLM's automatic prefix caching,
N parallel candidates cost close to one prefill instead of N -- the efficiency story behind
using several coder agents rather than one big call.
"""

from __future__ import annotations

from swecrew.sandbox import Workspace
from swecrew.task import Task

SYSTEM = (
    "You are an expert software engineer fixing a real bug in a real repository. "
    "You only ever propose changes to files whose current content you have been shown. "
    "You are precise about exact text and whitespace, because your edits are applied as "
    "literal search-and-replace, not as freeform text."
)


def build_context(task: Task, workspace: Workspace, files: list[str], settings_max_chars: int) -> str:
    parts = [f"## Problem statement\n{task.problem_statement.strip()}\n"]
    budget = settings_max_chars
    for path in files:
        content = workspace.read_file(path)
        if content is None:
            continue
        block = f"\n## File: {path}\n```\n{content}\n```\n"
        if len(block) > budget:
            block = block[: max(200, budget)] + "\n... (truncated)\n```\n"
        parts.append(block)
        budget -= len(block)
        if budget <= 0:
            break
    return "".join(parts)


def localize_prompt(task: Task, candidate_files: list[str]) -> list[dict[str, str]]:
    listing = "\n".join(f"- {f}" for f in candidate_files)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"## Problem statement\n{task.problem_statement.strip()}\n\n"
            f"## Candidate files in this repository\n{listing}\n\n"
            "Pick the files most likely to need changes to fix this problem. Return JSON with "
            "`files` (a list of paths copied exactly from the candidate list above, most relevant "
            "first) and a one-sentence `reasoning`.")},
    ]


def coder_prompt(context: str, attempt: int, prior_feedback: str | None = None) -> list[dict[str, str]]:
    user = context + (
        "\n\nPropose a fix as a list of search/replace edits. Each edit's `search` must be copied "
        "EXACTLY (including whitespace and indentation) from the file content shown above, and must "
        "be long enough to be unique in the file. Use `search: \"\"` only to create a brand-new file. "
        "Keep edits minimal and focused on the bug. Return JSON with `edits` "
        "(list of {path, search, replace}) and a one-sentence `summary`."
    )
    if attempt:
        user += f"\n\n(This is attempt #{attempt + 1}: consider a different, independent approach.)"
    if prior_feedback:
        user += f"\n\nA previous attempt failed:\n{prior_feedback}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def debug_prompt(context: str, prior_summary: str, failure_output: str) -> list[dict[str, str]]:
    user = context + (
        f"\n\nA previous fix attempt (\"{prior_summary}\") did not make the tests pass. "
        f"Test output (tail):\n```\n{failure_output[-3000:]}\n```\n\n"
        "Propose a corrected list of search/replace edits (against the ORIGINAL file content shown "
        "above, not against the previous attempt's edits). Return JSON with `edits` "
        "(list of {path, search, replace}) and a one-sentence `summary`."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def reproduce_prompt(task: Task, candidate_files: list[str]) -> list[dict[str, str]]:
    listing = "\n".join(f"- {f}" for f in candidate_files[:15])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"## Problem statement\n{task.problem_statement.strip()}\n\n"
            f"## Some files in this repository\n{listing}\n\n"
            "Write ONE new pytest test function that reproduces this bug (it should currently FAIL). "
            "Return JSON with `test_path` (a new file path such as `tests/test_swecrew_repro.py`) and "
            "`content` (the full file content, including imports).")},
    ]
