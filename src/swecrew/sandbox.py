"""Workspace: an isolated, git-tracked copy of the target repo.

Using git for the whole lifecycle (init/commit/diff/checkout/clean) is far more robust than
hand-rolled diffing:
  * ``diff()`` after edits is always a correct unified patch, in the format SWE-bench and every
    code-review tool expects -- no custom diff code to get wrong.
  * ``reset()`` between candidate attempts is one command, so N candidates share one clone
    instead of N full copies.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import venv
from dataclasses import dataclass
from pathlib import Path

from academy_core import get_logger
from swecrew.edits import apply_edits
from swecrew.models import FileEdit, TestOutcome
from swecrew.task import Task

log = get_logger(__name__)
GIT_ENV_AUTHOR = {"GIT_AUTHOR_NAME": "swecrew", "GIT_AUTHOR_EMAIL": "swecrew@local",
                  "GIT_COMMITTER_NAME": "swecrew", "GIT_COMMITTER_EMAIL": "swecrew@local"}


def _run(cmd: list[str], cwd: Path, timeout: float, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=full_env, check=False)


def _rmtree_force(path: Path) -> None:
    def on_error(func, target, exc_info):  # noqa: ANN001 - shutil callback signature
        try:
            Path(target).chmod(stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=on_error)


class WorkspaceError(RuntimeError):
    pass


@dataclass
class Workspace:
    path: Path
    _tmp: tempfile.TemporaryDirectory | None = None
    venv_python: str | None = None

    # ------------------------------------------------------------------ setup
    @classmethod
    def create(cls, task: Task, work_dir: str | None = None, clone_timeout: float = 60.0) -> Workspace:
        tmp = None
        if work_dir:
            root = Path(work_dir)
            root.mkdir(parents=True, exist_ok=True)
        else:
            tmp = tempfile.TemporaryDirectory(prefix="swecrew-")
            root = Path(tmp.name)
        target = root / "repo"

        if task.repo_path and task.repo_path.is_dir():
            source = task.repo_path.resolve()
            if source == target.resolve() or source in target.resolve().parents:
                raise WorkspaceError(
                    f"work_dir {target} would be created inside repo_path {source}; use an unrelated work_dir")
            shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git"))
        elif task.repo_url:
            result = _run(["git", "clone", "--depth", "1", task.repo_url, str(target)], root, clone_timeout)
            if result.returncode != 0:
                raise WorkspaceError(f"git clone failed: {result.stderr[-500:]}")
        else:
            raise WorkspaceError("task has no repo_path or repo_url")

        ws = cls(target, _tmp=tmp)
        ws._ensure_git_baseline(task.base_commit)
        return ws

    def _ensure_git_baseline(self, base_commit: str | None) -> None:
        if not (self.path / ".git").exists():
            _run(["git", "init", "-q"], self.path, 30)
        if base_commit:
            checkout = _run(["git", "checkout", "-q", base_commit], self.path, 30)
            if checkout.returncode != 0:
                log.warning("could not check out base_commit %s: %s", base_commit, checkout.stderr[-300:])
        _run(["git", "add", "-A"], self.path, 60)
        _run(["git", "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", "swecrew: baseline"],
             self.path, 60, GIT_ENV_AUTHOR)

    # ------------------------------------------------------------------ per-candidate lifecycle
    def reset(self) -> None:
        """Fully revert to the baseline commit, including new files, without touching HEAD.

        Deliberately plain ``-fd`` (not ``-x``): a target repo's own gitignored build artifacts
        (e.g. an editable install's ``*.egg-info``, written once by ``prepare_env``) must survive
        resets between candidates. Bytecode-cache staleness is instead prevented at the source,
        by disabling ``.pyc`` writing for test runs (see ``run_tests``)."""
        _run(["git", "reset", "-q", "--hard", "HEAD"], self.path, 30)
        _run(["git", "clean", "-q", "-fd"], self.path, 30)

    def apply(self, edits: list[FileEdit]) -> list[str]:
        applied = apply_edits(self.path, edits)
        return [a.path for a in applied]

    def _mark_changes(self) -> None:
        # --intent-to-add records new files (as empty blobs) so `git diff` shows their full
        # content as added, without actually staging anything -- reset() stays a clean revert.
        _run(["git", "add", "-A", "-N"], self.path, 30)

    def diff(self) -> str:
        self._mark_changes()
        result = _run(["git", "diff", "--no-color"], self.path, 30)
        return result.stdout

    def changed_files(self) -> list[str]:
        self._mark_changes()
        result = _run(["git", "diff", "--name-only"], self.path, 30)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def read_file(self, rel_path: str, max_bytes: int = 60_000) -> str | None:
        target = self.path / rel_path
        if not target.is_file():
            return None
        try:
            data = target.read_bytes()[:max_bytes]
            return data.decode("utf-8", errors="replace")
        except OSError:
            return None

    # ------------------------------------------------------------------ dependencies + tests
    def prepare_env(self, timeout: float) -> str:
        """Best-effort isolated venv with the target repo installed. Never fatal: falls back
        to the interpreter running this process, which already has torch/httpx/etc. available."""
        if self.venv_python:
            return self.venv_python
        venv_dir = self.path.parent / ".venv-task"
        try:
            venv.EnvBuilder(with_pip=True, symlinks=(sys.platform != "win32")).create(venv_dir)
            python = str(venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
            _run([python, "-m", "pip", "install", "-q", "pytest"], self.path, timeout / 3)
            if (self.path / "pyproject.toml").exists() or (self.path / "setup.py").exists():
                _run([python, "-m", "pip", "install", "-q", "-e", "."], self.path, timeout / 2)
            elif (self.path / "requirements.txt").exists():
                _run([python, "-m", "pip", "install", "-q", "-r", "requirements.txt"], self.path, timeout / 2)
            self.venv_python = python
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("env setup failed, falling back to the host interpreter: %s", exc)
            self.venv_python = sys.executable
        return self.venv_python

    def run_tests(self, cmd: str | None, fail_to_pass: list[str], pass_to_pass: list[str],
                 timeout: float, python: str | None = None) -> TestOutcome:
        python = python or sys.executable
        targets = [*fail_to_pass, *pass_to_pass]
        if cmd:
            shell_cmd = cmd
        elif targets:
            shell_cmd = f'{_quote(python)} -m pytest -q {" ".join(_quote(t) for t in targets)}'
        else:
            shell_cmd = f"{_quote(python)} -m pytest -q"
        # PYTHONDONTWRITEBYTECODE guards against a real hazard: the same repo path is reused
        # across candidates (reset -> edit -> test), and on filesystems with coarse mtime
        # resolution, Python's default timestamp-based .pyc invalidation can miss an edit that
        # lands within the same tick as the previous compile -- silently testing stale code.
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            result = subprocess.run(shell_cmd, shell=True, cwd=self.path, capture_output=True,  # noqa: S602
                                    text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired as exc:
            return TestOutcome(ran=True, timed_out=True, error="test run timed out",
                               stdout_tail=_tail((exc.stdout or "") + (exc.stderr or "")))
        except OSError as exc:
            return TestOutcome(ran=False, error=str(exc))
        output = result.stdout + "\n" + result.stderr
        f2p_passed, f2p_total = _count_test_results(output, fail_to_pass)
        p2p_passed, p2p_total = _count_test_results(output, pass_to_pass)
        if not targets:
            # No explicit test ids: use pytest's own summary as a single aggregate signal.
            passed, total = _parse_pytest_summary(output)
            f2p_passed, f2p_total = passed, total
        return TestOutcome(ran=True, returncode=result.returncode, fail_to_pass_passed=f2p_passed,
                          fail_to_pass_total=f2p_total or len(fail_to_pass), pass_to_pass_passed=p2p_passed,
                          pass_to_pass_total=p2p_total or len(pass_to_pass), stdout_tail=_tail(output))

    def cleanup(self, keep: bool = False) -> None:
        if keep:
            return
        if self.venv_python and self.venv_python != sys.executable:
            _rmtree_force(self.path.parent / ".venv-task")
        if self._tmp is not None:
            try:
                self._tmp.cleanup()
            except OSError:
                _rmtree_force(Path(self._tmp.name))


def _quote(value: str) -> str:
    return f'"{value}"' if " " in value else value


def _tail(text: str, n: int = 4000) -> str:
    return text[-n:]


def _count_test_results(output: str, node_ids: list[str]) -> tuple[int, int]:
    """Heuristic: with ``-q`` pytest only names tests it FAILED/ERRORed on, so anything not in
    that set (and not a full collection failure) is assumed to have passed."""
    if not node_ids:
        return 0, 0
    failed = _failed_ids(output)
    collection_failed = "collected 0 items" in output or "ERROR collecting" in output.lower()
    if collection_failed and not failed:
        return 0, len(node_ids)  # nothing could run at all
    passed = sum(1 for node_id in node_ids if node_id not in failed and not any(node_id in f for f in failed))
    return passed, len(node_ids)


def _failed_ids(output: str) -> set[str]:
    import re

    return set(re.findall(r"^(?:FAILED|ERROR) (\S+)", output, re.MULTILINE))


def _parse_pytest_summary(output: str) -> tuple[int, int]:
    import re

    match = re.search(r"(\d+) passed", output)
    passed = int(match.group(1)) if match else 0
    failed_match = re.search(r"(\d+) failed", output)
    error_match = re.search(r"(\d+) error", output)
    failed = int(failed_match.group(1)) if failed_match else 0
    errored = int(error_match.group(1)) if error_match else 0
    return passed, passed + failed + errored
