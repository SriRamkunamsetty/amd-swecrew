"""SweCrew orchestrator.

    load task -> workspace (git baseline) -> repo map -> localize files
       -> shared context (reused verbatim -> vLLM prefix caching makes N candidates ~1 prefill)
       -> N coder candidates in parallel, each tested in isolation (reset -> apply -> run tests)
       -> pick the best by *executed* outcome -> bounded debug/repair rounds on it
       -> final diff

Every stage is bounded by one Deadline, and the workspace is always cleaned up. If nothing
verified better than doing nothing, the result is an empty patch with confidence 0 -- never a
guess dressed up as a fix.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from academy_core import Deadline, LLMClient, get_logger
from swecrew.agents import CoderAgent, DebuggerAgent, LocalizerAgent, select_best
from swecrew.agents.selector import most_promising
from swecrew.config import Settings
from swecrew.models import Candidate, FixResult
from swecrew.prompts import build_context
from swecrew.repo_map import build_repo_map, extract_traceback_paths, rank_by_relevance
from swecrew.sandbox import Workspace, WorkspaceError
from swecrew.task import Task

log = get_logger(__name__)


@dataclass
class _State:
    trace: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)

    def log(self, msg: str) -> None:
        self.trace.append(f"{time.monotonic() - self.started:6.2f}s {msg}")
        log.debug(msg)


class SweCrew:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None):
        self.settings = settings or Settings()
        self.llm = llm if llm is not None else (LLMClient() if self.settings.use_llm else None)

    async def aclose(self) -> None:
        if self.llm is not None:
            await self.llm.aclose()

    async def __aenter__(self) -> SweCrew:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ public API
    async def fix(self, task: Task, budget_s: float | None = None) -> FixResult:
        deadline = Deadline(budget_s or task.time_budget_s or self.settings.time_budget_s)
        state = _State()
        s = self.settings

        try:
            workspace = await asyncio.to_thread(Workspace.create, task, s.work_dir)
        except WorkspaceError as exc:
            state.log(f"workspace creation failed: {exc}")
            return FixResult(summary=f"could not prepare workspace: {exc}", elapsed_s=deadline.elapsed,
                             trace=state.trace)

        try:
            return await self._fix_in_workspace(task, workspace, deadline, state)
        finally:
            await asyncio.to_thread(workspace.cleanup, s.keep_workspace)

    async def _fix_in_workspace(self, task: Task, workspace: Workspace, deadline: Deadline,
                                state: _State) -> FixResult:
        s = self.settings
        llm_ok = bool(self.llm is not None and s.use_llm and await self.llm.available())
        state.log(f"llm_available={llm_ok}")

        repo_map = await asyncio.to_thread(build_repo_map, workspace.path)
        hints = extract_traceback_paths(task.problem_statement)
        rank_by_relevance(repo_map, task.problem_statement, hints)
        state.log(f"repo map: {len(repo_map.files)} files")

        localizer = LocalizerAgent(self.llm if llm_ok else None)
        files = await localizer.select(task, repo_map, s.max_candidate_files, s.max_localized_files,
                                       deadline.child(s.llm_localize_timeout_s))
        state.log(f"localized files: {files}")

        if not llm_ok:
            state.log("no LLM available: cannot author a patch, returning an empty result")
            return self._result(task, workspace, None, [], deadline, state, llm_used=False)

        python = None
        if s.install_deps and deadline.remaining > s.env_setup_timeout_s + 10:
            python = await asyncio.to_thread(workspace.prepare_env, s.env_setup_timeout_s)
            state.log(f"env ready: {python}")

        context = build_context(task, workspace, files, s.max_context_chars)
        coder = CoderAgent(self.llm)  # type: ignore[arg-type]
        debugger = DebuggerAgent(coder)

        candidates = await self._generate_candidates(coder, context, deadline, state)
        for candidate in candidates:
            await self._evaluate(workspace, task, candidate, python, deadline, state)

        # Repair loop: always iterates on the *most promising* attempt so far (even one that
        # fixes nothing yet), but only ``select_best`` at the end decides what gets submitted.
        basis = most_promising(candidates)
        rounds = 0
        while (basis is not None and not (basis.outcome and basis.outcome.all_green)
               and rounds < s.max_debug_rounds and deadline.remaining > s.llm_debug_timeout_s + s.test_timeout_s):
            rounds += 1
            state.log(f"debug round {rounds} on {basis.source} (score={basis.outcome.score if basis.outcome else None})")
            repaired = await debugger.repair(context, basis, deadline.child(s.llm_debug_timeout_s))
            if repaired is None:
                break
            await self._evaluate(workspace, task, repaired, python, deadline, state)
            candidates.append(repaired)
            basis = most_promising(candidates)

        best = select_best(candidates)
        return self._result(task, workspace, best, candidates, deadline, state, llm_used=True)

    # ------------------------------------------------------------------ internals
    async def _generate_candidates(self, coder: CoderAgent, context: str, deadline: Deadline,
                                    state: _State) -> list[Candidate]:
        s = self.settings
        temps = (s.coder_temperatures or [0.0])[: s.num_candidates] or [0.0]
        while len(temps) < s.num_candidates:
            temps.append(temps[-1])
        tasks = [coder.propose(context, i, temp, deadline.child(s.llm_coder_timeout_s))
                for i, temp in enumerate(temps[: s.num_candidates])]
        results = await asyncio.gather(*tasks)
        candidates = [c for c in results if c is not None]
        state.log(f"coder proposed {len(candidates)}/{len(tasks)} candidates")
        return candidates

    async def _evaluate(self, workspace: Workspace, task: Task, candidate: Candidate, python: str | None,
                        deadline: Deadline, state: _State) -> None:
        def run() -> None:
            workspace.reset()
            try:
                workspace.apply(candidate.edits)
            except Exception as exc:  # noqa: BLE001 - a bad edit is a failed candidate, not a crash
                candidate.outcome = None
                state.log(f"{candidate.source}: edit application failed: {exc}")
                return
            candidate.patch = workspace.diff()
            timeout = min(self.settings.test_timeout_s, max(5.0, deadline.remaining - 2.0))
            candidate.outcome = workspace.run_tests(task.test_cmd or self.settings.default_test_cmd,
                                                    task.fail_to_pass, task.pass_to_pass, timeout, python)

        await asyncio.to_thread(run)
        if candidate.outcome:
            state.log(f"{candidate.source}: score={candidate.outcome.score} "
                      f"f2p={candidate.outcome.fail_to_pass_passed}/{candidate.outcome.fail_to_pass_total} "
                      f"p2p={candidate.outcome.pass_to_pass_passed}/{candidate.outcome.pass_to_pass_total}")

    def _result(self, task: Task, workspace: Workspace, best: Candidate | None, candidates: list[Candidate],
               deadline: Deadline, state: _State, llm_used: bool) -> FixResult:
        if best is None:
            return FixResult(patch="", confidence=0.0, summary="no verified fix found",
                             tests=self._test_dict(None), elapsed_s=deadline.elapsed, llm_used=llm_used,
                             trace=state.trace, candidates_tried=len(candidates))
        try:
            workspace.reset()
            workspace.apply(best.edits)
            patch = workspace.diff()
            files_changed = workspace.changed_files()
        except Exception as exc:  # noqa: BLE001
            state.log(f"final re-apply failed: {exc}")
            patch, files_changed = best.patch, []
        confidence = 0.95 if best.outcome and best.outcome.all_green else 0.4
        return FixResult(patch=patch, confidence=confidence, summary=best.summary, files_changed=files_changed,
                         tests=self._test_dict(best.outcome), elapsed_s=deadline.elapsed, llm_used=llm_used,
                         trace=state.trace, candidates_tried=len(candidates))

    @staticmethod
    def _test_dict(outcome) -> dict[str, int]:  # noqa: ANN001
        if outcome is None:
            return {}
        return {"fail_to_pass_passed": outcome.fail_to_pass_passed, "fail_to_pass_total": outcome.fail_to_pass_total,
                "pass_to_pass_passed": outcome.pass_to_pass_passed, "pass_to_pass_total": outcome.pass_to_pass_total}
