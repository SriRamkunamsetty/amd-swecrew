"""Localizer: picks which files to show the coder.

The LLM only ever *chooses among* real candidate paths (never invents one); the deterministic
ranking in ``repo_map`` is the fallback and the safety net when there is no LLM at all.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from academy_core import Deadline, LLMClient, LLMError, get_logger
from swecrew.prompts import localize_prompt
from swecrew.repo_map import RepoMap
from swecrew.task import Task

log = get_logger(__name__)


class LocalizeResult(BaseModel):
    files: list[str] = Field(default_factory=list)
    reasoning: str = ""


class LocalizerAgent:
    def __init__(self, llm: LLMClient | None):
        self.llm = llm

    async def select(self, task: Task, repo_map: RepoMap, max_candidates: int, max_files: int,
                     deadline: Deadline | None = None) -> list[str]:
        candidates = repo_map.top(max_candidates)
        heuristic = [f.path for f in candidates[:max_files]]
        if self.llm is None or not await self.llm.available():
            return heuristic
        try:
            result = await self.llm.complete_json(
                localize_prompt(task, [f.path for f in candidates]), LocalizeResult,
                max_tokens=300, deadline=deadline,
            )
        except LLMError as exc:
            log.info("localizer llm skipped: %s", exc)
            return heuristic
        valid_paths = {f.path for f in candidates}
        picked = [p for p in result.files if p in valid_paths]
        return (picked or heuristic)[:max_files]
