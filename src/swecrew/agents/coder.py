"""Coder: proposes a set of edits. Never allowed to invent an answer without checking --
that check happens later, in the sandbox, not here. A coder call that fails or returns
malformed JSON simply yields no candidate; the orchestrator degrades gracefully.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from academy_core import Deadline, LLMClient, LLMError, get_logger
from swecrew.models import Candidate, FileEdit
from swecrew.prompts import coder_prompt, debug_prompt

log = get_logger(__name__)


class EditItem(BaseModel):
    path: str
    search: str = ""
    replace: str = ""


class PatchProposal(BaseModel):
    edits: list[EditItem] = Field(default_factory=list, max_length=8)
    summary: str = ""


class CoderAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def propose(self, context: str, attempt: int, temperature: float,
                      deadline: Deadline | None = None) -> Candidate | None:
        try:
            result = await self.llm.complete_json(
                coder_prompt(context, attempt), PatchProposal, max_tokens=2000, deadline=deadline,
            )
        except LLMError as exc:
            log.info("coder attempt %d failed: %s", attempt, exc)
            return None
        if not result.edits:
            return None
        edits = [FileEdit(e.path, e.search, e.replace) for e in result.edits]
        return Candidate(edits=edits, summary=result.summary, source=f"coder#{attempt}", temperature=temperature)

    async def repair(self, context: str, prior: Candidate, failure_output: str,
                     deadline: Deadline | None = None) -> Candidate | None:
        try:
            result = await self.llm.complete_json(
                debug_prompt(context, prior.summary, failure_output), PatchProposal, max_tokens=2000,
                deadline=deadline,
            )
        except LLMError as exc:
            log.info("debugger repair failed: %s", exc)
            return None
        if not result.edits:
            return None
        edits = [FileEdit(e.path, e.search, e.replace) for e in result.edits]
        return Candidate(edits=edits, summary=result.summary or prior.summary, source="debugger",
                         temperature=prior.temperature, debug_rounds=prior.debug_rounds + 1)
