"""DebuggerAgent: a thin wrapper that hands a failing candidate's traceback back to the coder
for a bounded number of repair rounds. Kept as its own module so the orchestrator's control
flow reads as one role per line, and so a smarter debugger (e.g. bisecting which edit in a
multi-file patch caused the regression) can be dropped in later without touching the coder.
"""

from __future__ import annotations

from swecrew.agents.coder import CoderAgent
from swecrew.models import Candidate


class DebuggerAgent:
    def __init__(self, coder: CoderAgent):
        self.coder = coder

    async def repair(self, context: str, candidate: Candidate, deadline=None) -> Candidate | None:
        if candidate.outcome is None:
            return None
        failure = candidate.outcome.stdout_tail or candidate.outcome.error or "(no output captured)"
        return await self.coder.repair(context, candidate, failure, deadline=deadline)
