"""The crew: Localizer, Coder, Debugger, Selector. Each is a thin, testable role around the
shared LLMClient; none of them can write files or run tests themselves -- the orchestrator
owns the Workspace, so every proposal is verified the same way regardless of which role made it.
"""

from swecrew.agents.coder import CoderAgent
from swecrew.agents.debugger import DebuggerAgent
from swecrew.agents.localizer import LocalizerAgent
from swecrew.agents.selector import select_best

__all__ = ["CoderAgent", "DebuggerAgent", "LocalizerAgent", "select_best"]
