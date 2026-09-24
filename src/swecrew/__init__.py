"""SweCrew: a test-driven multi-agent software engineering crew.

Philosophy shared with the other mini-challenges in this monorepo: the LLM proposes inside a
grounded context (real files, real tracebacks), and deterministic code is what verifies and
decides. Agents "check each other" by *executing* the target repository's tests, not by
agreeing in conversation.
"""

from swecrew.config import Settings
from swecrew.models import FixResult
from swecrew.orchestrator import SweCrew
from swecrew.task import Task

__all__ = ["FixResult", "Settings", "SweCrew", "Task"]
__version__ = "0.1.0"
