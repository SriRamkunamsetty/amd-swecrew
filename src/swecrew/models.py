from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileEdit:
    """A search/replace block. ``search == ""`` means "create the file with this content"."""

    path: str
    search: str
    replace: str


@dataclass
class TestOutcome:
    ran: bool
    returncode: int | None = None
    fail_to_pass_passed: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_passed: int = 0
    pass_to_pass_total: int = 0
    stdout_tail: str = ""
    timed_out: bool = False
    error: str = ""

    @property
    def score(self) -> float:
        """Higher is better. Regressions in previously-passing tests are penalised hard."""
        broke = self.pass_to_pass_total - self.pass_to_pass_passed
        return self.fail_to_pass_passed - 2.0 * broke

    @property
    def all_green(self) -> bool:
        return (self.ran and not self.timed_out and self.fail_to_pass_passed == self.fail_to_pass_total
                and self.pass_to_pass_passed == self.pass_to_pass_total and self.fail_to_pass_total > 0)


@dataclass
class Candidate:
    edits: list[FileEdit]
    summary: str = ""
    source: str = "coder"
    temperature: float = 0.0
    outcome: TestOutcome | None = None
    patch: str = ""
    debug_rounds: int = 0


@dataclass
class FixResult:
    patch: str = ""
    confidence: float = 0.0
    summary: str = ""
    files_changed: list[str] = field(default_factory=list)
    tests: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    llm_used: bool = False
    trace: list[str] = field(default_factory=list)
    candidates_tried: int = 0

    def to_output(self, include_debug: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "patch": self.patch,
            "model_patch": self.patch,  # SWE-bench-style alias
            "diff": self.patch,          # extra alias for an unknown official field name
            "confidence": round(self.confidence, 3),
            "files_changed": self.files_changed,
            "tests": self.tests,
            "summary": self.summary,
        }
        if include_debug:
            out |= {
                "elapsed_s": round(self.elapsed_s, 2),
                "llm_used": self.llm_used,
                "candidates_tried": self.candidates_tried,
                "trace": self.trace,
            }
        return out
