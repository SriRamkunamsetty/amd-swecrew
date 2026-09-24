"""Wall-clock budgets.

Every harness call has a hard time limit and exceeding it scores zero, so all
long-running work takes a Deadline and checks it instead of hoping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Deadline:
    seconds: float
    start: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start

    @property
    def remaining(self) -> float:
        return max(0.0, self.seconds - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.remaining <= 0.0

    def fraction_used(self) -> float:
        return min(1.0, self.elapsed / self.seconds) if self.seconds > 0 else 1.0

    def child(self, seconds: float, reserve: float = 0.0) -> Deadline:
        """A sub-budget capped by what is left on this one (minus a reserve)."""
        return Deadline(max(0.0, min(seconds, self.remaining - reserve)))

    def timeout(self, cap: float, reserve: float = 0.0) -> float:
        """A per-operation timeout that never outlives the parent budget."""
        return max(0.05, min(cap, self.remaining - reserve))
