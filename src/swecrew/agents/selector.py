"""Selector: picks the winning candidate by *executed* outcome, never by the model's own
confidence. Ties break toward the smallest diff (Occam's razor for patches).
"""

from __future__ import annotations

from swecrew.models import Candidate


def _rank_key(c: Candidate) -> tuple[float, int]:
    diff_size = sum(len(e.search) + len(e.replace) for e in c.edits)
    return (c.outcome.score, -diff_size)  # type: ignore[union-attr]


def most_promising(candidates: list[Candidate]) -> Candidate | None:
    """The best-scoring tested candidate so far, with no safety threshold. Used to decide what
    the debugger should try to repair next -- an attempt that fixes nothing yet is still worth
    iterating on, even though it must never be *submitted* as-is (see ``select_best``)."""
    tested = [c for c in candidates if c.outcome is not None and c.outcome.ran]
    return max(tested, key=_rank_key) if tested else None


def select_best(candidates: list[Candidate]) -> Candidate | None:
    """The candidate to actually submit: never a net regression, never a no-op dressed up as
    a fix. ``score > 0`` requires fixing strictly more than it breaks (see ``TestOutcome.score``)."""
    best = most_promising(candidates)
    if best is None or best.outcome is None:
        return None
    return best if best.outcome.score > 0 or best.outcome.all_green else None
