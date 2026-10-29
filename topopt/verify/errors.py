"""Errors of `topopt verify` (CONTRACTS.md section 16).

`UserError` (from `topopt.config`) is bad input or bad usage: exit code 1.
`VerifyError` is a run that could not complete: exit code 2. The message is one or two lines and holds the facts and the options.
`SolverUnavailable` (a `UserError`) is a linear solver that cannot run here: none at all, or the one that the user named. It is
exit code 1: the text holds the facts about every factor and the options.
`UnsolvableMesh` (a `UserError`) is a part that the tet model cannot solve: a body with no support, a load region that
no body touches, a singular matrix. It is exit code 1, with the facts and the options in the text.
"""
from __future__ import annotations

from ..config import UserError


class VerifyError(RuntimeError):
    """The check could not run to the end (a missing extra, a meshing failure, a memory refusal, an empty region)."""


class VerifyUnavailable(VerifyError):
    """An optional package (gmsh, scikit-fem, or cholespy used directly) is missing. The text names the install step."""


class SolverUnavailable(UserError):
    """No linear solver can run here (`auto`), or the solver that the user named cannot (exit code 1, not a crash).

    `facts` holds the note of every factor (`{name: {"available", "note"}}`) and the name that was asked for.
    """

    def __init__(self, message: str, facts: dict | None = None):
        super().__init__(message)
        self.facts = dict(facts or {})


class MemoryRefused(VerifyError):
    """The memory estimate is above the limit. `facts` holds the numbers and the override."""

    def __init__(self, message: str, facts: dict | None = None):
        super().__init__(message)
        self.facts = dict(facts or {})


class EmptyRegion(VerifyError):
    """A region matches no mesh node or no boundary facet. `facts` holds the region, the band and the nearest distance."""

    def __init__(self, message: str, facts: dict | None = None):
        super().__init__(message)
        self.facts = dict(facts or {})


class UnsolvableMesh(UserError):
    """The mesh that the check reads cannot be solved as a loaded and supported part (exit code 1, not a crash).

    `facts` holds the numbers (bodies, the region, the solver, the error text of the solver).
    """

    def __init__(self, message: str, facts: dict | None = None):
        super().__init__(message)
        self.facts = dict(facts or {})


class SingularSystem(UnsolvableMesh):
    """A direct solver stopped on a zero pivot, a matrix that is not positive definite, or a singular matrix."""
