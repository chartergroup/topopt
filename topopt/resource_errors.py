"""Resource errors in the post step (CONTRACTS.md section 19, "Resource errors").

A fallback of the mesh step is meant for a geometry problem (a mesh that is not closed, a boolean that fails). It gives a part that differs
from the part of the normal path. A memory error or a failed thread start is not a geometry problem: the same input gives the normal part
when the resource is there. So a handler of the mesh step calls `reraise_if_resource` as its first statement, and the step stops with
`MeshResourceError`, which names the cause and the step.

The steps after `result/final.stl` is written (the part-only cut, the stress report) give a file or a block that the run promises. A resource
error there must not leave the run at exit 0 without it, so their handlers call `reraise_if_resource(exc, step, missing=...)` first, and the
post step stops with `PostResourceError`, which names the step, the cause and what this pass did not write (#447, #449).

`is_resource_error` is the one test. It looks at the type, at `errno`, at the text, and at the `__cause__` / `__context__` chain.
"""
from __future__ import annotations

import errno
from collections import deque

# Texts of a failed allocation or a failed thread start, in lower case. Sources: CPython `threading.Thread.start` ("can't start new thread"),
# the C++ runtime of a native library ("bad_alloc"), the C library (strerror of EAGAIN and ENOMEM; "failed to map segment from shared object":
# the loader could not map a native library, for example libllvmlite of numba under an address-space cap, #449).
RESOURCE_TEXTS = ("can't start new thread", "bad_alloc", "resource temporarily unavailable", "cannot allocate memory",
                  "failed to map segment from shared object")
RESOURCE_ERRNOS = (errno.ENOMEM, errno.EAGAIN)


OPTIONS = "Options: run one heavy job at a time, or give the run more memory; or use a coarser voxel size in a new run."


class ResourceStopError(RuntimeError):
    """A resource error that stops the post step. `step` names the step, `cause` the original error as text.

    The post step adds `route` (how to repeat only the post step on the run folder) when it knows the run folder; `str()` then ends with it.
    """

    route: str | None = None

    def __init__(self, step: str, cause: str, text: str):
        self.step, self.cause = step, cause
        super().__init__(text)

    def __str__(self) -> str:
        return super().__str__() + (f" {self.route}" if self.route else "")


class MeshResourceError(ResourceStopError):
    """A memory error or a failed thread start in the mesh step: no part was written."""

    def __init__(self, step: str, cause: str):
        super().__init__(step, cause, f"the mesh step ran out of a resource in {step} ({cause}). No part was written, because a fallback here "
                                      f"would give a different part. {OPTIONS}")


class PostResourceError(ResourceStopError):
    """A memory error or a failed thread start in a step after `result/final.stl` is written. `missing` names what this pass did not write."""

    def __init__(self, step: str, cause: str, missing: str):
        self.missing = missing
        super().__init__(step, cause, f"the post step ran out of a resource in {step} ({cause}). result/final.stl is written, but {missing} "
                                      f"is not, so the run stops here and does not end without it. {OPTIONS}")


def _one_is_resource(exc: BaseException) -> bool:
    if isinstance(exc, (MemoryError, ResourceStopError)):   # MemoryError has subclasses, for example the array memory error of numpy
        return True
    if isinstance(exc, OSError) and exc.errno in RESOURCE_ERRNOS:
        return True
    try:
        text = str(exc).lower()   # every type: a C++ library can raise its own class with the text (rtree: RTreeError 'std::bad_alloc')
    except Exception:   # a `__str__` that raises: no text to test
        return False
    return any(t in text for t in RESOURCE_TEXTS)


def first_resource_error(exc: BaseException | None) -> BaseException | None:
    """The first resource error of `exc` and its `__cause__` / `__context__` chain (`exc` first, then the causes), or None."""
    seen: set[int] = set()
    todo = deque([exc])
    while todo:
        e = todo.popleft()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        if _one_is_resource(e):
            return e
        todo.extend((e.__cause__, e.__context__))
    return None


def is_resource_error(exc: BaseException | None) -> bool:
    """True when `exc`, or an exception in its `__cause__` / `__context__` chain, is a memory error or a failed thread start."""
    return first_resource_error(exc) is not None


def cause_text(exc: BaseException) -> str:
    """The first resource error of the chain as `Type: text`. For a `ResourceStopError`, the cause that it holds."""
    found = first_resource_error(exc) or exc
    if isinstance(found, ResourceStopError):
        return found.cause
    return f"{type(found).__name__}: {found}"


def reraise_if_resource(exc: BaseException, step: str, missing: str | None = None) -> None:
    """Raise a `ResourceStopError` from `exc` when it is a resource error; return otherwise. Call it first in a handler of the post step.

    Without `missing` (the mesh step) the error is a `MeshResourceError`. With `missing` (a step after `result/final.stl` is written: what this
    pass then does not write) it is a `PostResourceError`. A `ResourceStopError` from a step inside passes on with the name of the inner step;
    a `MeshResourceError` from a mesh helper that a later step calls (the clean-up of the part-only cut) becomes a `PostResourceError` that names
    both steps, because the mesh step itself had finished.
    """
    if isinstance(exc, MeshResourceError) and missing is not None:
        raise PostResourceError(f"{exc.step}, in {step}", exc.cause, missing) from exc
    if isinstance(exc, ResourceStopError):
        raise exc
    if is_resource_error(exc):
        if missing is None:
            raise MeshResourceError(step, cause_text(exc)) from exc
        raise PostResourceError(step, cause_text(exc), missing) from exc
