"""Refine step: the markup page with the final part and a stress overlay.

Import is lazy, so `import topopt.refine` stays cheap.
"""


def build_refine(*args, **kwargs):
    """See `topopt.refine.build.build_refine`."""
    from .build import build_refine as _build

    return _build(*args, **kwargs)
