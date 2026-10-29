"""#437: a resource error in the mesh step stops the run; a geometry problem keeps its fallback (CONTRACTS.md section 19, "Resource errors").

A `MemoryError` or a failed thread start in a step of the SDF mesh step used to go into the fallback of that step. The run then ended with exit 0
and a part that differs from the part of the normal path. Each handler of the step now calls `reraise_if_resource` first.
The tests inject the errors into the call that each handler guards, on a small plate case (a deck on a design space).
"""
import ast
import errno
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from topopt import interfaces, post, smooth_sdf as ss  # noqa: E402
from topopt.resource_errors import MeshResourceError, cause_text, is_resource_error, reraise_if_resource  # noqa: E402

from test_sdf_shared_planes import NY, plate_case  # noqa: E402

MEMORY = MemoryError("Unable to allocate 14.7 MiB for an array with shape (642023, 3) and data type float64")
THREAD = RuntimeError("can't start new thread")
BOTH = pytest.mark.parametrize("exc", [MEMORY, THREAD], ids=["MemoryError", "no-thread"])


# -- the test --
def test_is_resource_error_kinds():
    assert is_resource_error(MemoryError())
    assert is_resource_error(MEMORY)
    assert is_resource_error(np.core._exceptions._ArrayMemoryError((3,), np.dtype(float)))   # the subclass of MemoryError that numpy raises
    assert is_resource_error(THREAD)
    assert is_resource_error(OSError(errno.ENOMEM, "Cannot allocate memory"))
    assert is_resource_error(OSError(errno.EAGAIN, "Resource temporarily unavailable"))
    assert is_resource_error(RuntimeError("std::bad_alloc"))
    assert is_resource_error(OSError("Resource temporarily unavailable"))   # no errno, only the text
    assert is_resource_error(MeshResourceError("a step", "MemoryError: x"))
    assert is_resource_error(OSError(errno.ENOMEM, "x"))      # the errno alone, with a text that is no resource text
    assert is_resource_error(OSError(errno.EAGAIN, "x"))


class RTreeError(Exception):
    """A stand-in with the name and the base of `rtree.exceptions.RTreeError`: a plain Exception."""


def test_a_library_class_with_a_resource_text_is_a_resource_error():
    exc = RTreeError('Error in "Index_InsertData": std::bad_alloc')   # the text that rtree raises when its C++ code runs out of memory
    assert is_resource_error(exc)
    with pytest.raises(MeshResourceError) as got:
        reraise_if_resource(exc, "the snap to the region faces")
    assert "RTreeError" in str(got.value) and "std::bad_alloc" in str(got.value)


def test_the_real_rtree_error_is_a_resource_error():
    rtree = pytest.importorskip("rtree")
    assert is_resource_error(rtree.exceptions.RTreeError('Error in "Index_InsertData": std::bad_alloc'))
    assert not is_resource_error(rtree.exceptions.RTreeError("a bad geometry"))


def test_the_message_names_the_first_resource_error_of_the_chain():
    try:
        try:
            raise MemoryError("Unable to allocate 1 MiB")
        except MemoryError as inner:
            raise ValueError("the boolean could not be made") from inner
    except ValueError as outer:
        assert cause_text(outer) == "MemoryError: Unable to allocate 1 MiB"
        with pytest.raises(MeshResourceError) as got:
            reraise_if_resource(outer, "the exact boolean step")
    assert "(MemoryError: Unable to allocate 1 MiB)" in str(got.value) and "the boolean could not be made" not in str(got.value)
    assert cause_text(got.value) == "MemoryError: Unable to allocate 1 MiB"


def test_is_resource_error_is_false_for_other_errors():
    assert not is_resource_error(None)
    assert not is_resource_error(ValueError("not a closed mesh"))
    assert not is_resource_error(OSError(errno.ENOENT, "No such file or directory"))
    assert not is_resource_error(RuntimeError("the boolean failed"))
    assert not is_resource_error(OSError(errno.EPERM, "Operation not permitted"))
    assert not is_resource_error(RuntimeError("a bad geometry"))


def test_is_resource_error_walks_the_chain():
    try:
        try:
            raise MemoryError("x")
        except MemoryError as inner:
            raise ValueError("wrapped") from inner      # __cause__
    except ValueError as caught:
        assert is_resource_error(caught)
    try:
        try:
            raise RuntimeError("can't start new thread")
        except RuntimeError:
            raise KeyError("in the handler")            # __context__
    except KeyError as caught:
        assert is_resource_error(caught)
    a, b = ValueError("a"), ValueError("b")             # a loop in the chain ends
    a.__cause__, b.__cause__ = b, a
    assert not is_resource_error(a)
    deep = ValueError("0")
    for i in range(1, 3000):                            # no count limit
        nxt = ValueError(str(i))
        nxt.__cause__ = deep
        deep = nxt
    root = deep
    while root.__cause__ is not None:
        root = root.__cause__
    root.__cause__ = MemoryError()
    assert is_resource_error(deep)


def test_reraise_if_resource():
    reraise_if_resource(ValueError("x"), "a step")                  # returns
    with pytest.raises(MeshResourceError) as got:
        try:
            raise MEMORY
        except MemoryError as exc:
            reraise_if_resource(exc, "the snap to the region faces")
    err = got.value
    assert isinstance(err, RuntimeError) and err.__cause__ is MEMORY and err.step == "the snap to the region faces"
    text = str(err)
    assert "the snap to the region faces" in text and "Unable to allocate 14.7 MiB" in text and "MemoryError" in text
    assert "No part was written" in text and "different part" in text
    with pytest.raises(MeshResourceError) as again:                 # an error from an inner step passes on with the inner name
        reraise_if_resource(err, "the volume change of the snap")
    assert again.value is err and again.value.step == "the snap to the region faces"


# -- each handler of smooth_sdf.make_mesh_sdf --
def boss():
    b = trimesh.creation.cylinder(radius=1.5, height=4.0, sections=32)
    b.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    b.apply_translation([16.0, 5.0, 5.0])    # inside the design space: no plane is crossed
    return b


def caller_of(depth=2):
    return sys._getframe(depth).f_code.co_name


class Hit:
    """Raise `exc` at the call that a handler guards. `how` is a name in smooth_sdf, replaced by a function that raises (when `when` is true)."""

    def __init__(self, monkeypatch, name, exc, when=None):
        self.calls, self.name, self.exc, self.when = 0, name, exc, when
        real = getattr(ss, name)

        def wrapper(*a, **k):
            self.calls += 1
            if self.when is None or self.when(self, sys._getframe(1)):
                raise self.exc
            return real(*a, **k)
        monkeypatch.setattr(ss, name, wrapper)


class FromFn(Hit):
    """Raise at the `nth` (1 based; None: every) call made from the function `fn` of smooth_sdf."""

    def __init__(self, monkeypatch, name, exc, fn, nth=None, where=None):
        self.in_fn, self.fn, self.nth, self.where = 0, fn, nth, where
        super().__init__(monkeypatch, name, exc, self._when)

    def _when(self, hit, frame):
        if frame.f_code.co_name != self.fn or (self.where and not self.where(frame)):
            return False
        self.in_fn += 1
        return self.nth is None or self.in_fn == self.nth


def _inject_replace(name):
    return lambda mp, exc: Hit(mp, name, exc)


def _plain_case(tmp_path):
    return plate_case(tmp_path), {}


def _fill_case(tmp_path):
    return plate_case(tmp_path), {"params": ss.SdfParams(envelope_fill="on")}


def _interface_case(tmp_path):
    return plate_case(tmp_path), {"extra_frozen": [boss()]}


HANDLERS = [
    # id, step, injection, case builder, the fact that a geometry error leaves
    ("simplify", "the float32 clean-up (simplify)", lambda mp, e: FromFn(mp, "_manifold_of", e, "_cleanup_float32"), _plain_case,
     lambda i: "error" in i["volume_steps_mm3"]["cleanup"] and "not a closed manifold" in i["exact"]["cleanup_error"]),
    ("envelope", "the envelope of the design space", _inject_replace("make_envelope"), _plain_case,
     lambda i: "the envelope was not made" in i["envelope_cut"]["reason"]),
    ("fill-growth", "the growth to the design space surface", _inject_replace("grow_to_surface"), _fill_case,
     lambda i: "the growth to the design space surface failed" in i["envelope_fill"]["reason"]),
    ("field-limit", "the limit of the field to the envelope", _inject_replace("envelope_sdf"), _plain_case,
     lambda i: "the field was not limited to the envelope" in i["envelope_cut"]["field_limit_note"]),
    ("region-surfaces", "the exact surfaces of the regions", lambda mp, e: FromFn(mp, "_load_manifold", e, "union_of"), _plain_case,
     lambda i: "the exact surfaces of the regions were not made" in i["snap"]["error"]),
    ("shared-planes", "the planes shared with the regions", _inject_replace("shared_planes"), _plain_case,
     lambda i: "were not made" in i["plane_clip"]["note"]),
    ("flush", "the move of the mesh onto the shared planes", _inject_replace("flush_to_planes"), _plain_case,
     lambda i: "error" in i["envelope_cut"]["flush"]),
    ("snap", "the snap to the region faces", _inject_replace("snap_into_regions"), _plain_case,
     lambda i: "error" in i["snap"]),
    ("plane-cut", "the cut at the shared planes", _inject_replace("clip_to_planes"), _plain_case,
     lambda i: "error" in i["plane_clip"]),
    ("envelope-cut", "the cut to the envelope", _inject_replace("cut_to_envelope"), _plain_case,
     lambda i: "error" in i["envelope_cut"]),
    ("exact-boolean", "the exact boolean step", lambda mp, e: FromFn(mp, "_load_manifold", e, "build"), _plain_case,
     lambda i: bool(i["exact"].get("error"))),
    ("interface-volume", "the volume with the interface shapes",
     lambda mp, e: FromFn(mp, "_manifold_of", e, "build", where=lambda fr: "body" in fr.f_locals), _interface_case,
     lambda i: "with_interfaces" not in i["volume_steps_mm3"] and "ValueError" in i["exact"]["interface_volume_error"]),
    ("snap-volume-change", "the volume change of the snap",
     lambda mp, e: FromFn(mp, "_drop_voids", e, "build", where=lambda fr: fr.f_locals.get("snap") is False), _plain_case,
     lambda i: "volume_change_error" in i["snap"]),
]


def run_case(tmp_path, case, level=logging.WARNING):
    (args, kwargs, _planes), extra = case(tmp_path)
    logging.disable(level)
    try:
        return ss.make_mesh_sdf(*args, **kwargs, **extra)
    finally:
        logging.disable(logging.NOTSET)


@pytest.mark.parametrize("name,step,inject,case,fact", HANDLERS, ids=[h[0] for h in HANDLERS])
@BOTH
def test_a_resource_error_in_a_step_stops_the_mesh_step(tmp_path, monkeypatch, name, step, inject, case, fact, exc):
    hit = inject(monkeypatch, exc)
    with pytest.raises(MeshResourceError) as got:
        run_case(tmp_path, case)
    assert hit.calls > 0
    assert got.value.step == step and step in str(got.value)       # the step of the handler, not of an outer one
    assert got.value.__cause__ is exc and str(exc) in str(got.value)


@pytest.mark.parametrize("name,step,inject,case,fact", HANDLERS, ids=[h[0] for h in HANDLERS])
def test_a_geometry_error_keeps_the_fallback_and_the_fact(tmp_path, monkeypatch, name, step, inject, case, fact):
    hit = inject(monkeypatch, ValueError("the mesh is not a closed manifold"))
    v, f, info = run_case(tmp_path, case)
    assert hit.calls > 0
    assert len(v) > 0 and len(f) > 0
    assert fact(info), info


# -- the other callers: the keep-out cut, the interface step --
def test_a_resource_error_in_the_keep_out_cut_stops_the_step(tmp_path, monkeypatch):
    cyl = trimesh.creation.cylinder(radius=2.0, height=10, sections=16)
    cyl.export(tmp_path / "hole.stl")
    verts, faces = np.asarray(cyl.vertices, float), np.asarray(cyl.faces, np.int64)
    box = trimesh.creation.box(extents=(20, 20, 20))
    box_v, box_f = np.asarray(box.vertices, float), np.asarray(box.faces, np.int64)
    monkeypatch.setattr(post, "_to_manifold", lambda *a, **k: (_ for _ in ()).throw(ValueError("not a manifold")))
    out = post.subtract_keep_out(box_v, box_f, [tmp_path / "hole.stl"])   # a geometry error: the mesh is kept as it is
    assert out[0] is box_v and not out[2].get("keep_out_cut") and "ValueError" in out[2]["keep_out_error"]   # and the fact says why
    monkeypatch.setattr(post, "_to_manifold", lambda *a, **k: (_ for _ in ()).throw(MEMORY))
    with pytest.raises(MeshResourceError) as got:
        post.subtract_keep_out(box_v, box_f, [tmp_path / "hole.stl"])
    assert got.value.step == "the cut of the keep-out regions"


def test_a_resource_error_in_the_interface_step_stops_the_step(monkeypatch):
    class Ctx:
        has_stock = False
        keep_out_missing = ()

        def steps(self, kind):
            return []

    def boom(*a, **k):
        raise MEMORY

    monkeypatch.setattr(ss, "_manifold_of", boom)
    box = trimesh.creation.box(extents=(2, 2, 2))
    with pytest.raises(MeshResourceError) as got:
        interfaces.apply_step(Ctx(), np.asarray(box.vertices, float), np.asarray(box.faces, np.int64))
    assert got.value.step == "the exact interface shapes (union and cut)"
    monkeypatch.setattr(ss, "_manifold_of", lambda *a, **k: (_ for _ in ()).throw(ValueError("not a manifold")))
    v, f, res = interfaces.apply_step(Ctx(), np.asarray(box.vertices, float), np.asarray(box.faces, np.int64))
    assert res.applied is False and "ValueError" in res.error      # the geometry fallback stays


# -- the guard: a new handler cannot forget the call --
# The handlers that change the written part or a file or block that the run promises. Each calls `reraise_if_resource` as its first statement.
# file, top-level function, number of handlers.
IN_SCOPE = {
    ("smooth_sdf.py", "make_mesh_sdf"): 12,        # the steps of the mesh step: envelope, fill growth, field limit, region surfaces, planes, flush, snap, cuts, booleans, volumes
    ("smooth_sdf.py", "_cleanup_float32"): 1,      # simplify; also called by the interface step and by part_only
    ("post.py", "subtract_keep_out"): 1,           # the keep-out cut of the blur path
    ("post.py", "_mesh_method"): 1,                # SDF -> blur: a resource error stops the step, any other error goes to blur (section 19)
    ("post.py", "mesh_step"): 1,                   # the rest of the step (the blur path has no inner handler): a clear message, not a bare MemoryError
    ("interfaces.py", "apply_step"): 1,            # the exact interface shapes of final.stl
    # the steps after final.stl is written: a resource error stops the post step with PostResourceError (#447, #449)
    ("part_only.py", "make"): 1,                   # result/part_only.stl
    ("stress_report.py", "stress_section"): 1,     # the stress block of metrics.json
    ("stress_report.py", "_stress_section"): 1,    # the thresholded stress estimate of the stress block
}
# The handlers that do not change the written part: (number of handlers, reason). A handler that is not here and not above fails the test,
# and so does a new handler in one of these functions (the number changes).
OUT_OF_SCOPE = {
    ("post.py", "_verdict"): (1, "the verdict is a fact; it changes no part"),
    ("post.py", "_mesh_record"): (2, "the region checks and the label checks are facts about the part (2 handlers)"),
    ("post.py", "resume_route"): (1, "builds the text of the route; any problem reading the folder only changes the text"),
    ("post.py", "_load_paths"): (1, "the load path facts are facts about the part"),
    ("post.py", "finish"): (1, "the outer handler: it writes the status `error` and raises again"),
    ("interfaces.py", "render_shapes"): (1, "draws the cut shapes for the preview: no written part"),
    ("smooth_sdf.py", "physical_ram_bytes"): (1, "reads the RAM size for the guard text: no part"),
    ("interfaces.py", "mesh_body"): (1, "returns None for a mesh that is not a closed manifold; the callers measure facts"),
    ("interfaces.py", "integrity"): (1, "the integrity check is a fact about the part"),
    ("part_only.py", "cut"): (2, "the contact facts (2 handlers): with `part_only_min_film_mm` set, they raise (and `make` stops on a resource "
                                 "error); else the file does not depend on them"),
    ("stress_report.py", "read_verify"): (1, "reads a number of verify/verify.json; a read error means no number"),
    ("stress_report.py", "read_verified_file"): (1, "reads a name of verify/verify.json; a read error means no name"),
    ("stress_report.py", "read_verify_coverage"): (1, "reads two keys of verify/verify.json; a read error means no keys"),
    ("stress_report.py", "_stl_connectivity"): (1, "the connectivity of final.stl is a fact about the part"),
    ("stress_report.py", "load_path_facts"): (1, "the load path facts are facts about the part (as `post._load_paths`)"),
    ("stress_report.py", "refresh_from_verify"): (1, "a reader that changes only texts of an existing stress block"),
}


def _handlers(path):
    """`{(file, top-level function): [handler, ...]}` for each handler that can catch a resource error (Exception, BaseException, bare)."""
    tree = ast.parse(Path(path).read_text())
    found = {}

    def catches_all(h):
        t = h.type
        names = [] if t is None else [e.id for e in (t.elts if isinstance(t, ast.Tuple) else [t]) if isinstance(e, ast.Name)]
        return t is None or any(n in ("Exception", "BaseException", "MemoryError", "RuntimeError", "OSError") for n in names)

    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(top):
                if isinstance(node, ast.ExceptHandler) and catches_all(node):
                    found.setdefault((Path(path).name, top.name), []).append(node)
                if isinstance(node, ast.With):   # contextlib.suppress(Exception)
                    for item in node.items:
                        call = item.context_expr
                        if isinstance(call, ast.Call) and getattr(call.func, "attr", getattr(call.func, "id", "")) == "suppress":
                            found.setdefault((Path(path).name, top.name), []).append(node)
    return found


def _calls(node, name):
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == name for n in ast.walk(node))


def _all_handlers():
    out = {}
    for f in ("smooth_sdf.py", "post.py", "interfaces.py", "part_only.py", "stress_report.py"):
        out.update(_handlers(ROOT / "topopt" / f))
    return out


def test_every_handler_of_the_mesh_step_calls_reraise_if_resource_first():
    found = _all_handlers()
    for key, expected in IN_SCOPE.items():
        hs = found.get(key, [])
        assert len(hs) == expected, f"{key}: {len(hs)} handlers, the table says {expected}"
        for h in hs:
            first = h.body[0]
            assert isinstance(h, ast.ExceptHandler) and h.name, f"{key} line {h.lineno}: the handler must name the exception"
            assert (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
                    and getattr(first.value.func, "id", None) == "reraise_if_resource"), f"{key} line {h.lineno}: the first statement is not reraise_if_resource"
            args = first.value.args
            assert args and isinstance(args[0], ast.Name) and args[0].id == h.name, f"{key} line {h.lineno}: the call must get the caught name `{h.name}`"


def test_no_handler_in_the_files_of_the_step_is_unlisted():
    known = set(IN_SCOPE) | set(OUT_OF_SCOPE)
    unlisted = sorted(set(_all_handlers()) - known)
    assert not unlisted, (f"handlers in {unlisted} catch a resource error. Call reraise_if_resource first and add the function to IN_SCOPE, "
                          "or add it to OUT_OF_SCOPE with the reason")
    found = _all_handlers()
    assert set(OUT_OF_SCOPE) <= set(found), "a listed out-of-scope function has no such handler any more: remove it from the table"
    for key, (count, _reason) in OUT_OF_SCOPE.items():
        assert len(found[key]) == count, f"{key}: {len(found[key])} handlers, the table says {count}: is the new one a handler of the mesh step?"


# -- the post step on a small recorded run --
def fixed_run(tmp_path, **settings):
    """The run of `test_post.make_hole_run` with an exact STL for the fixed region, so that the snap runs. Return the run folder."""
    from test_post import make_hole_run

    run, labels = make_hole_run(tmp_path)
    h, org = run["h"], np.asarray(run["origin"])
    for name, lo, hi in (("fixed", (0, 0, 0), (3, 24, 24)), ("load", (24, 10, 10), (27, 14, 14)), ("design", (0, 0, 0), (27, 24, 24))):
        trimesh.creation.box(bounds=[org + np.array(lo) * h, org + np.array(hi) * h]).export(tmp_path / f"{name}.stl")
    run["yaml_path"] = str(tmp_path / "x.yaml")
    run["regions"] = {"fixed": ["fixed.stl"], "load": {"t": "load.stl"}, "design": "design.stl"}
    run["region_files"] = {"fixed": ["fixed.stl"], "load": {"t": "load.stl"}, "design": "design.stl"}
    run.update(settings)
    (tmp_path / "run.json").write_text(json.dumps(run))
    return tmp_path


def _no_blur(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the blur path ran after a resource error")
    monkeypatch.setattr(post, "make_mesh", refuse)


@BOTH
def test_a_resource_error_in_an_inner_step_fails_the_post_step_with_no_blur(tmp_path, monkeypatch, caplog, exc):
    (tmp_path / "ok").mkdir()
    normal = post.finish(fixed_run(tmp_path / "ok"))
    assert normal["mesh"]["method"] == "sdf" and normal["mesh"]["snap"].get("moved") is not None and normal["mesh"]["fallback"] is None
    (tmp_path / "bad").mkdir()
    run_dir = fixed_run(tmp_path / "bad")
    old = b"an older final.stl of this folder"
    (run_dir / "result").mkdir()
    (run_dir / "result" / "final.stl").write_bytes(old)
    Hit(monkeypatch, "snap_into_regions", exc)
    _no_blur(monkeypatch)
    with caplog.at_level(logging.WARNING), pytest.raises(MeshResourceError) as got:
        post.finish(run_dir)
    assert got.value.step == "the snap to the region faces" and got.value.__cause__ is exc
    assert "mesh fallback" not in caplog.text and "The old Gaussian path" not in caplog.text     # #445: the text of a fallback of the mesh step
    st = json.loads((run_dir / "status.json").read_text())
    assert st["state"] == "error" and st["step"] == "postprocess"
    assert "the snap to the region faces" in st["message"] and str(exc) in st["message"] and "No part was written" in st["message"]
    assert "--resume" in st["message"] and f"--out {run_dir}" in st["message"]
    assert "options of the first run" in st["message"]      # this folder has no checkpoint: the text cannot read the options and says so
    assert (run_dir / "result" / "final.stl").read_bytes() == old and not (run_dir / "result" / "metrics.json").exists()   # nothing of this pass


@BOTH
def test_a_resource_error_outside_an_inner_handler_names_the_sdf_step(tmp_path, monkeypatch, exc):
    run_dir = fixed_run(tmp_path)
    Hit(monkeypatch, "binary_sdf", exc)
    _no_blur(monkeypatch)
    with pytest.raises(MeshResourceError) as got:
        post.finish(run_dir)
    assert got.value.step == "the SDF mesh step" and json.loads((run_dir / "status.json").read_text())["state"] == "error"


@pytest.mark.parametrize("exc", [ValueError("not a closed mesh"), RuntimeError("the field cannot be made")], ids=["ValueError", "RuntimeError"])
def test_an_error_of_the_sdf_path_that_is_no_resource_error_still_goes_to_blur(tmp_path, monkeypatch, exc):
    run_dir = fixed_run(tmp_path)
    Hit(monkeypatch, "binary_sdf", exc)
    m = post.finish(run_dir)
    mesh = m["mesh"]
    assert mesh["method"] == "blur" and mesh["requested"] == "sdf"
    assert mesh["fallback"] == {"error": f"{type(exc).__name__}: {exc}"}      # the old fact: no other key
    assert json.loads((run_dir / "status.json").read_text())["state"] == "done"


def test_a_geometry_error_in_the_snap_keeps_its_fact(tmp_path, monkeypatch):
    run_dir = fixed_run(tmp_path)
    Hit(monkeypatch, "snap_into_regions", ValueError("not a closed mesh"))
    m = post.finish(run_dir)
    assert m["mesh"]["method"] == "sdf" and m["mesh"]["fallback"] is None and "error" in m["mesh"]["snap"]   # as before: the fact records it


def test_a_resource_error_in_the_blur_path_fails_the_post_step(tmp_path, monkeypatch):
    run_dir = fixed_run(tmp_path, mesh_method="blur")
    cyl = trimesh.creation.cylinder(radius=2.0, height=10, sections=16)
    cyl.export(tmp_path / "hole.stl")
    run = json.loads((run_dir / "run.json").read_text())
    run["regions"]["keep_out"] = ["hole.stl"]
    run["region_files"]["keep_out"] = ["hole.stl"]
    (run_dir / "run.json").write_text(json.dumps(run))
    monkeypatch.setattr(post, "_to_manifold", lambda *a, **k: (_ for _ in ()).throw(MEMORY))
    with pytest.raises(MeshResourceError):
        post.finish(run_dir)
    st = json.loads((run_dir / "status.json").read_text())
    assert st["state"] == "error" and "different part" in st["message"]
    assert not (run_dir / "result" / "final.stl").exists()


def test_a_failed_post_step_of_a_continued_run_is_a_failed_run(tmp_path, monkeypatch):
    """A post step that repeats in a folder with an older result: the resource error ends the step with the status `error`. The older files stay
    (the step writes `final.stl` only after the mesh), but `run_status` and `run_result` report the failure, not the older result."""
    from topopt_mcp import server

    run_dir = fixed_run(tmp_path)
    post.finish(run_dir)
    old = (run_dir / "result" / "final.stl").read_bytes()
    run = json.loads((run_dir / "run.json").read_text())
    run["mesh_method"] = "blur"
    (run_dir / "run.json").write_text(json.dumps(run))

    def boom(*a, **k):
        raise MEMORY
    monkeypatch.setattr(post, "make_mesh", boom)
    with pytest.raises(MeshResourceError) as got:
        post.finish(run_dir)
    assert got.value.step == "the blur mesh method" and "No part was written" in str(got.value)
    st = json.loads((run_dir / "status.json").read_text())
    assert st["state"] == "error" and "Unable to allocate 14.7 MiB" in st["message"]
    assert (run_dir / "result" / "final.stl").read_bytes() == old     # the older file is not touched
    assert server.run_status(str(run_dir))["state"] == "error"
    res = server.run_result(str(run_dir), preview=False)
    assert res["done"] is False and "post-process failed" in res["message"] and "final_stl" not in res


def test_the_exit_code_of_a_resource_error_is_not_zero():
    from topopt.runjob import exit_code_of

    assert exit_code_of(MeshResourceError("a step", "MemoryError: x")) == 2


def test_the_blur_record_names_a_failed_keep_out_cut(tmp_path, monkeypatch):
    run_dir = fixed_run(tmp_path, mesh_method="blur")
    trimesh.creation.cylinder(radius=2.0, height=10, sections=16).export(tmp_path / "hole.stl")
    run = json.loads((run_dir / "run.json").read_text())
    run["regions"]["keep_out"] = ["hole.stl"]
    run["region_files"]["keep_out"] = ["hole.stl"]
    (run_dir / "run.json").write_text(json.dumps(run))
    monkeypatch.setattr(post, "_to_manifold", lambda *a, **k: (_ for _ in ()).throw(ValueError("not a manifold")))
    m = post.finish(run_dir)
    assert m["mesh"]["keep_out_cut"] is False and "ValueError" in m["mesh"]["keep_out_error"]


def test_topopt_run_ends_with_exit_2_and_the_status_names_the_step(tmp_path, monkeypatch):
    """The CLI path: a resource error in the SDF step of the post step. `main` returns 2, `status.json` is `error`, no `final.stl`."""
    from topopt.cli import main
    from test_optimize import make_case

    yaml_path = make_case(tmp_path, h=2.0, iterations=3)
    out = tmp_path / "run"
    Hit(monkeypatch, "make_mesh_sdf", MEMORY)
    _no_blur(monkeypatch)
    assert main(["run", str(yaml_path), "--out", str(out), "--quiet"]) == 2
    st = json.loads((out / "status.json").read_text())
    assert st["state"] == "error"
    assert "the SDF mesh step" in st["message"] and "Unable to allocate 14.7 MiB" in st["message"] and "--resume" in st["message"]
    assert not (out / "result" / "final.stl").exists()


# -- N2: a __str__ that raises --
class BadStr(Exception):
    def __str__(self):
        raise ValueError("no text")


def test_an_exception_whose_str_raises_is_no_resource_error_and_does_not_hide_a_cause():
    assert not is_resource_error(BadStr())
    try:
        try:
            raise MemoryError("x")
        except MemoryError as inner:
            raise BadStr() from inner
    except BadStr as outer:
        assert is_resource_error(outer)       # the cause is found, the broken text is skipped
        assert cause_text(outer) == "MemoryError: x"


# -- N3: the names of the steps outside the SDF path --
def _mesh_step(run, monkeypatch=None):
    labels = np.full((4, 4, 4), post.DESIGN, np.uint8)
    return post.mesh_step(run, np.ones((4, 4, 4), np.float32), labels, np.zeros(3), 1.0)


def test_the_blur_path_has_its_own_step_name(monkeypatch):
    monkeypatch.setattr(post, "make_mesh", lambda *a, **k: (_ for _ in ()).throw(MEMORY))
    with pytest.raises(MeshResourceError) as got:
        _mesh_step({"mesh_method": "blur"})
    assert got.value.step == "the blur mesh method" and "the blur mesh method" in str(got.value)


def test_the_set_up_of_the_interface_shapes_has_a_step_name(monkeypatch):
    monkeypatch.setattr(interfaces, "from_run", lambda *a, **k: (_ for _ in ()).throw(THREAD))
    with pytest.raises(MeshResourceError) as got:
        _mesh_step({"interfaces": [{"name": "x"}], "mesh_method": "blur"})
    assert got.value.step == "the set-up of the interface shapes"


def test_the_exact_interface_step_has_a_step_name(monkeypatch):
    monkeypatch.setattr(interfaces, "from_run", lambda *a, **k: object())
    monkeypatch.setattr(post, "_mesh_method", lambda *a, **k: (np.zeros((0, 3)), np.zeros((0, 3), np.int64), {}, {}))
    monkeypatch.setattr(interfaces, "apply_step", lambda *a, **k: (_ for _ in ()).throw(MEMORY))
    with pytest.raises(MeshResourceError) as got:
        _mesh_step({"interfaces": [{"name": "x"}]})
    assert got.value.step == "the exact interface step"


# -- M1: the route passes the resume check --
def _failed_first_run(tmp_path, monkeypatch, *flags):
    """`topopt run` of the small cantilever with the given run options; the SDF step fails with a MemoryError. Return (yaml, out, status message)."""
    from topopt.cli import main
    from test_optimize import make_case

    yaml_path = make_case(tmp_path, h=2.0, iterations=3)
    out = tmp_path / "run"
    with monkeypatch.context() as mp:
        mp.setattr(ss, "make_mesh_sdf", lambda *a, **k: (_ for _ in ()).throw(MEMORY))
        mp.setattr(post, "make_mesh", lambda *a, **k: (_ for _ in ()).throw(MEMORY))   # for a run with --mesh-method blur
        assert main(["run", str(yaml_path), "--out", str(out), "--quiet", "--no-wait", *flags]) == 2
    return yaml_path, out, json.loads((out / "status.json").read_text())["message"]


def _route_args(message):
    import shlex

    route = message.split("To repeat only the post step on this run: `")[1].split("`")[0]
    return shlex.split(route)[1:] + ["--quiet", "--no-wait"]


def test_the_route_of_a_blur_run_has_the_option_and_passes_the_resume_check(tmp_path, monkeypatch):
    from topopt.cli import main

    _yaml, out, message = _failed_first_run(tmp_path, monkeypatch, "--mesh-method", "blur")
    assert "--mesh-method blur" in message and "run_topopt has no argument" in message and "MCP: run_topopt with resume=true" not in message
    assert main(_route_args(message)) == 0
    assert json.loads((out / "result" / "metrics.json").read_text())["mesh"]["method"] == "blur"
    assert json.loads((out / "status.json").read_text())["state"] == "done"


def test_the_route_of_a_run_with_the_envelope_fill_option_has_it(tmp_path, monkeypatch):
    from topopt.cli import main

    _yaml, out, message = _failed_first_run(tmp_path, monkeypatch, "--envelope-fill", "on")
    assert "--envelope-fill on" in message and "--mesh-method" not in message
    assert main(_route_args(message)) == 0
    assert json.loads((out / "run.json").read_text())["envelope_fill"] == "on"


def test_the_route_of_a_run_without_run_options_names_the_mcp_tool(tmp_path, monkeypatch):
    from topopt.cli import main

    _yaml, out, message = _failed_first_run(tmp_path, monkeypatch)
    assert "--mesh-method" not in message and "--envelope-fill" not in message and "--solver" not in message
    assert "(MCP: run_topopt with resume=true.)" in message
    assert main(_route_args(message)) == 0
