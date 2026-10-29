"""Displacement and alignment limits as constraint rows (A5, CONTRACTS.md section 31).

Two kinds of row, both linear in the displacement `u` of the free dofs:

  displacement   the mean displacement of the nodes of a region, along a direction or as the
                 norm of the mean vector, in mm.
  alignment      the relative shift (mm) and the relative tilt (rad) of two regions. Each region
                 gets a least-squares rigid fit (translation `t` and small rotation `theta` about
                 the node mean). A rigid motion of the whole body gives zero shift and zero tilt.

A region is a `regions.measure` entry (a probe STL: a node set, no label, no force) or a load
region. The node set is the nodes of all voxels of the region (the 8-corner rule of
`loads.build_forces`). The nodes that touch no active voxel have no dof and are left out.

Each row is `y = G u` with a sparse `G`. The row value is `|y|` (the smooth norm, below) or
the signed `y`. The gradient is one adjoint solve per row and case: `K lam = G^T w`, with `w`
the slope of the value with respect to `y`, and `d value / d rho = - lam^T dK/d rho u`
(`DesignPoint.solve_adjoint` and `stiffness_derivative`; the factor or hierarchy of the forward
solve is reused). A row has no explicit density term.

Smooth norm. The row value is `sqrt(|y|^2 + eps^2)` with `eps = EPS_FRACTION * limit`. The
plain norm has no slope at `y = 0` (KB pitfall `norm-constraint-nonsmooth-at-zero`). The guard
changes the value by less than `eps^2 / (2 |y|)`: `eps` at `y = 0`, and 5e-5 of the limit
at `|y| = limit`.

Row form: `g = value / limit - 1 <= 0` (linear form, price `MOTION_PRICE`). A row that cannot
be met is a fact in the run log (`Formulation.unmet`), not an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .config import UserError
from .formulation import NOT_MET, Constraint, Response
from .loads import node_ids

# Row name prefix -> unit. The prefix tells the form of the value: displacement, shift and tilt are
# the smooth norm of `y`; displacement_max is `y` and displacement_min is `-y` (signed, no guard).
# The row "perimeter" (A4c, topopt/perimeter.py) is in the list for the unit only: `physical_value` returns its value as it is.
ROW_UNITS = {"displacement": "mm", "displacement_max": "mm", "displacement_min": "mm",
             "shift": "mm", "tilt": "rad", "perimeter": "mm2"}
EPS_FRACTION = 1e-2     # eps of the smooth norm, as a fraction of the limit
MOTION_PRICE = 1000.0   # price of a violation in MMA
GRAD_WRT = "rho_stiff"  # the gradient is with respect to the density of the stiffness (the eroded one in robust mode)
RANK_TOL = 1e-9         # a fit matrix with an eigenvalue below RANK_TOL times the largest one has lost a rank
HINT = "options: raise settings.volume_fraction, relax the limit, or move the region onto the load path"
# With the mass objective (A2) a row on a probe region does not hold material, so the volume option does not apply.
MASS_HINT = "options: relax the limit, move the region onto the load path, or add a compliance_limit or a stress limit"
# The A1 cause text ("... this limit does not fit at this volume") is not established for these rows.
CAUSE = "the other rows are met, and this limit is not met at the end of the run"

VOID, FIXED_DOF = -2, -1   # entries of the dof map: no dof, or a fixed dof. A free dof has its index (>= 0).


# --- nodes of a region ---------------------------------------------------


def dof_map(shape, free_dofs, fixed_dofs) -> np.ndarray:
    """For each dof of the grid: its place among the free dofs, FIXED_DOF, or VOID (no element)."""
    n_dof = 3 * int(np.prod(np.asarray(shape, dtype=np.int64) + 1))
    pos = np.full(n_dof, VOID, dtype=np.int64)
    pos[np.asarray(fixed_dofs, dtype=np.int64)] = FIXED_DOF
    pos[np.asarray(free_dofs, dtype=np.int64)] = np.arange(len(free_dofs))
    return pos


@dataclass
class RegionNodes:
    """The nodes of one region that carry dofs, and the facts about the rest."""
    name: str
    nodes: np.ndarray          # flat node numbers (sorted, unique)
    coords: np.ndarray         # (n, 3) positions in mm
    n_listed: int              # nodes of the voxels, before the nodes without a dof are dropped
    n_fixed: int               # nodes with all 3 dofs fixed (they count with displacement 0)

    @property
    def n_void(self) -> int:
        return self.n_listed - self.nodes.size


def region_voxels(config, problem, name) -> np.ndarray:
    """Bool mask (nx, ny, nz) of the voxels of a load region or a measure region."""
    from .voxelize import inside_mask, load_mesh   # trimesh: only when a measure STL is read
    for i, lr in enumerate(config.regions.load):
        if lr.name == name:
            return problem.load_region_ids == i
    for mr in config.regions.measure:
        if mr.name == name:
            mask = inside_mask(load_mesh(mr.path), np.asarray(problem.origin), problem.h, problem.shape)
            if not mask.any():
                raise UserError(
                    f"measure region '{name}' has no voxel inside the grid at voxel_size {problem.h}. "
                    f"Use a smaller voxel_size, make the region larger, or put it on the part")
            return mask
    raise UserError(f"region '{name}' is not a measure or a load region")


def region_nodes(config, problem, pos, name) -> RegionNodes:
    """Nodes of region `name` that have a dof, with positions. `pos` is `dof_map(...)`."""
    listed = node_ids(region_voxels(config, problem, name))
    dofs = 3 * listed[:, None] + np.arange(3)
    keep = pos[3 * listed] != VOID
    nodes = listed[keep]
    ijk = np.stack(np.unravel_index(nodes, tuple(int(s) + 1 for s in problem.shape)), axis=1)
    coords = np.asarray(problem.origin, dtype=float) + problem.h * ijk
    n_fixed = int(np.all(pos[dofs[keep]] == FIXED_DOF, axis=1).sum())
    return RegionNodes(name, nodes, coords, int(listed.size), n_fixed)


# --- rigid fit and rows --------------------------------------------------


def skew(v) -> np.ndarray:
    """Matrix S with S @ w = v x w."""
    v = np.asarray(v, dtype=float)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def fit_matrix(coords):
    """The 3 x 3 matrix `M = sum(|r|^2 I - r r^T)` of the nodes about their mean, and the rank of the fit.

    The least-squares rigid fit `u_k ~ t + theta x r_k` (`r_k` = position minus the node mean) has
    `t = mean(u)` and `M theta = sum(r_k x u_k)`. The fit matrix `[I, -skew(r_k)]` of all nodes
    has rank `3 + rank(M)`: 6 needs nodes that are not on one line. Returns `(centre, M, rank)`.
    """
    coords = np.asarray(coords, dtype=float)
    centre = coords.mean(axis=0)
    r = coords - centre
    M = (r ** 2).sum() * np.eye(3) - r.T @ r
    ev = np.linalg.eigvalsh(M)
    rank = 3 + int((ev > RANK_TOL * max(ev.max(), 0.0)).sum()) if ev.max() > 0 else 3
    return centre, M, rank


def _mean_rows(cols, n_free) -> sp.csr_matrix:
    """Sparse (3, n_free): the mean displacement vector. `cols` is (N, 3), the free index or < 0."""
    n = cols.shape[0]
    a = np.broadcast_to(np.arange(3), cols.shape)
    ok = cols >= 0
    return sp.csr_matrix((np.full(ok.sum(), 1.0 / n), (a[ok], cols[ok])), shape=(3, n_free))


def _rotation_rows(coords, centre, Minv, cols, n_free) -> sp.csr_matrix:
    """Sparse (3, n_free): theta = M^-1 sum(r_k x u_k)."""
    r = coords - centre
    S = np.zeros((len(r), 3, 3))
    S[:, 0, 1], S[:, 0, 2], S[:, 1, 0] = -r[:, 2], r[:, 1], r[:, 2]
    S[:, 1, 2], S[:, 2, 0], S[:, 2, 1] = -r[:, 0], -r[:, 1], r[:, 0]
    B = np.einsum("ij,kjl->kil", Minv, S)             # (N, 3, 3): theta_i = sum_k,a B[k,i,a] u[k,a]
    rows = np.broadcast_to(np.arange(3)[None, :, None], B.shape)
    colsb = np.broadcast_to(cols[:, None, :], B.shape)
    ok = colsb >= 0
    return sp.csr_matrix((B[ok], (rows[ok], colsb[ok])), shape=(3, n_free))


@dataclass
class RigidFit:
    """Sparse maps from the free displacements to the rigid motion of one region."""
    centre: np.ndarray    # node mean, mm
    translation: sp.csr_matrix   # (3, n_free): t, the mean displacement
    rotation: sp.csr_matrix      # (3, n_free): theta, rad
    rank: int
    n_nodes: int


def rigid_fit(reg: RegionNodes, pos, n_free, need_rotation=True) -> RigidFit:
    """Rigid-fit maps of a region. Raises nothing for a rank below 6: the caller decides."""
    cols = pos[3 * reg.nodes[:, None] + np.arange(3)]
    centre, M, rank = fit_matrix(reg.coords)
    rotation = None
    if need_rotation and rank == 6:
        rotation = _rotation_rows(reg.coords, centre, np.linalg.inv(M), cols, n_free)
    return RigidFit(centre, _mean_rows(cols, n_free), rotation, rank, int(reg.nodes.size))


def transverse_axes(axis):
    """Two unit vectors across `axis` (the same choice as the pipeline: e1 from x, or from y near x)."""
    h = np.asarray(axis, dtype=float)
    h = h / np.linalg.norm(h)
    ref = np.array([1.0, 0.0, 0.0]) if abs(h[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - h * (ref @ h)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(h, e1)


def alignment_rows(fit_a: RigidFit, fit_b: RigidFit, axis):
    """Sparse (2, n_free) maps of the shift (mm) and the tilt (rad) of region A relative to B.

    With `d = c_A - c_B` and `omega = (theta_A + theta_B) / 2`, the shift is the part across
    `axis` of `(t_A - t_B) + skew(d) omega`: the relative displacement of the two rigid motions
    at the midpoint of the two centres. The tilt is the part across `axis` of `theta_A - theta_B`.
    The twist (about `axis`) and the shift along `axis` are not limited.
    """
    d = fit_a.centre - fit_b.centre
    Sd = skew(d)
    dt, dr, sr = fit_a.translation - fit_b.translation, fit_a.rotation - fit_b.rotation, \
        fit_a.rotation + fit_b.rotation
    shift, tilt = [], []
    for e in transverse_axes(axis):
        shift.append(sp.csr_matrix(e[None, :]) @ dt + sp.csr_matrix((0.5 * (e @ Sd))[None, :]) @ sr)
        tilt.append(sp.csr_matrix(e[None, :]) @ dr)
    return sp.vstack(shift, format="csr"), sp.vstack(tilt, format="csr")


# --- the rows ------------------------------------------------------------


@dataclass
class MotionRow:
    """One constraint row: `y = G u` of one load case, and the form of its value."""
    name: str
    case: int
    G: sp.csr_matrix
    limit: float
    sense: str = "abs"      # abs: the smooth norm of y; max: y; min: -y (y has one entry)
    eps: float = 0.0

    def y(self, u) -> np.ndarray:
        return np.asarray(self.G @ u).reshape(-1)

    def value_and_slope(self, y):
        """The row value and its slope `w` with respect to `y`."""
        if self.sense == "abs":
            s = math.sqrt(float(y @ y) + self.eps ** 2)
            return s, y / s
        sign = 1.0 if self.sense == "max" else -1.0
        return sign * float(y[0]), np.array([sign])

    def measure(self, point) -> float:
        """The value of the row at `point` (the value that `evaluate` gives), with no adjoint solve."""
        return self.value_and_slope(self.y(point.U[:, self.case]))[0]

    def physical(self, U) -> float:
        """The physical value for the displacements `U` (n_free, n_cases): `|y|`, or `y` (max), or `-y` (min)."""
        y = self.y(U[:, self.case])
        return float(np.linalg.norm(y)) if self.sense == "abs" else (1.0 if self.sense == "max" else -1.0) * float(y[0])

    def evaluate(self, point) -> Response:
        u = point.U[:, self.case]
        value, w = self.value_and_slope(self.y(u))
        rhs = np.asarray(self.G.T @ w).reshape(-1)
        if not rhs.any():     # y = 0 (smooth norm), or a region with no free dof: the slope is 0
            return Response(value, np.zeros(point.n_design), GRAD_WRT)
        lam = point.solve_adjoint(rhs)
        return Response(value, -point.stiffness_derivative(lam, u), GRAD_WRT)


@dataclass(frozen=True)
class RowSpec:
    """One row of the config: its name, load case, limit, and where its operator comes from."""
    name: str
    case: str
    limit: float
    sense: str
    source: object      # the DisplacementLimit or AlignmentLimit of the config
    part: str           # displacement | shift | tilt

    @property
    def price(self) -> float:
        return self.source.price or MOTION_PRICE


def row_specs(config) -> list:
    """The rows of `config.limits` in order: displacement rows, then alignment rows (shift, then tilt)."""
    lim, out = config.limits, []
    for L in lim.displacement:
        kind = "displacement" if L.sense == "abs" else f"displacement_{L.sense}"
        out += [RowSpec(f"{kind}:{L.name}:{c}", c, L.limit_mm, L.sense, L, "displacement")
                for c in L.cases]
    for L in lim.alignment:
        for part, limit in (("shift", L.shift_limit_mm), ("tilt", L.tilt_limit_rad)):
            if limit is not None:
                out += [RowSpec(f"{part}:{L.name}:{c}", c, limit, "abs", L, part) for c in L.cases]
    return out


class MotionSet:
    """The rows of the `limits` of a config, for one model.

    `MotionSet(config).bind(problem, free_dofs)` builds the node sets and the maps, checks the
    rigid fits, and returns the facts (a list of text lines). `constraints()` makes one
    `Constraint` for each row and case. A set that is not bound yet binds to the model of the
    first `DesignPoint` it sees, and `log` (if given) gets the facts then. Names:
    `displacement:<name>:<case>`, `shift:<name>:<case>`, `tilt:<name>:<case>`.
    """

    def __init__(self, config, log=None):
        self.config = config
        self.specs = row_specs(config)
        self.rows: list[MotionRow] = []
        self.facts: list[str] = []
        self.log = log
        self._bound_to = None

    def bind_model(self, model) -> list:
        """Bind to `model`, send the facts to `log` (if any), and return them."""
        facts = self.bind(model.problem, model.free_dofs, key=model)
        if self.log is not None:
            for line in facts:
                self.log("fact: " + line)
        return facts

    def bind(self, problem, free_dofs, key=None) -> list:
        """Build the operators for the voxel grid of `problem`. Raise `UserError` for a row that cannot work. Return the facts."""
        pos = dof_map(problem.shape, free_dofs, problem.fixed_dofs)
        return self.bind_nodes(pos, len(free_dofs),
                               lambda name: region_nodes(self.config, problem, pos, name), key)

    def bind_nodes(self, pos, n_free, lookup, key=None) -> list:
        """Build the operators from a dof map and a region lookup. Raise `UserError` for a row that cannot work. Return the facts.

        This is the part of `bind` that does not depend on the element. `pos` maps each dof of the model to its place among
        the `n_free` free dofs, to `FIXED_DOF`, or to `VOID`. `lookup(name)` returns the `RegionNodes` of the region `name`
        (positions in mm). The voxel grid builds them with `region_nodes`; the P2 tet model of `topopt verify` builds
        them from its own nodes. The rows, the rigid fits, the rank checks and the facts are the same for both.
        """
        names = [c.name for c in self.config.load_cases]
        regions, fits, ops, facts = {}, {}, {}, []

        def region(name, what):
            if name not in regions:
                reg = lookup(name)
                if reg.nodes.size == 0:
                    raise UserError(
                        f"{what} cannot work: region '{name}' has no node that carries a dof "
                        f"({reg.n_listed} nodes, all of them outside the model). Options: move the "
                        f"region onto the part, or make it larger")
                regions[name] = reg
                facts.extend(region_facts(reg))
            return regions[name]

        def fit(name, what):
            if name not in fits:
                fits[name] = rigid_fit(region(name, what), pos, n_free)
            if fits[name].rank < 6:
                raise UserError(rank_message(what, name, fits[name]))
            return fits[name]

        def operator(L, part):
            if (id(L), part) not in ops:
                if part == "displacement":
                    reg = region(L.region, f"displacement row '{L.name}'")
                    T = _mean_rows(pos[3 * reg.nodes[:, None] + np.arange(3)], n_free)
                    G = sp.csr_matrix(np.asarray(L.direction)[None, :]) @ T if L.direction else T
                    if G.nnz == 0:
                        facts.append(all_fixed_fact(f"displacement row '{L.name}'", reg))
                    ops[(id(L), part)] = sp.csr_matrix(G)
                else:
                    what = f"alignment row '{L.name}'"
                    fa, fb = fit(L.region_a, what), fit(L.region_b, what)
                    shift, tilt = alignment_rows(fa, fb, L.axis)
                    if shift.nnz == 0:
                        facts.append(all_fixed_fact(what, regions[L.region_a], regions[L.region_b]))
                    ops[(id(L), "shift")], ops[(id(L), "tilt")] = shift, tilt
            return ops[(id(L), part)]

        rows = []
        for spec in self.specs:
            eps = EPS_FRACTION * spec.limit
            rows.append(MotionRow(spec.name, names.index(spec.case), operator(spec.source, spec.part),
                                  spec.limit, spec.sense, eps))
        self.rows, self.facts, self._bound_to = rows, facts, key
        return facts

    def row(self, index, model) -> MotionRow:
        if self._bound_to is not model:
            self.bind_model(model)
        return self.rows[index]

    def physical_values(self, U) -> dict:
        """`{row name: physical value}` for the displacements `U` (n_free, n_cases) of a design. Needs a bound set."""
        return {r.name: r.physical(U) for r in self.rows}

    def constraints(self) -> list:
        """One `Constraint` for each row, in the order of `row_specs`. Works before `bind`."""
        out = []
        hint = MASS_HINT if self.config.settings.objective == "mass" else HINT
        for i, spec in enumerate(self.specs):
            def evaluate(point, i=i):
                return self.row(i, point.model).evaluate(point)
            # stage_aware: the run log notes a beta stage that ends with the row not met. No `measure`: the step
            # acceptance of the stress rows does not check these rows (measured: CONTRACTS section 31, plan section 7).
            out.append(Constraint(spec.name, evaluate, limit=spec.limit, price=spec.price, hint=hint,
                                  cause=CAUSE, stage_aware=True))
        return out


# --- facts and errors ------------------------------------------------------


def region_facts(reg: RegionNodes) -> list:
    """Facts about one region: nodes without a dof, and fixed nodes. No fact blocks a run."""
    out = []
    if reg.n_void:
        out.append(f"region '{reg.name}': {reg.n_void} of {reg.n_listed} nodes touch no solid or design "
                   f"voxel and have no displacement; they are left out. Options: to measure them, put "
                   f"the region on solid or on a load part; otherwise no change")
    if reg.n_fixed:
        out.append(f"region '{reg.name}': {reg.n_fixed} of {reg.nodes.size} nodes are fixed; their "
                   f"displacement is 0 and it counts in the mean and in the fit. Options: to leave "
                   f"them out, cut the region away from the fixed part; otherwise no change")
    return out


def all_fixed_fact(row, *regions) -> str:
    names = " and ".join(f"'{r.name}'" for r in regions)
    return (f"{row}: every node of {names} is fixed, so the value is always 0 and the row cannot "
            f"steer the design. Options: move the region off the fixed region, or drop the row")


def rank_message(row, region, fit: RigidFit) -> str:
    """The `UserError` text for a region whose rigid fit has rank 6 - k: the fact and the options."""
    where = {1: "one node", 2: "two nodes"}.get(fit.n_nodes, f"{fit.n_nodes} nodes on one line")
    return (f"{row} cannot work: the rigid fit of region '{region}' has rank {fit.rank} of 6 "
            f"({where}). A rotation needs nodes that are not on one line. Options: a larger region, "
            f"a region that is not on one line, or a displacement row in its place")


def check_rows(config, problem, free_dofs) -> list:
    """The checks of the rows before a solve, for `topopt check`. Returns the facts; raises `UserError`."""
    if not (config.limits.displacement or config.limits.alignment):
        return []
    return MotionSet(config).bind(problem, free_dofs)


def motion_set(config, model=None, log=None) -> MotionSet:
    """The `MotionSet` of `config.limits`. With `model`, the checks run now and `log` gets the facts.

    `.constraints()` gives the rows. Without a model the set binds at the first evaluation (and
    `log` gets the facts then). A config with no row does nothing: no bind, no allocation.
    """
    ms = MotionSet(config, log)
    if model is not None and ms.specs:
        ms.bind_model(model)
    return ms


# --- report ---------------------------------------------------------------


def physical_value(name, value, limit) -> float:
    """The value of a row without the guard of the smooth norm.

    The history keeps `sqrt(|y|^2 + eps^2)` for the rows displacement, shift and tilt (it is
    the value that MMA sees). This gives back `|y|`. The signed rows (`displacement_max`:
    `y`, `displacement_min`: `-y`) have no guard: their value is returned as it is.
    """
    if name.split(":", 1)[0] not in ("displacement", "shift", "tilt"):
        return float(value)
    eps = EPS_FRACTION * limit
    return math.sqrt(max(float(value) ** 2 - eps ** 2, 0.0))


def guarded_value(name, physical, limit) -> float:
    """The value that the row gives for a physical value: the inverse of `physical_value`."""
    if name.split(":", 1)[0] not in ("displacement", "shift", "tilt"):
        return float(physical)
    return math.sqrt(float(physical) ** 2 + (EPS_FRACTION * limit) ** 2)


def rows_block(history, nominal=None) -> list:
    """The `rows` block of `metrics.json`: each A5 row with its limit and its value at the start and the end.

    Reads the first and the last record of `history` (a list of history dicts). The values are
    physical: `|y|` for the rows displacement, shift and tilt (the guard is taken out, see
    `physical_value`), `y` for displacement_max and `-y` for displacement_min, in the unit of
    the row. `met` is `g_end <= NOT_MET`, the rule of the run log.

    `nominal` (robust mode, A3b): `{row name: physical value}` of the nominal final design, from the
    solve of `robust.finish_run`. `value_end`, `g_end` and `met` are then the nominal design (the part
    that you print, like `compliance_final`), and `value_end_eroded` is the value that the optimizer
    saw at the end. `value_start` is the eroded design of iteration 0 (like `compliance_initial`).
    """
    if not history:
        return []
    first = {c["name"]: c for c in history[0].get("constraints") or []}
    rows = []
    for c in history[-1].get("constraints") or []:
        kind = c["name"].split(":", 1)[0]
        if kind not in ROW_UNITS:
            continue
        v0 = first.get(c["name"], c)["value"]
        row = {"name": c["name"], "kind": kind, "unit": ROW_UNITS[kind], "limit": c["limit"],
               "value_start": physical_value(c["name"], v0, c["limit"]),
               "value_end": physical_value(c["name"], c["value"], c["limit"]),
               "g_end": c["g"], "met": bool(c["g"] <= NOT_MET)}
        if nominal and c["name"] in nominal:
            row["value_end_eroded"] = row["value_end"]
            row["value_end"] = float(nominal[c["name"]])
            row["g_end"] = guarded_value(c["name"], row["value_end"], c["limit"]) / c["limit"] - 1.0   # as the row g
            row["met"] = bool(row["g_end"] <= NOT_MET)
        rows.append(row)
    return rows
