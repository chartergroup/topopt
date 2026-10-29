"""Optional tetrahedral mesher (gmsh). See CONTRACTS.md section 14.

    from topopt.tet import probe, mesh_stl
    print(probe().line())
    mesh = mesh_stl("result/final.stl", size=2.5)
    mesh.nodes, mesh.tets, mesh.facts

This package does not import gmsh when you import it. gmsh is GPL-2.0-or-later and the
user installs it (the `tet` extra). The core of topopt never imports gmsh. One core module, `topopt.apply_marks`, imports
`topopt.tet.surface` (numpy, scipy and numba only) for the seam of its splice and the surface facts of its report.
"""
from .mesher import MeshingCrashed, MeshingFailed, TetMesh, default_size, mesh_stl, tet_quality
from .gmsh_probe import STATES, TetProbe, TetUnavailable, probe, require_gmsh

__all__ = [
    "STATES", "MeshingCrashed", "MeshingFailed", "TetMesh", "TetProbe", "TetUnavailable",
    "default_size", "mesh_stl", "probe", "require_gmsh", "tet_quality",
]
