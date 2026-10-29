# Print checks, the STEP file and the tet mesher

[Back to the README](../../README.md)

## Check the printed part

After a run, check that the part is good to print:

```
.venv/bin/python tools/check_part.py runs/hook examples/hook/hook.yaml
```

The tool checks these points: watertight mesh, one body, fit on the print bed, the screwdriver path for each
keep-out region, thin members, material at the load tip, and the mass.
Other options: `--bed`, `--pitch`, `--min-thick`. Run it with `--help` to see all options.
The exit code is 0 if all checks pass, and 1 if one check fails.

`--ref` compares with a reference `metrics.json` (mass, volume, compliance and the stiffness of each load case must agree within 1 %; the triangle count is printed only). It is optional.
The reference must come from the same design and the same mesh method. A run of an older build (a recorded run) is a `blur` run, and a new
run uses `sdf` (the default method), so the mass of the Quick start run is about 4.9 % lower and the check against such a reference fails
(exit code 1, `matches reference within 1%`, with every other check passing). To compare with it, make the run with `--mesh-method blur`:

```
topopt run examples/hook/hook.yaml --out runs/hook_blur --mesh-method blur
.venv/bin/python tools/check_part.py runs/hook_blur examples/hook/hook.yaml --ref <reference run>/result/metrics.json
```

On one Linux x86_64 host this gave every number equal to the recording (within 0.01 %) and exit code 0.

## Your printer (the catalog)

You name the printer; topopt fills the print bed and the printer type. The catalog that ships with topopt (`topopt/data/printers.yaml`) has 15
printers (Bambu Lab, Prusa, Creality, Elegoo and Voron), with the facts from the page of the maker: the bed, the build height, the
temperatures, the enclosure, the nozzle material and the extruder. Every row names its source. A value that the maker page does not give
is `null` (unknown), never a guess. Ask Claude "find my printer: Bambu P1S" (the tool `find_printer`), or write the model in the YAML:

```yaml
print: {printer_model: bambu-lab-p1s}     # an id of the catalog, or a name such as "Bambu P1S"
```

- The model sets `bed_mm` and `printer_type` when the YAML does not give them; a key that the YAML gives wins. `topopt check` and the print
  report name the printer and say which values came from it (`profile:bambu-lab-p1s`), and they compare the height of the part with the
  build height of the printer. A name that fits two printers, or none, is an error that lists the candidates or the close names.
- A printer that is not in the catalog: ask Claude to look up the page of the maker and to save the values (the tool `save_printer`). Claude
  shows you the values and the page address first. The printer goes to `topopt-printers.yaml` in the work folder (the same fields as the
  catalog; `source_url` or `source: user` names the source; a value that is not known is `null`). You can edit the file by hand. A user
  printer never replaces a printer of the catalog. A fault in the file is shown as a fact; the catalog still works. The command line
  (`topopt run`, `topopt check`) reads the file in the same work folder as the server: `TOPOPT_WORKSPACE`, else `TOPOPT_DEFAULT_WORKSPACE`, else the current directory.
- The rules are in [CONTRACTS.md](../../CONTRACTS.md), section 18.7.

## Your filament (the catalog)

You name the filament; topopt fills the modulus and the density. The catalog that ships with topopt (`topopt/data/materials.yaml`) has 69
filaments of six makers, with the numbers from the data sheet of the maker: the tensile modulus and strength in the layer plane and across
the layers, the density, the temperatures, the enclosure, the nozzle that it needs and the drying setting. Every row names its source and
the day it was opened. A value that no sheet states is `null` (unknown), never a guess. The catalog has no price. Ask Claude "find my
filament: Bambu PLA Basic" (the tool `find_material`), or write the filament in the YAML:

```yaml
material: {filament: polymaker-polylite-abs}     # an id of the catalog, or a name such as "PolyLite ABS"
```

- The filament sets `E` (the lower of the XY and the Z modulus of its sheet) and `density` when the YAML does not give them; a key that the
  YAML gives wins. `nu` stays the assumed 0.35, because no data sheet gives it. `topopt check` and `run.json` say where each number came from
  (`filament:<id>`, `yaml` or `assumed`) and keep the facts of the filament. A name that fits two filaments, or none, is an error that lists the
  candidates or the close names. A family alone ("PLA") fits many filaments, so it is not enough.
- The numbers of a data sheet describe printed test bars of one maker and one set of print settings, not your part.
- A filament that is not in the catalog: ask Claude to look up the data sheet of the maker and to save the values (the tool `save_material`).
  Claude shows you the values and the page address first. The filament goes to `topopt-filaments.yaml` in the work folder (the same fields as
  the catalog; `source_url` or `source: user` names the source; a value that is not known is `null`). You can edit the file by hand. A user
  filament never replaces a filament of the catalog.
- Which filaments does my printer print? Ask Claude (the tool `printer_materials`). Each filament is `ok`, `care` (it can work: for example a
  hardened nozzle or a chamber heater is needed), `no` (a stated need is above a stated limit of the printer as it ships) or `unknown`
  (a value is missing), and each reason names the two values that it compares. These are facts, not a stop: `topopt check` prints a
  `material fit` line when the YAML names a printer, and a `no` does not stop a run.
- The rules are in [CONTRACTS.md](../../CONTRACTS.md), section 18.8.

## Print report

After each run, `result/metrics.json` has a `print` section. It gives, for the build direction in the YAML and for
each of the six axis directions, the overhang area, the plate contact area, a support-volume estimate and the bed fit
(with a turn about the build axis when the turn adds at least 10 mm of margin to the bed edges, or when the part does not fit without it), and one thinnest-member estimate (1 percent of the surface has a
thinner chord) with the surface area below 0.8, 1.2 and 2.0 mm. An optional `print:` block in the YAML sets `printer_model`, `build_direction`, `bed_mm`,
`printer_type` and `self_supporting_angle_deg`. Without it the report uses 220 x 220 mm, `fdm` and +z, and labels them
"assumed". `topopt check` shows whether the design-space box fits the bed. The numbers are facts and estimates, not a gate.
`python tools/print_section.py runs/hook/result/final.stl` prints the same table for any STL.
The rules and the keys are in [CONTRACTS.md](../../CONTRACTS.md), section 18.

## Print check

`topopt printcheck <STL file or run folder>` checks a finished part for printing. It needs no solve and no extra package. For each check it gives the value, the
limit with its source, a state (`pass`, `fail` or `unknown`) and a reason: the STL as the file holds it (closed, bodies, closed voids), the bed fit, the overhang,
the support estimate, the plate contact and the thin walls. A limit is a key of the `print:` block (`min_wall_mm`, `max_closed_void_mm3`, `max_bodies`,
`max_overhang_area_mm2`, `max_support_volume_mm3`, `min_plate_contact_mm2`; the keys are in CONTRACTS.md section 32) or `--set KEY=VALUE`. A check with no limit gives
the fact and the state `unknown`. By default the command only reports (exit code 0); `--strict` is an opt-in gate (exit code 4 when a check that has a limit fails).
The MCP tool is `print_check`. A print pass is not a structural pass. The rules are in [CONTRACTS.md](../../CONTRACTS.md), section 32.

```
topopt printcheck runs/hook --set min_wall_mm=1.2 --strict
```

## STEP file (optional)

`topopt step <run folder>` writes `result/final.step`: the final mesh as a STEP file, so that a CAD program can open it, cut holes in it and join it to
other parts. It needs the extra `step` (OpenCASCADE, the package `cadquery-ocp-novtk`, about 250 MB; macOS with Apple silicon, macOS Intel, Linux x86_64):

```
uv pip install -e ".[step]"
topopt step runs/hook
```

The file is faceted: every triangle of the mesh is a flat face of a closed solid, in mm. A CAD program can measure it, cut it and combine it. It cannot edit the
surfaces, because the optimizer shape has none. The faces of the fixed and load regions become single flat faces. Before it wraps the mesh,
the command simplifies it with manifold3d. The tolerance is the voxel size of the run / 20 (set `--tolerance-mm X`; `0` keeps every triangle; a run folder with no voxel size is not simplified). On the hook that is
a largest distance of 0.1 mm to the STL, and 70 percent fewer triangles. `--no-merge-planes` keeps one face for each triangle, `--part-only` wraps `result/part_only.stl`,
`--schema AP242` writes the newer schema, `--out FILE` writes elsewhere (it replaces an existing `.step` or `.stp` file only; any other existing file, and the input STL, give an error with the options; the report is then next to the new file, so the run folder keeps no new file).
`--estimate-only` simplifies, prints the expected faces, file size, time and memory, and writes nothing. It works without OpenCASCADE.

The command checks the model before it writes the file (it is valid, and every solid has a positive volume), reads the file back and checks it: the file holds the solids and shells that the command built, OpenCASCADE reads a valid and closed solid, the unit is mm, the schema is as asked.
No check has a threshold. A failed check is a fact in the report (`result/final.step.json`, next to the STEP file) and exit code 3; the file is written anyway. The report also gives the sampled distance between the STEP surface and the STL, and the volume of the file against the STL.
"Valid" is the opinion of OpenCASCADE, whose reader turns faces that point the wrong way; another CAD program can differ.
The MCP tool is `export_step`; the call returns when the file is written, so a large export holds it for minutes (use `estimate_only` first). Without the extra, the command prints the install line and exits with code 1.

Measured on Linux x86_64 (22 threads, shared with other jobs; a Mac differs, mostly in time). "Distance" is the largest sampled distance between the STEP surface and the STL.

| Mesh | Setting | Faces | STEP file | Time | Peak memory | Distance | Volume change |
|---|---|---|---|---|---|---|---|
| short hook, 20,586 triangles | every triangle kept (`--tolerance-mm 0 --no-merge-planes`) | 20,586 | 22 MB | 20 s | 0.8 GB | < 1e-6 mm | 0.00 % |
| short hook, 20,586 triangles | default (voxel size / 20, merged planes) | 9,017 | 10 MB | 12 s | 0.6 GB | 0.16 mm | -0.21 % |
| short hook, 20,586 triangles | voxel size / 10 | 4,619 | 5 MB | 8 s | 0.6 GB | 0.33 mm | -0.97 % |
| hook, 62,954 triangles | every triangle kept (`--tolerance-mm 0 --no-merge-planes`) | 62,954 | 69 MB | 57 s | 1.7 GB | < 1e-6 mm | 0.00 % |
| hook, 62,954 triangles | default (voxel size / 20, merged planes) | 18,756 | 20 MB | 23 s | 1.0 GB | 0.10 mm | -0.07 % |
| hook, 62,954 triangles | voxel size / 10 | 8,993 | 10 MB | 15 s | 0.9 GB | 0.24 mm | -0.36 % |
| hook, finer mesh, 172,188 triangles | every triangle kept (`--tolerance-mm 0 --no-merge-planes`) | 172,188 | 193 MB | 180 s | 3.1 GB | < 1e-6 mm | 0.00 % (valid solid: no) |
| hook, finer mesh, 172,188 triangles | default (voxel size / 20, merged planes) | 35,558 | 39 MB | 39 s | 1.1 GB | 0.12 mm | +0.05 % |
| hook, finer mesh, 172,188 triangles | voxel size / 10 | 8,905 | 9 MB | 13 s | 0.7 GB | 0.22 mm | +0.01 % |
| hook, finer mesh, 322,438 triangles | every triangle kept (`--tolerance-mm 0 --no-merge-planes`) | 322,438 | 363 MB | 385 s | 5.5 GB | < 1e-6 mm | 0.00 % |
| hook, finer mesh, 322,438 triangles | default (voxel size / 20, merged planes) | 18,082 | 19 MB | 26 s | 0.9 GB | 0.12 mm | -0.04 % |
| hook, finer mesh, 322,438 triangles | voxel size / 10 | 8,078 | 9 MB | 17 s | 0.7 GB | 0.28 mm | -0.14 % |
| hook, finest mesh, 565,276 triangles | every triangle kept (`--tolerance-mm 0 --no-merge-planes`) | 565,276 | 641 MB | 861 s | 9.2 GB | < 1e-6 mm | 0.00 % |
| hook, finest mesh, 565,276 triangles | default (voxel size / 20, merged planes) | 17,264 | 19 MB | 29 s | 0.9 GB | 0.12 mm | -0.12 % |
| hook, finest mesh, 565,276 triangles | voxel size / 10 | 8,151 | 9 MB | 20 s | 0.9 GB | 0.29 mm | -0.37 % |
| a larger part A (not shipped), 529,876 triangles | default (voxel size / 20, merged planes) | 85,103 | 94 MB | 102 s | 1.8 GB | 0.05 mm | not recorded |
| a larger part B (not shipped), 925,248 triangles | default (voxel size / 20, merged planes) | 355,585 | 402 MB | 613 s | 5.8 GB | 0.08 mm | not recorded |

With the default setting the cost follows the number of faces after the simplification, and that follows the part. The hook meshes (one shape at different triangle sizes) stay near 9,000 to 36,000 faces,
but two larger parts (not shipped) do not: 85,103 faces and 102 s for part A, 355,585 faces and 10 minutes for part B. With every triangle kept the file is about 1.1 KB for each face and the memory grows with it (565,276 triangles: 641 MB, 14 minutes, 9.2 GB).
`topopt step <run folder> --estimate-only` prints the faces, the file size, the time and the memory before the work. Nothing is refused.

Details: [CONTRACTS.md](../../CONTRACTS.md), section 42. Licence of the OpenCASCADE wheel: [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

## Tet mesher (optional)

`topopt.tet` meshes a closed STL (for example `result/final.stl`) to tetrahedra. The next steps
(stress check, buckling) build on it. There is no CLI flag, MCP tool or YAML key for it yet: call it from Python.

**License.** The mesher uses gmsh. gmsh is GPL-2.0-or-later. You install it yourself with the `tet` extra.
The core of topopt (license: MIT, see `LICENSE`) never imports it (a test checks this), and the repo does not include it.
Install it only if its license fits your use.

```
uv pip install -e ".[tet]"
```

The extra has a platform marker: PyPI has gmsh wheels for macOS and Linux x86_64 only.
On other platforms (for example Linux arm64) the probe says `missing`.

On Linux, the gmsh wheel is built with a GUI. It needs system libraries that a slim image can lack:
`libglu1-mesa libxft2 libxinerama1` (Debian and Ubuntu: `sudo apt install libglu1-mesa libxft2 libxinerama1`).
Without root, run `apt-get download libglu1-mesa libopengl0 libxft2 libxinerama1`, unpack each file with
`dpkg-deb -x <file> <dir>`, and put `<dir>/usr/lib/x86_64-linux-gnu` in `LD_LIBRARY_PATH` before Python starts.
There is also a GUI-free build of gmsh that needs no system library (a dev build, `4.15.0.dev1+nox`, Linux x86_64 only).
The probe prints its install command.

**Probe.** `topopt.tet.probe()` never raises. It returns a `TetProbe` with `state`, `version`, `detail`, `fix`:

| state | meaning |
|---|---|
| `ok` | gmsh imports and finds its library. `version` has the version. |
| `missing` | gmsh is not installed. `fix` has the install command. On a platform that the extra does not cover (for example Linux arm64), `fix` lists other options: gmsh from conda-forge (another OCC build, so its meshes differ) or a source build. |
| `load_failed` | gmsh is installed but does not load. On Linux, `detail` lists ALL missing system libraries and `fix` has the Debian package names. On macOS, `fix` says to run `otool -L` on libgmsh (the arm64 wheel needs macOS 12 or newer and links only system libraries). |

`probe().line()` gives one line of text. `probe_subprocess()` does the same in a child Python, so your process
does not load libgmsh. The `environment` tool of the MCP server reports the same data in its field `tet`.

**API.**

```python
from topopt.tet import probe, mesh_stl, TetMesh

print(probe().line())
mesh = mesh_stl("runs/hook/result/final.stl", size=2.5)   # size in mm; threads=1 by default
mesh.nodes    # (n, 3) float64, mm
mesh.tets     # (m, 4) int64, positive volume
mesh.facts    # dict, see below
path = mesh.save("hook_mesh")          # writes hook_mesh.npz (nodes float64, tets int64, facts as JSON)
same = TetMesh.load(path)
```

`mesh_stl(path, size=None, threads=1, optimize="Netgen", eps_factor=0.1, max_attempts=10, size_field=None, min_edge_mm=0.05, isolate=True, fallback_optimize=("Gmsh+Netgen", "Gmsh"), log=None)`.

- `size`: target edge length in mm. If you give none, it is 2.5 times the mean edge of the STL (`default_size`).
  The STL of a run made with `blur` has edges of about half a voxel, so the default is about 1.15 to 1.25 voxels
  (hook: 1.496 mm for 1.2 mm voxels; ladle: 3.744 mm for 3.25 mm voxels). The `sdf` STL has fewer short edges at the region faces, so its mean edge is larger:
  at c080f62 the hook gives 1.94 mm (1.6 voxels; 1.75 mm before the move of the vertices to the region faces), the ladle 4.28 mm (1.3 voxels; 3.36 mm) and the
  bracket of `examples/interfaces` 2.16 mm (2.2 voxels). `topopt verify` prints the size. The mesh has edges up to 1.5 times `size` inside the part.
  The pipeline notes give the rule h <= d_min / 3, with d_min the thinnest member, and used 2.5 mm for a larger part (not shipped).
  For an STL from a CAD tool the default can be a poor guide: pass `size`.
- `size_field`: optional function `f(P) -> sizes` for points `P` (n, 3) in mm. It makes the surface finer in a region (for example near a load).
  It is combined with the retry patches by `np.minimum`.
- `min_edge_mm`: the shortest edge that the surface preparation accepts. It is absolute (not scaled with `size`). Lower it for a part smaller than about 5 mm. Marching cubes can leave clusters of vertices 1e-5 mm apart on a flat face that lies on the node lattice. If an edge under the limit stays after the preparation, the preparation welds the input once with the same limit and tries again. `facts["surface_weld_retry"]` says what it did. The weld can fail. For example, it can move a vertex by more than the limit, make more fold edges, or give a surface that is not a closed manifold. The preparation then cleans the surface that it refused: it collapses the short edges again, and welds the few edges that stay (`facts["surface_clean_retry"]`). The collapse and the weld can make fold edges (two faces on top of each other); the cleaning then runs the fold repair of the first try, and the facts list it (`fold_repairs`). The limit is the radius for the vertices that are welded. A vertex whose faces all go (the apex of a spike on short edges) is removed, and the facts give its distance to the cleaned surface (`removed_vertex_distance_max_mm`). If that does not help either, the refusal stays and its text names the reason of each try. When gmsh then fails on a surface that a retry made, the error names the retry and the first refusal.
- `optimize="Netgen"` raises the minimum tet quality a lot (hook, size 2.5: 0.226 with it, 0.021 without it).
  Other values: None (no optimizer), `"Relocate3D"`, `"Gmsh"` (the default optimizer of gmsh during meshing), or steps joined
  by `+`, for example `"Gmsh+Netgen"`.
- **Isolation.** By default (`isolate=True`) gmsh runs in a child Python (`python -m topopt.tet.worker`). Your process
  never loads libgmsh (good for an MCP server), and a crash of gmsh kills only the child. The prepared surface goes
  to the child and the nodes and tets come back as .npz files in a temp folder. The meshes are the same as in-process
  (same hash). Cost: the first call asks a child for the probe (about 0.1 s, kept for later calls), and each attempt
  adds about 0.1 s (`facts["seconds_isolation"]`: 0.09 to 0.15 s on the hook, Linux x86_64). `isolate=False` runs gmsh in
  your process: a crash then kills it, and calls from several threads run one after the other.
- **Known crash.** gmsh 4.15.2 on Linux x86_64 crashes with SIGSEGV in the Netgen optimizer on the hook `final.stl` at size 0.6
  to 0.75 mm (0.8 mm and larger work). The HXT mesh at these sizes holds a degenerate tet (quality 2e-27); the crash probably comes from it (not proven).
  If the child dies on a signal, `mesh_stl` tries `fallback_optimize` on the same surface (default `"Gmsh+Netgen"`,
  then `"Gmsh"`) and records it in the facts: `optimize` (asked for), `optimize_used`, `crashes` (attempt, signal, size, optimize).
  If every setting crashes, it raises `MeshingCrashed` (a `MeshingFailed`) with the signal, the size, the optimize settings
  and the options. Measured on the hook, 1 thread, HXT (tets, minimum quality; no tet below 0.05 unless noted):

  | size mm | no optimizer | `Netgen` | `Gmsh` | `Gmsh+Netgen` |
  |---|---|---|---|---|
  | 0.5 | 701,943; 6e-11; 9,763 below 0.05 | 541,114; 0.461; 41 s | 670,796; 0.347; 2 s | 600,639; 0.315; 29 s |
  | 0.6 | 328,952; 2e-27; 1,650 below | SIGSEGV | 317,264; 0.184; 1 s | 289,238; 0.244; 16 s |
  | 0.65 | 283,561; 2e-27; 1,586 below | SIGSEGV | 273,338; 0.126; 1 s | 250,418; 0.158; 11 s |
  | 0.7 | 255,333; 2e-27; 1,553 below | SIGSEGV | 246,015; 0.094; 1 s | 226,220; 0.143; 10 s |
  | 0.75 | 221,132; 2e-18; 713 below | SIGSEGV | 213,579; 0.182; 1 s | 196,601; 0.204; 11 s |
  | 0.8 | 150,359; 2e-9; 30 below | 125,614; 0.214; 6 s | 145,554; 0.167; 0.5 s | not run |
  | 1.0 | 77,201; 0.001; 25 below | 66,740; 0.215; 3.5 s | 74,508; 0.169; 0.3 s | not run |
  | 1.2 | 43,540; 0.002; 10 below | 39,049; 0.202; 1.7 s | 41,740; 0.166; 0.2 s | not run |

  The times are the gmsh step. `Relocate3D` alone did not remove the slivers (0.6 and 0.75: still about 1,600 and 700 below 0.05).
  Delaunay (algorithm 1) with Netgen also meshed 0.6 and 0.75 (qmin 0.25 and 0.20, 21 s and 15 s) but topopt tries HXT first.
  The crash was seen on one STL; other STLs may crash at other sizes. The study ran on one machine and no Mac.
- The STL is first coarsened to `size` with a surface decimator that keeps it a closed manifold.
  It changes the volume a little (`volume_ratio_stl`, the hook: 0.974 at 2.5 mm, 0.993 at 1.2 mm; `deviation_from_input_mm` gives the distances).
  If gmsh reports a self-intersection, the surface is prepared again with a fine size around the reported points. A mesh with a
  tet of no volume or that the stress check cannot invert (four nodes in one plane) is treated the same way: that attempt fails and the next one refines at the tet.
  If the new preparation is refused, the error says what gmsh did on the surface of the attempt before.
- With `isolate=False`, one mesh at a time per process: gmsh keeps one global state, so calls from several threads run
  one after the other. It works off the main thread.
- **Repeatability.** One thread gives the same mesh on every run (same machine, same gmsh version: the same `sha256`).
  With more threads (`threads=4`, no hard cap) the HXT mesher can give a different mesh on each run, with another
  tet count too. Measured on Linux x86_64 with gmsh 4.15.2, on the hook `final.stl` at size 0.8 mm (about 125,600 tets):
  1 thread gave the same hash in all runs; 4 threads gave a different hash in each of three runs (125,922 to 126,138 tets,
  quality min 0.210 to 0.214). The gmsh step took 5.6 s at 1 thread and 5.6 to 5.7 s at 4 threads, so 4 threads gave no gain.
  A small mesh (8,000 tets) gave the same hash twice at 4 threads, so the effect depends on the size.
  A different OCC build of gmsh (conda-forge) gives a different mesh even with one thread. Mac results are not measured.
  Store the mesh (`mesh.save(...)`) and its hash if you need the same mesh again.
- Facts: `tets`, `nodes`, `quality_min`, `quality_mean`, `quality_p01`, `below_0p1`, `below_0p05`,
  `volume_fraction_below_0p05` (quality is the mean ratio, 1 for a regular tet; 0.05 is the cut of the P2 report),
  `volume_mm3`, `volume_stl_mm3`, `volume_ratio` (tet volume divided by the volume of the prepared surface: 1 within 1e-12;
  a mesh is accepted if it is within 1e-3), `volume_ratio_stl` (divided by the volume of the STL), `deviation_from_input_mm`,
  `stl_sha256`, `inverted_fixed` (tets that had to be reordered, normally 0), `threads`, `repeatable_expected`
  (True at 1 thread; a prediction), `sha256`, `size`, `size_given`, `size_field_given`, `algorithm`, `attempts`,
  `failed_attempts`, `optimize`, `optimize_used`, `crashes`, `isolated`, `gmsh_version`, `seconds`, `seconds_surface`,
  `seconds_gmsh`, `seconds_isolation`, `surface_faces`.
- Errors, all `topopt.config.UserError`: `TetUnavailable` if the probe is not `ok` (the text has the state and the fix),
  `SurfaceRejected` if the STL is not a closed manifold, `MeshingFailed` if gmsh fails, `MeshingCrashed` (a `MeshingFailed`)
  if gmsh dies on a signal with every optimize setting, and a plain `UserError` for a bad argument or a file that is not an STL.
