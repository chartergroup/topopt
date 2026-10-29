# Refine page (3D markup tool)

The refine page shows the final part with a stress overlay.
You place add, remove and smooth marks on the part and export them as JSON.
The page does not change the mesh. A later step applies the marks.

The tool comes from the markup tool of an earlier project of the same author. It is adapted for topopt runs.

## Build

A run makes the page by itself at the end of the run, after the status is `done`.
The file is `<run_dir>/result/refine.html`. A failure in this step only logs one warning line.

To build the page for a finished run by hand:

    topopt refine <run_dir>

The command prints the path of the page.
It rebuilds the model from the YAML file named in `run.json` (a relative path starts at the run folder).
If the YAML is at another place, `--yaml FILE` gives it. `refine` compares the YAML with `run.json` and prints one `note:` line
for each difference (forces, weights, case names, region files), the same lines as `topopt verify`. The page data has the key
`yaml_vs_run` too. A difference does not stop the step. The material (`E`, `nu`) is not in `run.json`, so it is not compared.
It stops with an error if the YAML or the STL files changed after the run.
Decimation acts only above 150,000 faces.

The step also writes `<run_dir>/result/stress.npz`:
the voxels with density 0.5 or more, their centres in the mm frame of `final.stl`,
the von Mises stress for each load case (`vm_case_<i>`), and `vm_max` (the maximum over the cases).

## Stress

- One solve for each load case. The stiffness is the stiffness of the last snapshot
  (`E_min + rho^p (E - E_min)`, as in the optimizer).
- The stress is the von Mises stress of `D0 * strain` at the centre of each voxel.
  `D0` uses the full modulus `E`. The stress is not scaled by `rho^p`.
  Boundary voxels have a density below 1. A scale by `rho^p` would hide the stress at the surface.
- Each vertex of the mesh gets the value of the nearest voxel centre. The page shows the maximum over the load cases.
- The colour scale goes from 0 to `stress_max_mpa`. This is the 99th percentile of the surface values.
  A few hot voxels do not wash out the colours. The scale is in the info line and in the legend.
  Values above the scale have the top colour.
- Loads and clamps are spread over whole voxels. The stress near them is only a rough guide.

## Open

The dashboard shows a "Refine" button in the result panel when the page exists.
The button opens `/refine` of the dashboard server in a new tab.
With `topopt run ... --dashboard`, the browser also opens `/refine` by itself after the run.
You can also double-click `refine.html` (file://, Chrome or Safari).
The page holds its own copy of three.js r128 and its OrbitControls (`topopt/refine/vendor/`, about 0.6 MB of the page).
It needs no network, like the rest of the dashboard.
A page that an older topopt made loads three.js from cdn.jsdelivr.net. The dashboard puts the copies in its place when it serves
such a page. To open it as a file with no network, make it again: `topopt refine <run_dir>`.

## Mark

- Choose Add (green), Remove (red) or Smooth (blue). Set radius / amount / strength / mirror in the Tool box (these are the defaults for new marks).
- Click the part (no drag) = point mark. Shift + drag along the part = stroke (surface points about 2 mm apart). Orbit drags never place marks.
- Click a mark in the list to select it, fly to it, and edit radius, amount (add/remove) or strength (smooth), mirror, note, or delete it.
- Mirror is OFF by default. If it is ON, the part is symmetric about y = 0 and the ghost on the other side is also applied.
- Ctrl/Cmd+Z undo, M toggles mark mode, "Clear all" asks for confirmation. Marks autosave to localStorage (per STL hash and per server port).
- "colour by stress" is ON when the page opens. Use the check box to switch it off.

## Save

"Download marks JSON" gives `marks_<run name>_<YYYYMMDD-HHMM>.json` in the download folder of the browser.
"Import JSON" continues a session. "Copy JSON" copies to the clipboard.

Schema: `{"version":1,"source_stl","source_sha256","units":"mm","created","marks":[{id,type,points:[[x,y,z]],normals:[[nx,ny,nz]],radius_mm,amount_mm,strength,mirror,note}]}`.
All coordinates are in the mm frame of `result/final.stl` (the view is centred internally only).
`type` is `add`, `remove` or `smooth`.
`amount_mm` is null for smooth. `strength` is null for add and remove.
`normals` are outward surface normals of the (decimated) display mesh.
`source_sha256` is the SHA-256 of `final.stl`. Use it to check that the STL did not change.

Optional keys (added with C3b; the page does not write them and ignores them on import): a `smooth` mark can carry `smooth_mode`
(`auto`, `tube`, `iso`, `fair`), a `remove` mark can carry `remove_mode` (`cut`, `shave`), and a mark can carry a `cleanup` object with
the facts of a candidate. `python -m topopt.cleanup list` (CONTRACTS.md section 29.3) writes a marks file with these keys for the
spurs, bumps, struts and loose bodies that it finds. The page can import such a file to show the marks; this was not run.

Round trip: an import followed by an export on the page drops `smooth_mode`, `remove_mode` and `cleanup`. The import keeps only the
keys that the page knows (`template.html`). So a FAIR or tube mark of a list becomes a plain smooth mark, and `apply_marks` uses its
call default for it. Apply the file of the list command as it is, not a file that the page exported.

Drawing size: the page draws one sphere for each point. A list can have many points (the ladle list at 0.5 mm: 68,937 points in 10
marks; the hook list at 0.6 mm: 2,098 points in 4 marks; both measured with the code of this branch), and a long list draws slowly.

## Mark semantics

Each mark is a point or a polyline (stroke). Its region is the set of points within `radius_mm` of the point or of the polyline.
If `mirror` is true, apply the same mark again with y -> -y (also ny -> -ny).
The falloff must be smooth (for example cosine or Gaussian from 1 at the path to 0 at `radius_mm`).

- **add**: union a local outward offset of `amount_mm` (along the surface normal) within the region, with smooth falloff.
- **remove**: subtract a local inward offset of `amount_mm` within the region, with smooth falloff.
  Keep the fixed and load regions and the keep-out shapes of the YAML. Keep a minimum member size that the printer can make.
- **smooth**: local SDF smoothing. Replace the signed distance field with a Gaussian-filtered version inside the region,
  with sigma tied to the `radius_mm` x `strength` neighbourhood (strength 1 to 3, larger = stronger), blended in with smooth falloff. Then re-mesh.

The page does not apply the marks to the mesh. The MCP tool `marks_to_hints` (and `python -m topopt.feedback hints`)
reads the marks file and gives facts and options for each mark, with a proposed YAML and region STL files in a new folder
(CONTRACTS.md section 21). It uses `points`, `radius_mm` and `mirror`. It does not use `normals`, `amount_mm`, `strength` or `note`.
The MCP tool `apply_marks` (and `python -m topopt.apply_marks`) applies the marks to the finished part in a new folder (CONTRACTS.md
section 29). It uses `points`, `radius_mm` (not in `smooth_mode` tube), `amount_mm` (add: the neck offset; remove: the depth in `remove_mode` shave only),
`strength` (smooth) and `mirror`. It does not use `normals` or `note`.

## Limits

The marks sit on the decimated display mesh (decimation only above 150,000 faces; the volume stays within about 0.5% of the source).
Points can be up to a fraction of a mm off the true surface.
