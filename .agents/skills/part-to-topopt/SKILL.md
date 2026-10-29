---
name: part-to-topopt
description: Use when the user describes a physical part in plain words (for example "a wall hook for a T-shirt", "a bracket for a 2 kg shelf", "a phone stand"), or gives an STL of their own part ("here is my part", "keep the screw holes"), and wants it optimized or made lighter with topopt. Turns the description or the STL into a valid problem YAML and region STL files, checks them, and prints the assumptions. Do not use when the user already has the region STL files and the problem YAML.
---

# Part description to topopt problem

Goal: build a valid problem YAML and region STLs that match what the user wants.
Before the run, state the assumptions: write them down (Step 5 has the format).
Ask the user when an answer changes the part (for example the load, the mounting or the size). For a small detail, make a sensible assumption and write it down.
The load changes the shape of the part, so the load has no default. If the user does not say what the part holds or carries (no object, no mass and no force), ask for it before the run. Do this also when the user says "run it". You can give a typical value in the question. Then wait for the answer. If the user names the object (for example a coat or a plant pot), use a typical mass for it and write it in the assumptions.
The tools give facts, estimates and options. You decide the method. The user decides the load, the mounting and the size.
When the user wants a result and you have the answers to your questions, you may start the optimizer and watch it through the MCP tools or `topopt run`.

Units are mm, N, MPa, g. The MCP server instructions carry the short version of this guidance for any client. This file has the detail.

Where things run. Files go in the work folder: the folder where the client runs (`runs/<name>/`).
This skill is in `.claude/skills/part-to-topopt/` and, as a copy, in `.agents/skills/part-to-topopt/` (Codex, Gemini, Cursor and VS Code read that folder).
A path below starts at the repo root, unless the text says "in the folder of this skill" (the folder that holds this `SKILL.md`; the Agent Skills rule for files next to a skill).
In the Claude Code plugin, the repo root is `${CLAUDE_PLUGIN_ROOT}`.
In the Claude Code plugin `topopt`, the tools of the MCP server (`build_regions`, `check_problem`, `run_topopt`, ...) work with no setup, and the plugin has its own Python environment.
To run a Python command in it, use this prefix (Claude Code fills in the two paths):

```
CLAUDE_PLUGIN_DATA="${CLAUDE_PLUGIN_DATA}" sh "${CLAUDE_PLUGIN_ROOT}/scripts/topopt-env.sh" python ...
```

In a source checkout of topopt (no plugin), use `sh scripts/topopt-env.sh python ...` in the repo root: it uses the same environment as the MCP server of the client configs. `.venv/bin/python` also works after the dev install.

## Two entries

- A part in words: Step 1 to Step 5 below.
- Your own STL: the user has a file of the part (an exported CAD file). Use the section "Your own STL" (after Step 2). It takes the place of Step 1 and uses Step 2 to Step 5.

## Step 1. Infer the part (think, do not write long text)

Decide each item from the description.

1. Function: what does the part hold or carry?
2. Mounting: which faces are fixed? A wall plate, screw bosses, or a clamp ring. Make the fixed region a thick plate or ring, 3 mm or more.
   - A fixed region is a full clamp. To fix only some axes (a roller or a slider), name them in `fixed_axes` (see Step 2). A per-axis region is a solid body in the part: its nodes are fixed along the named axes only, so it carries load along the other axes and adds its stiffness (outside the part, a slab of one voxel makes a half model 12 to 22 percent too stiff, and a slab of 3 mm 40 percent). For a symmetry plane, make the slab one voxel thick and put it inside the part, or give the full symmetric part and use `mirror`. The supports together need to block all six rigid-body modes (three rollers on three planes do, one roller does not). If they do not, the engine stops before the run and names the free modes.
   - A part that sits on a soft mount (a rubber foot, a spring clip) has a spring instead of a clamp: add the mount body to `load` and a `springs` entry for it, `{region: <load name>, stiffness: [kx, ky, kz]}` in N/mm (the total for the region, global axes). The spring acts on all faces of that body that look into free space or a keep-out region, so give the body a free face only where the mount touches the ground. The body is solid and carries load. A spring can replace the `fixed` region (a part on springs only is valid), but a very soft spring makes a part that is nearly free: the check prints a warning.
   - A part that hangs on a rod, a pin or a tube that is not in the grid has a joint: add the body that the rod goes through (a sleeve, a boss) to `load`, and a `joints` entry `{region: <load name>, to: [x, y, z], section: {kind: round, d: 4}}`. `to` is the fixed far end of a beam (mm), in any direction. Sections: `round {d}`, `tube {od, id}`, `rect {b, h}`, `props {A, Iy, Iz, J}` (mm). The faces of the body that look into free space are bonded to one rigid body, and an Euler-Bernoulli beam joins that body to the far point. A joint can replace the `fixed` region (a part on joints only is valid). Optional keys: `origin` (the start of the beam), `orient` (the section y axis of a section that is not round), `E`, `nu`, `bond_stiffness_per_area`. The check prints the beam length and L/d (the model has no shear: a beam with L/d 10 is 0.6 percent more flexible than the model says, L/d 5 2.3 percent, L/d 3 6.3 percent, L/d 2 14 percent, L/d 1 57 percent), the bond stiffness with its source, `rho` and the bond share. The result has the wrench on the part and the ground reaction of each joint. The solvers pardiso, cholesky, amg and `gpu-mg` take joints (and springs); `gpu-mg` takes per-axis supports (`fixed` with `axes`) too; in its case C (a device without 64-bit float) the joint term is computed in 32-bit float: `check` gives the ratio of the beam to the bond stiffness and the cases that were measured, and a run that stops gives the ratio and the options (another solver, a device with 64-bit float, a stiffer beam: a larger section or a shorter beam). The voxel buckling check gives the fact and the options.
3. Load: where does the force act? A load region is a forced-solid body. Make it the WHOLE solid feature that the force acts on (a hook tip, a bowl, a shelf lip). A small patch on a thin wall is a poor load region: the optimizer can cut it off and remove it. Give its force in N as `[Fx, Fy, Fz]`.
   - If the description does not say what the part holds or carries (no object, no mass and no force), ask the user for it. Do not start a run on a load that you chose.
   - Static force = mass in kg x 9.81. Example: 0.2 kg gives 1.96 N.
   - Add a second load case for abuse: 5 to 10 times the static force, in a different direction (a pull, a knock, a side force). Give it weight 1.0.
   - Give the static case a bigger weight only if you want it to dominate.
   - A moment (a twist or a bending couple) is `moment: [Mx, My, Mz]` in N mm in a force entry. A force that acts away from its region is a remote force: add `at: [x, y, z]` in mm. Both are spread over the nodes of the load region, with no added stiffness (a distributing coupling, RBE3-like). A rigid coupling is not available. A stiff link can be a load region of solid voxels.
   - A pressure (N/mm2, MPa) is a `pressures` entry of a load case: `{region: <load name>, value_mpa: 0.2}`. It acts on the faces of that load region that look into free space or a keep-out region. A positive value pushes into the part. A pressure on a fixed region or on the design space is refused, with the reason. Shape the load region so that only the wanted face is free.
4. Dimensions: use real sizes. Use common sizes for the object (a screw is 4 to 5 mm, a shelf is 200 mm deep). A sketch, a photo or a drawing with no dimension gives proportions, not a size. Get the scale from one real dimension that the user gives, or from a known object in the picture (a standard screw, a coin, a known product). When there is none, ask for one before the run: the size is one of the answers that change the part (the top of this file), and a guessed size can give, for example, a hook that does not fit. Take the other sizes from the proportions of the sketch. Write each number that you took or guessed as an assumption (origin `inferred` in `record_brief`), and show the list to the user. The part has to fit the print bed. Take the bed size from `print.bed_mm` of the spec or the problem YAML, or from the printer model (`print.printer_model`; see "The printer" below). If nothing is given, the tools use 220 x 220 mm (the default, labelled "assumed"). `topopt check` gives the fit of the design-space box as a fact, and the `print` section of `metrics.json` gives it for the final mesh.
5. Keep-out regions: screw holes with clearance for the screw head and the screwdriver, the space of the user's hand, and the space of the held object. Keep-out regions overlap the design space.
6. Material: PLA or PETG. Use E 3500, nu 0.35, density 1.24 unless the user names another material. If the user names the filament they have, call `find_material` and put `filament: <id>` in the `material` block (see "The filament" below): the profile gives E and density.
7. Volume fraction: it is the mean density of the FREE design voxels (without the fixed, load and solid regions, which are solid on top). Use 0.2 for a light part, 0.25 default, 0.3 for a part with a big abuse load. The YAML keeps your value, as CONTRACTS.md and the engine read it. The helper prints the basis once, then the free and the forced-solid voxels, the target volume in mm3 and the target mass in g of the run (the numbers of `topopt check`), and the share of all active voxels (for information only; it is written nowhere).
8. Solid bodies: a body that must stay solid, but has no clamp and no force, goes in `solid` (for example a cap or a rim). The builder writes it as `regions.solid` in the problem YAML. The engine freezes it at density 1. A load region keeps its voxels where it overlaps a solid body (priority: keep-out, fixed, load, solid, design).
9. Use in the world: think about how the part touches the objects around it. A sketch or a description often does not show this. For example, a hook for a hanging object needs a lip, so the object does not slide off. A part that nests in another object needs clearance around that object. Add these features to the spec, and write them in the assumptions.
10. Symmetry: if the part and its loads are symmetric about a plane, you may add the optional `mirror` key (see "The `mirror` key"). It makes the design mirror-symmetric. It is a choice for the part, not a default.

Coordinate rule: pick one frame for all regions. Put the mount at x = 0, with the part out in +x, and z up.
Make the design space a box (or a prism) that wraps all the load and fixed regions with a margin.
The design space must touch both the fixed regions and the load regions.
A load region and a fixed region must not overlap each other. Load regions can overlap the design space.

### Load inventory, verify and ask, buckling

Do this before the first run. Write the result in the `assumptions` list (Step 2) and in the brief (`record_brief`).
The size of the inventory follows the part. There are three levels:

- A simple part with clear loads (a wall hook, a shelf bracket): one or two lines. Example: "Shirt at the tip (1.96 N); the wall plate holds it; an abuse pull of 30 N." Then go on.
- The user gives exact loads: use them. Check that each interface that carries force has a load or a support. Ask only about a gap.
- A part that joins other parts or carries a person: fill each column for each interface (a bore, a socket, a face, a clamp, a tip). Look at the neighbour parts. A force that a person puts on one part can reach this part through the assembly.

| Column | Content |
|---|---|
| Interface | each place where the part touches another part or the world |
| Neighbour | the part or the person on the other side |
| Load sources | weight, the user, the neighbour parts, the assembly (preload, press fit), heat |
| Service cases | static, operation, fatigue (cycles), impact, misuse: one line each, or "not applicable" with the reason |
| Standard | a test standard for this kind of product, or "none known" |
| Brief trace | each requirement about strength, stiffness or life, mapped to a load case or marked "not modelled, because ..." |
| Model | a load case, a support, a joint, a spring, a row, or "outside the tool" |

Worked inventory for a part that joins other parts: a bracket that holds a motor on a flange and bolts to a frame.

| Interface | Neighbour | Load sources | Service cases | Standard | Model |
|---|---|---|---|---|---|
| Four bolt bosses | Frame | Motor weight (4 kg x 9.81 = 39 N); reaction of the motor torque (6 N m) | Static; torque peak (twice the rated torque); fatigue from vibration; a drop in transport | Product standard, if the user has one | Weight and torque as two load cases; the bosses are `fixed` |
| Motor flange | Motor | Motor weight; bolt preload | Static; misuse (a person pulls on the cable, 50 N) | None known | The flange is a `load` region with a moment; preload is outside the tool |

Brief trace: "the motor must not move more than 0.5 mm" maps to the displacement limit of the torque case. "Must last 10 years" is fatigue: not modelled, because the tool has no fatigue analysis; the user must check it.

Verify and ask. Before the first run, say what you are not sure of: a load you guessed, a support you simplified, a requirement of the brief that you did not model. If an answer changes the part, ask the user. A sub-agent returns the questions to its caller. If you go on with a guess, write it as a guess (origin `inferred` in `record_brief`) and say it again in the final review. A plan item that you wrote and did not do is a fact for the final review.

Buckling. Turn on the check (`buckling: {check: true}`, CONTRACTS.md section 33) when the part has members in compression and one of these: a low volume fraction (below about 0.3), a metal, slender struts or thin walls. When the check gives a low load factor, add the row `buckling.min_factor` (CONTRACTS.md section 39; direct solver only). The `buckling` block is a top-level key of the spec; the helper copies it into the problem YAML. Report the lowest factor of each load case in the final review, or say why the check did not run.

## The printer

The user names the printer; the tools fill the bed and the printer type. When the user says which printer they print on ("I print on a Prusa MK4S"), call `find_printer` with the name, show the user the values of the profile (`derives`, `profile`), and put `printer_model: <id>` in the `print` block of the spec. A key that the spec gives (`bed_mm`, `printer_type`) wins over the profile. With `ambiguous`, ask which printer. With `none`, follow the skill `start-here`, Step 1b ("A printer that is not in the catalog"): look up the maker page, show the values with their source, and call `save_printer` after the user agrees. A value that the profile does not give is `null`: say that it is unknown and do not guess it.

## The filament

The user names the filament; the catalog gives the modulus and the density. When the user says which spool they print with ("PolyLite PETG"), call `find_material` with the name and show the user the values of the profile (`derives`, `profile`). Put `filament: <id>` in the `material` block of the spec: the YAML then gets E (the lower of the XY and Z modulus of the data sheet) and density from the profile, and nu stays the assumed 0.35, because no data sheet gives it. A key that the spec gives (`E`, `nu`, `density`) wins over the profile. With `ambiguous`, ask which filament (a family such as "PLA" gives `ambiguous`). With `none`, follow the skill `start-here`, Step 1b ("A filament that is not in the catalog"): look up the data sheet of the maker, show the values with their source, and call `save_material` after the user agrees. A value that the profile does not give is `null`: say that it is unknown and do not guess it. A sheet that gives no tensile modulus but a bending modulus gives E from it, and `derives` says so. A filament with no modulus at all needs `E` in the block. A density that the profile does not give is the median of its family in the catalog, labelled `assumed`; tell the user.

`printer_materials` shows what the printer can print, as facts (`ok`, `care`, `no`, `unknown`, with the reasons). `topopt check` and `check_problem` give the same answer for the filament of the YAML in the key `material` (`compat`) when the YAML names a printer. These are facts: tell the user, and the user decides. A `no` does not stop a run.

## Preferences

The reply of `environment` has the key `preferences`. When `exists` is true, the file `topopt-preferences.yaml` in the work folder holds what the user said once: `print` (printer model, printer type, bed, build direction), `material` (E, nu, density, filament, name), `filaments` (the filaments the user has), `quality` (the voxel size) and `outputs`. You may use these values in the spec. You decide, and you tell the user which values came from there.

- Leave out the `material` block, the `print` block or `settings.voxel_size`, and `build_regions` takes the missing one from the preferences. It says so in `preferences_facts`: tell the user each fact in one line, for example "material from your preferences: PETG, E 2100 MPa".
- Or write the values in the spec yourself. A block that the spec gives is used as it is: `build_regions` does not add a value of the preferences to it. A partial block (for example `material: {E: 1800}`) keeps the defaults of the problem YAML for the keys that it does not give. Write the whole block if you want the other values of the preferences in it.
- What the user says for this part wins over the preferences. If the user names another bed or another material, put it in the spec, and say that it differs from the preferences. `check_problem` also names a difference between a YAML and the preferences in `preferences_facts`. That is a fact, and the YAML stays as it is.
- After a run, offer the outputs that the preferences list in `outputs`: `renders` (`render_part`: the images that the user gets; your own look in `start-here` is a separate check), `step` (`export_step`) and `verify` (`verify_run`). Give the cost where you know it. The user decides. The final STL and the part card are made in every run.
- When the preferences have `print.printer_model`, `build_regions` writes `print: {printer_model: <id>}` into a spec that has no `print` block, and the fact names the printer and the values that it gives. Tell the user, for example "printer from your preferences: Bambu P1S, bed 256 x 256 mm, build height 256 mm".
- When the preferences have `material.filament`, `build_regions` writes `material: {filament: <id>}` into a spec that has no `material` block, and the fact says what the run takes (E, nu and density, each with its source). Tell the user, for example "filament from your preferences: Polymaker PolyLite PETG, E 1898.7 MPa". If the user has several filaments (`filaments.available`), ask which one is for this part.
- When `exists` is false, the skill `start-here` (Step 1b) asks the user once and `save_preferences` keeps the answers. If the user gives a new value for later runs ("my bed is now 300 mm"), call `save_preferences` with it.

## Step 2. Write the spec file

Write `runs/<name>/spec.yaml`. Use a short lower-case name with no spaces.

```yaml
name: shelf_bracket
regions:
  design:                        # one shape: a list of primitives
    - box: {min: [0, 0, 0], max: [80, 30, 60]}
    - subtract: {sphere: {center: [80, 15, 60], radius: 6}}
  fixed:                         # name -> list of primitives
    wall_plate:
      - box: {center: [-2, 15, 30], size: [4, 36, 66]}
  keep_out:
    screw_hole:                  # today a name with "hole" in it marks a screw hole (path checks)
      - cylinder: {p0: [-3, 15, 50], p1: [6, 15, 50], radius: 2.25}
  load:                          # whole feature, with a force in load_cases
    shelf_edge:
      - capsule: {p0: [70, 5, 6], p1: [70, 25, 6], radius: 3}
  solid:                         # optional: solid, no clamp, no force
    end_cap:
      - box: {min: [75, 0, 0], max: [80, 30, 12]}
# springs: [{region: foot, stiffness: [0, 0, 80]}]   # optional: an elastic support on a load region (N/mm, total). `fixed` may then be empty.
# joints: [{region: sleeve, to: [60, 10, 6], section: {kind: round, d: 4}}]   # optional: a beam from a load region to a fixed point (mm). `fixed` may then be empty.
# fixed_axes: {side_wall: [y]}   # optional: the fixed region `side_wall` fixes only y (a roller, a solid body in the part). Others are full clamps.
load_cases:                      # same format as CONTRACTS.md sections 2 and 26.2 (moment, at, pressures)
  - {name: static, weight: 1.0, forces: [{region: shelf_edge, force: [0, 0, -19.6]}]}
  - {name: knock,  weight: 1.0, forces: [{region: shelf_edge, force: [30, 0, -60]}]}
material: {E: 3500, nu: 0.35, density: 1.24}      # optional, these are the defaults; or {filament: <id>} for a filament of the catalog (see "The filament")
assumptions:                     # optional: short strings, copied into the problem YAML and run.json
  - "Shelf load 2 kg, so 19.6 N down"
  - "Screw holes M4, clearance 2.25 mm radius"
settings: {volume_fraction: 0.25, iterations: 80, voxel_size: auto}   # fraction of FREE voxels; or target_mass_g: 30 (grams the user asks for) in place of volume_fraction: one of the two keys
print: {bed_mm: 180}            # optional: add only the keys that the user gave (CONTRACTS.md section 18)
```

The `print` block takes `printer_model` (the id or the name of a printer; it gives `bed_mm` and `printer_type` when the block does not, and the print report compares the part height with the build height of the printer), `build_direction` (default `[0, 0, 1]`), `bed_mm` (default 220), `printer_type` (`fdm` default, `sla`, `lpbf`, `sls`) and `self_supporting_angle_deg` (default by printer type, 45). It also takes limit keys that `topopt printcheck` and the tool `print_check` compare with the finished STL: `min_wall_mm`, `min_wall_area_pct`, `max_closed_void_mm3`, `max_bodies`, `max_overhang_area_mm2`, `max_support_volume_mm3`, `min_plate_contact_mm2` (CONTRACTS.md section 32). Add a limit key only when the user gives a limit. A key that the user did not give stays out of the spec, so the report and `topopt check` label it "assumed". Do not copy the defaults into the spec: that hides the label.

The `settings` block takes every key of the problem YAML `settings` (the cards of the keys: `topopt levers`, or the resource `topopt://levers`). The builder copies each key into the YAML as you wrote it, and the YAML loader checks the value. For example: `objective` (`mass` makes `volume_fraction` the start density), `solver`, `penal`, `move_limit`, `beta_schedule`, `case_combine`, `mesh_method`, `objective_scale`, `filter_radius`, `min_member_mm`, `filter_solid_neighbours`, `stop_rules`, `perimeter_max_mm2`, `robust`, `eta_eroded`, `max_grid_voxels`, `max_active_voxels`, the `stress_*` keys, the `gpu_*` keys and `part_only_min_film_mm`. Do not add a key to the YAML by hand after the build. Some keys have a rule of their own:

- `voxel_size`, `iterations`, `volume_fraction` and `target_mass_g` are written by the builder from your value: `voxel_size` is the size of the grid it built (a number, `auto`, or `--voxel`), `iterations` defaults to 80, and `volume_fraction` defaults to 0.25. Give one of `volume_fraction` and `target_mass_g`.
- `target_active` and `allow_large_grid` are read by the builder only. The YAML does not have them.
- A key that is not in the list of the YAML stops the build before it writes a file. The message names the key and the known keys ("unknown key 'objectve' in 'settings'. Known keys: ... Options: use a known key, or remove 'objectve'"). Write the key under a known name, or remove it.

The top level works the same way. The builder copies each top-level key of the problem YAML that it does not write itself, as you wrote it, and the YAML loader checks it: `springs`, `joints`, `assumptions`, `mirror`, `print`, `limits`, `buckling`, `interfaces`, `acceptance` and `bore_checks`. A `null` is the same as no key. The builder writes `name`, `regions`, `load_cases`, `settings` and `material` itself; `material` takes `E`, `nu` and `density` only. `fixed_axes` is read by the builder only. A key that is in none of these lists, and a `material` key other than `E`, `nu` and `density`, stops the build before it writes a file. The message names the key and the known keys. The spec has no `measure` or `not_printed` region role (the builder refuses it), so a `limits` row names a load region.

Primitives (every item is one key):

| Key | Parameters |
|---|---|
| `box` | `min` + `max`, or `center` + `size` |
| `cylinder` | `p0`, `p1` (end centres), `radius` |
| `sphere` | `center`, `radius` |
| `capsule` | `p0`, `p1`, `radius` (cylinder with round ends) |
| `bowl` | `center`, `radius` (outer), `wall`, `open` (direction the cup opens to, default `[0,0,1]`, so the shell is below the centre): a hollow hemisphere |
| `prism` | `plane` (`xy`, `xz` or `yz`), `points` (polygon, 3 or more), `extent` ([min, max] along the third axis) |
| `stl` | The user's own STL as a solid: `path`, optional `plug_holes`, `screw_holes`, `ring_mm` (see "Your own STL"). A closed mesh, read as mm. Not the `stl` kind of the `interfaces` block |
| `screw_ring` | The frozen ring round one screw hole of an STL, for `fixed`, `load` or `solid`: `path`, `hole`, optional `ring_mm` |
| `subtract` | One primitive to cut out of the items before it |

Items in one region join (union). Each `fixed`, `load`, `solid` and `keep_out` name becomes one STL.
Do not give a region the name of a primitive (for example `box`, `stl` or `screw_ring`). Use a clear name such as `ladle_bowl`.
Each `load` name is the name that `forces[].region` uses.
Use `voxel_size: auto` unless you have a reason. Section "Voxel size" explains the rule.

### The `mirror` key

`mirror` is an optional top-level key of the spec. The design is then mirror-symmetric about up to 3 planes. The builder copies the list into the problem YAML as it is. Each plane is `{normal: x|y|z, at_mm: <mm>}`, with an optional `align: auto|face|centre`. The planes are axis-aligned and have distinct axes.

```yaml
mirror:
  - {normal: y, at_mm: 0.0}      # the part is symmetric about the plane y = 0
```

Facts and options:

- The engine moves the grid origin along the plane axis by less than one voxel, so that the plane is a grid plane. It tries the plane on a voxel face and through the voxel centres, and it keeps the parity with fewer label mismatches (voxels whose mirror image has another label). `align: face` or `align: centre` sets the parity. `build_regions`, `topopt check` and `run.json` give both counts.
- A design voxel with no design partner is an orphan. Its mirror image is outside the grid, or it is empty, keep-out, fixed or load. An orphan stays a free variable. The tools give the count, the share and the cause. Regions that are symmetric about the plane give 0 orphans. When all or most design voxels are orphans (for example when `at_mm` is far from the part), the tools say that the mirror has no or little effect, give the extent of the grid along the plane axis, and name the options: check `at_mm`, move the plane, make the regions symmetric about it, or remove the key.
- The loads are not mirrored. For each load case, the tools give `|f - R f| / |f|` about each plane (0 is symmetric, 2 is antisymmetric). A case that is not symmetric, for example a `pull_off` case with a side force, gives the best symmetric design for that load and not the best design. Options: leave the plane out, make the load symmetric, or keep it and tell the user.
- A plane that is not axis-aligned, a repeated axis and more than 3 planes are refused with a one-line message. The option is to re-orient the part (the regions and the loads) so that the plane is axis-aligned.
- The mirror gives no speed-up. The finite element model and the optimizer keep the full size. The run time is the same.

### The `assumptions` key

`assumptions` is a list of short strings. The builder copies it into the problem YAML. The engine ignores it. `run.json` keeps it.
It does not block a run. It is a fact list for the user.
Write one line for each of these:

- Critical dimensions: the sizes that the part must keep (a hole spacing, a lip height, a gap).
- Scale of a sketch: the dimension or the object that gave the scale, and each size taken from the proportions.
- Loads: the source of each force (for example "0.2 kg x 9.81 = 1.96 N down") and the abuse case.
- Keep-outs: what stays empty and why.
- Minimum printable dimension: the thinnest member you accept. The tools give a fact: members thinner than about one voxel or one filter radius (the larger) are not resolved.
  `settings.min_member_mm` (optional, in the spec `settings`) asks for a minimum member. The engine derives the filter radius from it by one rule (`filter_radius = 1.877 * min_member_mm`: a wall of the requested thickness just survives the filter). It is an estimate, not a guarantee: in one calibration case the 5th percentile of the interior thickness was 8 to 22 percent below the request, a skin at a free face of the design space was about 1 voxel, and a thin rod of the request size is removed. The radius is large, so `topopt check` also prints the filter size and the filter memory (the matrix is part of the memory estimate), and a `min feature:` line that says in mm what the radius keeps (walls and rods, interior and free faces). An explicit `filter_radius` wins. `topopt check` prints the request, the derived radius and the facts with options (for example a request below 2 voxels).
  `settings.filter_solid_neighbours` (optional): `density_one` (default) lets the filter see fixed, load and solid voxels as density 1. `ignore` leaves them out, so a design voxel next to them is not pulled to solid. `topopt check` prints a `solid pull:` line with the number of design voxels that the default rule pulls (a filtered density above 0.1 from solid neighbours alone) and the largest pull.
  `settings.robust` (optional): `off` (default), `eroded` or `wall`. Eroded mode keeps the volume and the printed shape on the nominal design and computes the stiffness on a thinner (eroded) design, so a member that is too thin for the filter radius gets no credit. `settings.eta_eroded` (optional, between 0.5 and 1, default 0.55) sets the erosion threshold. Write both keys in the spec `settings`; the builder copies them into the problem YAML. In an eroded run `compliance_final` and the stiffness in `metrics.json` are the printed (nominal) design, and the history curve is the eroded design. `topopt check` prints the `robust:` lines and the facts with options (the radius rule is for the default threshold; the last beta; a solver without a parity test). `wall` is the eroded form aimed at the print wall: it takes the wall w from `print.min_wall_mm` (a YAML error with options without it), sets `eta_eroded` to 0.7 unless you write it, derives the filter radius R = max(w / L(eta_e), 1.5 voxels) (R = 1.52 mm at w 1.2 mm; a written `filter_radius` wins and `min_member_mm` is only reported), and pads the density filter with void so that a wall at a face of the design space is as thick as an inner wall. On the hook example (voxel 0.5 mm, wall 1.2 mm) the share of the surface area under 1.2 mm fell from 4.145 to 0.000 percent for +7.9 percent compliance and 1.07 times the run time. When `print.min_wall_mm` is set and `robust` is not `wall`, `topopt check` gives a fact with these numbers; the mode is your choice, and it is not on by itself.

### The `perimeter_max_mm2` setting

`settings.perimeter_max_mm2` is optional (in the spec `settings`; a number above 0, in mm2). It adds one constraint row: the surface area `P` of the part stays at or below the bound. Use it when the user gives a limit on the surface (for example a coating area, or a print time that follows the surface). Without the key the run is the same as before. The builder copies the key into the problem YAML.

- `P` is the area of the voxel faces between the part and the void (a staircase surface; in one measured problem about 1.24 times the area of `final.stl`). Solid regions (fixed, load, solid, spring) count as part. Faces to keep-out voxels and to the outside of the grid count as void. The unit is mm2.
- `topopt check` prints a `perimeter:` line with the bound, the fixed part (solid voxels next to void), the smallest possible `P` (a minimum cut of the design space) and the bound over the smallest `P`. `check_problem` gives the same numbers under the key `perimeter`. The ratio is a fact. A bound near the smallest `P` leaves little material to spend, so the stiffness drops or another row (volume, compliance) stays unmet.
- The engine refuses a bound below the smallest `P`: no design with density 0 or 1 can meet it (a gray design can be a little below it, only near void). The message gives the bound, the fixed part, the smallest `P` (rounded up to 0.1 mm2, so the engine accepts the shown value as a bound) and the options: raise the bound, reduce the solid regions, make the design space bigger. The filter can keep `P` above the smallest value, so the smallest `P` is a floor and not a promise.
- A mass objective keeps its volume ceiling: the row limits the surface and not the material (in the measured problems the empty design had the smallest `P`). The row has price 1000 and a compliance limit has price 100, so a tight bound can leave a compliance limit not met; the run log says so and gives the option to raise the bound.
- `run.json` (`perimeter`), `history.jsonl`, `metrics.json` (`constraints`, `rows`) and `run.log` give `P` at the start and at the end, and say if the bound is met.

## Your own STL

Use this when the user has an STL of the part. It takes the place of Step 1. Step 2 to Step 5 stay. CONTRACTS.md sections 45, 45.1 and 45.2 have the detail.

1. Put the file in the work folder, or use its absolute path. Call `find_features` with the path. It reads the file and changes nothing.
2. Check the size. An STL has no unit: topopt reads every file as mm. The reply has `mesh.size_mm`. If the size looks like metres or inches (for example a hand-size part that is 0.2 across), ask the user. There is no scale key: the user exports the file again in mm. A mesh that is not closed, or has inward faces, is an error with the cause. The builder does not repair a mesh.
3. Show the holes and the big flat faces in a short table, one line each (equal holes in one line): `H1` d 4 mm, 4 mm long, through, at (30, 10, 0); `F1` 760 mm2, normal (0, 0, -1). The ids are stable for one file.
4. Ask only what changes the problem. Do not ask what the file shows.
   - Which holes take a screw? Each gets a frozen ring (`screw_holes`).
   - Which holes should be filled, so that the optimizer may use that material? (`plug_holes`.) A hole that you do not name stays as the file has it.
   - What is mounted: which holes (a `screw_ring` item in `fixed`) or which face (a `box` on that face)? Where does the load act, and how large? The load has no default (top of this file).
   - May the part grow? A growth box is one more item in the design region. Ask how far.
5. Write the spec (below), call `build_regions` and `check_problem`, and fix errors as in Step 4.
6. Say the assumptions (Step 5). Add: the size that you read in mm, the thickness t of each ring and why (the report has one line for each screw hole), the holes that you plugged, and the growth box.

```yaml
name: bracket
regions:
  design:
    - stl: {path: bracket.stl, screw_holes: [H2, H3], plug_holes: [H4]}    # the part: a ring round H2 and H3, H4 filled
    - box: {min: [0, 0, 10], max: [60, 30, 18]}                            # optional: a growth box
  fixed:
    wall:
      - screw_ring: {path: bracket.stl, hole: H2}                          # screwed to the wall at H2
  load:
    pull:
      - screw_ring: {path: bracket.stl, hole: H3}                          # the force acts on the ring of H3
load_cases:
  - {name: static, weight: 1.0, forces: [{region: pull, force: [0, 0, -30]}]}
settings: {volume_fraction: 0.4, iterations: 80, voxel_size: auto}
```

- `path`: a relative path starts at the work folder for the tool `build_regions`, and at the folder of the spec file for the script.
- `plug_holes`: hole ids, or `all`. The plug fills the wall of the hole. A chamfer at the mouth and a drill point stay.
- `screw_holes`: hole ids, or `all`. Each hole gets a `tube` interface (CONTRACTS.md section 23): a solid ring that the optimizer does not remove, and the hole stays free and round. No support and no force act on this ring: for that, use a `screw_ring` item in `fixed` or `load`. A hole is plugged or it takes a screw, not both. A notch and a cavity take no ring.
- The thickness t of a ring: 0.5 x the hole diameter, at least 2 voxels (the default). `ring_mm` sets it: a number for all rings of the item, or `{H2: 2.5}` for one hole. It is a method setting: set it when the user asks for a thicker or thinner ring, and say so. A user who gives a wall thickness gives a requirement: use it as `ring_mm` and say it.
- `screw_ring`: the same ring as a region item. Give it the same `ring_mm` as the `screw_holes` of that hole (a `WARNING:` says when the two differ).
- Another exact interface (a bore, a boss, a face) is the spec key `interfaces`: a list as in CONTRACTS.md section 23. The builder copies it into the YAML, before the rings. `references/patterns.md` has the rules of the `interfaces` block, also for these rings: what to set, and what to read in `check_problem` and `run_result`.
- A ring that leaves the part (a hole near an edge) gets a `note:` with the volume: the exact union adds that material outside the part. A keep-out region wins over a ring (`WARNING:`).

## Step 3. Build and check

The MCP tool `build_regions` does this from a spec given as data (a dict, or the YAML text). The same code runs as a script:

```
CLAUDE_PLUGIN_DATA="${CLAUDE_PLUGIN_DATA}" sh "${CLAUDE_PLUGIN_ROOT}/scripts/topopt-env.sh" python "${CLAUDE_PLUGIN_ROOT}/.claude/skills/part-to-topopt/build_regions.py" runs/<name>/spec.yaml --out runs/<name>
```

A relative `path` of an `stl` or `screw_ring` item starts at the work folder for the tool, and at the folder of the spec file for the script.
The script is a thin wrapper for `topopt/regions_builder.py`. (In a source checkout: `.venv/bin/python .claude/skills/part-to-topopt/build_regions.py runs/<name>/spec.yaml --out runs/<name>`.)
The command writes the STL files and `runs/<name>/<name>.yaml`. Then it runs the topopt input layer.
It prints the grid, the voxel counts for each region, the active voxel count, the volume basis (free and forced-solid voxels, the target volume and mass), the free dofs, two time estimates (the model value, and the model value scaled by the fill of the part), and a memory estimate line (GiB against the RAM of the machine).
Use `--check` to test a spec and write nothing. Use `--voxel H` to force a voxel size.
The last line is `RESULT: OK` or `RESULT: FAILED`.
The MCP tool `build_regions` and `topopt check` also give the `derived` numbers: the force of each load case in N (vector and size), the design block in mm3, the target mass in g, the mass of the full block in g, the voxel size and the minimum feature size.
State these numbers with your assumptions. Ask the user before the run when an answer changes the part (the top of this file has the rule).
After `build_regions`, call `record_brief` with the brief and your assumptions. The case file keeps them beside the YAML with the hashes of its input files, and `run_topopt` says when an input file changed since (CONTRACTS.md section 30).

Also run the official check once: the MCP tool `check_problem` with `runs/<name>/<name>.yaml` (the CLI gives the same data: `topopt check runs/<name>/<name>.yaml`).

The check prints a `solver:` line. On Linux x86_64 without `pypardiso` (`--json` gives the same data as JSON) it also prints a `recommendation:` line. The facts in it:

- The run uses the AMG solver. PARDISO is faster. The factor depends on the problem (`speedup_factor` in the line). The line gives the time estimate for this problem with and without PARDISO (a rough model, the same one as for the Mac).
- `install_command` installs the optional extra `pardiso` into the environment that runs topopt.
- `pypardiso` pulls Intel packages (about 456 MB in a fresh venv). `mkl`, `tbb` and `tcmlib` have the Intel Simplified Software License. `intel-openmp` and `intel-cmplr-lib-ur` have the Intel EULA for Developer Tools, which has more limits (for example, no use in a hosted service). `umf` is Apache-2.0. The `mkl` version is not pinned. The `license_note` field has the same text. Tell the user this before you install.
- If the action is `fix_pypardiso_load`, pypardiso is already installed but does not load. The `error` field has the cause (often the MKL library). A new install often does not fix a load error. The `error` field gives the cause.

If the check prints this line, offer the install to the user, with the speed-up as the reason. When the user agrees, run `install_command`. If the user declines, the run goes on with AMG.

## Step 4. Fix errors

Read the message. Change the spec. Run again. If the same problem stays, tell the user what you tried.
The auto retry grows the voxel only for a voxel-count guard failure. A bigger voxel does not fix other errors (for example `separate pieces`). Fix the spec.
Some messages are `WARNING:` lines. They are facts and do not stop the build. Decide if you act on them.

A guard stop is not an error in the spec. The guards are `max_grid_voxels` and `max_active_voxels` (the engine) and the RAM block (`allow_large_grid`). They are method settings: you decide with the facts and report what, why and the price. An override uses more memory and time. The risk: a run above the RAM can freeze the machine (a 0.5 mm whole-domain grid crashed a 16 GB Mac).

1. Read the numbers of the message, the memory estimate (`check_problem.memory_estimate`, or the line that the helper prints) and the RAM of the machine.
2. The options of the message have names: the override (for example `settings.max_grid_voxels`) and the voxel size that fits. Take one.
3. Tell the user what you took, why and the price (memory, time, a coarser voxel). If the estimate is above the RAM, say the risk with it.
4. If the user gave the voxel size, it is a requirement (its origin in `record_brief` is `user`): a larger voxel loosens it, so ask the user first and bring these facts.

| Message | Fix |
|---|---|
| `WARNING: region '...' is small or thin` | The region has fewer than 30 voxels, or it is less than 3 voxels across. The optimizer can cut it off. Make it the whole solid feature, or use a smaller voxel size. A small region is allowed. |
| `WARNING: force is lost: ...` | Load nodes touch fixed nodes. The solver drops their force. The warning gives the lost N and percent (`lost_force` in the check data, per load case and per region). Move the load region off the fixed region (keep a gap of one voxel or more), shrink the overlap, or accept it. Tell the user the number. |
| `load region '...' has no force` | Add a force in `load_cases`, or move the region to `solid`. |
| `min_member_mm ... voxels` (a fact, with options) | The request is below 2 voxels at this voxel size. Use a smaller voxel size (3 voxels across), a bigger request, or accept it. |
| `settings.filter_radius ... is set, so it is used` (a fact) | Both `filter_radius` and `min_member_mm` are set. The explicit radius wins. Remove one of them, or keep both. |
| `settings.perimeter_max_mm2 is ... no design with density 0 or 1 can have a perimeter below ...` | The bound is below the smallest possible `P` (a minimum cut). Raise the bound to the value in the message or above (the value is rounded up to 0.1 mm2, so the engine accepts it), reduce the solid regions, or make the design space bigger. |
| `the supports leave ... rigid-body modes free` | The roller axes do not block every motion of the part. The message names the free motions. Add axes, or add a support that blocks them. |
| `... cannot carry the moment` | The nodes of the load region lie on a line along the moment axis. Make the region wider across that axis, or use a force pair. |
| `pressure on load region '...': no face ...` | No face of the region looks into free space or a keep-out region. Shape the region so that the loaded face is free. |
| `WARNING: forced-solid voxels are ... percent` | Fixed, load and solid regions are more than 60 percent of the model. Little is left to optimize. Make them smaller, or make the design space bigger. |
| `ERROR: the active voxels make N separate pieces ... (`topopt run` stops with this error)` | A piece has no support: no fixed voxel (and no spring or joint region) in it, because a region shares no face with the rest of the part. The run would stop. Move the region, or make the design space bigger so they touch. |
| `WARNING: the active voxels make N separate pieces ... each holds a fixed voxel` | A fact. Every piece holds a support, so each piece is fixed and the run works. A load on one piece does not reach the other pieces. If you want one part, make the pieces touch. |
| `note: region planes off the voxel grid: N of M change the part ...` (also in `warnings` and `check.region_planes`) | A fact, and most parts have it. A voxel belongs to a region when its centre is inside the region, so a flat face whose plane is not on a voxel face moves to the nearest voxel face. The note gives the voxel size, the grid origin (the min corner of all regions: it moves when such a region moves), and one item for each plane that changes the part: the region, the plane, and how much larger or smaller than your shape the region is. It matters where two regions should meet at one plane (a wall and the part, a keep-out and a surface). Move or resize the region so that the plane is on a voxel face, choose a voxel size that divides the distance from the grid origin, or extend a keep-out region past the part. `check.region_planes` has all planes, and `changes_part` is false for a plane under a region of higher priority or outside the part. Nothing is snapped for you. |
| `WARNING: no fixed voxel touches a design voxel` | A fact. The fixed region shares no face with the design space. The message gives the distance and says if every design voxel is in a piece that holds a support (then the part is held through load or solid voxels, and the run is not stopped). If you want the plate to touch the design, overlap them by 1 or 2 mm. If the path through a solid body is what you want, keep it. |
| `no voxel left` | A keep-out region covers a whole region. Shrink the keep-out. |
| `WARNING: the estimate is above your time budget of ... s (settings.stop_rules.time_budget_s)` | A fact, and only for a YAML that sets `settings.stop_rules.time_budget_s`: the budget is the user's own. It is a stop rule, not a note, and the message says what it does in this YAML. With the stop rules on (a `stop_rules` mapping turns them on), the run stops at the first pass after the budget (rule `time_budget`), and the part is the design of that pass. With `enabled: false` the budget is set but the rules are off, so the run does not stop at it. The options: a larger voxel (the voxel size that the message names), another budget (ask the user), or all passes with the stop rules off (`enabled: false` in the YAML, or `run_topopt(stop_rules=false)` for one run). |
| `grid would have ... voxels at voxel_size ..., the limit is ...` or `model has ... active voxels` | The engine guard (`max_grid_voxels`, `max_active_voxels`). Options: a larger `voxel_size`, or a higher setting in `settings` of the spec if the machine has the memory. Decide with the facts (see above). With `auto` the helper grows the voxel 8 percent and tries again, up to 3 times. |
| `ERROR: the memory estimate ... is above the RAM` | The estimate is above the RAM of this machine. A run can crash it. Options: a larger voxel, or `allow_large_grid: true` in `settings` to build anyway. Decide with the facts and say the risk (see above). |
| `ERROR: the grid alone needs about ... GiB` | The grid is too big for the RAM before any solve. Options: a larger voxel, or `allow_large_grid: true` (and maybe `max_grid_voxels`) in `settings`. Decide with the facts and say the risk (see above). |
| `SPEC ERROR ...` | The message names the region and the item, or the key path of the value (for example `load_cases[0].forces[1] must be a mapping, for example {region: tip, force: [0, 0, -2]}, got int 5`). It says what the spec has and what it expects. Fix that key. |
| `not a closed solid` | A subtract removed everything, or a primitive has zero size. For an `stl` item: the file is not a closed solid (the message counts the open edges), its faces point inward, or a plug left the mesh open. The builder does not repair a mesh: ask the user for a closed file. |
| `...screw_holes: ... is also in plug_holes`, `... not a closed ring`, `... a cavity` | A hole is plugged or it takes a screw. A notch (an arc) and a cavity take no ring: leave them out of `screw_holes` (a list, not `all`). |
| `...: a screw ring is a frozen shape` | A ring is for the part. Put the `stl` item in `design`, `fixed`, `load` or `solid`, and a `screw_ring` item in `fixed`, `load` or `solid`; not in `keep_out` or `subtract`. |
| `note: ... the ring of H1 ... outside the part` | A fact. The hole is near an edge or a face, so part of the ring is outside your part, and the exact union adds that material. Tell the user the volume, or give a thinner ring with `ring_mm`. |
| `WARNING: ... the ring of H1 ... overlaps the keep_out region` | The keep-out region wins: the ring is cut there. Move the keep-out region, or give a thinner ring. |
| `WARNING: ... ring_mm: ... under 2 voxels` | The ring is less than 2 voxels thick at this voxel size. Use a smaller voxel, a thicker ring, or accept it. |

## Voxel size (time and memory)

The helper has no time limit of its own. The only time budget is the one of the user, `settings.stop_rules.time_budget_s` in the YAML. The helper reports the time as an estimate and you choose. Tell the user the estimate in plain words, and say that it is an estimate: a run can take less or more time. Where the reply has a band, give the band.
The cost depends on the number of free dofs, which is about 3 x the number of active voxels.
The shape of the part also matters. A compact part costs more than a slender part, because the solver needs more fill-in for it.

| Run (measured 2026-10-11) | Active | Free dofs | Estimate | Real | Real / estimate |
|---|---|---|---|---|---|
| T-shirt hook (compact, fill 0.36) | 39,944 | 115,836 | 145 s | 148 s | 1.0 |
| Ladle (slender, fill 0.06 to 0.09) | 33,407 | 114,780 | 143 s | 38 s | 0.27 |

"Fill" is active voxels divided by grid voxels. The model estimate is the number for a compact part: the hook ran 1.0 times it and the slender ladle ran 0.27 times it (a factor of about 3.7). Both are runs on Apple silicon. On a shared Linux host, experiment E1 (a run-time study) found finished PARDISO runs with 50,000 free dofs and more at a median of 2.19 times the model total, so the model is not an upper bound. `check_problem` gives the measured "up to" value in `derived.time_estimate_up_to_s` and `derived.memory_estimate_up_to_gib` (facts only; null on macOS and below 10,000 free dofs, where E1 found the time estimate high).
The time model is calibrated on Apple silicon with the Accelerate Cholesky solver. For `amg`, the check scales the time per iteration with a rough factor (CONTRACTS.md section 4); use its estimate for the solver that runs.

Facts and options:

- The helper prints the model value and the model value scaled by the fill of the part (an estimate each). The scale goes from 0.27 (fill 0.085 or less) to 1.0 (fill 0.30 or more).
- When the YAML sets `settings.stop_rules.time_budget_s` and the estimate is above it, the helper prints a warning with a voxel size that would fit. The helper does not change the voxel and does not stop the build. The run is different: `time_budget_s` is a stop rule. A `stop_rules` mapping turns the stop rules on (the rule `converged` too), and the run stops at the first pass that ends after the budget. The part is the design of that pass. The clock of the rule is the run time at the end of a pass: it counts the setup and the passes, not the post step, but the estimate counts all three. So a run with an estimate just above the budget can still make all its passes. With `enabled: false` the run does not stop at the budget. Tell the user which case it is, and give the options of the table (a larger voxel, another budget, or all passes with the stop rules off). Without a budget of the user there is no time warning: the estimate is a fact.
- Voxel size against time: free dofs grow as 1 / voxel^3, and the time per iteration grows about as dofs^2.4. A voxel 10 percent bigger gives about 25 percent fewer dofs and about half the time per iteration.
- Default aim of `auto`: `voxel = cbrt(active_volume_mm3 / target_active)`, rounded up to 0.05 mm. The active volume is design + fixed + load + solid, minus keep-out.
  The default `target_active` aims at an estimate of about 120 s for 80 iterations (the model scaled by the fill; an aim, not a limit), at most 41,000 active voxels. A compact part gets about 36,000 (estimate about 120 s). A slender part gets 41,000 (the scaled estimate is 70 to 85 s, and the model value is 190 to 230 s).
- To aim elsewhere, set `settings.target_active` (with `voxel_size: auto`) or give a `voxel_size` number. The helper uses your value as it is.
- Memory: the helper prints a rough memory estimate (grid arrays, the Cholesky factor and the filter) and the RAM of the machine. A 0.5 mm whole-domain grid crashed a 16 GB Mac. The helper blocks only when the estimate is above the RAM. `settings.allow_large_grid: true` (a real boolean) overrides this block. Read the estimate against the RAM first (Step 4).
- On Linux, the RAM that the helper shows is the host RAM or the cgroup (container) memory limit, whichever is lower (`topopt.cli.physical_ram_bytes`). It is the size of the limit, not the free memory. Other processes in the container can use a part of it.
- The memory model is fitted to the peak RSS of measured Cholesky runs. The AMG backend uses less.
- The engine has its own guards: `max_grid_voxels` (5,000,000 by default) and `max_active_voxels` (100,000 by default). To go beyond, set them in `settings` of the spec (Step 4 has the facts to read first). The error message names the setting.
- A thin feature needs about 3 voxels across to survive. If it is thinner, make it thicker in the spec, use a smaller voxel, or accept a rough result.
- A thin feature can need a finer voxel than the run has. When it does, `check_problem` has the key `fine_pass` (`recommend` is true; `topopt check` prints one line): the voxel that the part needs (`voxel_mm`, one value: the thickness is twice the radius of the largest sphere inside the region, over 2.8, rounded down to 0.05 mm; the fine pass tests it on its own grid and goes down in steps of 0.05 mm if a region has too few voxels across there), the feature that sets it (`limiting`: 2.8 voxels across a thin region, 3 across `settings.min_member_mm`), and estimates of the time and the memory of a fine pass. The option is a fine pass. It runs the same problem on the finer voxel and starts from the design of a finished run: `run_topopt(yaml_path, out_dir=NEW, fine_from=PARENT)`, or `run_topopt(yaml_path, out_dir, fine_pass=true)` for the first run and the fine pass in one call (the fine pass goes to `<out_dir>-fine`, and the reply names both folders). The same YAML works. The voxel of the fine pass is the voxel of the part, or `fine_voxel`. The fine pass ends when its loss has settled (the change of the objective between passes is at most `fine_tol`, or `settings.fine_pass.objective_change` in the YAML, default 0.001), or at the maximum `fine_passes` (default: the pass count of the parent run); `stop_rules=false` turns every early stop off. `check_problem.fine_pass.guards` says if the fine grid fails `max_active_voxels`, `max_grid_voxels` or the memory, with the finest voxel that passes (`fine_voxel`). In two measured parts the result came within 0.5 % and 2.1 % of the compliance of a run from zero at the fine voxel, with fewer passes at the fine voxel. The first run is extra time, and every time and memory number is an estimate. Give the user the numbers and the choice. A fine pass starts only when you call it.

## Patterns (read on demand)

Read `references/patterns.md` in the folder of this skill when the part is a bowl, cup, scoop, ladle, dish or container; a handle or a grip; or has a surface that a neighbour touches or that needs an exact size (a socket, a seat, a press fit, a sealing face, an end trim, a bore or a boss with machining stock). It has three patterns, and the `interfaces` block of the spec.

## Pre-flight checklist (do this in your head before the helper)

1. Does each load region cover the whole feature, with about 3 voxels across at the planned voxel size?
2. Does each load region have a force in `load_cases`? Does a no-force body use `solid`?
3. Do fixed, load, solid and design shapes touch or overlap each other? Is there one body only?
4. Are the forced-solid regions (fixed + load + solid) less than about 60 percent of the active volume? (Above this the helper warns.)
5. Is the volume fraction the fraction of the FREE voxels, and is it between 0.15 and 0.4?
6. Today the screw-hole check finds holes by "hole" in the keep-out name. Does each screw hole have it, and no other keep-out?
7. Is the part smaller than the bed (`print.bed_mm`, or the bed of `print.printer_model`; the default is 220 mm) in two directions, and lower than the build height of the printer, if it has one? Are all numbers in mm and N?
8. Does the abuse case have its own direction and a weight?
9. Does the part work with the objects around it (a lip for a hanging object, clearance for a nested object)?
10. Does each surface that a neighbour touches (a socket, a seat, a press fit) have `full_surface: true`, and does each end face have `radius_mm`? (`references/patterns.md`)
11. For an STL of the user: did you read the size in mm, show the holes, and ask which holes take a screw, which are plugged and what is mounted? Does the summary say the ring thickness and the plugged holes?

## Step 5. Print the summary

Write a short summary for the audience in plain words. Use at most 10 lines.

- What the part is, and its size in mm.
- How it is held (fixed regions, in words).
- The loads: the force in N, its direction, and why (for example "0.2 kg x 9.81 = 1.96 N down"). Name the abuse case. For a moment, a remote force or a pressure, give the value and where it acts. The stiffness figure of such a case is `n/a`: say so.
- What stays empty (keep-out regions) and why.
- Material and the volume fraction. If a value came from the preferences file, say so.
- The assumptions you made and the open questions (one line each). Use the `assumptions` list and the `derived` numbers.
- The time estimate in plain words, and any WARNING line. Give the voxel size, the active voxel count and the memory estimate when the user asks.
- The YAML path, and how the run starts: the MCP tool `run_topopt` with that path
  (the CLI: `topopt run runs/<name>/<name>.yaml --out runs/<name>_run --dashboard`).
  `run_topopt` returns the dashboard URL. Give it to the user.

Write the summary in the user's language, in plain words, with short sentences.

## After the first finished run: offer the stress check

When the first run of a part has finished, you can offer the stress check once. It answers "will it break under these loads?" with a second model of the finished part (`verify_run`: another mesh, quadratic tets, one line for each load case). Say what it costs: it needs the extra `verify`, and `verify_run` with `estimate_only` (the default) gives the memory and the time first and starts nothing. Say that a yes or no needs an allowable stress, and the user gives it (`allowable_mpa`). Without one, the check gives the stress values only, and `topopt://materials/reference` has typical data sheet values. This is an offer: the user decides, and a "no" ends it for this part. When the job is done, `verify_status` has `plain`: one plain line for each load case, with the source of the allowable and the margin. Give these lines to the user as they are. A margin is a ratio of two numbers.

## Worked example

`examples/hook_spec.yaml` in the folder of this skill rebuilds the T-shirt hook. It uses a `prism` for the design space and two cylinders for each screw hole.

## Blind test record (read on demand)

`references/blind-test.md` in the folder of this skill is the record of a blind test of this skill (2026-10-12, a gravy ladle). Read it only when you check how this skill did in a test.
