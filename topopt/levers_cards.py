"""The table of lever cards (CONTRACTS.md section 41): one card for each YAML key and each engine tool.

This file is data. A row has no logic. `topopt/levers.py` reads the table and makes the reference text (the MCP resource
`topopt://levers` and the command `topopt levers`). `tests/test_levers.py` checks the table against the keys that the loader
accepts (`topopt.config.KEY_BLOCKS`) and against the CLI subcommands and the MCP tools.

How to add a card: a new YAML key gets a card with `group="yaml"` and its key path (`settings.penal`, `load_cases[].weight`). A key that
only repeats the effect of another key may be listed in `also` of that card (for example the `<number>_source` keys). A new engine tool
gets a card with `group="tool"` and the name of the MCP tool or of the CLI subcommand. Write only what the code, CONTRACTS.md or a
measured run says. A field that no source fills says "not measured" or "no fact yet": it is not a guess.

Fields:
- `kind`: `requirement` (a user requirement), `method` (a method setting) or `both` (it follows its origin in the brief record).
- `layer`: 1 a global setting (a new run), 2 continue or branch a run, 3 the post step only, 4 a local edit of the finished part.
- `symptoms`: words the user says, and the fact in the result that detects each (a key of `metrics.json`, `run_result` or a tool reply).
- `evidence`: measured rows (change, effect, price, part class, run ids). `confidence` is `measured` when there is a row with run ids,
  `code` when the card comes from the code or CONTRACTS.md only, `untested` when nobody ran the lever.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

KINDS = ("requirement", "method", "both")
LAYERS = (1, 2, 3, 4)
CONFIDENCE = ("measured", "code", "untested")
GROUPS = ("yaml", "tool")
# The documents that a fact path starts from (CONTRACTS.md section 41): `metrics` is result/metrics.json (what `run_result` reads), `run` is run.json,
# `history` is a record of history.jsonl, `check_problem` is the reply of the tool, `print_check` the reply of the tool, `verify` the result that
# `verify_status` gives, `ratchet` a record of the ratchet log that `ratchet_status` gives.
FACT_SOURCES = ("metrics", "run", "history", "check_problem", "print_check", "verify", "ratchet")


@dataclass(frozen=True)
class Symptom:
    """A thing the user says (`words`, comma-separated) and the fact in the result that detects it (`detect`, free text).

    `facts` is the machine-readable form of `detect`: the exact key paths of the detecting facts, in the one path form of CONTRACTS.md
    section 41: `<source>.<key path>`, with `[]` for every entry of a list and `<name>` for any key of a mapping (a case, a region).
    The sources are in `FACT_SOURCES`. A bare tool name (a card of a tool, or a tool in `READ_ONLY_TOOLS`) says that the tool gives the
    fact. `facts` is empty where no fact exists (`detect` then says "no fact yet" or names the error text).

    `when` (optional text) says which value of the facts makes the symptom true: a flag that is set, a list that is not empty, a
    value that is null or not null. It names the key of a fact. A symptom with no `when` is read from `detect` alone. A `when` needs `facts`.
    """
    words: str
    detect: str
    facts: tuple[str, ...] = ()
    when: str = ""


@dataclass(frozen=True)
class Evidence:
    """One measured row: what was changed, what the result did, what it cost, on which part class, in which runs."""
    change: str
    effect: str
    price: str
    part_class: str
    runs: str


@dataclass(frozen=True)
class Card:
    key: str                                   # a key path (`settings.penal`), or an engine tool (`verify_run`, `run_topopt:resume`)
    group: str                                 # `yaml` or `tool`
    kind: str                                  # requirement | method | both
    layer: int                                 # 1 to 4
    effect: str                                # one sentence: what it does
    confidence: str                            # measured | code | untested
    values: str = ""                           # type, unit, default, range or typical values, interactions
    cost: str = ""                             # the cost of one try, if it is not the cost of the layer
    call: str = ""                             # how to use it, if it is not the call of the layer
    symptoms: tuple[Symptom, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    also: tuple[str, ...] = ()                 # more key paths that this card covers (the same effect)
    kind_note: str = ""                        # for `both`: when it is which
    price_note: str = ""                       # how to read the price of this lever: which measure to trust, the noise


WALL_HOOK = "hook 0.5 mm (wall rule)"   # the part class of the step 1 measurement of the wall rule (#328)
F19_HOOK, F19_BRACKET, F19_LADLE = "hook 80 passes (F1-9)", "bracket (F1-9)", "ladle (F1-9)"   # part classes of the F1-9 measurement: the wall rule against the rod rule
# The part classes of the evidence rows (`Evidence.part_class` is a key of this dict). A row is a fact about this class of part only.
PART_CLASSES = {
    "phone stand": "a small desk stand: voxel 0.7 mm, 80 passes, real GPU (gpu-mg), code of main f545ee7. The stiffness values are field values "
                   "of the optimizer (the printed part was not analysed again). FREF is the stand without `min_member_mm`. Run-to-run noise for "
                   "a tiny input change is about 0.1 %, so a difference of 1 % or more is above the noise. The run ids (E1 to E7, F0 to F7) are the "
                   "runs of the F-series map of the phone stand (not in this repository). Weight, tap and side are the mass and the stiffness "
                   "under the tap load and the side load.",
    "hook": "examples/hook/hook_short.yaml of this repository: voxel 2.0 mm, 40 passes, CPU, solver pardiso (auto), code of origin/main 4f6d39d. "
            "Four kinds of run ids. `E1/<id>` are the 60 full runs of the E1 series (the run folders are not in this repository): "
            "the base run B0 took 24 s, and the price of a key is the ratio against B0. `P7-1/l3/<name>` are the resume runs of PR-1 of P7 "
            "(a finished run, then `--resume --allow-changed-inputs`); their base run took 27.0 s. `P7-1/r3/<name>` are the "
            "continuations of round 3 of PR-1 (`--continue-from` a copy of B0 with `--more-iterations`, CPU, 4 threads, solver pardiso); "
            "the host was loaded by other jobs, so a wall time is good to a factor of about 1.5, and a price in these rows is "
            "the ratio against B0 again. `E2/<id>` are the runs of the E2 series (same code, CPU, 4 threads, one run at a time, "
            "the host was loaded by other jobs, so a wall time is a range): the base B0 again (its `final.stl` is "
            "byte-equal to the B0 of E1), one key changed from B0, or from B0s (B0 with a symmetric `pull_off` load, Fy = 0) for the mirror pair; "
            "`E2/T-...` are the tool calls on a copy of the finished B0. The E1 bands apply to the `E2` rows of this class. Base values of B0: optimizer "
            "stiffness 1294.7 N/mm for the case `hang` and 1316.8 for `pull_off`; P2 stiffness 1062.8 and 909.2 N/mm (0.82 and 0.69 of the "
            "optimizer value); mass 25.72 g.",
    "hook fine": "examples/hook/hook.yaml of this repository: voxel 1.2 mm, 80 passes, the default beta schedule, CPU. The run ids `E1/H...` are "
                 "runs of the E1 series; the base run H0 took 261 s (247 s of loop). `E2/MIR` is `examples/hook/hook_mirror.yaml` as it is, against H0. It repeats five effects of the 2 mm series: the sign held in 10 of 10 comparisons, "
                 "the size changes with the grid.",
    "drone arm": "a 5-inch drone arm (measured 2026-10-26; the files are not in this repository): voxel 0.7 mm in the fine pass and the cold run, 1.0 mm in the parent, "
                 "59,571 elements and 199,041 free dofs at 0.7 mm, 20,496 active voxels at 1.0 mm, 3 load cases (weights 9, 1, 1), solver amg, 80 passes in the parent and in the cold "
                 "run (81 snapshots each), CPU. The host had 22 cores shared with other jobs (load average 11 to 65), so a wall time is good to a factor of 2 or more: read the pass "
                 "counts first. The run ids `FP/arm-...` are runs of the fine-pass study (the folders are not in this repository). `weighted` is the weighted compliance sum of "
                 "the engine (lower is better): parent 1.0 mm 623.67 (953 s), cold run at 0.7 mm 596.95 (3462 s, peak memory 2.11 GiB, grayness 0.2235, volume 0.30739). The grayness "
                 "of the arm is 0.22 in every run, so the rule `converged` cannot fire on it.",
    "hook fine pass": "examples/hook/hook_short.yaml of this repository (solver pardiso, load cases `hang` and `pull_off`), as a fine pass: the parent has voxel 2.0 mm and 40 passes "
                      "(41 snapshots, weighted 1.3778, 52 s wall), the fine pass and the cold run have voxel 1.2 mm; the cold run has 41 snapshots (weighted 1.3612, grayness 0.0164, "
                      "184 s wall, 915 s CPU). CPU, a host with 22 cores shared with other jobs (load average 11 to 65), so a wall time is a range. The run ids `FP/hook-...` are "
                      "runs of the fine-pass study (the folders are not in this repository).",
    "bracket example": "examples/interfaces/bracket.yaml of this repository: voxel 1.0 mm, 40 passes, CPU. Run ids as for the `P7-1/l3/<name>` runs of the hook.",
    "bracket": "examples/interfaces/bracket.yaml of this repository (CONTRACTS.md section 23): voxel 1.0 mm, grid 40 x 20 x 20, 12352 active voxels (10064 free design voxels), `volume_fraction` 0.3, "
               "40 passes, default filter radius (1.5 voxels), default beta schedule [[0,1],[30,2],[50,4],[65,8]], `penal` 3, one load case `pull_down` (15 N down on `tip`), interfaces (a boss, a bore "
               "and a machined face). CPU, 4 threads, solver pardiso (auto), code of origin/main f86ac04. The run ids `E3/BR-<id>` are the runs of the E3 series (the run folders are "
               "not in this repository), and a price is the ratio against the base BR-B0. The base is a grey design: its 40 passes end at beta 2 (beta 8 starts at pass 65), so BR-B0 has grayness "
               "0.129 (the verdict test is 0.1), objective change 0.000254, state `iteration_limit` and `last_beta_stage` not met (the verdict text says that the beta schedule has not reached its "
               "last stage). Every row of this class is a change of that grey design; the class `bracket-fitted` has the same keys on a base that reaches beta 8. Base values of BR-B0: optimizer "
               "stiffness 642.1 N/mm (case `pull_down`, named `pull` in the rows), P2 stiffness 632.8 N/mm (0.986 of the optimizer value; P2 meshes `nominal.stl`, the finished part), volume of "
               "`final.stl` 5398.2 mm3 (mass 6.69 g; the mass ratio of a row is the volume ratio of `final.stl`). Noise: the engine repeats byte for byte (BR-B0 and BR-B1 have byte-equal `final.stl`); "
               "three runs with one input changed by 0.05 to 0.1 % (`volume_fraction` +-0.05 % and `penal` +0.1 %: BR-N1, BR-N2, BR-N3) moved the optimizer stiffness by at most 0.045 %, the mass by "
               "0.037 % and P2 by 2.09 %, and flipped 0 to 0.07 % of the design voxels. These bands are the largest deviation of the three runs, with no margin. A row marked `inside the noise` has "
               "the optimizer ratio, the mass ratio and the P2 ratio inside them. A run took 18 s (20 passes) to 93 s (160 passes) of wall time, 25.3 to 36.0 s for 40 passes, and the P2 check adds 5 to "
               "14 s (host load average 18 to 40, so a wall time is a range).",
    "bracket-fitted": "the same file as `bracket` with `beta_schedule` [[0,1],[15,2],[25,4],[32,8]]: the default schedule scaled to the 40 passes (the rule of the branch P7-4, which is not in f86ac04; run "
                      "BR-BS40s). Beta 8 starts at pass 32, so this base reaches the last beta stage: grayness 0.020, objective change 0.00096 (threshold 1e-4, not met), state `iteration_limit`, "
                      "`last_beta_stage` met. Against BR-B0 the base has optimizer x1.107, mass x1.008, P2 x1.037. Base values of BR-BS40s: optimizer stiffness 710.9 N/mm, P2 stiffness 656.0 N/mm "
                      "(0.923 of the optimizer value), volume of `final.stl` 5444.0 mm3 (6.75 g). The run ids `E3/BR-S...` are runs on this base; `BR-IT20s`, `BR-IT80`, `BR-SR80` and `BR-SR160` "
                      "are runs that the table of the example base also holds, and a row of this class gives their ratios against BR-BS40s. Noise: two runs with `volume_fraction` +-0.05 % (BR-SN1, "
                      "BR-SN2) moved the optimizer stiffness by at most 0.08 %, the mass by 0.059 % and P2 by 1.476 %, and flipped 0.07 and 0.13 % of the design voxels; this base has no repeat run "
                      "(the engine repeated byte for byte on the other bases). A row marked `inside the noise` has the optimizer, mass and P2 ratios inside these bands.",
    "ladle-4mm": "a coarse copy of examples/ladle/ladle.yaml (voxel 3.25 mm, 80 passes, three load cases) with `voxel_size` 4.0 mm and no other change. The copy exists because the run on the grid of "
                 "the example (part class `ladle`, run LAD-B0) took 401 s, which is more than the few minutes that the brief of the series allows for one try. Grid 55 x 24 x 22, 19021 active voxels "
                 "(18014 free design voxels), load cases `full_scoop`, `scrape`, `side_stir` (named scoop, scrape and stir in the rows; weights 1), keep-out regions, `volume_fraction` 0.10628, 80 "
                 "passes, default filter radius (1.5 voxels) and default beta schedule (beta 8 from pass 65: it fits 80 passes). CPU, 4 threads, solver pardiso, code f86ac04. The run ids `E3/LAD-<id>` "
                 "are runs of the E3 series (the folders are not in this repository); a price is the ratio against the base LAD-D0. LAD-D0 ends `converged` (objective change 4.8e-05) with grayness "
                 "0.014, mass 231.31 g (volume of `final.stl` 186540.9 mm3), optimizer stiffness 267.7 / 408.1 / 166.2 N/mm and P2 stiffness 284.1 / 449.1 / 173.4 N/mm (1.061, 1.100 and 1.043 of the "
                 "optimizer values). The three cases often move in opposite directions, so a row also gives `objective`: base objective over run objective, where the objective is the weighted "
                 "compliance sum of the engine (`compliance_final`; weights 1, 1, 1; 8.3155 for the base). The P2 objective is the same sum with the P2 stiffness of each case in place of the "
                 "optimizer one (the case compliance times k_optimizer / k_P2): this is a definition of the E3 report, not an engine value. Noise: the engine repeats byte for byte (LAD-D0 and "
                 "LAD-D1); three runs with one input changed by 0.05 to 0.1 % (LAD-N1, LAD-N2, LAD-N3) moved the optimizer stiffness of a case by at most 0.439 % (the objective by 0.20 %), the mass "
                 "by 0.084 % and the P2 stiffness of a case by 1.54 % (the P2 objective by 1.3 %), and flipped 0.6 to 1.3 % of the design voxels. A row marked `inside the noise` has every optimizer "
                 "value (the cases and the objective), the mass ratio and every P2 value inside these bands. A run took 43 to 81 s of wall time (20 to 40 passes), 119 to 157 s (80 passes) and 272 to "
                 "276 s (160 passes), the loop takes 1.36 s per pass, and the P2 check adds 7.9 to 13.4 s (host load average 20 to 38).",
    "ladle": "examples/ladle/ladle.yaml as it is: voxel 3.25 mm, grid 68 x 30 x 27, 36542 active voxels, 80 passes, three load cases, code f86ac04. Only three filter rows are measured on this grid, one "
             "run each (LAD-XFR1, LAD-XFR2, LAD-XMM2; 330 to 398 s of wall time, and the P2 check adds 16 to 19 s). The base LAD-B0 ends `converged` (objective change 1.5e-05) with grayness 0.012; "
             "it took 400.7 s of wall time (381.2 s in the loop), mass 236.81 g, optimizer stiffness 280.8 / 424.9 / 175.4 N/mm, P2 stiffness 304.1 / 479.8 / 186.4 N/mm (1.083, 1.129 and 1.063 of "
             "the optimizer values), objective 7.9586. The measures and the case names are those of `ladle-4mm`. This grid has no noise set of its own: `inside the noise` uses the bands of "
             "`ladle-4mm`. For the same number of voxels the member in mm differs from the 4.0 mm copy (6.5 mm against 8.0 mm for 2 voxels), so the rows of the two classes are not the same "
             "physical request.",
    F19_HOOK: "examples/hook/hook.yaml of this repository with `voxel_size` 2.0 mm (grid 41 x 18 x 33, 8982 active voxels, 7651 free design voxels), 80 passes, the default beta schedule (beta 8 from pass 65), two load cases (`hang` weight 234, `pull_off` weight 1), CPU, 4 threads, solver pardiso (auto), code of origin/main 74a1e46. The request is 4.0 mm = 2 voxels in every row. The run ids `F1-9/hookA-<arm>` are runs of the F1-9 measurement (the run folders are not in this repository); a price is the ratio against the base run hookA-BASE (default filter, radius 3 mm = 1.5 voxels): optimizer stiffness 1297.4, 1344.1 N/mm (`hang`, `pull_off`), objective 1.3624 (`compliance_final`), P2 stiffness 1041.0, 921.9 N/mm, P2 objective 1.8397, mass 25.71 g, grayness 0.022, state `converged`, 36.6 s. A stiffness ratio and an objective ratio are base over run, so a value below 1 is a loss; the mass ratio is run over base. The P2 objective is the sum of weight x |F|^2 / P2 stiffness over the cases (`topopt verify --no-sensitivity`; a definition of the F1-9 report, as in the class `ladle-4mm`, not an engine value). Thickness is the local thickness of the design voxels: the diameter of the largest ball in the material that holds a voxel (density 0.5 of the last pass, grid refined by 2, bias about a quarter of a voxel). `interior` is a design voxel at least 3 voxel centres from an inactive voxel, and the rest is the face layer, where the filter has one side. A voxel is rod-like when the material inside a ball of three times the request is mostly spread along one axis, and round when its cross-section is as wide as it is high (the covariance eigenvalue ratio l3/l2 is 0.5 or more); the method was checked on synthetic rods and bars only. A row says if its optimizer objective, its P2 objective and its mass are outside the band of the class. Noise: this grid has no noise set of its own; the bands of the class `hook` are used (optimizer stiffness +-0.3 %, mass +-0.1 %, P2 +-3.5 %). The same file with 40 passes and the schedule [[0,1],[16,2],[26,4],[34,8]] (`hook_short.yaml`, the class `hook`) repeated the base B0 (optimizer stiffness 1294.7 and 1316.8 N/mm, mass 25.72 g) and the optimizer ratios of the row E1/MM40 (runs F1-9/hookB-BASE and F1-9/hookB-WALL).",
    F19_BRACKET: "examples/interfaces/bracket.yaml of this repository as it is (CONTRACTS.md section 23): voxel 1.0 mm, grid 40 x 20 x 20, 12352 active voxels (10064 free design voxels), `volume_fraction` 0.3, 40 passes, the default beta schedule, one load case `pull_down` (named `pull` in the rows), interfaces, CPU, 4 threads, solver pardiso (auto), code of origin/main 74a1e46. The request is 2.0 mm = 2 voxels in every row. The run ids `F1-9/bracket-<arm>`; a price is the ratio against the base run bracket-BASE, which repeated BR-B0 of the class `bracket`: optimizer stiffness 642.1 N/mm, mass 6.69 g, grayness 0.129, state `iteration_limit` (a grey design), P2 stiffness 646.4 N/mm (632.8 N/mm for BR-B0: the P2 base is 2.1 % higher with the verify code of this revision, at the edge of the band). A stiffness ratio and an objective ratio are base over run, so a value below 1 is a loss; the mass ratio is run over base. The P2 objective is the sum of weight x |F|^2 / P2 stiffness over the cases (`topopt verify --no-sensitivity`; a definition of the F1-9 report, as in the class `ladle-4mm`, not an engine value). Thickness is the local thickness of the design voxels: the diameter of the largest ball in the material that holds a voxel (density 0.5 of the last pass, grid refined by 2, bias about a quarter of a voxel). `interior` is a design voxel at least 3 voxel centres from an inactive voxel, and the rest is the face layer, where the filter has one side. A voxel is rod-like when the material inside a ball of three times the request is mostly spread along one axis, and round when its cross-section is as wide as it is high (the covariance eigenvalue ratio l3/l2 is 0.5 or more); the method was checked on synthetic rods and bars only. A row says if its optimizer objective, its P2 objective and its mass are outside the band of the class. Noise: the bands of the class `bracket` are used (optimizer 0.045 %, mass 0.037 %, P2 2.09 %).",
    F19_LADLE: "examples/ladle/ladle.yaml of this repository as it is: voxel 3.25 mm, grid 68 x 30 x 27, 36542 active voxels, 80 passes, three load cases (`full_scoop`, `scrape`, `side_stir`; named scoop, scrape and stir in the rows; weights 1), CPU, 4 threads, solver pardiso (auto), code of origin/main 74a1e46. The request is 6.5 mm = 2 voxels in every row. The run ids `F1-9/ladle-<arm>`; a price is the ratio against the base run ladle-BASE, which repeated LAD-B0 of the class `ladle`: optimizer stiffness 280.8, 424.9, 175.4 N/mm, objective 7.9586, mass 236.81 g, grayness 0.012, state `converged`, 483.8 s (host load average 25 to 34); P2 stiffness 302.5, 477.0, 185.3 N/mm, P2 objective 7.2041 (304.1 / 479.8 / 186.4 N/mm and 7.9586 for the objective of LAD-B0; the difference is 0.5 % in each case). A stiffness ratio and an objective ratio are base over run, so a value below 1 is a loss; the mass ratio is run over base. The P2 objective is the sum of weight x |F|^2 / P2 stiffness over the cases (`topopt verify --no-sensitivity`; a definition of the F1-9 report, as in the class `ladle-4mm`, not an engine value). Thickness is the local thickness of the design voxels: the diameter of the largest ball in the material that holds a voxel (density 0.5 of the last pass, grid refined by 2, bias about a quarter of a voxel). `interior` is a design voxel at least 3 voxel centres from an inactive voxel, and the rest is the face layer, where the filter has one side. A voxel is rod-like when the material inside a ball of three times the request is mostly spread along one axis, and round when its cross-section is as wide as it is high (the covariance eigenvalue ratio l3/l2 is 0.5 or more); the method was checked on synthetic rods and bars only. A row says if its optimizer objective, its P2 objective and its mass are outside the band of the class. Noise: the class `ladle` has no noise set of its own and uses the bands of `ladle-4mm` (optimizer objective 0.20 %, mass 0.084 %, P2 objective 1.3 %).",
    "larger part": "a larger part that this repository does not ship (a part with two tube joints and no fixed region).",
    "hook_stress": "examples/hook/hook_stress.yaml of this repository: voxel 2.4 mm (grid 34 x 15 x 28, 5406 active voxels), 220 passes, `objective: mass` (no volume row: "
                   "`volume_fraction` 0.3 is the start density), two load cases (`hang`, 1.96 N, stress limit 1.5 MPa; `pull_off`, 30 N at 45 degrees, stress limit 0.5 MPa), `move_limit` 0.03, "
                   "`stress_q` 40, `stress_power` 0.5, `stress_scale` auto, `stress_step_acceptance` true. CPU, 4 threads, solver pardiso (auto), code of origin/main 4f6d39d. The run ids "
                   "`E2/<id>` are the runs of the E2 series (the run folders are not in this repository). The base is the MEAN of 6 runs of the file as it is "
                   "(E2/ST0 to ST5): mass 33.09 g, optimizer stiffness 874.3 N/mm (`hang`) and 1079.7 N/mm (`pull_off`), P2 stiffness 770.5 and 813.3 N/mm. These runs do NOT repeat "
                   "(the same YAML gave 6 designs; the cause was not isolated). Their range against the mean: mass x0.983 to x1.009, optimizer hang x0.989 to x1.013, optimizer "
                   "pull_off x0.986 to x1.014, P2 hang x0.973 to x1.038, P2 pull_off x0.962 to x1.044, rejected tries of the step acceptance 34 to 43. A change of 0.1 % in one key "
                   "(E2/ST2, E2/ST3) moved the end state as much as the whole range. A row marked `inside the noise` has the three optimizer values (hang, pull_off, mass) inside "
                   "that range. Each setting is one run, except where the repeat is named (E2/SQ20r, E2/SV50r, E2/SM60r agree with their first runs within 0.4 % in mass). "
                   "The wall seconds follow the load of the host (load average 12 to 43 on 22 cores), not the key.",
    "hook_buckling-2mm": "a coarse copy of examples/hook/hook_buckling.yaml (that file is `hook.yaml` + `buckling.min_factor: 150` at voxel 1.2 mm, 80 passes, and took 1469 s): "
                         "`hook_short.yaml` (voxel 2.0 mm, 40 passes, schedule [[0,1],[16,2],[26,4],[34,8]]) + `buckling: {min_factor: 150}`; CPU, 4 threads, solver pardiso, code of "
                         "origin/main 4f6d39d. The run ids are `E2/BK...`. The reference is B0, the same case with no buckling row (loop 16.55 s; the lowest "
                         "factors of its printed design, from `topopt buckling`: 1401.87 for `hang` and 123.085 for `pull_off`); a price is the ratio against B0. A buckling run repeats "
                         "byte for byte (E2/BK0 and E2/BK1), so the bands of the part class `hook` apply: optimizer stiffness +-0.3 %, mass +-0.1 %, P2 +-3.5 %. The loop seconds follow "
                         "the load of the host (load average 22 to 36 on 22 cores); the ratio against B0 is the more portable figure.",
    "not printed (CONTRACTS)": "the numbers of CONTRACTS.md section 9.1 (`regions.not_printed`, `result/part_only.stl`, the film setting) that the work of P4b-2, P4b-3 and P4b-6 recorded: "
                               "the hook example with `regions.not_printed` (voxel 2 mm; `hook_short.yaml`) or the bar of `tests/test_part_only.py` (voxel 1 mm); the row says which. "
                               "They are CONTRACTS-recorded: P7 read them and did not run them again, and the run folders are not in this repository. Times are of the post step "
                               "(Linux x86_64, 2 threads), on a quiet host or on a loaded one (load average 30 on 22 cores).",
    WALL_HOOK: "the T-shirt hook of examples/hook/hook.yaml at voxel 0.5 mm (564,298 active voxels), 80 passes, with print.min_wall_mm 1.2 mm; step 1 of the wall rule was a research prototype of the void padding "
               "(not in main), the second row is the code of main. Solver gpu-mg on one Intel Arc iGPU of a shared Linux x86_64 host. The default run of the same part is the base (radius 0.75 mm = 1.5 voxels). "
               "The share is the part of the surface area whose wall is under 1.2 mm. The run folders are not in this repository.",
}

# The cost of one try on the hook, by layer (measured: E1 series and the PR-1 runs; part class `hook`; layer 2: the continuations of round 3).
LAYER_COSTS = {
    1: "a run of 40 passes at voxel 2 mm took 15 to 37 s wall, median 21 s over 47 runs (the load of the host changed the same run by a factor of about 2). "
       "The loop takes 0.40 s per pass (median) for 7651 free design voxels, 1.58 s per pass at 18256 (voxel 1.5 mm) and 3.09 s per pass at 34220 "
       "(`hook.yaml`, voxel 1.2 mm, 80 passes: 261 s). The post step, the print check and the refine page add 3.9 s (median). A stress run (`hook_stress.yaml`, voxel 2.4 mm, 220 passes, "
       "two stress rows) took 45 to 105 s (the seconds follow the host load, not the key). A run with the buckling row (2 mm, 40 passes) took 177 to 235 s, 10.4 to 13.8 times a "
       "plain run, also with the row far from binding; 12 modes took 498 s (E2).",
    2: "a continuation (`continue_from` a finished hook run, 10 or 20 more passes) took 14.3 to 25.7 s wall in 4 runs (the host load was 29 to 42 on 22 cores), against 34.0 s "
       "for a new run of the same 40 passes in the same hour: the fixed part (start, build, pass K again, the post step, the refine page) is of the size of the passes, so "
       "10 and 20 more passes do not differ above the noise of the host. A continuation with a changed key of the passes (volume_fraction 0.25, 10 passes) took 15.1 s. "
       "A resume of a run that did not finish costs the passes that are left (not measured).",
    3: "the post step alone took 4.5 to 5.6 s at voxel 2 mm and 10.5 to 13.8 s at 1.2 mm (E1/PO_*). The resume route (`--resume --allow-changed-inputs`) took 6.1 to 16.9 s "
       "(P7-1/l3 runs; the top of the range is the buckling check).",
    4: "`print_check` 5.5 s, `cleanup_list` 3.0 s, `apply_marks` 2.5 s, `marks_to_hints` 9.6 s, `compare_runs` 0.6 s, `topopt verify` 12 s (default call with the sensitivity; the call `--no-sensitivity` of the E1 series took a median of 4.7 s, "
       "3.1 to 18.4 s, over 54 runs; 3.6 to 3.9 s on B0 in three runs of mine), `topopt buckling` 6.4 s, `topopt refine` 2.3 s, `topopt render` 6.1 s (P7-1 runs on the finished hook run). "
       "E2, on copies of the finished B0 on a loaded host (load average 12 to 43): `print_check` 3.2 to 3.9 s, `cleanup_list` 0.63 s (estimate) and 3.1 s, `apply_marks` 2.4 to 3.1 s, "
       "`compare_runs` 0.12 to 0.67 s, `ratchet_run` 3.7 to 4.3 s for the estimate and 15.5 s (rule `mass`) or 23.3 s (rule `stiffness`) for a real job, `topopt buckling` 6.8 to 16.6 s "
       "at voxel 2 mm (47.3 s and 1.8 GB at 1.2 mm).",
}

# How to read the numbers of the evidence rows (shown in the reference header).
PRICE_TEXT = (
    "A price is the change of a result against the base run when one key changed. Rows with the part class `hook` give three measures. `optimizer` is "
    "`stiffness_cases[].N_per_mm` of the run: the voxel field, with the penal of the run itself. `P2` is `topopt verify <run> --no-sensitivity` on `final.stl` "
    "(quadratic tets, independent of the voxel model; the printed part is the P2 model; a median of 4.7 s per check on the hook). `mass` is `mass_final_g` of `final.stl`. "
    "`hang` and `pull_off` are the two load cases of the hook. The optimizer value is not a safe price for four keys: `penal`, a filter radius of 1 voxel "
    "or less (no filter), `mesh_method` and `voxel_size`. For these the optimizer and P2 differ in size or sign, and the card says to read P2.\n\n"
    "Noise of the hook (the base run B0, and four runs with one input changed by 0.05 to 0.1 %): the optimizer stiffness moved at most 0.12 % (hang) and 0.17 % "
    "(pull_off), the mass 0.08 %, and 0.72 % of the design voxels changed state; P2 moved 0.7 % (hang) and 3.3 % (pull_off). A row marked `inside the noise` has all "
    "three optimizer values inside +-0.3 % (stiffness) and +-0.1 % (mass). A normal step of `move_limit`, `penal` or the filter radius changes the layout (`flip` "
    "is the share of the solid design voxels of B0 whose state at density 0.5 differs: 5 to 21 % for these steps) and moves the stiffness by up to about 2 %. "
    "One run for each setting cannot tell that from a lever effect: read a difference below about 2 % as small, unproven, and a P2 difference below about 4 % as "
    "inside the P2 noise. `gray` is the grayness of the last pass. A row is one run of one part class and a fact about that class. "
    "The part classes `bracket`, `bracket-fitted`, `ladle-4mm` and `ladle` (E3) give the same three measures. Their price strings name the case (`pull` for the bracket; `scoop`, `scrape`, `stir` "
    "for the ladle) and, for the ladle, `objective` (base objective over run objective: the weighted compliance sum of the engine, and for P2 the sum with the P2 stiffness of each case; the E3 "
    "report defines the P2 objective, the engine does not). Each of these classes has its own base; the bands are in its description (the class `ladle` uses the bands of `ladle-4mm`), and `inside the noise` there means that the optimizer "
    "values, the mass ratio and the P2 values are all inside the bands of that class. The bracket base `bracket` is a grey design (it ends at beta 2); `bracket-fitted` is the same file with the "
    "schedule fitted to its passes, and a card gives the row of both where their signs differ. A card with rows on the hook, the bracket and the ladle says in one sentence where the sign is the same "
    "on the three parts and where it differs; three parts do not make a general law. "
    "The part classes `hook_stress` and `hook_buckling-2mm` (E2) have their own bases and "
    "noise: a stress run does not repeat, so its price is the ratio against the MEAN of 6 base runs and `inside the noise` is the range of those 6 runs; a buckling run repeats byte for "
    "byte and has the bands of `hook`. The part class says which."
)


# Tools that only report. They change no result and judge none, so they have no card (the reference lists them).
READ_ONLY_TOOLS = {
    "run_status": "state of the run that goes on",
    "run_result": "the facts of a finished run (the source of the symptom facts in the cards)",
    "list_runs": "the run folders",
    "run_history": "the facts of earlier runs",
    "verify_status": "state of a verify job",
    "ratchet_status": "the log of a ratchet",
    "record_brief": "saves the brief and the origin of each assumption",
    "save_preferences": "keeps the printer model, the print bed, the material (also the filament), the filaments the user has, the quality and the outputs of the user in the work folder (a preference does not change a run by itself)",
    "find_printer": "finds a printer by name in the shipped catalog and the user printers of the work folder (the facts of a printer; it changes nothing)",
    "save_printer": "keeps a printer that is not in the catalog, with its sources, in the work folder (an entry changes no run until a YAML names it; a replaced entry changes the next check or run of a YAML that names it)",
    "find_material": "finds a filament by name in the shipped catalog and the user filaments of the work folder (the facts of a filament; it changes nothing)",
    "save_material": "keeps a filament that is not in the catalog, with its sources, in the work folder (an entry changes no run until a YAML names it; a replaced entry changes the next check or run of a YAML that names it)",
    "printer_materials": "lists the filaments by state (ok, care, no, unknown) for a printer, with the reasons (facts from the stated values; it changes nothing and stops no run)",
    "case_status": "shows the brief and what changed",
    "environment": "versions, solver backends, RAM, cores",
    "team_guide": "the role playbooks",
    "show_problem": "shows the problem as a 3D view in the chat, with the estimate of check_problem (it changes nothing)",
    "levers": "prints these cards (the command of this reference)",
    "dashboard": "shows a run folder in the browser",
    "doctor": "what this install is and what is wrong with it",
    "gpu-check": "checks the GPU solver gpu-mg on this machine",
    "export_step": "writes a faceted STEP file of a finished run (no pass runs; the design and the run files do not change)",
    "find_features": "lists the holes, bosses and flat faces of an STL file (it reads the file and writes nothing; the numbers are facts of the mesh)",
    "report": "writes the part card result/report.md again from the files of the run folder (no pass runs; the design does not change)",
}


# --- shared text of the table (data, not logic) ------------------------------------------------------------------------------
# Part classes of the evidence rows. The run ids of "P7-1/l3/<name>" are the runs of PR-1.
STAND, HOOK, HOOK_FINE, BRACKET, LUG = "phone stand", "hook", "hook fine", "bracket example", "larger part"
BR_E3, BR_FIT, LAD4, LAD_EX = "bracket", "bracket-fitted", "ladle-4mm", "ladle"   # the part classes of the E3 series: bracket (grey base), bracket with the fitted schedule, ladle at 4.0 mm, ladle at 3.25 mm
HOOK_STRESS, HOOK_BUCKLING = "hook_stress", "hook_buckling-2mm"   # the stress case and the coarse buckling copy of the E2 series
BEFORE_F4_27B = "measured before F4-27b (no snap at the region faces)"   # the label of the rows that were measured with the old mesh of the `sdf` method
NOT_PRINTED = "not printed (CONTRACTS)"   # the numbers that CONTRACTS.md section 9.1 recorded (P4b-2, P4b-3, P4b-6); P7 did not run them again
RESUME_CALL = ("edit the key; `run_topopt` with the same `out_dir`, `resume=true`, `allow_changed_inputs=true` "
               "(CLI: `topopt run <yaml> --out <run> --resume --allow-changed-inputs`)")
RESUME_COST = "seconds, no new pass (6.1 to 16.9 s on the hook example, longer with the buckling check; the host load changed; its base run took 27.0 s)"


def _resume_row(change: str, effect: str, seconds: str, run: str, part_class: str = HOOK) -> Evidence:
    """A row of the post-step route. The row is plain text; the helper only fixes the shared words."""
    return Evidence(change, effect, f"{seconds}, no new pass (history, checkpoint and snapshots equal by sha256)", part_class, f"P7-1/l3/{run}")


def _hook(change: str, effect: str, price: str, run: str, part_class: str = HOOK) -> Evidence:
    """A row of the E1 series: one key changed from the base run B0 or H0, the ratios against that base run.

    `optimizer` is `stiffness_cases[].N_per_mm` of the run (the voxel field, with the penal of the run itself), `P2` is `topopt verify
    --no-sensitivity` on final.stl (quadratic tets, the independent model), `hang` and `pull_off` are the two load cases. The row is plain text.
    """
    return Evidence(change, effect, price, part_class, "E1/" + run.replace(", ", ", E1/"))


def _e2(change: str, effect: str, price: str, run: str, part_class: str = HOOK) -> Evidence:
    """A row of the E2 series: one key changed from the base of its part class, the ratios against that base.

    The base of `hook_stress` is the mean of 6 runs of the unchanged file; the base of `hook` and `hook_buckling-2mm` is the run B0 (the mirror pair B0s and
    the 1.2 mm pair H0 say so in the row). The columns are as in `_hook`. The row is plain text.
    """
    return Evidence(change, effect, price, part_class, "E2/" + run.replace(", ", ", E2/"))


def _e3(change: str, effect: str, price: str, run: str, part_class: str) -> Evidence:
    """A row of the E3 series: one key changed from the base of its part class (BR_E3, BR_FIT, LAD4, LAD_EX), the ratios against that base.

    The bracket has one load case (`pull` in the price); the ladle has three (`scoop`, `scrape`, `stir`) and an `objective` ratio (base objective over run objective). The
    price gives the optimizer ratios, the mass ratio and the P2 ratios; `inside the noise` is set when all of them are inside the bands of the part class. The row is plain text.
    """
    return Evidence(change, effect, price, part_class, "E3/" + run.replace(", ", ", E3/"))


def _f19(change: str, effect: str, price: str, run: str, part_class: str) -> Evidence:
    """A row of the F1-9 measurement (Refs #150): one filter key changed from the base run of its part class, the ratios against that base.

    The columns are as in `_e3`; the ratios are stiffness ratios (base over run) and the mass ratio is run over base. The row is plain text.
    """
    return Evidence(change, effect, price, part_class, "F1-9/" + run)


def _st(change: str, effect: str, price: str, run: str) -> Evidence:
    """A row of the stress case (part class `hook_stress`)."""
    return _e2(change, effect, price, run, HOOK_STRESS)


def _bk(change: str, effect: str, price: str, run: str) -> Evidence:
    """A row of the buckling case (part class `hook_buckling-2mm`)."""
    return _e2(change, effect, price, run, HOOK_BUCKLING)


# Said where a card names the grayness as a symptom (E3): the grayness alone does not show that the last beta stage was missed.
GRAY_NOT_ENOUGH = ("A grayness below 0.1 alone does not show that the beta schedule reached its last stage: E3/LAD-IT40 (40 passes, default schedule, the run ends at beta 2) has grayness 0.085, "
                   "below the verdict test of 0.1, and its objective is x0.730 and its P2 objective x0.923 of the base; the verdict's `last_beta_stage` (not met) shows it")


# The noise of the stress case, said on each card of a stress key (the part class `hook_stress` has the full text).
STRESS_NOISE = ("Noise: a stress run does not repeat (6 runs of the unchanged file: optimizer pull_off stiffness +-1.4 %, P2 pull_off stiffness -3.8 % / +4.4 %, mass 32.54 to "
                "33.40 g), and a change of 0.1 % in one key moved the end state as much as that range. A row marked `inside the noise` has hang, pull_off and mass inside "
                "the range of the 6 runs; any other row is one sample, and a move that repeated in a second run says so.")


# I6 (CONTRACTS.md section 5): two facts of every run. `metrics.load_paths` says if the finished mesh joins each load region to a support; `metrics.threshold_stability`
# gives the voxel volume and the body count of the thresholded design at the levels 0.4, 0.5 and 0.6. Each is a fact only: the engine sets no limit for either.
_LOAD_PATH_STL = "metrics.load_paths.regions.<region>.final_stl"
STL_JOIN_SYMPTOM = Symptom(
    "floating or not attached in the STL, the mesh does not join the load region, a gap in the part",
    f"`{_LOAD_PATH_STL}.connected_to_fixed` for each load region (false: no body of `final.stl` holds a point of the load region and a point of a fixed region), with "
    "`touches_a_body` and `bodies_with_the_region`; `metrics.load_paths.final_stl.bodies` counts the bodies of the mesh. "
    "`metrics.load_paths.regions.<region>.thresholded_design.connected_to_fixed` is the same test on the thresholded voxels, before the mesh step",
    facts=(f"{_LOAD_PATH_STL}.connected_to_fixed", f"{_LOAD_PATH_STL}.touches_a_body", f"{_LOAD_PATH_STL}.bodies_with_the_region"),
    when=f"`{_LOAD_PATH_STL}.connected_to_fixed` is false")
STABILITY_SYMPTOM = Symptom(
    "the shape changes with the cut, a grey part, thin members that come and go at another threshold",
    "`metrics.threshold_stability.levels[]`: the voxel volume (`volume_mm3`) and the body count (`bodies`) of the thresholded design at the levels 0.4, 0.5 and 0.6 of the "
    "last snapshot. A voxel with a density between two levels is solid at one and void at the other, so a volume or a body count that differs between the levels is the part "
    "that depends on the cut. The engine sets no limit for the difference",
    facts=("metrics.threshold_stability.levels[].volume_mm3", "metrics.threshold_stability.levels[].bodies"))


# The rows of the E1 series by key (plain data). The reading of the numbers is in PRICE_TEXT.
_E1_PENAL = (
    _hook('2.0 (from 3.0)', 'gray 0.033; flip 13.4% - optimizer value up, P2 value down', 'optimizer hang x1.020, pull_off x1.063, mass x1.001; P2 hang x0.970, pull_off x0.955', 'P20'),
    _hook('2.5', 'gray 0.026; flip 8.2%', 'optimizer hang x1.008, pull_off x1.029, mass x1.000; P2 hang x0.959, pull_off x0.962', 'P25'),
    _hook('3.5', 'gray 0.025; flip 12.9%', 'optimizer hang x0.986, pull_off x0.968, mass x1.000; P2 hang x0.991, pull_off x0.954', 'P35'),
    _hook('4.0', 'gray 0.023; flip 20.7% - layout moves most (flip), P2 pull lowest of the four', 'optimizer hang x0.975, pull_off x0.961, mass x1.000; P2 hang x0.954, pull_off x0.913', 'P40'),
    _hook('3.0 against 4.0 (the reading of P40 against B0, inverted)', 'the phone-stand rows say that 3 is 13 to 33 % stiffer than 4; on the hook the optimizer gives less and P2 gives more than the optimizer', 'optimizer hang x1.026, pull_off x1.041; P2 hang x1.048, pull_off x1.095', 'P40, B0'),
)
_E1_MIN_MEMBER = (
    _hook('1.0 (filter 1.877 mm = 0.94 voxel)', 'gray 0.001; flip 32.9%; islands removed 5; converged - filter has no neighbour: byte-equal final.stl to FR20; 2 warning lines', 'optimizer hang x1.009, pull_off x1.008, mass x0.995; P2 hang x0.937, pull_off x0.878', 'MM10'),
    _hook('1.6 (filter 3.003 mm = 1.50 voxel)', 'gray 0.025; flip 0.3% - same radius as the default; 1 warning line', 'optimizer hang x1.000, pull_off x0.999, mass x1.000; P2 hang x0.996, pull_off x0.986; inside the noise (all three optimizer values)', 'MM16'),
    _hook('2.0 (filter 3.754 mm = 1.88 voxel)', 'gray 0.054; flip 15.4% - byte-equal final.stl to FR375; 1 warning line', 'optimizer hang x0.947, pull_off x0.976, mass x0.994; P2 hang x0.953, pull_off x0.987', 'MM20'),
    _hook('3.0 (filter 5.631 mm = 2.82 voxel)', 'gray 0.090; flip 52.9%; islands removed 1 - 1 warning line (1.5 voxels)', 'optimizer hang x0.836, pull_off x0.904, mass x0.977; P2 hang x0.862, pull_off x0.881', 'MM30'),
    _hook('4.0 (filter 7.508 mm = 3.75 voxel)', 'gray 0.112; flip 64.7%; islands removed 1 - no warning (2 voxels)', 'optimizer hang x0.805, pull_off x0.834, mass x0.974; P2 hang x0.836, pull_off x0.859', 'MM40'),
    _hook('6.0 (filter 11.262 mm = 5.63 voxel)', 'gray 0.166; flip 66.2%; islands removed 1', 'optimizer hang x0.691, pull_off x0.716, mass x0.967; P2 hang x0.805, pull_off x0.816', 'MM60'),
    _hook('4.0 (filter 7.508 mm = 6.26 voxels) at voxel 1.2 mm, 80 passes', 'gray 0.120; flip 72.9%; verdict iteration_limit at pass 80; the same sign as MM40 at 2 mm', 'optimizer hang x0.765, pull_off x0.770, mass x0.983; P2 hang x0.844, pull_off x0.803', 'HMM40', HOOK_FINE),
)
_E1_FILTER_RADIUS = (
    _hook('2.0 mm = 1 voxel', 'gray 0.001; flip 32.9%; islands removed 5; converged - no filter effect (radius not above 1 voxel)', 'optimizer hang x1.009, pull_off x1.008, mass x0.995; P2 hang x0.937, pull_off x0.878', 'FR20'),
    _hook('3.754 mm (= 1.877 x 2.0)', 'gray 0.054; flip 15.4% - byte-equal to MM20', 'optimizer hang x0.947, pull_off x0.976, mass x0.994', 'FR375'),
    _hook('4.0 mm = 2 voxel', 'gray 0.061; flip 18.1%', 'optimizer hang x0.935, pull_off x0.972, mass x0.992; P2 hang x0.942, pull_off x0.985', 'FR40'),
    _hook('6.0 mm = 3 voxel', 'gray 0.096; flip 54.1%; islands removed 1', 'optimizer hang x0.824, pull_off x0.891, mass x0.975; P2 hang x0.866, pull_off x0.899', 'FR60'),
)
_E1_ROBUST = (
    _hook('eroded, eta_eroded default 0.55', 'gray 0.025; flip 4.5% - metrics.robust: eroded compliance is 2.0 % above the nominal one; 1 warning line', 'optimizer hang x0.999, pull_off x0.979, mass x0.997; P2 hang x0.979, pull_off x0.947; optimizer values within about 2 %: small, unproven', 'RE55'),
    _hook('eroded, eta_eroded 0.65', 'gray 0.030; flip 8.5% - 1 warning line', 'optimizer hang x0.990, pull_off x0.958, mass x0.995; P2 hang x0.978, pull_off x0.940', 'RE65'),
)
_E1_BETA = (
    _hook('[[0,1],[20,2],[32,4]] (ends at beta 4)', 'gray 0.084; flip 3.6%', 'optimizer hang x0.899, pull_off x0.943, mass x0.991; P2 hang x0.956, pull_off x0.966', 'BSa'),
    _hook('[[0,1],[10,2],[18,4],[26,8],[34,16]] (ends at beta 16)', 'gray 0.002; flip 6.0%', 'optimizer hang x1.025, pull_off x1.005, mass x1.001; P2 hang x0.975, pull_off x0.937', 'BSb'),
    _hook('[[0,1],[30,2],[50,4],[65,8]] (CONTRACTS default, run ends at beta 2)', 'gray 0.138; flip 8.6%; islands removed 1 - verdict: schedule has not reached its last stage', 'optimizer hang x0.801, pull_off x0.869, mass x0.972; P2 hang x0.924, pull_off x0.960', 'BSc'),
    _hook('[[0,8]] (beta 8 from pass 0)', 'gray 0.341; flip 36.9%; islands removed 1 - thresholded volume only 0.71 of the allowed volume', 'optimizer hang x0.483, pull_off x0.562, mass x0.815; P2 hang x0.616, pull_off x0.619', 'BSd'),
)
# The runs with 20 and 80 passes and the schedule unchanged or scaled are rows of `iterations` and of `beta_schedule` (the same runs).
_E1_MOVE = (
    _hook('0.05 (from 0.2)', 'gray 0.026; flip 15.6%', 'optimizer hang x0.979, pull_off x0.999, mass x1.001; P2 hang x0.995, pull_off x0.999; optimizer values within about 2 %: small, unproven', 'ML05'),
    _hook('0.1', 'gray 0.024; flip 4.9%', 'optimizer hang x0.994, pull_off x1.008, mass x1.000; P2 hang x0.974, pull_off x0.977; optimizer values within about 2 %: small, unproven', 'ML10'),
    _hook('0.4', 'gray 0.030; flip 5.8%', 'optimizer hang x0.983, pull_off x1.000, mass x0.999; P2 hang x0.969, pull_off x0.981; optimizer values within about 2 %: small, unproven', 'ML40'),
)
_E1_ITER = (
    _hook('20 (schedule unchanged: ends at beta 2)', 'gray 0.142; flip 8.3%; islands removed 2 - half the wall time', 'optimizer hang x0.791, pull_off x0.834, mass x0.973; P2 hang x0.928, pull_off x0.875', 'IT20'),
    _hook('20 with the schedule scaled by 0.5: [[0,1],[8,2],[13,4],[17,8]]', 'gray 0.028; flip 9.8% - half the wall time', 'optimizer hang x0.989, pull_off x0.967, mass x1.000; P2 hang x0.969, pull_off x0.900', 'IT20s'),
    _hook('80 (schedule unchanged)', 'gray 0.024; flip 0.1%; converged - verdict converged (objective change 5.7e-6)', 'optimizer hang x1.000, pull_off x1.001, mass x1.000; P2 hang x1.010, pull_off x1.020; inside the noise (all three optimizer values)', 'IT80'),
    _hook('80 with the schedule scaled by 2: [[0,1],[32,2],[52,4],[68,8]]', 'gray 0.022; flip 13.1%', 'optimizer hang x1.002, pull_off x1.019, mass x0.999; P2 hang x0.991, pull_off x1.051; optimizer values within about 2 %: small, unproven', 'IT80s'),
    _hook('40 at voxel 1.2 mm (schedule unchanged: ends at beta 2)', 'gray 0.093; flip 3.6%; verdict iteration_limit at pass 40; the same sign as IT20 at 2 mm', 'optimizer hang x0.876, pull_off x0.906, mass x0.991; P2 hang x0.981, pull_off x0.963', 'HIT40', HOOK_FINE),
)
_E1_ITER_FOR_BETA = tuple(replace(e, change="iterations " + e.change) for e in _E1_ITER[:4])
_E1_STOP = (
    _hook('true with iterations 80', 'gray 0.025; flip 0.1%; stopped_by_rule - stopped by rule converged at pass 48; compare with IT80', 'optimizer hang x1.001, pull_off x1.001, mass x1.000; P2 hang x0.996, pull_off x0.998; inside the noise (all three optimizer values)', 'SR80'),
)
_E1_OSCALE = (
    _hook('first (from stage)', 'gray 0.033; flip 2.3%', 'optimizer hang x0.984, pull_off x0.990, mass x0.996; P2 hang x0.990, pull_off x1.011; optimizer values within about 2 %: small, unproven', 'OSF'),
)
_E1_COMBINE = (
    _hook('max (from sum)', 'gray 0.026; flip 5.3% - run.json objective_form is max', 'optimizer hang x1.003, pull_off x0.987, mass x1.000; P2 hang x0.990, pull_off x0.953; optimizer values within about 2 %: small, unproven', 'CCM'),
)
_E1_VOXEL = (
    _hook('3.0 mm with filter_radius 3.0 mm (1 voxel)', 'gray 0.001; flip n/a (grid differs) - grid 27 x 12 x 22; mass ratio vs engine target differs', 'optimizer hang x0.738, pull_off x0.695, mass x0.963; P2 hang x0.726, pull_off x0.768', 'VX30f'),
    _hook('1.5 mm with filter_radius 3.0 mm (2 voxel)', 'gray 0.048; flip n/a (grid differs) - grid 54 x 24 x 44; about 3.8 x the loop time', 'optimizer hang x0.991, pull_off x1.096, mass x1.087; P2 hang x1.167, pull_off x1.180', 'VX15f'),
)
_E1_MESH = (
    _hook('blur (from sdf)', 'gray 0.025; flip 0.0% - same snapshots as B0 (byte-equal); only final.stl differs', 'optimizer hang x1.000, pull_off x1.000, mass x1.097; P2 hang x0.571, pull_off x0.784', 'MBb'),
    _hook('sdf -> blur, the post step alone on a copy of the finished base run (finish() of topopt.post; not a documented command)', 'byte-equal final.stl to the full run MBb; mass 28.21 g against 25.72 g; triangles 27682 against 20586', 'finish() 4.47 s and 4.49 s with blur, 5.56 s and 4.57 s with sdf (mesh step 0.85 to 1.12 s against 2.33 to 2.43 s; the print check of 2.6 to 3.6 s is in both)', 'PO_blur1, PO_blur2, PO_sdf1, PO_sdf2'),
    _hook('the same at voxel 1.2 mm', 'mass sdf 25.34 g, blur 26.66 g (sdf is x0.950 of blur here, x0.912 at 2 mm)', 'finish() 13.75 s with sdf, 10.51 s with blur; the full run H0 took 261 s', 'PO_H_sdf, PO_H_blur', HOOK_FINE),
)
_E1_FSN = (
    _hook('ignore (from density_one)', 'gray 0.023; flip 5.5%', 'optimizer hang x0.996, pull_off x1.009, mass x1.001; P2 hang x0.996, pull_off x1.018; optimizer values within about 2 %: small, unproven', 'FSI'),
)
_E1_VF = (
    _hook('0.15 (from 0.20)', 'gray 0.024; flip 30.6%', 'optimizer hang x0.737, pull_off x0.760, mass x0.853; P2 hang x0.664, pull_off x0.660', 'VF15'),
    _hook('0.25', 'gray 0.022; flip 25.5%', 'optimizer hang x1.242, pull_off x1.207, mass x1.147; P2 hang x1.190, pull_off x1.143', 'VF25'),
    _hook('0.30', 'gray 0.022; flip 50.0%', 'optimizer hang x1.469, pull_off x1.376, mass x1.292; P2 hang x1.369, pull_off x1.250', 'VF30'),
    _hook('0.25 at voxel 1.2 mm, 80 passes', 'gray 0.016; flip 26.0%; converged', 'optimizer hang x1.232, pull_off x1.178, mass x1.144; P2 hang x1.209, pull_off x1.139', 'HVF25', HOOK_FINE),
)
_E1_KEEPOUT = (
    _hook('block 1 (28..40 x -14..14 x -4..6 mm, 3360 mm3), vf 0.2 unchanged', 'gray 0.026; flip 10.1% - the block holds no solid voxel of the base design', 'optimizer hang x0.940, pull_off x0.955, mass x0.969; P2 hang x0.927, pull_off x0.955', 'K1'),
    _hook('block 1, vf 0.21162 (mass-keeping formula)', 'gray 0.025; flip 7.4%', 'optimizer hang x0.999, pull_off x1.002, mass x1.000; P2 hang x0.974, pull_off x0.974; inside the noise (all three optimizer values)', 'K1v'),
    _hook('block 2 (8..24 x -14..14 x 0..14 mm, 6272 mm3), vf 0.2', 'gray 0.026; flip 12.3% - the block holds no solid voxel of the base design', 'optimizer hang x0.889, pull_off x0.906, mass x0.940; P2 hang x0.864, pull_off x0.862', 'K2'),
    _hook('block 2, vf 0.22283', 'gray 0.026; flip 5.1%', 'optimizer hang x0.996, pull_off x1.003, mass x1.000; P2 hang x0.973, pull_off x0.967', 'K2v'),
    _hook('block 3 (26..38 x -14..14 x -16..-8 mm, 2688 mm3), vf 0.2', 'gray 0.028; flip 84.1% - block holds 134 solid voxels of B0 (on the load path)', 'optimizer hang x0.776, pull_off x0.865, mass x0.982; P2 hang x0.791, pull_off x0.907', 'K3'),
    _hook('block 3, vf 0.20682', 'gray 0.027; flip 83.3%', 'optimizer hang x0.800, pull_off x0.902, mass x1.003; P2 hang x0.822, pull_off x0.955', 'K3v'),
    _hook('block 1, vf 0.2 at voxel 1.2 mm, 80 passes', 'gray 0.017; flip 9.3%; converged', 'optimizer hang x0.947, pull_off x0.952, mass x0.969; P2 hang x0.938, pull_off x0.931', 'HK1', HOOK_FINE),
    _hook('block 1, vf 0.21136 (mass-keeping formula) at voxel 1.2 mm', 'gray 0.018; flip 4.0%; converged', 'optimizer hang x1.000, pull_off x0.998, mass x1.000; P2 hang x0.998, pull_off x0.971', 'HK1v', HOOK_FINE),
)
# The keep-out runs are rows of `regions.keep_out` and of `settings.volume_fraction` (the same runs).
_E1_KEEPOUT_FOR_VF = tuple(replace(e, change="keep-out " + e.change) for e in _E1_KEEPOUT)
_E1_FORCE = (
    _hook('x 0.5 : [10.445, 2.605, -10.445] N', 'gray 0.024; flip 14.2% - byte-equal final.stl to LH20', 'optimizer hang x1.046, pull_off x0.888, mass x0.998; P2 hang x1.010, pull_off x0.858', 'LP05'),
    _hook('x 2 : [41.78, 10.42, -41.78] N', 'gray 0.021; flip 51.7% - byte-equal final.stl to LH05 and WP4', 'optimizer hang x0.776, pull_off x1.099, mass x1.001; P2 hang x0.788, pull_off x1.067', 'LP20'),
    _hook('x 0.5 : [0, 0, -0.98] N', 'gray 0.021; flip 51.7% - byte-equal final.stl to LP20', 'optimizer hang x0.776, pull_off x1.099, mass x1.001', 'LH05'),
    _hook('x 2 : [0, 0, -3.92] N', 'gray 0.024; flip 14.2% - byte-equal final.stl to LP05', 'optimizer hang x1.046, pull_off x0.888, mass x0.998', 'LH20'),
    _hook('x 2 for both cases', 'gray 0.025; flip 0.0% - byte-equal to B0; compliance x 4', 'optimizer hang x1.000, pull_off x1.000, mass x1.000; inside the noise (all three optimizer values)', 'LA20'),
    _hook('pull_off force x0.5, x1, x2: the stress, which the stiffness ratios do not show', 'the layout depends on the ratio of the weighted loads only; the load level sets the stress', 'P2 von Mises p99.9 of pull_off: 0.99 MPa (x0.5), 1.66 MPa (base), 3.08 MPa (x2 on pull_off)', 'LP05, B0, LP20'),
)
_E1_WEIGHT = (
    _hook('4.0 (from 1.0)', 'gray 0.021; flip 51.7% - byte-equal final.stl to LP20', 'optimizer hang x0.776, pull_off x1.099, mass x1.001', 'WP4'),
)


# --- top level, regions, load cases, material --------------------------------------------------------------------------------
# --- the rows of the E2 series by key. Each number is from the run folders (`result/metrics.json`, `verify/verify.json`, `history.jsonl`, `run.log`) or from the
# tables of the report. Stress rows (`_st`): the ratios are against the MEAN of the 6 base runs (E2/ST0 to ST5). Other rows: against B0 (the pairs say so). -----------------------------
_E2_STRESS_LIMIT = (
    _st("`pull_off` limit 0.35 MPa (tighter; from 0.5)", "relaxed peak 0.332 MPa (limit 0.35, met); thresholded voxel peak 0.371 MPa with 8 voxels above the limit (grey design 0.418 MPa, 50 above); "
        "only 1 of 4 stage ends feasible; rejected tries 7 (base 34 to 43); P2 p99.9 1.295 MPa, inside the base range 1.225 to 1.333",
        "optimizer hang x1.795, pull_off x1.691, mass x1.461; P2 hang x1.754, pull_off x1.472", "SL035"),
    _st("`pull_off` limit 0.7 MPa (looser)", "relaxed peak 0.673 MPa (met); thresholded voxel peak 0.677 MPa, none above; P2 p99.9 1.622 MPa (the base range is 1.225 to 1.333)",
        "optimizer hang x0.695, pull_off x0.695, mass x0.852; P2 hang x0.696, pull_off x0.725", "SL07"),
    _st("`pull_off` limit 1.0 MPa (looser)", "relaxed peak 0.960 MPa (met); thresholded voxel peak 0.968 MPa, none above; P2 p99.9 2.235 MPa",
        "optimizer hang x0.426, pull_off x0.477, mass x0.708; P2 hang x0.440, pull_off x0.523", "SL10"),
    _st("`hang` limit 0.04 MPa (tighter; from 1.5; the row was not binding, its relaxed peak in the base is 0.054 MPa)", "the row binds (relaxed `hang` peak 0.0394 MPa, met); the `pull_off` row stays met; "
        "P2 p99.9 of `hang` 0.0625 MPa (base 0.0770 to 0.0911); rejected tries 80",
        "optimizer hang x1.095, pull_off x1.030, mass x1.035; P2 hang x1.112, pull_off x1.038", "SH004"),
    _st("`hang` limit 3.0 MPa (looser; the row was not binding)", "nothing that the optimizer can use changed; rejected tries 60 (base 34 to 43: the count is noisier than the 6 base runs show)",
        "optimizer hang x1.012, pull_off x1.001, mass x1.004; P2 hang x0.991, pull_off x0.995; inside the noise (all three optimizer values)", "SH30"),
)
_E2_STRESS_Q = (
    _st("20 (from 40)", "relaxed peak 0.474 MPa (met); thresholded voxel peak 0.486 MPa; 1 grey voxel above the limit (base 27 to 41); rejected tries 17 (base 34 to 43). The repeat E2/SQ20r gave mass x0.958 "
        "(31.69 g), optimizer hang x1.114, pull_off x1.207 and 17 rejected tries: the move repeats. The mesh step removed 1 loose body in each run (81.6 mm3 in SQ20, 0.0016 mm3 in SQ20r; "
        "`metrics.mesh.extra_bodies_removed`)",
        "optimizer hang x1.113, pull_off x1.208, mass x0.954; P2 hang x0.974, pull_off x1.127", "SQ20, SQ20r"),
    _st("80", "relaxed peak 0.478 MPa (met); 109 grey voxels above the limit (base 27 to 41); rejected tries 104 (base 34 to 43): a higher q cost mass and tries and gave no margin",
        "optimizer hang x1.020, pull_off x0.945, mass x1.120; P2 hang x1.058, pull_off x0.977", "SQ80"),
)
_E2_STRESS_POWER = (
    _st("0.35 (from 0.5)", "relaxed peak 0.486 MPa (met); grey voxel peak 0.552 MPa, 13 grey voxels above the limit; rejected tries 57",
        "optimizer hang x1.073, pull_off x1.015, mass x1.054; P2 hang x1.033, pull_off x0.981", "SP035"),
    _st("0.75", "relaxed peak 0.471 MPa (met); grey voxel peak 0.738 MPa (base 0.565 to 0.627) and 106 grey voxels above the limit; rejected tries 65",
        "optimizer hang x0.990, pull_off x1.007, mass x1.012; P2 hang x1.036, pull_off x1.024", "SP075"),
    _st("1.0", "a grey design: gray 0.270 (base 0.072 to 0.080); relaxed peak 0.443 MPa (met, below the limit); grey voxel peak 0.823 MPa with 571 grey voxels above the limit; "
        "thresholded peak 0.539 MPa; rejected tries 80",
        "optimizer hang x0.816, pull_off x0.957, mass x1.088; P2 hang x1.182, pull_off x1.346", "SP10"),
)
_E2_STRESS_SCALE = (
    _st("1.0 (from auto)", "the row is a bound (the KS aggregate is at or above the peak): relaxed peak 0.450 MPa (limit 0.5, met); 9 grey voxels above the limit, none in the thresholded design; rejected tries 36",
        "optimizer hang x1.143, pull_off x1.125, mass x1.060; P2 hang x1.136, pull_off x1.094", "SS1"),
    _st("0.5 (below 1: the optimizer holds twice the limit)", "relaxed peak 0.957 MPa against the limit 0.5 MPa, and `relaxed.met` is true; voxel peaks 1.283 MPa (grey) and 0.957 MPa (thresholded), 770 and 583 "
        "voxels above the limit; P2 p99.9 1.716 MPa; the run log ends with `stress peak above the limit: stress:pull_off relaxed peak 0.957 MPa, limit 0.5 MPa (+91.39%)`; `topopt check` printed no fact for this value (before F4-28)",
        "optimizer hang x0.454, pull_off x0.464, mass x0.761; P2 hang x0.523, pull_off x0.567", "SS05"),
    _st("2.0 (above 1: the optimizer holds half the limit)", "relaxed peak 0.336 MPa (limit 0.5); the `pull_off` row is NOT met (g +0.4755), 0 of 4 stage ends feasible, `feasible` false, while the voxel peaks "
        "(0.414 MPa grey, 0.366 MPa thresholded) are below the limit; `topopt check` has a fact for this value (the relaxed peak is held to at most 0.5 times the limit)",
        "optimizer hang x2.118, pull_off x1.899, mass x1.916; P2 hang x2.104, pull_off x1.637", "SS2"),
)
_E2_STEP_ACCEPTANCE = (
    _st("false (from true)", "both rows met, 4 of 4 stage ends feasible; relaxed peak 0.485 MPa; no effect found; the history has no `step_acceptance` field, so there is no count of rejected tries",
        "optimizer hang x1.001, pull_off x1.016, mass x1.003 (pull_off is 0.1 % above the range of the 6 base runs: 1096.467 N/mm against a maximum of 1095.152 N/mm); P2 hang x1.005, pull_off x1.014", "SA0"),
)
_E2_MOVE = (
    _st("0.015 (from 0.03) on the stress case", "relaxed peak 0.487 MPa (met); rejected tries 45 (base 34 to 43)",
        "optimizer hang x1.020, pull_off x1.020, mass x1.016; P2 hang x0.976, pull_off x0.965", "SM15"),
    _st("0.06 on the stress case", "relaxed peak 0.470 MPa (met); rejected tries 67. The repeat E2/SM60r gave optimizer hang x1.038, pull_off x1.082, mass x0.995 and 73 rejected tries: the move repeats "
        "(pull_off stiffness +8 to +9 % at the same mass, 1.6 to 2 times the rejected tries)",
        "optimizer hang x1.044, pull_off x1.091, mass x0.997; P2 hang x1.047, pull_off x1.075", "SM60, SM60r"),
)
_E2_FILTER_SOLID = (
    _st("ignore (from density_one) on the stress case", "relaxed peak 0.482 MPa (met); mass 33.53 g (0.4 % above the base maximum); the voxel and P2 measures are inside the base range",
        "optimizer hang x1.039, pull_off x1.042, mass x1.013; P2 hang x1.018, pull_off x1.016", "SFI"),
)
_E2_ITER = (
    _st("110 with the schedule [[0,1],[30,2],[50,4],[70,8]] (from 220) on the stress case", "2 bodies in final.stl; the `pull_off` row NOT met (g +0.0053); 2 of 4 stage ends feasible; `topopt verify` stopped "
        "(exit code 2: the tet volume differs from the STL volume), so there is no P2 value; 110 passes were too few for this case",
        "optimizer hang x1.047, pull_off x0.951, mass x1.193", "SI110"),
)
_E2_VF = (
    _st("0.5 (from 0.3) under `objective: mass`: the value is the start density only", "relaxed peak 0.475 MPa (met); rejected tries 53; the voxel peaks are inside the base range. The repeat E2/SV50r gave mass x0.902 "
        "(29.86 g), optimizer pull_off x1.218, P2 pull_off x1.166 and 54 rejected tries: the move repeats",
        "optimizer hang x1.025, pull_off x1.218, mass x0.901; P2 hang x1.068, pull_off x1.161", "SV50, SV50r"),
)


_E2_ETA = (
    _e2("0.60 with `robust: eroded` (E1 has 0.55 and 0.65)", "gray 0.026; flip 7.2%; the eroded compliance is 5.1 % above the nominal one; 1 warning line (the eroded design keeps a wall of 1.98 mm, 0.99 voxels, at the filter radius "
        "of 3 mm: a limit below 2 voxels is at the grid scale, so robust mode has little effect on the member size); the robust part took 0.25 s",
        "optimizer hang x0.995, pull_off x0.967, mass x0.995; P2 hang x0.994, pull_off x0.964", "RE60"),
    _e2("0.75", "gray 0.034; flip 11.7%; the eroded compliance is 27.0 % above the nominal one; 1 warning line (a wall of 2.65 mm, 1.3 voxels, at the filter radius of 3 mm); the robust part took 0.27 s",
        "optimizer hang x0.982, pull_off x0.941, mass x0.996; P2 hang x0.975, pull_off x0.944", "RE75"),
)
_E2_PERIMETER = (
    _e2("30000 mm2 (the final perimeter of B0 is 16030.6 mm2: the row is not active)", "gray 0.025; flip 0.0%; final.stl byte-equal to B0 (only the compliance of the last pass differs, in the 7th digit)",
        "optimizer hang x1.000, pull_off x1.000, mass x1.000; P2 hang x1.000, pull_off x1.000; inside the noise (all three optimizer values)", "PM30"),
    _e2("14000 mm2 (2.07 times the smallest possible perimeter, 6776.0 mm2 by `topopt check`)", "the row binds (final 14002.6 mm2, g +0.00019); gray 0.023; flip 37.8%; P2 falls more than the optimizer and outside the P2 band",
        "optimizer hang x0.918, pull_off x0.939, mass x1.000; P2 hang x0.843, pull_off x0.846", "PM14"),
    _e2("11000 mm2 (1.62 times the smallest possible)", "the row binds (final 11009.8 mm2) and the volume row goes slack (volume used 0.145 against 0.2); 2 bodies; gray 0.123; flip 67.8%",
        "optimizer hang x0.417, pull_off x0.473, mass x0.852; P2 hang x0.566, pull_off x0.519", "PM11"),
)
_E2_SOLID = (
    _e2("an 8 x 8 x 8 mm block (64 voxels, centre x 29, y 0, z -6 mm), volume_fraction 0.2 unchanged", "gray 0.024; flip 34.1%; the target volume rose by 409.6 mm3 (22889.6 to 23299.2); `compare_runs` shows "
        "'64 voxel labels differ: design->load 64'",
        "optimizer hang x0.996, pull_off x0.974, mass x1.020; P2 hang x1.000, pull_off x0.992", "SOL1"),
    _e2("the same block, volume_fraction 0.19325 (keeps the target volume 22889.6 mm3)", "gray 0.023; flip 39.2%",
        "optimizer hang x0.956, pull_off x0.948, mass x0.999; P2 hang x0.962, pull_off x0.973", "SOL1v"),
)
_E2_DISPLACEMENT = (
    _e2("`tip` in `pull_off`, limit 0.1 mm (B0 has 0.02397 mm: the row is not active)", "gray 0.025; flip 0.0%; the run equals B0 within the E1 bands (final.stl bytes differ, the compliance of the last pass +2.6e-5)",
        "optimizer hang x1.000, pull_off x1.000, mass x1.000; P2 hang x1.006, pull_off x1.016; inside the noise (all three optimizer values)", "LDa"),
    _e2("limit 0.022 mm (8 % under the value of B0)", "met (value 0.02177 mm); the stiffness moves from `hang` to `pull_off`; gray 0.031; flip 61.6%",
        "optimizer hang x0.665, pull_off x1.064, mass x0.998; P2 hang x0.701, pull_off x1.084", "LDb"),
    _e2("limit 0.019 mm (21 % under the value of B0)", "NOT met (value 0.02070 mm, g +0.0896, `feasible` false); the run log says `constraint not met: displacement:tip_pull:pull_off value 0.020702 limit 0.019 (+8.96% of the limit)`; "
        "gray 0.025; flip 61.9%",
        "optimizer hang x0.570, pull_off x1.112, mass x1.003; P2 hang x0.603, pull_off x1.125", "LDc"),
)
_E2_MIRROR = (
    _e2("plane y = 0, `align` auto with the real `pull_off` load (a side part, Fy +5.21 N; loads are not mirrored); auto chose `face` (50 label mismatches against 1140 for `centre`; 12 orphan voxels)",
        "gray 0.024; flip 25.9%; `run.mirror.end.rho_deviation` 0.0658; mirror error of final.stl 0.462 mm (B0 without a mirror: 4.99 mm); the grid origin moved along y from -17.5 to -18.0 mm, so `compare_runs` says "
        "the grids differ",
        "optimizer hang x0.992, pull_off x0.886, mass x1.002; P2 hang x1.010, pull_off x0.869", "MI2"),
    _e2("plane y = 0, `align` centre (forced), real load", "gray 0.025; 570 orphan voxels (7.5 % of the design voxels) and one more y layer in the grid (19 against 18; flip not comparable); "
        "`run.mirror.end.rho_deviation` 0.275; mirror error of final.stl 3.36 mm",
        "optimizer hang x0.997, pull_off x0.946, mass x0.998; P2 hang x1.000, pull_off x0.890", "MI2c"),
    _e2("plane y = 0, auto (face), symmetric load (`pull_off` force with Fy = 0); the reference is B0s", "gray 0.023; flip 18.2% against B0s; `run.mirror.end.rho_deviation` 0.0353; mirror error 0.471 mm (B0s: 3.49 mm)",
        "optimizer hang x0.990, pull_off x0.995, mass x1.000; P2 hang x0.979, pull_off x0.931", "MI2s"),
    _e2("plane y = 0, centre, symmetric load; the reference is B0s", "gray 0.024; 570 orphan voxels; `run.mirror.end.rho_deviation` 0.108; mirror error 1.94 mm (B0s: 3.49 mm)",
        "optimizer hang x0.992, pull_off x0.995, mass x0.998; P2 hang x0.989, pull_off x0.964", "MI2cs"),
    _e2("`hook_mirror.yaml` as it is (plane y = 0, auto, voxel 1.2 mm, 80 passes, real load) against `hook.yaml` (E1/H0)", "gray 0.017; converged; auto chose `centre` (0 label mismatches against 134 for `face`), 0 orphans; "
        "`run.mirror.end.rho_deviation` 3.3e-16; mirror error of final.stl 0.201 mm (H0: 7.91 mm); loop 388 s against 247 s of H0 on a host with a different load, so the ratio is not a price of the key",
        "optimizer hang x1.016, pull_off x0.921, mass x1.001; P2 hang x1.029, pull_off x0.866", "MIR", HOOK_FINE),
)
_E2_ALIGN = (_E2_MIRROR[0], _E2_MIRROR[1], _E2_MIRROR[4])


_E2_BK_MIN = (
    _bk("150, the value of the public `hook_buckling.yaml` (B0 has the printed factors 1401.87 for `hang` and 123.085 for `pull_off`)",
        "the printed design has the lowest factors 1621.7 and 148.3 (+15.7 % and +20.5 %); the row holds the gray `pull_off` factor at 149.49 at the last pass (g +0.0034, met); the stage end 4 to 8 was NOT met (g +0.0407) "
        "and the last stage brought it back; flip 9.2 % from B0; loop 171.5 s and 181.4 s (10.4 and 11.0 times the 16.55 s of B0); the two runs have byte-equal final.stl",
        "optimizer hang x0.997, pull_off x1.004, mass x0.999; P2 hang x0.980, pull_off x0.917", "BK0, BK1"),
    _bk("150.15 (+0.1 % of the request)", "the printed factors 1603 and 146.7; the gray `pull_off` factor at the last pass 161.7 against 149.5 (+8.2 %), g -0.0716 against +0.0034, the stage end 4 to 8 met; flip 0.72 % from BK0; "
        "loop 206.2 s (12.5 times B0)",
        "optimizer hang x0.996, pull_off x1.006, mass x0.998; P2 hang x0.968, pull_off x0.922", "BK2"),
    _bk("100 (under the factor 123.1 of B0: the row is slack, g -0.3192 at the last pass)", "the printed factors 1493 and 154.1 (+6.5 % and +25 %): the design still moved (flip 6.2 % from B0); loop 189.4 s "
        "(11.4 times B0); the row ran at all 41 passes for `pull_off` and at 11 for `hang`",
        "optimizer hang x0.999, pull_off x0.998, mass x1.000; P2 hang x1.005, pull_off x1.033; inside the noise (all three optimizer values)", "BKfL"),
    _bk("250", "the printed factors 2219 and 293.6 (+58 % and +139 %); the gray `pull_off` factor of the row 269.2 (g -0.0713, slack); flip 14.6 % from B0; loop 228.9 s (13.8 times B0)",
        "optimizer hang x0.987, pull_off x0.997, mass x0.998; P2 hang x0.967, pull_off x0.915", "BKfH"),
)
_E2_BK_Q = (
    _bk("20 (from 40)", "the printed factors 1611 and 147.9; flip 0.39 % from BK0; loop 192.4 s (11.6 times B0): no effect found",
        "optimizer hang x0.997, pull_off x1.004, mass x0.999; P2 hang x0.979, pull_off x0.937", "BKq20"),
    _bk("80", "the printed factors 1622 and 148.3; flip 0.00 % from BK0 (the STL bytes differ); the stage end 4 to 8 NOT met (g +0.0421); loop 188.3 s (11.4 times B0): no effect found",
        "optimizer hang x0.997, pull_off x1.004, mass x0.999; P2 hang x0.967, pull_off x0.921", "BKq80"),
)
_E2_BK_MODES = (
    _bk("modes 2 (from 6; in the row and in the check)", "final.stl byte-equal to BK0: the design does not change; the loop is 184.5 s (11.1 times B0) against 171.5 s and 181.4 s, so 2 modes save no time",
        "optimizer hang x0.997, pull_off x1.004, mass x0.999; P2 hang x0.980, pull_off x0.917", "BKm2"),
    _bk("modes 12", "final.stl byte-equal to BK0; the loop is 490.4 s (29.6 times B0; 2.7 to 2.9 times BK1 and BK0) and the median pass 9.64 s against 3.38 s; wall 498 s",
        "optimizer hang x0.997, pull_off x1.004, mass x0.999; P2 hang x0.980, pull_off x0.917", "BKm12"),
    _e2("`topopt buckling <run> --modes 2` on a copy of B0 (hook_short, 2 mm)", "the lowest factors 1401.87 and 123.085 are the same as for 6 modes; 25 solves for each case against 40 and 54",
        "6.8 s for the command (the default took 9.8 s and 10.5 s); peak 564 MB", "T-bk-m2, T-bk-def, T-bk-def2"),
    _e2("`topopt buckling <run> --modes 12` on a copy of B0", "the lowest factors 1401.87 and 123.085 again; 78 and 81 solves; 12 modes found for each case",
        "16.6 s for the command (1.7 times the default); peak 581 MB", "T-bk-m12"),
)
_E2_BK_CHECK = (
    _e2("`topopt buckling <run>` with the settings of the run, on a copy of B0 (hook_short, 2 mm, 6 modes), two calls", "metrics.buckling is rewritten: the lowest factors 1401.87 (`hang`) and 123.085 (`pull_off`), the same in both calls; "
        "6 modes found for each case; 40 and 54 solves",
        "9.8 s and 10.5 s for the command (8.4 s and 9.2 s in `metrics.buckling.seconds`; the factorization 0.9 s and 1.0 s); peak 575 MB and 577 MB; the host load average was about 30", "T-bk-def, T-bk-def2"),
    _e2("`topopt buckling <run>` on a copy of H0 (the public hook, voxel 1.2 mm)", "the lowest factors 1381.09 and 137.004, 6 modes for each case; 53 and 55 solves",
        "47.3 s for the command (45.8 s in `metrics.buckling.seconds`); peak 1823 MB", "T-bk-H0", HOOK_FINE),
)
_E2_BK_DENSITY = (
    _e2("`--density gray` on a copy of B0 (2 mm)", "metrics.buckling.density gray: the lowest factors 1325.04 and 116.372 against 1401.87 and 123.085 for `printed` (5.5 % and 5.5 % lower)",
        "10.2 s for the command (the printed check 9.8 s and 10.5 s)", "T-bk-gray"),
    _e2("`--density gray` on a copy of H0 (1.2 mm)", "the lowest factors 1338.43 and 132.798 against 1381.09 and 137.004 for `printed` (3.1 % and 3.1 % lower)",
        "32.2 s for the command (the printed check 47.3 s); peak 1817 MB", "T-bk-H0g", HOOK_FINE),
)
_E2_BK_BUDGET = (
    _e2("`--max-gb 0.3` on a copy of B0 (the memory estimate is 0.6 GB)", "the check does not start: exit code 3, status `skipped`, the fact names the estimate and the budget, the options are a larger G or a larger voxel size; "
        "no factor",
        "1.3 s for the command", "T-bk-gb"),
    _e2("`--max-seconds 1` on a copy of B0", "both eigen solves stop at the budget (17 and 19 solves of 1.0 s); exit code 0; no factor (\"no positive load factor in any case\") and two warnings with the option to raise S",
        "5.0 s for the command", "T-bk-ms"),
)
_E2_VERIFY_BK = (
    _e2("`topopt verify --buckling` on a copy of B0 (tet model, 2 mm run)", "the tet model gives the lowest factors 1250.2 (`hang`) and 109.294 (`pull_off`), 0.892 and 0.888 times the voxel values 1401.87 and 123.085 "
        "(the KB value of the 1.2 mm hook is 0.786 and 0.713); the check adds the key `buckling` to the result",
        "6.6 s for the whole command, of which 2.1 s are the buckling part", "T-verifyb"),
)


_E2_PRINT = (
    _e2("`print_check` on a copy of B0 (hook_short, 2 mm run folder), no limits", "4 pass, 0 fail, 4 unknown (overhang 2,414 mm2, 19.17 % of the area below 45 degrees; a support estimate of 52,600 mm3; 122.5 mm2 on the plate; "
        "thinnest member 2.818 mm at p1: the numbers are in the text of the unknown checks); no file written into the run folder",
        "3.2 s and 3.9 s in two calls (the thickness estimate is 2.13 s of the first and 2.77 s of the second)", "T-print"),
    _e2("the same, with `max_overhang_area_mm2` 500 and `min_wall_mm` 1.2", "4 pass, 1 fail (overhang; the gate is `fail` only with `strict`), 3 unknown; the wall check stays unknown: the limit is under 3 times the voxel size "
        "of the run (2 mm), and the reply says that a pass needs a resolution of 0.3999 mm or less",
        "3.5 s", "T-print"),
)
_E2_CLEANUP = (
    _e2("`cleanup_list` with `estimate_only` on a copy of B0", "no search: the reply gives the detector grid (0.5 mm pitch, 2,165,566 voxels) and the memory model (733 MB); the judge is not available when the YAML has no "
        "`acceptance` block", "0.63 s", "T-cleanup"),
    _e2("`cleanup_list` on a copy of B0 that holds `verify/`", "found 1, listed 1, 15 near frozen regions excluded; the one mark is a bump, operator remove cut, confidence high, 49.875 mm3, 13.0 mm long, nearest frozen region "
        "6.75 mm; the stress ratio 0.402 comes from `verify/fields.npz`; marks.json and cleanup.json written", "3.1 s", "T-cleanup"),
    _e2("the same on a copy without `verify/`", "the same candidate with no stress facts, and the note that `verify/fields.npz` is missing with the option to run `topopt verify` first", "3.7 s", "T-cleanup-nov"),
)
_E2_APPLY = (
    _e2("`apply_marks` with the `remove` mark of `cleanup_list` on a copy of B0", "applied; volume -40.25 mm3 (-0.194 %, 81 % of the 49.875 mm3 that the detector listed), mass -0.05 g; 947 of 20586 triangles replaced (4.6 %), "
        "24494 triangles in the edited mesh (+19 %); 1 body, closed; the frozen regions unchanged; the parent check within tolerance; the run folder and final.stl not changed (edited.stl and apply_marks.json go to `out_dir`)",
        "3.1 s", "T-marks"),
    _e2("one `smooth` mark, radius 4 mm", "applied; volume -2.2 mm3 (-0.011 %), mass -0.003 g; 386 of 20586 triangles replaced (1.88 %); 1 body, closed", "2.4 s", "T-marks-smooth"),
    _e2("one `add` mark, radius 4 mm", "applied; volume +7.5 mm3 (+0.036 %), mass +0.009 g; 375 of 20586 triangles replaced (1.82 %); 1 body, closed", "2.7 s", "T-marks-add"),
)
_E2_COMPARE = (
    _e2("`compare_runs` B0 against RE60 (`robust: eroded`, `eta_eroded` 0.6), labels checked", "11 summary lines (compliance_final +1.9 %, pull_off stiffness -3.3 %, mass -0.5 %); `setup.run_json_diff` empty; "
        "`setup.yaml_diff` has 2 keys (`settings.robust`, `settings.eta_eroded`); the voxel labels are equal; no problem", "0.67 s", "T-compare"),
    _e2("the same, `check_labels` false", "11 summary lines as with the check, and `setup.labels_diff` (from the labels.bin of the two runs) still says the labels are equal; what changes is `inputs.<run>.labels_vs_disk`: "
        "`checked` false with the reason 'not requested (check_labels=false)', and the fact that the YAML and STL files on disk voxelize to the same labels as labels.bin is not stated", "0.12 s", "T-compare"),
    _e2("B0 against MI2 (a mirror run)", "12 summary lines; the grid origin moved (y -17.5 against -18.0 mm), so `setup.labels_diff.comparable` is false and no label difference is given; the summary line says "
        "`run.json differs in 32 keys` and names the first six; `mirror.planes[...]` is in `setup.run_json_diff`; `setup.yaml_diff` has 2 keys (`mirror[0].normal`, `mirror[0].at_mm`); a note says that the target volumes differ",
        "0.66 s", "T-compare"),
    _e2("B0 against SOL1 (a solid block)", "13 summary lines; `run.json differs in 4 keys` (`regions.solid`, `region_files.solid`, the target volume 22889.6 against 23299.2 mm3, the target mass); "
        "`64 voxel labels differ: design->load 64` (a solid region has the label `load`); the same note on the target volume", "0.65 s", "T-compare"),
)
_E2_RATCHET = (
    _e2("`ratchet_run` with `estimate_only` (default), rule `mass` (`must_fall`), the mark of `cleanup_list`, on a copy of B0", "candidates 1, judge calls 2, verify calls 0; the reply says that the time of printcheck and "
        "apply_marks is not in the verify estimate", "3.7 s", "T-ratchet"),
    _e2("the same with rule `stiffness` (`max_drop_pct` 5, a test value)", "candidates 1, judge calls 2, verify calls 2 at 3.29 s each (the estimate of the verify time only)", "4.3 s", "T-ratchet2"),
    _e2("a real job, rule `mass` (`must_fall`), `estimate_only` false", "decision ACCEPT: mass 25.6659 g against the parent 25.7159 g; the baseline was judged in 3.68 s and the candidate in 6.97 s (source 3.42 s, printcheck 3.54 s); "
        "peak 0.33 GB", "15.5 s in all: the start call 4.8 s and the background job 10.7 s", "T-ratchet-M"),
    _e2("a real job, rule `stiffness` (`max_drop_pct` 5; needs `verify`)", "decision ACCEPT: stiffness `hang` 1055.37 against 1062.78 N/mm (-0.70 %) and `pull_off` 904.594 against 909.182 N/mm (-0.50 %), both under the limit of 5 %; "
        "the baseline was judged in 8.4 s (printcheck 3.05 s, verify 5.33 s), the candidate in 10.26 s (apply 2.48 s, printcheck 2.93 s, verify 4.84 s); peak 0.53 GB",
        "23.3 s in all: the start call 4.6 s and the background job 18.7 s", "T-ratchet-S"),
)


# --- the rows of the E3 series by key: the bracket (grey base `bracket`, fitted schedule `bracket-fitted`) and the ladle (`ladle-4mm`, and the filter rows of the example grid `ladle`). Each number is
# from the run folders (`result/metrics.json`, `verify/verify.json`, `history.jsonl`, `verdict.json`, `run.log`) or from the tables of the report; the price strings follow the form of `_e3` and the bands of the part
# classes. Rows with the same sign on the fitted and on the grey bracket base are given for the fitted base only (the sentence of the card has the grey number). -----------------------------------------------
_E3_FILTER_RADIUS = (
    _e3('1.0 mm = 1 voxel (no filter)', 'gray 0.002; flip 31.9 %; 2 islands removed; converged (40 passes); objective change 9e-07; filter radius 1 mm = 1 voxel, 1 entry per row; no warning line',
        'optimizer pull x0.956, mass x1.009; P2 x0.964', 'BR-SFR1', BR_FIT),
    _e3('2.0 mm = 2 voxels', 'gray 0.058; flip 8.7 %; iteration_limit (40 passes); objective change 0.001; filter radius 2 mm = 2 voxels, 27 entries per row; no warning line',
        'optimizer pull x0.977, mass x0.998; P2 x1.011', 'BR-SFR2', BR_FIT),
    _e3('3.0 mm = 3 voxels', 'gray 0.107; flip 10.9 %; iteration_limit (40 passes); objective change 0.0019; filter radius 3 mm = 3 voxels, 93 entries per row; no warning line',
        'optimizer pull x0.926, mass x0.991; P2 x0.988', 'BR-SFR3', BR_FIT),
    _e3('1.0 mm = 1 voxel (no filter)', 'grey base (BR-B0 ends at beta 2, gray 0.129); gray 0.027; flip 33.8 %; 3 islands removed; iteration_limit (40 passes); objective change 8.6e-07; last beta stage not reached (ends at beta 2); filter radius 1 mm = 1 voxel, 1 entry per row; no warning line; the sign of the optimizer ratio depends on the base: up here, down on the fitted base (E3/BR-SFR1)',
        'optimizer pull x1.049, mass x1.003; P2 x0.987', 'BR-FR1', BR_E3),
    _e3('3.0 mm = 3 voxels', 'grey base (BR-B0 ends at beta 2, gray 0.129); gray 0.259; flip 15.4 %; iteration_limit (40 passes); objective change 0.00011; last beta stage not reached (ends at beta 2); filter radius 3 mm = 3 voxels, 93 entries per row; no warning line; P2 is outside its band here and inside it on the fitted base (E3/BR-SFR3)',
        'optimizer pull x0.835, mass x0.977; P2 x0.951', 'BR-FR3', BR_E3),
    _e3('4.0 mm = 1 voxel (no filter)', 'gray 0.004; flip 64.5 %; 27 islands removed; converged (80 passes); objective change 8.5e-07; filter radius 4 mm = 1 voxel, 1 entry per row; no warning line; the case `stir` is up in the optimizer and down in P2',
        'optimizer scoop x0.863, scrape x0.841, stir x1.102, objective x0.894, mass x0.979; P2 scoop x0.747, scrape x0.718, stir x0.925, objective x0.764', 'LAD-FR1', LAD4),
    _e3('8.0 mm = 2 voxels', 'gray 0.037; flip 42.2 %; iteration_limit (80 passes); objective change 0.0011; filter radius 8 mm = 2 voxels, 27 entries per row; no warning line; the cases move in opposite directions and the P2 objective stays inside its band',
        'optimizer scoop x1.023, scrape x1.001, stir x0.834, objective x0.953, mass x0.990; P2 scoop x1.084, scrape x1.057, stir x0.857, objective x0.996', 'LAD-FR2', LAD4),
    _e3('12.0 mm = 3 voxels', 'gray 0.066; flip 52.9 %; iteration_limit (80 passes); objective change 0.0016; filter radius 12 mm = 3 voxels, 93 entries per row; no warning line',
        'optimizer scoop x0.896, scrape x0.888, stir x0.671, objective x0.821, mass x0.978; P2 scoop x1.041, scrape x1.030, stir x0.808, objective x0.961', 'LAD-FR3', LAD4),
    _e3('3.25 mm = 1 voxel (no filter)', 'gray 0.008; flip 63.3 %; 35 islands removed; converged (80 passes); objective change 9.3e-07; filter radius 3.25 mm = 1 voxel, 1 entry per row; no warning line; the case `stir` is up in the optimizer and down in P2',
        'optimizer scoop x0.930, scrape x0.871, stir x1.046, objective x0.909, mass x0.977; P2 scoop x0.815, scrape x0.738, stir x0.932, objective x0.781', 'LAD-XFR1', LAD_EX),
    _e3('6.5 mm = 2 voxels', 'gray 0.031; flip 36.5 %; iteration_limit (80 passes); objective change 0.00043; filter radius 6.5 mm = 2 voxels, 27 entries per row; no warning line',
        'optimizer scoop x1.048, scrape x1.019, stir x0.903, objective x0.988, mass x0.995; P2 scoop x1.094, scrape x1.056, stir x0.948, objective x1.026', 'LAD-XFR2', LAD_EX),
)

_E3_MIN_MEMBER = (
    _e3('1.0 mm = 1 voxel', 'gray 0.052; flip 7.7 %; iteration_limit (40 passes); objective change 0.0011; filter radius 1.877 mm = 1.88 voxels, 27 entries per row; 1 warning line',
        'optimizer pull x0.981, mass x0.998; P2 x1.005', 'BR-SMM1', BR_FIT),
    _e3('2.0 mm = 2 voxels', 'gray 0.137; flip 13.3 %; iteration_limit (40 passes); objective change 0.0021; filter radius 3.754 mm = 3.75 voxels, 251 entries per row; no warning line',
        'optimizer pull x0.889, mass x0.986; P2 x0.977', 'BR-SMM2', BR_FIT),
    _e3('2.0 mm = 2 voxels', 'grey base (BR-B0 ends at beta 2, gray 0.129); gray 0.314; flip 21.6 %; iteration_limit (40 passes); objective change 7.2e-05; last beta stage not reached (ends at beta 2); filter radius 3.754 mm = 3.75 voxels, 251 entries per row; no warning line',
        'optimizer pull x0.755, mass x0.970; P2 x0.907', 'BR-MM2', BR_E3),
    _e3('4.0 mm = 1 voxel', 'gray 0.033; flip 39.1 %; iteration_limit (80 passes); objective change 0.00047; filter radius 7.508 mm = 1.88 voxels, 27 entries per row; 1 warning line',
        'optimizer scoop x1.020, scrape x1.003, stir x0.879, objective x0.968, mass x0.988; P2 scoop x1.055, scrape x1.034, stir x0.894, objective x0.993', 'LAD-MM1', LAD4),
    _e3('8.0 mm = 2 voxels', 'gray 0.074; flip 62.0 %; iteration_limit (80 passes); objective change 0.0017; filter radius 15.02 mm = 3.75 voxels, 251 entries per row; no warning line; the P2 loss is in the case `stir` (x0.617); `scoop` x1.008 and `scrape` x0.993 are inside the band',
        'optimizer scoop x0.817, scrape x0.813, stir x0.490, objective x0.697, mass x0.973; P2 scoop x1.008, scrape x0.993, stir x0.617, objective x0.856', 'LAD-MM2', LAD4),
    _e3('6.5 mm = 2 voxels', 'gray 0.070; flip 61.8 %; iteration_limit (80 passes); objective change 0.0019; filter radius 12.2 mm = 3.75 voxels, 251 entries per row; no warning line',
        'optimizer scoop x0.891, scrape x0.880, stir x0.664, objective x0.814, mass x0.981; P2 scoop x1.055, scrape x1.036, stir x0.816, objective x0.968', 'LAD-XMM2', LAD_EX),
)

_F19_MIN_MEMBER = (
    _f19("min_member_mm 4.0 (filter 7.508 mm = 3.75 voxels, the wall rule) against no value",
         "gray 0.113; flip 55.7 %; iteration_limit (80 passes); objective change 0.00012; thickness of the design material at density 0.5 (mm): p5 3.5, median 6.25, 11 % below 4 mm (base: p5 2.319, median 4.25, 49 %); interior (162 voxels): p5 5.25, 1 % below (base: p5 2.812, 44 %); round rod-like voxels from 4 to 7.03 mm thick: 202 (14 % of the material; base 10 %); outside the band of the class: optimizer yes, P2 yes, mass yes",
         "optimizer hang x0.803, pull_off x0.821, objective x0.812, mass x0.974; P2 hang x0.866, pull_off x0.844, objective x0.854", "hookA-WALL", F19_HOOK),
    _f19("filter_radius 4.272 mm = 1.068 x 4.0 (2.14 voxels, the rod rule) against no value",
         "gray 0.063; flip 25.9 %; converged (80 passes); objective change 9.6e-05; thickness of the design material at density 0.5 (mm): p5 2.062, median 5.25, 33 % below 4 mm (base: p5 2.319, median 4.25, 49 %); interior (101 voxels): p5 3.5, 22 % below (base: p5 2.812, 44 %); round rod-like voxels from 4 to 7.03 mm thick: 170 (11 % of the material; base 10 %); outside the band of the class: optimizer yes, P2 no, mass yes",
         "optimizer hang x0.918, pull_off x0.953, objective x0.935, mass x0.989; P2 hang x0.952, pull_off x0.980, objective x0.967", "hookA-ROD", F19_HOOK),
    _f19("filter_radius 3.75 mm (1.88 voxels; 0.94 x the request, the ratio of the client pipeline for a 4 mm gate) against no value",
         "gray 0.052; flip 19.8 %; converged (80 passes); objective change 7.6e-05; thickness of the design material at density 0.5 (mm): p5 2.312, median 5, 39 % below 4 mm (base: p5 2.319, median 4.25, 49 %); interior (97 voxels): p5 3.5, 26 % below (base: p5 2.812, 44 %); round rod-like voxels from 4 to 7.03 mm thick: 187 (12 % of the material; base 10 %); outside the band of the class: optimizer yes, P2 no, mass yes",
         "optimizer hang x0.945, pull_off x0.972, objective x0.958, mass x0.994; P2 hang x0.984, pull_off x0.995, objective x0.990", "hookA-G375", F19_HOOK),
    _f19("min_member_mm 2.0 (filter 3.754 mm = 3.75 voxels, the wall rule) against no value",
         "gray 0.314; flip 21.6 %; iteration_limit (40 passes); objective change 7.2e-05; thickness of the design material at density 0.5 (mm): p5 3, median 5.75, 1 % below 2 mm (base: p5 1.75, median 5.047, 9 %); interior (1746 voxels): p5 3.688, 0 % below (base: p5 3.125, 0 %); round rod-like voxels from 2 to 3.51 mm thick: 174 (6 % of the material; base 3 %); outside the band of the class: optimizer yes, P2 yes, mass yes",
         "optimizer pull x0.755, mass x0.970; P2 x0.894", "bracket-WALL", F19_BRACKET),
    _f19("filter_radius 2.136 mm = 1.068 x 2.0 (2.14 voxels, the rod rule) against no value",
         "gray 0.192; flip 8.4 %; iteration_limit (40 passes); objective change 0.00023; thickness of the design material at density 0.5 (mm): p5 1.938, median 5, 8 % below 2 mm (base: p5 1.75, median 5.047, 9 %); interior (1606 voxels): p5 3.375, 0 % below (base: p5 3.125, 0 %); round rod-like voxels from 2 to 3.51 mm thick: 86 (3 % of the material; base 3 %); outside the band of the class: optimizer yes, P2 no, mass yes",
         "optimizer pull x0.930, mass x0.990; P2 x0.988", "bracket-ROD", F19_BRACKET),
    _f19("min_member_mm 6.5 (filter 12.2 mm = 3.75 voxels, the wall rule) against no value",
         "gray 0.070; flip 61.8 %; iteration_limit (80 passes); objective change 0.0019; thickness of the design material at density 0.5 (mm): p5 6.297, median 12.188, 5 % below 6.5 mm (base: p5 3.25, median 8.938, 36 %); interior (2594 voxels): p5 8.734, 1 % below (base: p5 4.062, 22 %); round rod-like voxels from 6.5 to 11.4 mm thick: 431 (12 % of the material; base 10 %); outside the band of the class: optimizer yes, P2 yes, mass yes",
         "optimizer scoop x0.891, scrape x0.880, stir x0.664, objective x0.814, mass x0.981; P2 scoop x1.045, scrape x1.029, stir x0.812, objective x0.962", "ladle-WALL", F19_LADLE),
    _f19("filter_radius 6.942 mm = 1.068 x 6.5 (2.14 voxels, the rod rule) against no value",
         "gray 0.034; flip 38.1 %; iteration_limit (80 passes); objective change 0.00071; thickness of the design material at density 0.5 (mm): p5 5.688, median 9.242, 25 % below 6.5 mm (base: p5 3.25, median 8.938, 36 %); interior (2536 voxels): p5 5.688, 13 % below (base: p5 4.062, 22 %); round rod-like voxels from 6.5 to 11.4 mm thick: 396 (11 % of the material; base 10 %); outside the band of the class: optimizer yes, P2 yes, mass yes",
         "optimizer scoop x1.034, scrape x1.005, stir x0.887, objective x0.973, mass x0.994; P2 scoop x1.087, scrape x1.051, stir x0.941, objective x1.020", "ladle-ROD", F19_LADLE),
)

_E3_ITER = (
    _e3('20 passes (half of 40; default schedule, the run ends at beta 1)', 'grey base (BR-B0 ends at beta 2, gray 0.129); gray 0.176; flip 9.5 %; iteration_limit (20 passes); objective change 0.0077; last beta stage not reached (ends at beta 1); wall 18 s',
        'optimizer pull x0.939, mass x0.994; P2 x1.013', 'BR-IT20', BR_E3),
    _e3('20 passes with the schedule scaled to them: [[0,1],[8,2],[13,4],[16,8]]', 'grey base (BR-B0 ends at beta 2, gray 0.129); gray 0.025; flip 11.4 %; iteration_limit (20 passes); objective change not judged (5 of 6 passes at beta 8 needed); wall 20 s; the sign of the optimizer ratio differs from the fitted base',
        'optimizer pull x1.078, mass x1.011; P2 x1.020', 'BR-IT20s', BR_E3),
    _e3('80 passes (double; the default schedule fits 80 passes)', 'grey base (BR-B0 ends at beta 2, gray 0.129); gray 0.019; flip 4.3 %; converged (80 passes); objective change 3.6e-05; wall 45 s',
        'optimizer pull x1.114, mass x1.010; P2 x1.038', 'BR-IT80', BR_E3),
    _e3('20 passes with the schedule scaled to them (half of the 40 of the base)', 'gray 0.025; flip 9.3 %; iteration_limit (20 passes); objective change not judged (5 of 6 passes at beta 8 needed); wall 20 s',
        'optimizer pull x0.973, mass x1.003; P2 x0.984', 'BR-IT20s', BR_FIT),
    _e3('80 passes with the default schedule (double; beta 8 from pass 65)', 'gray 0.019; flip 5.2 %; converged (80 passes); objective change 3.6e-05; wall 45 s',
        'optimizer pull x1.006, mass x1.002; P2 x1.001', 'BR-IT80', BR_FIT),
    _e3('40 passes (half of 80; default schedule, the run ends at beta 2)', "gray 0.085; flip 19.5 %; iteration_limit (40 passes); objective change 0.0053; last beta stage not reached (ends at beta 2); wall 76 s; gray 0.085 is below the verdict test of 0.1 although the run did not reach the last beta stage: the verdict's `last_beta_stage` shows it",
        'optimizer scoop x0.725, scrape x0.742, stir x0.696, objective x0.730, mass x0.986; P2 scoop x0.918, scrape x0.933, stir x0.899, objective x0.923', 'LAD-IT40', LAD4),
    _e3('40 passes with the schedule scaled to them: [[0,1],[15,2],[25,4],[32,8]]', 'gray 0.014; flip 26.6 %; iteration_limit (40 passes); objective change 0.0017; wall 81 s',
        'optimizer scoop x0.869, scrape x0.902, stir x1.005, objective x0.926, mass x1.000; P2 scoop x0.870, scrape x0.902, stir x1.002, objective x0.926', 'LAD-IT40s', LAD4),
    _e3('160 passes (double; default schedule)', 'gray 0.014; flip 0.1 %; converged (160 passes); objective change 9.1e-06; wall 272 s',
        'optimizer scoop x1.001, scrape x1.001, stir x1.001, objective x1.001, mass x1.000; P2 scoop x1.001, scrape x1.000, stir x1.000, objective x1.000; inside the noise', 'LAD-IT160', LAD4),
    _e3('160 passes with the schedule scaled by 2: [[0,1],[60,2],[100,4],[130,8]]', 'gray 0.014; flip 22.6 %; converged (160 passes); objective change 5.2e-05; wall 276 s',
        'optimizer scoop x1.049, scrape x1.037, stir x1.025, objective x1.034, mass x1.001; P2 scoop x1.045, scrape x1.031, stir x1.021, objective x1.029', 'LAD-IT160s', LAD4),
)

# `beta_schedule` holds the rows that are about the schedule itself: the runs with the schedule scaled to the passes, and the 80-pass run on the fitted bracket (the schedule
# scaled to 80 passes is the default). The runs with the default schedule at other pass counts are the pairs of F4-22 and are on `iterations` only.
_E3_BETA_RUNS = ("E3/BR-IT20s", "E3/LAD-IT40s", "E3/LAD-IT160s")
_E3_ITER_FOR_BETA = tuple(replace(e, change="iterations " + e.change) for e in _E3_ITER
                          if e.runs in _E3_BETA_RUNS or (e.runs == "E3/BR-IT80" and e.part_class == BR_FIT))

_E3_BS = (
    _e3('[[0,1],[15,2],[25,4],[32,8]] (the default schedule scaled to the 40 base passes)', 'gray 0.020; flip 4.1 %; iteration_limit (40 passes); objective change 0.00096; wall 30 s; the base BR-B0 is grey (ends at beta 2, gray 0.129); this run reaches beta 8 from pass 32',
        'optimizer pull x1.107, mass x1.008; P2 x1.037', 'BR-BS40s', BR_E3),
)

_FINE_TOL_EVIDENCE = (Evidence("offline replay of the history of fine pass A (window 5, the verdict measures) with three tolerances",
                            "1e-3 stops at pass 21 with weighted 599.72 (+0.46 % against 596.95 for the run from zero of 80 passes); 5e-4 stops at pass 30 (596.71); 1e-4 does not stop in 30 passes",
                            "no run: a replay of the file `history.jsonl`; 9 of 30 passes saved at 1e-3", "drone arm", "FP/arm-A"),
                   Evidence("the same replay on the hook fine pass at 1.2 mm (beta 8 held), and on the arm fine pass B (beta 4 then 8)",
                            "hook: 1e-3 stops at pass 15 (weighted 1.39); 5e-4 and 1e-4 do not stop in 16 passes. Arm B does not stop in 30 passes at any of the three: the step of beta at pass 14 "
                            "moves the loss again",
                            "no run: a replay of the file `history.jsonl`", "hook fine pass", "FP/hook-fine-b8, FP/arm-B"))


_E3_STOP = (
    _e3('true with iterations 80, and true with iterations 160 (E3/BR-SR160 is the same design)', 'gray 0.019; flip 5.2 %; stopped_by_rule at pass 78 (78 passes); objective change 4.1e-05; both stop by the rule `converged` at pass 78, the design of E3/BR-IT80 (flip 0.0 % against BR-IT80); wall 48 s and 51 s, against 45 s for BR-IT80 and 89 s for BR-IT160 (host load differs)',
        'optimizer pull x1.006, mass x1.002; P2 x1.001', 'BR-SR80, BR-SR160', BR_FIT),
    _e3('true with iterations 80, and true with iterations 160 (E3/BR-SR160 is the same design)', 'grey base (BR-B0 ends at beta 2, gray 0.129); gray 0.019; flip 4.3 %; stopped_by_rule at pass 78 (78 passes); objective change 4.1e-05; both stop by the rule `converged` at pass 78, the design of E3/BR-IT80 (flip 0.0 % against BR-IT80); wall 48 s and 51 s, against 45 s for BR-IT80 and 89 s for BR-IT160 (host load differs)',
        'optimizer pull x1.114, mass x1.010; P2 x1.038', 'BR-SR80, BR-SR160', BR_E3),
    _e3('true with iterations 160', 'gray 0.014; flip 0.0 %; stopped_by_rule at pass 79 (79 passes); objective change 4.8e-05; stops by the rule `converged` at pass 79 of 160, the design of LAD-D0 (flip 0.0 % against LAD-D0); wall 142 s, against 272 s for LAD-IT160 (host load differs)',
        'optimizer scoop x1.000, scrape x1.000, stir x1.000, objective x1.000, mass x1.000; P2 scoop x1.001, scrape x1.001, stir x1.000, objective x1.001; inside the noise', 'LAD-SR160', LAD4),
)

_E3_PENAL = (
    _e3('2.0 (from 3.0)', 'gray 0.024; flip 11.5 %; iteration_limit (40 passes); objective change 0.00023',
        'optimizer pull x1.022, mass x0.995; P2 x0.991', 'BR-SP20', BR_FIT),
    _e3('4.0', 'gray 0.022; flip 9.6 %; iteration_limit (40 passes); objective change 0.00063',
        'optimizer pull x0.967, mass x1.000; P2 x0.978', 'BR-SP40', BR_FIT),
    _e3('2.0 (from 3.0)', 'grey base (BR-B0 ends at beta 2, gray 0.129); gray 0.151; flip 10.6 %; iteration_limit (40 passes); objective change 0.00019; last beta stage not reached (ends at beta 2)',
        'optimizer pull x1.049, mass x1.001; P2 x1.032', 'BR-P20', BR_E3),
    _e3('4.0', 'grey base (BR-B0 ends at beta 2, gray 0.129); gray 0.127; flip 5.7 %; iteration_limit (40 passes); objective change 0.0022; last beta stage not reached (ends at beta 2)',
        'optimizer pull x0.970, mass x1.000; P2 x1.007', 'BR-P40', BR_E3),
    _e3('2.0 (from 3.0)', 'gray 0.023; flip 66.7 %; iteration_limit (80 passes); objective change 0.00013',
        'optimizer scoop x1.216, scrape x1.156, stir x1.204, objective x1.169, mass x1.003; P2 scoop x1.211, scrape x1.144, stir x1.165, objective x1.150', 'LAD-P20', LAD4),
    _e3('4.0', 'gray 0.013; flip 35.1 %; converged (80 passes); objective change 3.1e-05',
        'optimizer scoop x0.878, scrape x0.902, stir x0.895, objective x0.900, mass x0.998; P2 scoop x0.886, scrape x0.911, stir x0.899, objective x0.907', 'LAD-P40', LAD4),
)

_E3_VF = (
    _e3('0.24 (-20 % of 0.3)', 'gray 0.021; flip 21.6 %; iteration_limit (40 passes); objective change 0.00076',
        'optimizer pull x0.818, mass x0.894; P2 x0.852', 'BR-SVF24', BR_FIT),
    _e3('0.36 (+20 %)', 'gray 0.022; flip 20.0 %; iteration_limit (40 passes); objective change 0.00013',
        'optimizer pull x1.149, mass x1.106; P2 x1.124', 'BR-SVF36', BR_FIT),
    _e3('0.08502 (-20 % of 0.10628)', 'gray 0.014; flip 36.0 %; iteration_limit (80 passes); objective change 0.00015',
        'optimizer scoop x0.759, scrape x0.772, stir x0.683, objective x0.747, mass x0.867; P2 scoop x0.752, scrape x0.762, stir x0.681, objective x0.739', 'LAD-VF085', LAD4),
    _e3('0.12754 (+20 %)', 'gray 0.015; flip 27.9 %; converged (80 passes); objective change 4.7e-05',
        'optimizer scoop x1.279, scrape x1.243, stir x1.283, objective x1.253, mass x1.131; P2 scoop x1.267, scrape x1.226, stir x1.255, objective x1.234', 'LAD-VF128', LAD4),
)

_E3_MESH = (
    _e3('blur (from sdf)', 'gray 0.020; flip 0.0 %; iteration_limit (40 passes); objective change 0.00096; same last snapshot as the base: only `final.stl` differs',
        'optimizer pull x1.000, mass x0.986; P2 x0.938', 'BR-SMBb', BR_FIT),
    _e3('blur (from sdf)', 'gray 0.014; flip 0.0 %; converged (80 passes); objective change 4.8e-05; same last snapshot as the base: only `final.stl` differs; one run, cause not isolated (of the P2 values); verify log: 1 body, watertight, 578 fixed nodes (1029 for sdf), area of the fixed region on the mesh 753 mm2 (1123 mm2), max displacement of `scrape` 29.7 mm (0.189 mm), `metrics.mesh.volume_error_pct` -1.65 % (+0.011 %); the blur mesh has no `mesh.sources` and no `mesh.checks` entry',
        'optimizer scoop x1.000, scrape x1.000, stir x1.000, objective x1.000, mass x0.980; P2 scoop x0.005, scrape x0.007, stir x0.071, objective x0.009', 'LAD-MBb', LAD4),
)


_TOP_LEVEL = (
    Card("name", "yaml", "method", 3, "Names the problem in titles and file names. It has no effect on the part.", "measured",
         values="non-empty text; required.", cost=RESUME_COST, call=RESUME_CALL,
         evidence=(_resume_row("name changed", "the numbers of metrics.json stay equal; run.json is rewritten", "9.1 s", "name"),)),
    Card("regions", "yaml", "requirement", 1,
         "The STL files of the problem: the design space, the supports, the loaded bodies, the voids and the frozen bodies. All STL files "
         "share one coordinate system in mm; the voxel grid is the box around all of them.", "code",
         values="mapping; keys on the cards `regions.*`. When regions overlap, the priority is keep_out, then fixed, then load, then solid, "
                "then design (CONTRACTS.md section 22a). A path is relative to the YAML file."),
    Card("load_cases", "yaml", "requirement", 1,
         "The load situations of the part. The objective adds the cases (`settings.case_combine: sum`) or takes the worst "
         "(`max`).", "measured",
         values="list of one case or more; each case has a unique `name` and its forces or pressures. A case that repeats another one "
                "(equal or opposite loads) costs a solve and adds nothing: `check_problem` gives it as a fact (`repeated_load_cases`).",
         symptoms=(Symptom("falls apart under the other load, breaks when pulled", "`metrics.stiffness_cases[].N_per_mm` of each case "
                           "(null with a reason for a case with a moment, a remote force or a pressure, and for a case that applies no load); `metrics.constraints`", facts=("metrics.stiffness_cases[].N_per_mm", "metrics.constraints[].g")),),
         evidence=(Evidence("a symmetric pair of side loads (+ and -)", "walls across the depth, fewer fork levels", "not given", STAND, "E4"),)),
    Card("material", "yaml", "requirement", 1, "The material of the whole part: one E, one nu, one density. The default is PLA.", "code",
         values="mapping E, nu, density; see the cards `material.*`. One material only; no material per region."),
    Card("assumptions", "yaml", "method", 3,
         "A list of short notes for people. The solver ignores it; it goes into run.json.", "measured",
         values="list of non-empty text.", cost=RESUME_COST, call=RESUME_CALL,
         evidence=(_resume_row("a note added", "the numbers of metrics.json stay equal; run.json is rewritten", "6.6 s", "assumptions"),)),
)

_REGIONS = (
    Card("regions.design", "yaml", "requirement", 1,
         "The space that the part can fill. Where the optimizer puts no material, the space stays empty.", "code",
         values="one STL path (not a list), closed mesh. Tighten: a smaller space; loosen: a larger one (more voxels, a longer run).",
         symptoms=(Symptom("does not fit, too big, too long", "`check_problem.print.box_extents_mm` and `print.options[].bed_fit` before the run; "
                           "`metrics.print.configured.bed_fit` after it", facts=("check_problem.print.box_extents_mm", "check_problem.print.options[].bed_fit", "metrics.print.configured.bed_fit")),)),
    Card("regions.fixed", "yaml", "requirement", 1,
         "The bodies that hold the part. A path clamps all three axes of every node of its voxels. Without a fixed region the YAML needs "
         "`springs` or `joints`, else the stiffness matrix is singular.", "code",
         values="list of STL paths (full clamp) or `{file, axes}` entries (a per-axis support). Required unless the YAML has `springs` or `joints`.",
         symptoms=(Symptom("floating, not attached, detached", "`metrics.verdict.facts.load_path.not_joined` (a load region with no path to a "
                           "support); the run stops with a `floating` error before the solve", facts=("metrics.verdict.facts.load_path.not_joined",)),
                   Symptom("force vanishes, lost force", "`check_problem.lost_force` and `run_result.lost_force` (`max_lost_percent`): force on "
                           "nodes that a load and a fixed region share", facts=("check_problem.lost_force.max_lost_percent", "run.lost_force.max_lost_percent")),
                   STL_JOIN_SYMPTOM),),
    Card("regions.fixed[].file", "yaml", "requirement", 1, "The STL of a per-axis support entry (a roller or a slider).", "code",
         values="STL path; the entry also needs `axes`.", also=("regions.fixed[].axes",)),
    Card("regions.load", "yaml", "requirement", 1,
         "Named bodies that carry a force, a pressure, a spring or a joint. A load region that no case uses is a frozen body that is "
         "not clamped.", "code",
         values="mapping name -> STL path, one or more; the order is kept (in an overlap the first region owns the voxel)."),
    Card("regions.keep_out", "yaml", "both", 1,
         "Voids: no material in these STL files (density 0). A hole or a clearance that the user names is a requirement. A slot that the "
         "agent adds as a hint for the layout is the agent's.", "measured",
         kind_note="requirement when the user names the void; method when you add it as a layout hint",
         values="list of STL paths, may be empty. A keep-out makes the free part of the design space smaller, and `settings.volume_fraction` "
                "applies to the free part, so the mass falls at the same fraction. To keep the mass: volume_fraction_new = (target_volume_mm3 - "
                "forced_solid_new_mm3) / free_design_new_mm3, where target_volume_mm3 is `derived.volume.target_volume_mm3` of the YAML WITHOUT the block, and the other two "
                "are `derived.volume.forced_solid_mm3` and `free_design_mm3` of `check_problem` on the YAML WITH the block (not the volume of the block: a block that "
                "touches a channel or the border removes less free volume than its own volume). With the formula the mass was back within 0.3 % in three blocks (E1/K1v, K2v, K3v). "
                "The price is the lost mass for a block over empty space (K1: mass x0.969, optimizer x0.940 / x0.955 at the same fraction; with the formula x0.999 / x1.002), "
                "and a geometry price for a block over the load path (K3v: x0.800 / x0.902 at equal mass, 134 solid voxels of the base design inside the block).",
         symptoms=(Symptom("wall across the depth, thick wall, slab", "no fact yet (look at `render_part`); full-depth walls are local optima "
                           "(F4-15)", facts=()),
                   Symptom("lighter than before, lost mass after a void", "`check_problem.derived.volume.target_mass_g` before and after the block (and `free_design_mm3`); "
                           "`metrics.mass_saved_pct` does not show the loss (its 'before' mass falls too: 71.1 % without the block, 70.7 / 70.3 / 70.8 % with blocks 1, 2, 3)",
                           facts=("check_problem.derived.volume.target_mass_g", "check_problem.derived.volume.free_design_mm3", "run.derived.volume.target_mass_g")),),
         evidence=(*_E1_KEEPOUT, Evidence("a 4 mm slot at mid-depth in a full-depth wall, same mass", "the wall splits into two posts",
                            "one slot: weight -1.8 %, tap -0.9 %. Two slots: weight +6.3 %, tap +5.2 %, side +2.6 % against no slot (F6a vs F3c); "
                            "the same slots on the F2a base: weight +6.2 %, tap +6.3 %, side +3.3 %; on F0 (a full rear wall): weight +26.5 %, "
                            "tap +4.6 %, side +14.7 %", STAND, "F4b, F5c, F6a, F7b, F7c (FINDINGS F4-15)"),
                   Evidence("a slot in a round trunk", "twin columns, less tree", "weight +0.1 %, tap -1.3 %: no gain", STAND, "F7a"),
                   Evidence("a 4 mm keep-out slot with volume_fraction unchanged", "the mass drops (the free part is smaller)",
                            "mass -2.7 %; the stand runs used volume_fraction 0.104 -> 0.109 to keep the mass", STAND, "FINDINGS F4-16"),
                   _resume_row("keep_out file removed, resume of the finished run", "NOT a new design: the saved density is mapped onto the "
                               "new voxels and the post step runs again; the optimizer does not run", "6.1 s", "keep_out_removed")),
         price_note="Read P2 for the price at equal mass: K1v and K2v are within 0.4 % in the optimizer and about 3 % low in P2 (at the edge of the P2 noise). The keep-out is respected in every run (`metrics.mesh.checks.keep_out.<file>.intersect_mm3` is 0.0). A keep-out is a user requirement: if you add one as a hint, recompute volume_fraction and report the old and the new target mass."),
    Card("regions.solid", "yaml", "both", 1,
         "Frozen bodies: density 1, no clamp, no load. They add material to the part where the user wants it (for example a boss).", "measured",
         kind_note="requirement when the user names the body; method when you add it as a layout hint",
         values="list of STL paths, may be empty; frozen voxels have the label LOAD with load id -1 (`compare_runs` shows a solid block as 'design->load'; there is no label `solid`). `volume_fraction` applies to "
                "the free voxels, so a solid block raises the target volume at the same fraction (SOL1: 22889.6 to 23299.2 mm3). The fraction that keeps the target volume is given by the formula on `regions.keep_out` "
                "(SOL1v: (22889.6 - 11160) / 60696 = 0.19325).",
         evidence=_E2_SOLID,
         price_note="One 8 mm block (centre x 29, y 0, z -6 mm) cost 4.4 % (hang) and 5.2 % (pull_off) of the optimizer stiffness at the same mass (SOL1v; P2 3.8 % and 2.7 %), and 2.6 % of the pull_off stiffness at +2.0 % mass "
                    "(SOL1, `volume_fraction` unchanged). The optimizer and P2 agree in sign. A pass is faster with fewer free voxels (median 0.21 s against 0.35 s for B0)."),
    Card("regions.measure", "yaml", "both", 1,
         "Named probe regions for the `limits` rows. A probe changes no label and carries no force.", "code",
         kind_note="follows the `limits` row that uses it",
         values="mapping name -> STL path; a name that is also a load region name is an error."),
    Card("regions.not_printed", "yaml", "requirement", 3,
         "Names bodies (files of fixed, load, solid or keep_out regions) that are in the problem but not in the print. The post step "
         "then also writes `result/part_only.stl` without them.", "measured",
         values="list of STL paths; each one is a file of a fixed, load, solid or keep_out region (matched by the resolved path); another path is an error. "
                "`verify_run` with `part_only=true` checks the file; `settings.part_only_min_film_mm` cuts the thin film at its contact faces.",
         cost=RESUME_COST, call=RESUME_CALL,
         evidence=(_resume_row("wall_plate.stl added to not_printed", "result/part_only.stl is new; metrics.part_only appears", "6.3 s",
                               "not_printed"),)),
)

_LOAD_CASES = (
    Card("load_cases[].name", "yaml", "method", 1,
         "The name of a load case. Rows (`limits`, `acceptance`) refer to it in their `cases` lists.", "code",
         values="non-empty text, unique."),
    Card("load_cases[].forces", "yaml", "requirement", 1,
         "The forces of one case, each on a load region. A case needs `forces` or `pressures`.", "code",
         values="list of entries `{region, force, moment, at, coupling}`; two forces on one region add up. See the cards `load_cases[].forces[].*`."),
    Card("load_cases[].pressures", "yaml", "requirement", 1,
         "The pressures of one case, each on the faces of a load region toward empty space.", "code",
         values="list of entries `{region, value_mpa}`; a pressure acts on a load region only (not on a fixed region or the design space)."),
    Card("load_cases[].weight", "yaml", "both", 1,
         "The weight of the case in the objective: the objective is the sum of weight times compliance (`case_combine: sum`), or the "
         "worst weighted case (`max`).", "measured",
         kind_note="requirement when the user ranks the cases; method when you balance them",
         values="number > 0, default 1.0. The hook example uses 234 for `hang` and 1 for `pull_off`. A weight w on a case acts as a force scale of sqrt(w) on it: byte-equal final.stl for `pull_off` weight 4 and force x2 (E1/WP4, LP20).",
         evidence=_E1_WEIGHT),
    Card("load_cases[].compliance_limit", "yaml", "requirement", 1,
         "An upper bound on the compliance of the case (N mm): a stiffness requirement. The case compliance stays at or below it, if the "
         "volume allows.", "code",
         values="number > 0, N mm (a text such as `5e-3` is read as a number). Price of the row 100. A limit that the design cannot meet at "
                "the given volume is not an error: the volume stays at its target and the row is unmet.",
         symptoms=(Symptom("too soft, flimsy, bends, wobbly", "`metrics.stiffness_cases[].N_per_mm`, `metrics.constraints` (the row and its "
                           "`g`) and `metrics.verdict.feasible`", facts=("metrics.stiffness_cases[].N_per_mm", "metrics.constraints[].g", "metrics.verdict.feasible")),)),
    Card("load_cases[].stress_limit", "yaml", "requirement", 1,
         "The final stress limit of the case in MPa. The optimizer keeps the relaxed von Mises stress at or below it. It is a target, not "
         "a guarantee, and not a check of the printed part.", "measured",
         values="number > 0, MPa; takes no `safety_factor`." + " `verify_run(gate=true)` reads this as the limit of the tet value (card `verify_run:gate`). " + "Read `metrics.stress.cases.<case>.summary` for the relaxed result and the voxel "
                "estimates of the part stress. The price is large and one-sided (E2, `pull_off` of the hook): a limit 30 % tighter cost x1.461 in mass and only the last stage end was feasible; "
                "limits 0.7 and 1.0 MPa gave x0.852 and x0.708 with the optimizer stiffness down to x0.695 and x0.477. A limit on the case that carries little stress binds only below its peak "
                "(`hang`: peak 0.054 MPa; a limit of 0.04 cost x1.035, a limit of 3.0 changed nothing beyond the noise).",
         symptoms=(Symptom("breaks, cracks, too weak at the root, stress too high",
                           "`run_result.metrics.stress.cases.<case>.summary` and `metrics.stress.cases.<case>.relaxed.met`; an independent check: `verify_run`",
                           facts=("metrics.stress.cases.<case>.summary", "metrics.stress.cases.<case>.relaxed.met", "verify_run"),
                           when="`relaxed.met` is false: the relaxed KS row of the optimizer is above the limit at the last pass (`relaxed.met_means` gives the rule; it is not a statement about the part)"),
                   Symptom("stress still too high in the part though the run says met", "the independent check `verify_run` (`verify.cases.<case>.von_mises.p99_9_by_volume_MPa`) against the limit; "
                           "the voxel estimates of the part are in `metrics.stress.cases.<case>.summary`; `relaxed.peak_above_limit` is true when the relaxed peak itself is above the limit",
                           facts=("verify.cases.<case>.von_mises.p99_9_by_volume_MPa", "metrics.stress.cases.<case>.summary", "metrics.stress.cases.<case>.relaxed.peak_above_limit"),
                           when="the P2 p99.9 stress of the case is above the limit; on the hook it was 2.45 to 2.67 times the limit (limit 0.5 MPa) while `relaxed.met` was true (E2/ST0 to ST5); "
                                "or `relaxed.peak_above_limit` is true (the relaxed peak is above the limit, with the row met or not)")),
         evidence=_E2_STRESS_LIMIT,
         price_note=STRESS_NOISE + " 'Met' is about the relaxed measure of the optimizer, not about the part: with the limit at 0.5 MPa the P2 p99.9 stress of `pull_off` was 1.225 to 1.333 MPa in the 6 base runs "
                    "and the voxel estimates were above the limit too (grey design 0.565 to 0.627 MPa, thresholded 0.495 to 0.524 MPa). The measures that follow the limit value are the relaxed peak and the "
                    "thresholded voxel peak (limit 0.35 / 0.5 / 0.7 / 1.0 MPa: relaxed peak 0.332 / 0.491 / 0.673 / 0.960 MPa, thresholded 0.371 / 0.513 / 0.677 / 0.968 MPa). The P2 p99.9 over the whole part did not "
                    "follow it (1.295 MPa at 0.35, inside the base range; 2.235 MPa at 1.0): it is set by the stress near the load region. The independent check and the voxel estimates, not the word met, tell "
                    "whether the printed part is below the limit."),
    Card("load_cases[].allowable_mpa", "yaml", "requirement", 1,
         "The allowable stress of the material in MPa. With `safety_factor` the limit is allowable / factor. Topopt has no default allowable: "
         "the user or you give it.", "code",
         values="number > 0, MPa. Data sheet ranges of printed PETG and PLA: the resource `topopt://materials/reference`." + " `verify_run(gate=true)` reads this as the limit of the tet value (card `verify_run:gate`)."),
    Card("load_cases[].safety_factor", "yaml", "requirement", 1,
         "Divides `allowable_mpa` to give the stress limit. Only with `allowable_mpa`; not with `stress_limit`.", "code",
         values="number > 0. Raise it to tighten the stress rule (with a report and its price: more mass); lower it only with the user's consent." + " `verify_run(gate=true)` reads this as the limit of the tet value (card `verify_run:gate`). The limit of the gate is allowable / factor."),
    Card("load_cases[].allowable_source", "yaml", "requirement", 3,
         "Free text: where the allowable comes from. The report and the ratchet show it. It changes no number.", "code",
         values="non-empty text. The passes do not read it, so a change is a resume of a finished run.", cost=RESUME_COST, call=RESUME_CALL),
    Card("load_cases[].forces[].region", "yaml", "requirement", 1, "The load region that carries the force.", "code",
         values="a key of `regions.load`."),
    Card("load_cases[].forces[].force", "yaml", "requirement", 1,
         "The total force on the region in N, [Fx, Fy, Fz]. On a user part it is a user requirement: change it only with the user's consent "
         "(you may tighten it, with a report and its price).", "measured",
         values="three finite numbers, N; the force is spread over the nodes of the region. The layout depends on the ratio w_i |F_i|^2 of the weighted cases only: all forces x2 give the same design (byte-equal, E1/LA20), and pull_off x2 equals hang x0.5 (E1/LP20, LH05). The load level sets the stress, not the layout; the stress is not in `stiffness_cases`, read P2 `von_mises`. A force on nodes that a fixed region shares is lost "
                "(a fact: `lost_force`). The effect is not monotone (see the last row). A case in which every force and moment is exactly 0 and no "
                "pressure is other than 0 applies no load: it adds nothing to the objective, and its weight and its stress limit do nothing. "
                "A case whose load vector on the free dofs is exactly 0.0 feels no load for the same "
                "reason: `load_reaches_free_dofs` is false for it, and the same rows refuse it (a case that loses only a part of its force "
                "keeps the `lost_force` fact and is not refused). It has one of two causes, and `load_reach` names it: the nodal forces of the case "
                "add up to 0.0 at every node (opposite forces on the same nodes cancel, or a force is so small that its nodal share is 0.0 in "
                "double precision), or every force lands on fixed dofs (the `lost_force` fact says 100 percent; a squeeze has a net force of 0, so "
                "its fact does not). A force so small that the compliance is 0.0 or subnormal in double precision (1e-300 N) stops the run at the "
                "first pass where that holds, next to the same rows, with the same options. "
                "A moment alone is a load, and so are two equal and opposite forces on two regions.",
         symptoms=(Symptom("wall instead of forks, rear trunk too wide", "no fact yet (look at `render_part`); the side load sets the width of "
                           "the rear trunk", facts=()),
                   Symptom("the load case does nothing, zero force, no load, nothing is optimized",
                           "`check_problem.derived.load_cases[].applies_load` is false for the case (every number of the YAML is 0), or "
                           "`check_problem.derived.load_cases[].load_reaches_free_dofs` is false (the nodal forces cancel, or every force lands on fixed dofs), and "
                           "`check_problem.warnings` has a line that starts "
                           "with \"fact: load case\" and says the case applies no load (the same line is in `topopt check` and at the start of `run.log`; "
                           "`run.derived.load_cases[].applies_load` and `run.derived.load_cases[].load_reaches_free_dofs` after the run). "
                           "The loader (the YAML numbers) and the problem builder (the load on the free dofs) refuse, before the run, a YAML in which "
                           "no case does work and the objective is compliance, and a YAML with a case with no load next to a `compliance_limit`, to "
                           "`case_combine: max` (compliance objective) or to `buckling.min_factor` or `buckling.check`",
                           facts=("check_problem.derived.load_cases[].applies_load", "run.derived.load_cases[].applies_load",
                                  "check_problem.derived.load_cases[].load_reaches_free_dofs", "run.derived.load_cases[].load_reaches_free_dofs"),
                           when="`applies_load` is false: every force and moment of the case is exactly 0 and it has no pressure other than 0; "
                                "or `load_reaches_free_dofs` is false: the load vector of the case is exactly 0.0 on every free dof (`load_reach` is `nodal_zero`: the nodal "
                                "forces add up to 0.0 at every node; or `fixed` or `fixed_net`: every force lands on fixed dofs)")),
         evidence=(*_E1_FORCE, Evidence("side push 0.6 -> 0.3 N", "clean Y forks; the rear trunk stays wide", "weight 1.27, tap 0.99 of FREF", STAND, "F1c"),
                   Evidence("side push 0.6 -> 1.2 N", "the rear trunk becomes a wide wall", "weight 0.84, tap 0.69 of FREF", STAND, "F1d"),
                   Evidence("side push 0.3 -> 0.15 N", "the rear trunk becomes a FULL wall: the effect is NOT monotone", "weight +5 %, tap +3 %",
                            STAND, "F4a vs F3c")),
         price_note="Hook rows (E1): the price of a load change is the move of the balance between the cases (pull_off x2: hang x0.776, pull_off x1.099 in the optimizer, P2 x0.788 / x1.067)."),
    Card("load_cases[].forces[].moment", "yaml", "requirement", 1,
         "A moment on the region in N mm about the node centroid of the region. With a moment the case stiffness `N_per_mm` is null.", "code",
         values="three numbers, N mm; `force` is then optional."),
    Card("load_cases[].forces[].at", "yaml", "requirement", 1,
         "The point (mm) where `force` acts: a remote force. It makes a moment about the region.", "code",
         values="three numbers, mm; needs `force`."),
    Card("load_cases[].forces[].coupling", "yaml", "method", 1,
         "How the force enters the region. Only `distributing` exists (the force and the moment are spread over the nodes). `rigid` is refused "
         "with a fact and options.", "code", values="`distributing` (the default)."),
    Card("load_cases[].pressures[].region", "yaml", "requirement", 1, "The load region that carries the pressure.", "code",
         values="a key of `regions.load`."),
    Card("load_cases[].pressures[].value_mpa", "yaml", "requirement", 1,
         "The pressure in MPa (N/mm2) on the faces of the load region toward empty space. Positive pushes into the body, negative pulls. "
         "With a pressure the case stiffness `N_per_mm` is null.", "code", values="a finite number."),
)

_MATERIAL = (
    Card("material.E", "yaml", "requirement", 1,
         "Young's modulus of the material in MPa. Stiffness and compliance numbers follow it, and the stress rows read it.", "code",
         values="number > 0, MPa; default 3500 (PLA)."),
    Card("material.nu", "yaml", "requirement", 1, "Poisson's ratio of the material.", "code", values="-1 < nu < 0.5; default 0.35."),
    Card("material.filament", "yaml", "requirement", 1,
         "The filament, by name. The tool finds it in the filament catalog and fills `material.E` (the lower of the XY and the Z modulus of its data sheet; "
         "the bending modulus of the sheet when it gives no tensile modulus, and the reason says so) "
         "and `material.density` when the YAML does not give them (their source reads `filament:<id>`). `material.nu` stays the assumed 0.35, because no data "
         "sheet gives it. A key that the YAML gives wins. `topopt check` and `check_problem` also say if the printer of `print.printer_model` can print the "
         "filament (a fact, not a stop).", "code",
         values="text: an id of the catalog (`polymaker-polylite-abs`), or a name or alias (`Bambu ABS`, `Prusament PETG`); case, punctuation and maker words do "
                "not matter. A family alone (`PLA`) names no product. A text that is not one filament is an error that names the candidates or the suggestions; "
                "`find_material(name)` shows the match, the candidates and the numbers that the profile gives. A filament that the catalog does not have comes "
                "from the user file `topopt-filaments.yaml` in the work folder (`save_material` writes it, with the source of each value). A filament whose sheet "
                "gives no modulus at all (no tensile and no bending modulus) is an error unless the YAML gives `E`. A sheet that gives a density range "
                "gives the middle of it. A density that the sheet does not give at all is the median density of the catalog rows of the same family, "
                "labelled `assumed` (the default 1.24 when the family has none). "
                "No default: without it the material is the default of the other keys (E 3500, nu 0.35, density 1.24: PLA)."),
    Card("material.density", "yaml", "requirement", 3,
         "Density in g/cm3. The optimizer reads it for mass numbers only (the mass objective, the mass in the reports).", "measured",
         values="number > 0; default 1.24 (PLA).", cost=RESUME_COST, call=RESUME_CALL,
         evidence=(_resume_row("density 1.24 -> 1.5", "mass_final_g 25.72 -> 31.11, mass_design_space_g 89.1 -> 107.78; nothing else changes",
                               "7.6 s", "density"),)),
)


# --- settings -----------------------------------------------------------------------------------------------------------------
_SETTINGS = (
    Card("settings", "yaml", "method", 1,
         "The numbers of the method: the grid, the allowed volume, the number of passes, the filter, the optimizer, the solver, the stop rules "
         "and the mesh step. Most keys are method settings; `volume_fraction`, `min_member_mm`, `objective` and `perimeter_max_mm2` can be "
         "requirements (see their cards); `target_mass_g` is a requirement.", "code",
         values="mapping; required keys `voxel_size` and `iterations`, and exactly one of `volume_fraction` and `target_mass_g`; the other keys are on the cards `settings.*`. A key that "
                "only the post step reads has layer 3; the cards say which."),
    Card("settings.voxel_size", "yaml", "method", 1,
         "The edge of the voxel cube in mm. It sets the smallest feature (about one voxel), the run time and the memory.", "measured",
         values="number > 0, mm; required. The grid is the box around all region files. A finer voxel with the same physical filter makes thin "
                "sheets, not more forks. The mass changes with the grid at the same fraction (final volume over the target 0.957, 0.906 and 1.067 at 3.0, 2.0 and 1.5 mm on the hook). Guards: `max_grid_voxels`, `max_active_voxels`. A finer voxel from the design of a finished run: `run_topopt:fine_from`; "
                "`check_problem.fine_pass` gives the voxel that the part needs.",
         cost="the run time grows with the voxel count; 0.5 mm with `min_member_mm` 1.5 took about 3 times the run time of 0.7 mm (E6 vs E7). On the hook: "
              "3.0 mm 9 s, 2.0 mm 21 s (median), 1.5 mm 78 s, 1.2 mm with 80 passes 261 s. On the ladle (80 passes): 4.0 mm took 119 s and 3.25 mm took 401 s of wall time (E3/LAD-D0, E3/LAD-B0), against checks "
              "of 55 s and 160 s (the rows of `check_problem`)",
         symptoms=(Symptom("blocky, jagged, stair steps, rough", "`check_problem.derived.min_feature_note` (the grid sets a floor of about one voxel) "
                           "and `metrics.mesh.local_loss`", facts=("check_problem.derived.min_feature_note", "metrics.mesh.local_loss.deep_lost_mm3")),
                   Symptom("too slow, takes too long", "`check_problem.time_estimate.total_s` and `check_problem.backends` (the estimate per solver)", facts=("check_problem.time_estimate.total_s", "check_problem.backends"))),
         evidence=(*_E1_VOXEL, Evidence("0.5 mm (from 0.7) with the same physical filter", "thin sheets, not more forks", "stiffness +10 to +51 %", STAND, "E5 vs E1"),
                   Evidence("0.5 mm (from 0.7) with `min_member_mm` 1.5", "about the same look as at 0.7", "stiffness +9 to +15 %; about 3 times the run time",
                            STAND, "E6 vs E7")),
         price_note="Read P2: the optimizer and P2 differ in size (1.5 mm: optimizer hang x0.991 / pull_off x1.096, P2 x1.167 / x1.180). A finer grid also changes the mass at the same volume_fraction. The printed part is the P2 model."),
    Card("settings.volume_fraction", "yaml", "both", 1,
         "The allowed material: the mean density of the FREE design voxels (design voxels minus fixed, load, solid and keep-out voxels). Fixed and "
         "load voxels are solid on top. The volume row is a ceiling.", "measured",
         kind_note="requirement when the user states a volume or share target (a mass in grams: `settings.target_mass_g`); method when you chose it",
         values="0 < v <= 1; required unless the YAML has `settings.target_mass_g` (the YAML has exactly one of the two; with a mass target the engine derives v). The target is (forced_solid_mm3 + v * free_design_mm3) of `check_problem.derived.volume`, and "
                "`target_mass_g` is that volume times the density. A keep-out makes the free part smaller, so the mass falls at the same v. To keep the mass: v_new = "
                "(target_volume_mm3 - forced_solid_new_mm3) / free_design_new_mm3, where target_volume_mm3 is `derived.volume.target_volume_mm3` of the YAML without the block "
                "and the other two are `derived.volume.forced_solid_mm3` and `free_design_mm3` of `check_problem` on the YAML with the block (`regions.keep_out` has the details). "
                "The rows K1 to K3v below measured it: the mass is back within 0.3 %. `metrics.mass_saved_pct` hides such a loss: its 'before' mass falls too (71.1 % without a block, 70.7 / 70.3 / 70.8 % with the three blocks). "
                "With `settings.objective: mass` it is only the start density, and it is not inert there: 0.5 in place of 0.3 gave mass x0.901 and pull_off stiffness x1.218 in two runs (E2/SV50, E2/SV50r, part class `hook_stress`).",
         symptoms=(Symptom("too heavy, too much material", "`metrics.mass_final_g`, `metrics.mass_saved_pct`, `check_problem.derived.volume.target_mass_g`", facts=("metrics.mass_final_g", "metrics.mass_saved_pct", "check_problem.derived.volume.target_mass_g")),
                   Symptom("too light, too thin, too weak", "`metrics.stiffness_cases[].N_per_mm`; `metrics.verdict.facts.volume.used_projected` "
                           "(how much of the allowed volume the run used)", facts=("metrics.stiffness_cases[].N_per_mm", "metrics.verdict.facts.volume.used_projected")),
                   Symptom("lighter than before after a void was added", "`check_problem.derived.volume.target_mass_g` before and after the block (and `free_design_mm3`); `metrics.mass_saved_pct` does not show the loss",
                           facts=("check_problem.derived.volume.target_mass_g", "check_problem.derived.volume.free_design_mm3")),
                   Symptom("volume left unused", "`metrics.verdict.facts.volume` (`used_projected`, `used_thresholded`)", facts=("metrics.verdict.facts.volume.used_projected", "metrics.verdict.facts.volume.used_thresholded"))),
         evidence=(*_E1_VF, *_E1_KEEPOUT_FOR_VF, Evidence("a 4 mm keep-out slot with volume_fraction unchanged", "the mass drops (the free part is smaller)",
                            "mass -2.7 %; the stand runs used volume_fraction 0.104 -> 0.109 to keep the mass", STAND, "FINDINGS F4-16"), *_E2_VF, *_E3_VF),
         price_note="The optimizer and P2 agree in sign (0.25: optimizer x1.242 / x1.207, P2 x1.190 / x1.143; 0.15: optimizer x0.737 / x0.760, P2 x0.664 / x0.660). The price is the free-part material (volume_fraction x the free design volume): x1.25 of it gave x1.242 / x1.207 and x0.75 gave x0.737 / x0.760 in the optimizer, about 1 : 1. The mass of the part moves less (x1.147 and x0.853). `check_problem` gives the target mass (`derived.volume.target_mass_g`) before the run. "
                    "Under `objective: mass` (the stress case, part class `hook_stress`) there is no volume row, and the value is the start density: 0.5 gave a lower mass (x0.901 and x0.902) and a higher pull_off stiffness "
                    "(x1.218 in both) than the mean of the 6 base runs, outside their range. "
                    "On three parts (E3; objective ratios, the hook from E1 with -25 % and +25 %, the bracket and the ladle with -20 % and +20 %; a sign counts only outside the noise band of that part): a lower fraction gives an "
                    "optimizer ratio down on all three (hook x0.749, bracket x0.810, fitted bracket x0.818, ladle x0.747) and a P2 ratio down on all three (x0.662, x0.858, x0.852, x0.739); a higher fraction gives both up on "
                    "all three (optimizer x1.225, x1.173, x1.149, x1.253; P2 x1.164, x1.154, x1.124, x1.234): same sign on the three parts in both measures. The mass moves less than the fraction (hook x0.853 and x1.147, fitted "
                    "bracket x0.894 and x1.106, ladle x0.867 and x1.131); on the bracket 2288 mm3 of the part is forced solid, so the fraction applies to the 10064 free voxels. " + STRESS_NOISE),
    Card("settings.target_mass_g", "yaml", "requirement", 1,
         "The mass in grams that the user asks for, in place of `settings.volume_fraction`. It sets the allowed material: the engine derives the volume fraction from it "
         "when it builds the problem, because the fraction needs the voxel grid (the forced-solid and the free design voxels). The YAML has exactly one of the two keys.", "code",
         values="number > 0, grams. Both keys, or neither, is refused (the text names the options). The derived fraction is "
                "(target_mass_g / density * 1000 - forced_solid_mm3) / free_design_mm3 of `check_problem.derived.volume`, with the density of `material.density`: a new density gives a new fraction, so a design that follows it needs a new run (a resume of a finished run keeps the design and re-computes the mass numbers). "
                "The engine refuses a target that is not above the mass of the forced-solid voxels (the fixed, load and solid regions, and the union interfaces), and a target above the mass of all active voxels solid; "
                "the text gives the limit in grams and the options. It refuses the key with `settings.objective: mass` (there the fraction is only the start density). "
                "`check_problem.derived.volume.target_mass_asked_g` is the asked mass; `derived.volume.target_mass_g` is the mass of the derived fraction (the same number to 3 decimals); "
                "`derived.volume_fraction` is the fraction, and `derived.volume_fraction_origin` says it comes from the mass. A run records the same keys in `run.json`. "
                "The final mesh does not weigh the asked mass: the thresholded voxels end near the target volume (`metrics.threshold_stability.levels[]` at 0.5), and the mesh step changes the volume again "
                "(`metrics.mesh.volume_final_mm3`; the card `settings.voxel_size` gives the ratio of the final volume to the target at three voxel sizes). `metrics.mass_asked_g` (only in a run with this key) is the asked mass in the result, next to `metrics.mass_final_g`; the end of the run log and the "
                "`run_status` summary of a done run state both in one line (\"final mesh 27.31 g against 30 g asked\"). Read the two against each other; the engine sets no tolerance, and you and the user decide what is close enough. "
                "A body of `regions.not_printed` is a fixed, load or solid region, so its voxels are forced solid and are in the asked mass, as in `derived.volume`: `result/part_only.stl` is lighter than the asked mass "
                "by about their mass (the hook tip: 225 voxels, 2.232 g at 2 mm), and the end line gives `metrics.part_only.mass_g` too. "
                "A changed asked mass is a change of a key of the passes: a resume refuses it, and `allow_changed_inputs` with `continue_from` takes it. A resume of a finished run "
                "with a changed mass runs no pass and keeps its design (the card `run_topopt:allow_changed_inputs` has the rows for `settings.volume_fraction`).",
         symptoms=(Symptom("under N grams, at most N g, weigh no more than N g, the mass limit, the weight target",
                           "`check_problem.derived.volume.target_mass_asked_g` (the asked mass) and `check_problem.derived.volume.target_mass_g` (the mass of the derived fraction) before the run; "
                           "`run.derived.volume.target_mass_asked_g` in the record; `metrics.mass_asked_g` and `metrics.mass_final_g` after it",
                           facts=("check_problem.derived.volume.target_mass_asked_g", "check_problem.derived.volume.target_mass_g",
                                  "run.derived.volume.target_mass_asked_g", "metrics.mass_asked_g", "metrics.mass_final_g")),)),
    Card("settings.iterations", "yaml", "method", 1,
         "The number of passes of the optimizer. The run makes all of them unless a stop rule ends it earlier.", "measured",
         values="integer >= 1; required. The run time is linear in it (20, 40 and 80 passes took 15, 21 and 32 s on the hook; E1). With the default beta schedule a run of fewer than 65 passes ends at a low beta (E1/IT20: x0.791 / x0.834); scale the schedule with the passes (E1/IT20s: x0.989 / x0.967 in 13 s). 80 passes ended `converged` with the same stiffness as 40 within the noise (E1/IT80). A resume of a finished run does not make it longer (the engine says so); `run_topopt:continue_from` with `more_iterations` adds passes in a new folder (layer 2: 10 or 20 more passes gave `converged` with the same stiffness within the noise, P7-1/r3/C10 and C20).",
         symptoms=(Symptom("grey, blurred, not finished, not converged", "`metrics.verdict.state` (`iteration_limit`), `metrics.verdict.converged`, "
                           "`metrics.verdict.measures.grayness.value`. " + GRAY_NOT_ENOUGH, facts=("metrics.verdict.state", "metrics.verdict.converged", "metrics.verdict.measures.grayness.value")),
                   STABILITY_SYMPTOM),
         evidence=(*_E1_ITER, *_E2_ITER, *_E3_ITER),
         price_note="Outside the noise the optimizer and P2 agree in sign. The unscaled schedule at 20 passes costs x0.791 / x0.834 in the optimizer and x0.928 / x0.875 in P2 (the run ends at beta 2 and the part stays grey); the schedule scaled with the passes costs x0.989 / x0.967 (E1/IT20 against E1/IT20s). The verdict `iteration_limit` at 40 passes is stricter than the result needs: E1/IT80 (`converged`) has the same stiffness within 0.1 %. "
                    "On the stress case (part class `hook_stress`) 110 passes with a schedule scaled to them were too few: 2 bodies, the `pull_off` row not met, and `topopt verify` could not mesh the part; "
                    "the public file has 220 passes. "
                    "On three parts (E3; objective ratios against the base of each part, the hook from E1; a sign counts only outside the noise band of that part): half the base passes with the default schedule gives an "
                    "optimizer ratio down on all three (hook x0.812, bracket x0.939, ladle x0.730): same sign on the three parts, with a P2 ratio down on the hook (x0.898) and the ladle (x0.923) and inside the band on the "
                    "grey bracket (x1.013). Half the base passes with the schedule scaled to them: the optimizer ratio is down on the hook (x0.978) and the ladle (x0.926), up on the grey bracket (x1.078) and down on the "
                    "fitted bracket (x0.973): the sign differs, and on the bracket it depends on the base; the P2 ratio is down on the hook (x0.931), the ladle (x0.926) and the fitted bracket (x0.984) and inside the band on the grey bracket "
                    "(x1.020). Double the base passes with the default schedule: the optimizer ratio is inside the band on the hook (x1.001) and the ladle (x1.001) and up on the bracket (x1.114; fitted x1.006), and the P2 ratio is "
                    "inside the band on the hook (x1.016), the ladle (x1.000) and the fitted bracket (x1.001) and up on the grey bracket (x1.038): the sign differs. In the E3 rows of this key no optimizer ratio and P2 ratio have "
                    "opposite signs outside both bands. "
                    "F4-22, what doubling the passes gave in pairs of runs on the same part (E3 sections 4 and 6.3; optimizer objective, P2 objective after it): when the smaller run had reached the last beta stage (8 pairs: hook "
                    "E1/B0 -> E1/IT80 and E1/IT80s; ladle E3/LAD-IT40s -> E3/LAD-D0, E3/LAD-D0 -> E3/LAD-IT160 and E3/LAD-IT160s; bracket E3/BR-BS40s -> E3/BR-IT80, E3/BR-IT80 -> E3/BR-IT160 and E3/BR-IT160s) the gain was "
                    "x1.000 to x1.080 in the optimizer and x1.000 to x1.080 in P2. When it had not (4 pairs: ladle E3/LAD-IT20 -> E3/LAD-IT40 and E3/LAD-IT40 -> E3/LAD-D0; bracket E3/BR-IT20 -> E3/BR-B0 and E3/BR-B0 -> E3/BR-IT80) "
                    "the gain was x1.064 to x1.391 in the optimizer and x0.987 to x1.287 in P2; the grayness of those four smaller runs was 0.085 to 0.176, and the verdict showed `last_beta_stage` not met on them and met on the "
                    "other eight. At the same end-of-run objective change of 0.0017 the hook (E1/B0, 6 design passes at beta 8) gained x1.001 (P2 x1.016) and the ladle (E3/LAD-IT40s, 8 design passes at beta 8) gained x1.080 "
                    "(P2 x1.080), so the objective change alone does not predict the gain. Further points of the same table: the bracket pair at 0.00025 (E3/BR-B0, last beta stage not reached) gained x1.114 (P2 x1.038); at 4.8e-05 or less "
                    "(`converged`, 15 design passes at beta 8) the ladle and the bracket gained x1.001 and x1.000 with the schedule unchanged and x1.034 and x1.004 with the schedule scaled by 2. A grayness below 0.1 does not show the missed stage either: E3/LAD-IT40 has grayness 0.085 and did not reach it. " + STRESS_NOISE),
    Card("settings.filter_radius", "yaml", "method", 1,
         "The radius R of the density filter in mm. At the projection level 0.5 a rod thinner than about 0.94 x R vanishes and a wall thinner than about "
         "0.53 x R (continuum values, interior of the design space). An explicit value wins over the radius that `min_member_mm` derives, and over the radius of `settings.robust: wall`.", "measured",
         values="number > 0, mm; default 1.5 x voxel_size (or 1.877 x min_member_mm); the same lever as `min_member_mm` in other units (E1/MM20 and E1/FR375 give byte-equal final.stl). A member of size v needs R = 1.877 x v for a wall (the wall rule of `min_member_mm`) "
                "and R = 1.068 x v for a rod (the rod rule); the rod rule gives walls thinner than v, so it loosens a request for a member size: use it only with the consent of the user "
                "(the price of the two rules is on the card `settings.min_member_mm`). `check_problem.derived.filter` gives the radius in voxels, "
                "the weights of the filter kernel, the padded grid of its FFT and its memory. `metrics.filter.facts` gives a line for a radius that is not above one voxel only when `min_member_mm` sets the radius; a `filter_radius` alone gives no line (E1/FR20): read `metrics.filter.size.radius_voxels`.",
         symptoms=(Symptom("webs, threads, thin members vanish", "`metrics.filter.min_feature_note` (from what size walls and rods survive; written in every run), `metrics.filter.size.radius_voxels` "
                          "(the radius in voxels) and `metrics.filter.facts` (a line only when `min_member_mm` is set too)",
                          facts=("metrics.filter.min_feature_note", "metrics.filter.size.radius_voxels", "metrics.filter.facts"),
                          when="`metrics.filter.size.radius_voxels` is 1 or less: no neighbour to average with, no filter (E1/FR20: 1.0). `metrics.filter.facts` is empty for a `filter_radius` alone (E1/FR20, E1/FR40)"),),
         evidence=(*_E1_FILTER_RADIUS, *_E3_FILTER_RADIUS),
         price_note="As `min_member_mm`: read P2 where the radius is 1 voxel or less (no filter: optimizer x1.009 / x1.008, P2 x0.937 / x0.878). The detecting facts are `metrics.filter.min_feature_note` and `metrics.filter.size.radius_voxels`; the print-check estimate of the thinnest member did not follow the radius above 1 voxel on the hook (3.00 mm at 1.88, 2 and 3.75 voxels; 2.82 mm at the default). "
                    "On three parts (E3; objective ratios against the base of each part, the hook from E1; a sign counts only outside the noise band of that part): "
                    "1 voxel (no filter): the optimizer ratio differs, up on the hook (x1.009) and the grey bracket (x1.049), down on the fitted bracket (x0.956) and the ladle (x0.894; at 3.25 mm x0.909); "
                    "the P2 ratio is down on the hook (x0.904), the fitted bracket (x0.964) and the ladle (x0.764; at 3.25 mm x0.781) and inside the band on the grey bracket (x0.987). "
                    "2 voxels: the optimizer ratio is down on the hook (x0.953), the bracket (x0.942; fitted x0.977) and the ladle (x0.953): same sign on the three parts; the P2 ratio differs, down on the hook "
                    "(x0.965), inside the band on the bracket (x1.020; fitted x1.011) and the ladle (x0.996) and up on the ladle at 3.25 mm (x1.026). "
                    "3 voxels: the optimizer ratio is down on the hook (x0.856), the bracket (x0.835; fitted x0.926) and the ladle (x0.821): same sign on the three parts; the P2 ratio is down on the hook "
                    "(x0.884), the grey bracket (x0.951) and the ladle (x0.961) and inside the band on the fitted bracket (x0.988). "
                    "F4-25, a radius of 1 voxel (five rows: E1/FR20, E3/BR-FR1, E3/BR-SFR1, E3/LAD-FR1, E3/LAD-XFR1): `metrics.filter.size.entries_per_row` is 1 (10064 entries on the bracket, one for each design voxel), "
                    "`metrics.filter.facts` is empty and `run.log` has no `warning:` line; `topopt check` prints 'filter size: about 1 entries per row' (BR-FR1). The grayness falls (hook 0.025 -> 0.001, "
                    "bracket 0.129 -> 0.027, fitted bracket 0.020 -> 0.002, ladle 0.014 -> 0.004, ladle at 3.25 mm 0.012 -> 0.008), the mesh step removes loose islands (`metrics.mesh.islands_removed`: hook 5, "
                    "bracket 3, fitted bracket 2, ladle 27, ladle at 3.25 mm 35), and the thinnest-member estimate of the print check is lower than the base by 0.44 to 0.54 mm (hook, bracket, fitted bracket, "
                    "ladle) and by 1.37 mm (ladle at 3.25 mm). The P2 ratio divided by the optimizer ratio is x0.897 (hook), x0.940 (grey bracket), x1.009 (fitted bracket), x0.854 (ladle) and x0.859 (ladle at 3.25 mm): "
                    "P2 is 6 to 15 % below the optimizer on four of the five rows and within 0.9 % of it on the fitted bracket."),
    Card("settings.penal", "yaml", "method", 1,
         "The SIMP power: stiffness grows with density to this power, so a grey density is a poor use of material. A lower value allows sheets "
         "and plates (the stiffness optimum); a higher value pushes toward members.", "measured",
         values="number >= 1; default 3.0 (the stand runs used 4).",
         symptoms=(Symptom("sheets, plates, walls instead of members", "no fact yet (look at `render_part`)", facts=()),),
         evidence=(*_E1_PENAL, *_E3_PENAL, Evidence("3 (from 4)", "sheets and plates (the stiffness optimum is sheets)", "stiffness +13 to +33 % (optimizer values of the stand; no P2 check)", STAND, "E3, E3 stage, E2"),
                   Evidence("3.5 (from 4)", "more small Y forks, the lowest sheet share, no sheets", "weight +7 %, tap +6 % (optimizer values)", STAND, "F3c vs F2a")),
         price_note="Read P2, not the optimizer value. The optimizer stiffness uses the penal of its own run, so it does not rank penal values: on the hook it says 2.0 is +2 % / +6 % and P2 says x0.970 / x0.955. In P2 no other value beats 3.0 on the hook (x0.91 to x0.99; P2 noise about 0.7 % for hang and 3.3 % for pull_off). The rows of the phone stand are optimizer values. A step of penal flips 8 to 21 % of the layout (E1/P25 to E1/P40). "
                    "On three parts (E3; objective ratios, the hook from E1; a sign counts only outside the noise band of that part): penal 2.0 gives an optimizer ratio up on all three (hook x1.041, bracket x1.049, fitted bracket "
                    "x1.022, ladle x1.169): same sign on the three parts; the P2 ratio differs: down on the hook (x0.962), up on the grey bracket (x1.032) and the ladle (x1.150), inside the band on the fitted bracket (x0.991). "
                    "Penal 4.0 gives an optimizer ratio down on all three (hook x0.968, bracket x0.970, fitted bracket x0.967, ladle x0.900): same sign on the three parts; the P2 ratio is down on the hook (x0.932), the fitted "
                    "bracket (x0.978) and the ladle (x0.907) and inside the band on the grey bracket (x1.007). On the ladle the optimizer and P2 agree (2.0: x1.169 and x1.150; 4.0: x0.900 and x0.907), and P2 puts 2.0 above "
                    "3.0 there (one run each)."),
    Card("settings.beta_schedule", "yaml", "method", 1,
         "The sharpness (beta) of the Heaviside projection by pass. A low beta gives a grey design; the last beta sets how black and white the "
         "result is.", "measured",
         values="list of [pass, beta]; the first pass is 0, passes rise, beta > 0; default [[0,1],[30,2],[50,4],[65,8]]. The passes are absolute: "
                "a run shorter than the last pass ends at a lower beta and can stay grey (the default schedule on 40 passes ends at beta 2: grayness 0.138, the verdict names it). Before the run "
                "`topopt check`, `check_problem` (a warning that starts with \"fact: settings.iterations\") and `run.log` say it, with two options "
                "(more `iterations`, or the schedule scaled to the passes) and the measured price; the run goes on. Scale the passes of the schedule with `iterations` (E1/IT20s). A beta jump can remove "
                "volume (see `settings.objective_scale`).",
         symptoms=(Symptom("grey, blurred, fuzzy", "`metrics.verdict.measures.grayness.value` (threshold 0.1) and "
                           "`metrics.verdict.measures.last_beta_stage`. " + GRAY_NOT_ENOUGH, facts=("metrics.verdict.measures.grayness.value", "metrics.verdict.measures.last_beta_stage.met")),
                   Symptom("volume dropped after a jump", "`history.jsonl` record `stage_end.volume.used_projected` at the end of each beta stage", facts=("history.stage_end.volume.used_projected",)),
                   Symptom("will it end grey, the run stops before the last beta", "before the run: the warning of `check_problem` that starts with \"fact: settings.iterations\" "
                           "(the same line is in `topopt check` and `run.log`)", facts=("check_problem.warnings",),
                           when="a warning that starts with \"fact: settings.iterations\" is in the list"),
                   STABILITY_SYMPTOM),
         evidence=(*_E1_BETA, *_E1_ITER_FOR_BETA, *_E3_BS, *_E3_ITER_FOR_BETA),
         price_note="A schedule that ends at beta 4 or less leaves a grey part; a start at beta 8 costs the most (x0.483 / x0.562). The optimizer and P2 differ in sign for a schedule that goes up to beta 16 (optimizer x1.025 / x1.005, P2 x0.975 / x0.937): read P2. With 20 passes the unscaled schedule costs x0.791 / x0.834 (optimizer) and the schedule scaled by 0.5 costs x0.989 / x0.967 (E1/IT20 and E1/IT20s). "
                    "On three parts (E3; objective ratios, the hook from E1; a sign counts only outside the noise band of that part): the schedule scaled to the passes reached beta 8 in every pass count tried (grayness "
                    "0.025, 0.020 and 0.020 on the bracket at 20, 40 and 160 passes; 0.020, 0.014 and 0.014 on the ladle). Against the default schedule at the same passes: 20 passes, bracket x1.078 against x0.939 "
                    "(P2 x1.020 against x1.013), ladle x0.823 against x0.525 (P2 x0.811 against x0.717; E3/LAD-IT20s and E3/LAD-IT20), hook x0.978 against x0.812 (P2 x0.931 against x0.898); ladle at 40 passes x0.926 against "
                    "x0.730 (P2 x0.926 against x0.923). The scaled schedule at half the base passes: the optimizer ratio is down on the hook (x0.978) and the ladle (x0.926), up on the grey bracket (x1.078) and down on the fitted "
                    "bracket (x0.973): the sign differs, and on the bracket it depends on the base. The schedule fitted to the 40 base passes (E3/BR-BS40s) gave the bracket x1.107 (P2 x1.037) against the example schedule, which ends at beta 2. F4-22 (E3): the gain of doubling the passes was x1.000 to x1.080 (optimizer and P2) with the last beta stage reached and up to x1.391 in the optimizer and x1.287 in P2 without it; the objective change alone does not predict the gain. The pairs, the numbers and the runs are on the card `settings.iterations`."),
    Card("settings.move_limit", "yaml", "method", 1,
         "The move limit of the optimizer (MMA): the largest change of a density in one pass. A smaller value is slower and steadier.", "measured",
         values="0 < m <= 1; default 0.2. With a stress limit, step acceptance halves it for a rejected try and grows it again up to this value.",
         symptoms=(Symptom("jumpy, collapses, oscillates", "`history.jsonl` field `change`; `metrics.verdict.measures.objective_change`", facts=("history.change", "metrics.verdict.measures.objective_change.value")),),
         evidence=(*_E1_MOVE, *_E2_MOVE),
         price_note="Inside or near the noise: a step of move_limit flips 5 to 16 % of the layout and moves the optimizer stiffness by up to 2.1 %, which one run cannot tell from another optimum. Read these rows as small, unproven. The detecting fact is the objective change of the last passes. "
                    "On the stress case (part class `hook_stress`) the file sets 0.03, the upper limit of the trial step of the step acceptance: 0.06 gave pull_off stiffness x1.091 and x1.082 at the same mass in two runs, "
                    "outside the range of the 6 base runs, and 1.6 to 2 times the rejected tries; 0.015 gave x1.020 in the stiffness and x1.016 in the mass, near the range. " + STRESS_NOISE),
    Card("settings.solver", "yaml", "method", 1,
         "The linear solver: auto (gpu-mg on a GPU that passes a self-test, else the CPU solver of the machine; if gpu-mg fails during a run, the run goes on with "
         "the CPU solver from its last checkpoint), cholesky (Apple silicon), pardiso (Linux x86_64 with the pardiso extra), amg, or gpu-mg (the GPU solver on any OpenCL device, "
         "also a CPU device; it has no fallback).", "code",
         values="auto | cholesky | pardiso | amg | gpu-mg; default auto. The loader does not check that the solver runs here; `check_problem.solver` does "
                "(`solver.resolved` is the solver that a run uses first, `solver.cpu_solver` the CPU solver, `solver.choice` the facts of the pick). "
                "The keys of the GPU solver are `settings.gpu_*`.",
         symptoms=(Symptom("too slow, takes too long", "`check_problem.backends` (the time estimate for each solver) and `environment`", facts=("check_problem.backends", "environment")),
                   Symptom("run refused, solver cannot run", "`check_problem.solver.error` and the warning that starts with 'settings.solver is'", facts=("check_problem.solver.error", "check_problem.warnings"))),),
    Card("settings.case_combine", "yaml", "method", 1,
         "How the load cases join in the objective: `sum` (the weighted sum of the compliances) or `max` (the worst weighted case).", "measured",
         values="sum | max; default sum. With one case both give the same run. The hook balances its cases by design (weights 234 : 1 give the weighted compliances 0.6943 and 0.6834), so `max` changes little there; a part with unbalanced cases is needed to measure it.",
         evidence=_E1_COMBINE),
    Card("settings.max_active_voxels", "yaml", "method", 1,
         "A guard, not physics: the run stops before it starts when the number of active voxels is above this number.", "code",
         values="integer >= 1; default 100000 (CONTRACTS.md: the Cholesky memory is about 4 GB at 80000 voxels on a 16 GB machine). A higher value "
                "allows a bigger problem and can need more memory than the machine has. You decide with the facts: `check_problem.memory_estimate` "
                "(`bytes` against `ram_bytes`; `environment` gives the RAM too) and the risk, which is that a run above the RAM can freeze the machine "
                "(a 0.5 mm whole-domain grid did on a 16 GB Mac). The options are a larger `voxel_size` (the error text suggests one) or a higher guard; "
                "report what you took, why and the price (memory, time, a coarser voxel). A change is a "
                "note on a resume, not a change of the inputs, so the resume fact of a finished run does not name it.",
         symptoms=(Symptom("run refused, too many voxels", "the error text names this key and the options; `check_problem.memory_estimate`", facts=("check_problem.memory_estimate",)),)),
    Card("settings.max_grid_voxels", "yaml", "method", 1,
         "A guard, not physics: the run stops before it starts when the grid (the box around all regions) is above this number of voxels.", "code",
         values="integer >= 1; default 5000000. Every region file widens the grid. You decide with the facts: `check_problem.memory_estimate` (`bytes` against "
                "`ram_bytes`; `environment` gives the RAM too) and the risk, which is that a run above the RAM can freeze the machine (a 0.5 mm whole-domain grid "
                "did on a 16 GB Mac). The options are a larger `voxel_size` or a higher guard; report what you took, why and the price (memory, time, a coarser "
                "voxel). `build_regions` has a guard of the same kind: it stops a build whose memory estimate is above the RAM, and `settings.allow_large_grid` of the "
                "spec lets the build go on. A change of this key is a note on a resume, so the resume fact of a finished run does not name it.",
         symptoms=(Symptom("run refused, grid too large", "the error text names this key; `check_problem.grid`", facts=("check_problem.grid.voxels",)),)),
    Card("settings.objective", "yaml", "both", 1,
         "`compliance` (default): the stiffest part for the allowed volume. `mass`: the lightest part that meets the stress and compliance "
         "limits; `volume_fraction` is then only the start density.", "code",
         kind_note="follows what the user asked for: a stiffest part for a mass, or the lightest part for a limit",
         values="compliance | mass. A mass run with no stress limit and no compliance limit is not refused: the engine says as a fact that no row holds "
                "material, and the run goes on with a volume ceiling at `volume_fraction`. A displacement or alignment row on a probe region does not "
                "hold material in a mass run. `case_combine: max` has no effect in a mass run."),
    Card("settings.stress_q", "yaml", "method", 1,
         "The KS sharpness of the stress rows: how close the aggregate is to the largest voxel stress.", "measured",
         values="number > 0; default 40. Only with a stress limit. On the hook stress case 20 gave less mass and more stiffness in two runs; 80 gave more mass and 104 rejected tries against 34 to 43 in the 6 base runs.",
         evidence=_E2_STRESS_Q,
         price_note=STRESS_NOISE + " q 20 moved several measures outside the range of the 6 base runs, all on the better side (two runs agree): the mass (x0.954) and the stiffness (x1.113 / x1.208), the relaxed peak "
                    "(0.474 MPa), both voxel peaks (0.523 and 0.486 MPa), the rejected tries (17 against 34 to 43) and the P2 pull_off stiffness (x1.127); no measure of the table is worse than the base range. "
                    "q 80 cost mass and rejected tries (x1.120 and 104 against 34 to 43) and gave no margin: the relaxed peak was 0.478 MPa and the grey design had 109 voxels above the limit."),
    Card("settings.stress_power", "yaml", "method", 1,
         "The relaxation power s of the stress rows (stress is scaled by density to the power s, so a void voxel carries no stress).", "measured",
         values="0 < s <= 1; default 0.5. Only with a stress limit. A high power makes the relaxed measure, the one the optimizer holds, a poor guide for the grey design: at 1.0 the relaxed peak "
                "was below the limit and the grey design had 571 voxels above it.",
         evidence=_E2_STRESS_POWER,
         price_note=STRESS_NOISE + " The mass moved up with both steps away from 0.5 (x1.054 at 0.35, x1.088 at 1.0); the grayness and the grey voxel peak are the facts that moved at 0.75 and 1.0."),
    Card("settings.stress_scale", "yaml", "both", 1,
         "The scale c of the stress rows. `auto` takes c at the first design of each beta stage, so the row is the relaxed peak at that moment. "
         "A number changes the stress limit that the optimizer holds: the row is c * S - 1 <= 0 (S is the KS aggregate over the limit), so it asks for S at or below 1/c. "
         "1 or more holds the relaxed peak below the limit; 0.5 holds twice the limit; 2.0 holds half of it.", "measured",
         kind_note="`auto`, the default, leaves the user's stress limit as it is. A number is the user's requirement whoever picks it, because it moves the limit: 1 or more tightens it (you may, with "
                   "a report and the price: 2.0 held half the limit and cost x1.916 in mass), and below 1 loosens it (only with the user's consent: 0.5 held twice the limit, the part carried 1.9 times the "
                   "limit, and the mass fell to x0.761). This follows the rule for a requirement (tighten with a report and the price, loosen only with consent) and not the origin of the key: the key is a setting, and its number moves the user's limit.",
         values="auto | number > 0; default auto. Only with a stress limit. With `auto` and a stage that starts far below the limit, c is small and "
                "the relaxed peak can go above the limit while the row is met (CONTRACTS.md section 17); the run log has a fact for it. `topopt check` prints a fact for a number above 1 (2.0: 'the "
                "relaxed peak is held to at most 0.5 times the limit') and one for a number below 1 (0.5: 'the row allows a relaxed peak up to 2 times the limit'), each with the words "
                "that the number tightens or loosens the user's stress limit, a requirement (F4-28), and nothing for 1.0 or auto (E2: SS1; SS05 and SS2 ran before the facts); the run log ends with the line `stress peak above the limit` when the relaxed "
                "peak is above the limit (met or not), and `metrics.stress.cases.<case>.relaxed.peak_above_limit` is true in the same cases (one test of the engine for both). A row that is NOT met in a run with a number above 1 is not a failure against the user's limit: the voxel peaks of E2/SS2 were below it.",
         symptoms=(Symptom("stress row not met though the part is under the limit, stress row met though the peak is above the limit", "`metrics.stress.cases.<case>.relaxed.met` with the peak and the limit "
                           "(`relaxed.peak_MPa`, `stress_limit_MPa`), `relaxed.peak_above_limit` and the voxel estimates in `summary`",
                           facts=("metrics.stress.cases.<case>.relaxed.met", "metrics.stress.cases.<case>.relaxed.peak_MPa", "metrics.stress.cases.<case>.stress_limit_MPa", "metrics.stress.cases.<case>.summary",
                                  "metrics.stress.cases.<case>.relaxed.peak_above_limit"),
                           when="`relaxed.met` is false with a `relaxed.peak_MPa` below `stress_limit_MPa` (a number above 1 holds less than the limit), or `relaxed.peak_above_limit` is true with "
                                "`relaxed.met` true (a number below 1 holds more than the limit; an `auto` scale far below 1 does the same: see `relaxed.c`)"),),
         evidence=_E2_STRESS_SCALE,
         price_note=STRESS_NOISE + " The price of a number is large and acts on the user's requirement: 0.5 gave mass x0.761 and a part at 1.9 times the limit (relaxed peak 0.957 MPa, 770 grey voxels above "
                    "the limit, P2 p99.9 1.716 MPa) while the row said met; 2.0 gave mass x1.916 and a row that is not met with the voxel peaks below the limit; 1.0 gave x1.060. The optimizer and P2 agree in "
                    "sign and differ in size: pull_off x0.464 (optimizer) and x0.567 (P2) at 0.5, x1.899 and x1.637 at 2.0, x1.125 and x1.094 at 1.0; hang x0.454 and x0.523, x2.118 and x2.104, x1.143 and x1.136."),
    Card("settings.stress_step_acceptance", "yaml", "method", 1,
         "Checks each optimizer step against the stress rows. A step that makes a row worse is tried again with the move limit halved "
         "(up to 6 tries in a pass).", "measured",
         values="true | false; default true. `false` takes every MMA step (CONTRACTS.md section 17: for comparisons; a reviewer's test without the rule "
                "collapsed). A run with no stress row does not use the rule. On the hook stress case at 0.5 MPa one run with `false` ended with both rows met and the mass of the base (E2/SA0); "
                "the history of such a run has no `step_acceptance` field, so there is no count of rejected tries.",
         symptoms=(Symptom("stress run jumpy, collapses", "`history.jsonl` field `step_acceptance` (`tries`, `accepted`) and the `fact: step acceptance` "
                           "lines of the run log", facts=("history.step_acceptance.tries", "history.step_acceptance.accepted")),),
         evidence=_E2_STEP_ACCEPTANCE,
         price_note=STRESS_NOISE + " The one run with `false` is at the edge of the range (pull_off 0.1 % above it: 1096.467 N/mm against 1095.152 N/mm) and every other measure is inside: no effect found at this limit. "
                    "The rule is not shown to be unneeded: in the review of the stress rows the example with the `pull_off` limit at 5 MPa ended without the rule "
                    "with its row not met (g +0.006 to +0.094), and a 1.2 mm run from a uniform start collapsed without it."),
    Card("settings.stop_rules", "yaml", "method", 1,
         "Lets the run end before the last pass: when the objective has settled and the run is feasible (`converged`), or when the run time reaches "
         "`time_budget_s`. Off by default.", "measured",
         values="false | true | mapping of `settings.stop_rules.*`. `topopt run --stop-rules` / `run_topopt(stop_rules=true)` wins over the YAML. A "
                "change of this key is a note on a resume, not a change of the inputs, so the resume fact of a finished run does not name it.",
         symptoms=(Symptom("takes too long, runs after it has settled", "`metrics.verdict.measures.objective_change` and `metrics.verdict.converged`", facts=("metrics.verdict.measures.objective_change.value", "metrics.verdict.converged")),
                   Symptom("the verdict does not follow my thresholds, the verdict is from other numbers", "`metrics.verdict.notes` (empty in a normal run) and `metrics.verdict.source`: when the run folder has no `verdict.json`, "
                           "or it does not match the history, the post step makes the verdict from the history with the default thresholds and says so in `notes`",
                           facts=("metrics.verdict.notes",), when="`metrics.verdict.notes` is not empty (`metrics.verdict.source` is then `history`)")),
         evidence=(*_E1_STOP, *_E3_STOP, _resume_row("objective_change 1e-4 -> 1e-2, resume of the finished run", "metrics.verdict.state iteration_limit -> converged; the part is the same",
                               "6.1 s", "stop_rules_thresholds"),),
         price_note="`stop_rules: true` with double the base passes on three parts (E3; objective ratios, a sign counts only outside the noise band of that part): the rule `converged` ended the hook run at pass 48 of 80 "
                    "(E1/SR80), the bracket runs at pass 78 (E3/BR-SR80 with 80 passes, E3/BR-SR160 with 160) and the ladle run at pass 79 of 160 (E3/LAD-SR160). The design is the one of the 80-pass run: flip 0.0 % against "
                    "E3/BR-IT80 and against E3/LAD-D0 (the hook run is x1.001 of its base). Against its own base the optimizer ratio is inside the band on the hook (x1.001) and the ladle (x1.000) and up on the bracket (x1.114 on the "
                    "grey base, x1.006 on the fitted base, the numbers of E3/BR-IT80): the sign differs. The 80-pass run ended `converged` too (E3/BR-IT80, E3/LAD-D0), so the rule ended a "
                    "160-pass run 82 and 81 passes early; the wall time was 51 s with the rule and 89 s without it on the bracket, and 142 s and 272 s on the ladle (the host load differs)."),
    Card("settings.stop_rules.enabled", "yaml", "method", 1, "Turns the stop rules on or off. A mapping turns them on unless it has `enabled: false`.", "code",
         values="true | false. A change is a note on a resume, so the resume fact of a finished run does not name it."),
    Card("settings.stop_rules.objective_change", "yaml", "method", 3,
         "The relative change of the objective between two passes that counts as settled. It sets the verdict of every run, and the stop of a run "
         "with the rules on.", "measured",
         values="number > 0; default 1e-4 (a text such as `1e-4` is read as a number). The default is not tuned for a part: give the number to the user. "
                "A larger value changes the verdict and leaves the part the same: the design stays as it is. "
                "With the rules on the key also decides when the run ends: that is a new run. A change is a note on a resume, so the resume fact "
                "of a finished run does not name it.",
         cost=RESUME_COST, call=RESUME_CALL,
         symptoms=(Symptom("not converged, did it finish", "`metrics.verdict.measures.objective_change` (value, threshold, met)", facts=("metrics.verdict.measures.objective_change.value", "metrics.verdict.measures.objective_change.met")),),
         evidence=(_resume_row("1e-4 -> 1e-2, resume of the finished run", "metrics.verdict.state iteration_limit -> converged", "6.1 s", "stop_rules_thresholds"),),
         price_note="F4-22 (E3): the end-of-run objective change alone does not predict the gain of doubling the passes. With the last beta stage reached the gain was x1.000 to x1.080 (optimizer and P2), without it up to "
                    "x1.391 in the optimizer and x1.287 in P2. The pairs, the numbers and the runs are on the card `settings.iterations`."),
    Card("settings.stop_rules.window", "yaml", "method", 3,
         "The number of passes in a row with a change at or below `objective_change`.", "code",
         values="integer >= 1; default 5. Same layers as `objective_change`, and a change is a note on a resume, so the resume fact does not name it.",
         cost=RESUME_COST, call=RESUME_CALL),
    Card("settings.stop_rules.grayness_max", "yaml", "method", 3,
         "The largest mean grayness mean(4 rho (1 - rho)) at the last pass that counts as black and white.", "code",
         values="0 < g <= 1; default 0.1. Same layers as `objective_change`, and a change is a note on a resume, so the resume fact does not name it.",
         cost=RESUME_COST, call=RESUME_CALL,
         symptoms=(Symptom("grey", "`metrics.verdict.measures.grayness.value` against its threshold. " + GRAY_NOT_ENOUGH, facts=("metrics.verdict.measures.grayness.value", "metrics.verdict.measures.grayness.met")),)),
    Card("settings.stop_rules.time_budget_s", "yaml", "method", 1,
         "Stops the run at the first pass that ends after this many seconds of run time (the time before a resume counts).", "code",
         values="number > 0 seconds; default none. A change is a note on a resume, so the resume fact of a finished run does not name it."),
    Card("settings.fine_pass", "yaml", "method", 1,
         "The settings of a fine pass that is made from this YAML (`--fine-from`, `--fine-pass`). One key for now: `objective_change`, the tolerance of the "
         "plateau stop `settled`. The key has no effect on the passes of a run that reads this YAML.", "code",
         values="mapping with the key `objective_change`; absent: the default tolerance. A change of this key is a note on a resume and on a continuation, "
                "not a change of the inputs, so the resume fact does not name it: a fine run keeps the value that it used (`run.fine_from.tolerance`)."),
    Card("settings.fine_pass.objective_change", "yaml", "method", 1,
         "The tolerance of the plateau stop `settled` of every fine pass made from this YAML. A fine pass ends when the largest relative change of the "
         "objective between two passes in the last `window` passes, at one beta, is at or below it and the constraints are met.", "measured",
         values="number > 0 and finite (a text such as `5e-4` is read as a number). Default `1e-3` (`config.FINE_OBJECTIVE_CHANGE`). `--fine-tol T` / "
                "`run_topopt(fine_tol=T)` wins for one run. The record `run.fine_from.tolerance_from` says which source gave the value: `yaml`, `option` or "
                "`default`. The window is `settings.stop_rules.window` (default 5). A smaller value runs more passes. A change of this key is a note on a resume "
                "and on a continuation, so the resume fact does not name it: the fine run keeps the value of its record.",
         cost="the passes after the loss is flat: a replay of the history of the arm run shows 9 passes less at 1e-3 than at 5e-4",
         call="edit `settings.fine_pass.objective_change` in the YAML; then `run_topopt(yaml_path, out_dir, fine_from=PARENT)` or `fine_pass=true`",
         symptoms=(Symptom("stopped too early, stopped too late, runs on after it is flat, the loss is still moving",
                           "`metrics.verdict.measures.objective_change.value` against `run.fine_from.tolerance`, and `metrics.verdict.state`",
                           facts=("run.fine_from.tolerance", "run.fine_from.tolerance_from"), when="the run is a fine pass: the key `fine_from` exists"),),
         evidence=_FINE_TOL_EVIDENCE),
    Card("settings.filter_solid_neighbours", "yaml", "method", 1,
         "How the filter treats solid voxels (fixed, load, solid): `density_one` counts them as density 1 (the pull of the support on its "
         "neighbours); `ignore` leaves them out.", "measured",
         values="density_one | ignore; default density_one. `check_problem.solid_pull` says how many design voxels get a filtered density above 0.1 from "
                "solid neighbours alone.",
         symptoms=(Symptom("thick root, material piles at the support", "`check_problem.solid_pull` (`voxels_above`, `share_above`, `b_max`)", facts=("check_problem.solid_pull.voxels_above", "check_problem.solid_pull.share_above", "check_problem.solid_pull.b_max")),),
         evidence=(*_E1_FSN, *_E2_FILTER_SOLID),
         price_note="Compliance case (part class `hook`, E1): the optimizer values moved within about 2 %, small and unproven. Stress case (part class `hook_stress`, E2): mass x1.013 and stiffness x1.039 / x1.042, "
                    "just outside the range of the 6 base runs, with the voxel and P2 measures inside it: a small move toward more mass and more stiffness (under 5 % on mass, stiffness and P2; "
                    "the report lists `ignore` under 'no effect found at this size'). " + STRESS_NOISE),
    Card("settings.min_member_mm", "yaml", "both", 1,
         "The smallest member that you want. It derives the filter radius: filter_radius = 1.877 x value (an estimate, not a guarantee; it sets "
         "the interior members, a skin at a free face can be thinner). An explicit `filter_radius` wins. With `settings.robust: wall` the radius comes from `print.min_wall_mm` and this key is only reported.", "measured",
         kind_note="requirement when a print rule or the user sets it; method when it is a look lever",
         values="number > 0, mm; default none. A fact appears when it is below 2 voxels. Memory: the filter is an FFT on the padded grid, so a larger radius only widens the pad. On the hook at voxel "
                "2 mm every value below 4.0 mm gets the warning 'A member needs 2 voxels or more' (the value 1.6 mm is the default filter radius, 1.5 voxels). "
                "`min_member_mm` v and `filter_radius` 1.877 v are one lever: byte-equal final.stl (E1/MM20, E1/FR375). A radius of 1 voxel or less is no filter "
                "(E1/MM10, E1/FR20 byte-equal): grayness 0.001, 5 loose islands removed by the mesh step. "
                "A wall of the value survives the filter and a rod needs about 1.76 x the value (4.0 mm asks for walls of 4.0 mm and rods of about 7.03 mm; `metrics.filter.min_feature_note` gives both sizes). "
                "`filter_radius` 1.068 x the value is the rod rule: it keeps rods of the value and it makes walls thinner than the value, so it loosens the request; use it only with the consent of the user. "
                "The two sizes are continuum values for the interior of the design space and not limits of a result: round members from the value to 1.76 x the value are also in the wall-rule results (F1-9; the cost of the two rules and what they leave are in the reading of the price below).",
         symptoms=(Symptom("webs, sheets, threads, thin sticks", "`metrics.filter.min_feature_note` (the size that survives; written in every run) and `metrics.filter.facts` (the warning lines "
                           "with options: `min_member_mm` under 2 voxels, a derived radius not above 1 voxel, both keys set). The print-check estimate `metrics.print.thinnest_member` did not follow the "
                           "filter on the hook (E1) and moved 4.8 % in the noise runs, so it does not detect a filter effect", facts=("metrics.filter.min_feature_note", "metrics.filter.facts"),
                           when="`metrics.filter.facts` is not empty (E1 at voxel 2 mm: a line for min_member_mm 1.0, 1.6, 2.0 and 3.0, none for 4.0 = 2 voxels)"),),
         evidence=(*_E1_MIN_MEMBER, *_E3_MIN_MEMBER, Evidence("1.5 mm at voxel 0.7 (filter 1.877 x value) against no value", "round members and Y forks in place of sheets and webs",
                            "tap -18 %, side -11 % against FREF; weight +2 %", STAND, "E7 = F0 vs E1 = FREF"),
                   Evidence("1.2 (from 1.5)", "same member look, finer forks; one small loose blob",
                            "weight 1.13, tap 0.94, side 0.99 of FREF (most of the price comes back)", STAND, "F1a"),
                   Evidence("2.0 (from 1.5)", "thicker, coarser, same layout", "0.85 / 0.67 / 0.74 of FREF (weight / tap / side)", STAND, "F1b"),
                   Evidence("1.0 (from 1.2, with a side load 0.3 N)", "about the same look, finer twigs", "weight +6 %, tap +7 %", STAND, "F3a vs F2a"),
                   Evidence("1.0 (from 1.2, side load 0.6 N)", "a wall with branches on its face, more webs: the look starts to break between 1.2 and 1.0",
                            "1.16 / 1.00 / 1.03 of FREF", STAND, "F2c"),
                   Evidence("4 mm on a larger part (user rule 2.3 mm)", "rods of about 7 mm: stricter than the user's print rule", "mass price not measured",
                            LUG, "case 1, F1-9"),
                   *_F19_MIN_MEMBER),
         price_note="Read P2 where the radius is 1 voxel or less: the optimizer says x1.009 / x1.008 (no filter), P2 says x0.937 / x0.878. Above 1 voxel the two agree in sign (min_member_mm 6.0: optimizer x0.691 / x0.716, P2 x0.805 / x0.816). A step to 2 voxels flips 15 to 18 % of the layout (E1/MM20, E1/FR40), and a radius of 3 voxels or more flips 53 to 66 %; read an optimizer difference below about 2 % as small, unproven. The printed part is the P2 model. "
                    "On three parts (E3; objective ratios, the hook from E1; a sign counts only outside the noise band of that part): a `min_member_mm` of 1 voxel (radius 1.88 voxels) gives an optimizer ratio down on all three "
                    "(hook x0.961, bracket x0.951, fitted bracket x0.981, ladle x0.968): same sign on the three parts, with the P2 ratio inside the band of its part on all four (x0.971, x1.001, x1.005, x0.993). "
                    "A `min_member_mm` of 2 voxels (radius 3.75 voxels, 251 entries per row against 19 at the default) gives optimizer and P2 ratios down on all three parts: same sign on the three parts. "
                    "F4-19, the 2-voxel member on five rows (optimizer objective / P2 objective / mass, grayness base -> run): hook x0.819 / x0.848 / x0.974, 0.025 -> 0.112 (E1/MM40); "
                    "bracket x0.755 / x0.907 / x0.970, 0.129 -> 0.314 (E3/BR-MM2, grey base); fitted bracket x0.889 / x0.977 / x0.986, 0.020 -> 0.137 (E3/BR-SMM2); "
                    "ladle at 4.0 mm x0.697 / x0.856 / x0.973, 0.014 -> 0.074 (E3/LAD-MM2); ladle at 3.25 mm x0.814 / x0.968 / x0.981, 0.012 -> 0.070 (E3/LAD-XMM2). The cost is 18.1, 24.5, 11.1, 30.3 and 18.6 % in the optimizer "
                    "and 15.2, 9.3, 2.3, 14.4 and 3.2 % in P2, outside the band of the part in all five. The 1-voxel member costs x0.951 to x0.981 in the optimizer, with P2 inside the band (above). "
                    "The warning 'A member needs 2 voxels or more to carry load' (a line of `metrics.filter.facts`) appears for a `min_member_mm` of 1 voxel on the hook, the bracket, the fitted bracket and the ladle "
                    "(E1/MM20, E3/BR-MM1, E3/BR-SMM1, E3/LAD-MM1); it does not appear for a `min_member_mm` of 2 voxels (E1/MM40, E3/BR-MM2, E3/BR-SMM2, E3/LAD-MM2, E3/LAD-XMM2), for an explicit `filter_radius` of "
                    "1, 2 or 3 voxels, or for the default radius. The thinnest-member estimate of the print check rose with the radius on the bracket (2.06 -> 2.12 and 2.51 mm at 1.88 and 3.75 voxels) and on the ladle "
                    "(2.89 -> 3.66 and 5.11 mm), and stayed at 3.00 mm on the hook (2.82 mm at the default)."
                    " F1-9, the wall rule against the rod rule at the same request of 2 voxels (`min_member_mm` v, radius 1.877 v, against `filter_radius` 1.068 v; the stiffness ratio of the wall-rule run over the rod-rule run as optimizer objective / P2 objective, then the mass ratio): hook.yaml at 2 mm (request 4.0 mm) x0.868 / x0.883, mass 0.985 (F1-9/hookA-WALL, F1-9/hookA-ROD); hook_short.yaml x0.867 / x0.877, mass 0.981 (F1-9/hookB-WALL, F1-9/hookB-ROD; hookB-WALL repeats E1/MM40 in the optimizer); bracket (request 2.0 mm) x0.812 / x0.906, mass 0.980 (F1-9/bracket-WALL, F1-9/bracket-ROD); ladle at 3.25 mm (request 6.5 mm) x0.837 / x0.943, mass 0.986 (F1-9/ladle-WALL, F1-9/ladle-ROD). The wall rule costs stiffness on every part, outside the bands of the class in the optimizer, in P2 and in mass, and the P2 loss is smaller than the optimizer loss. It also leaves less material thinner than the request: the share of the design material below the request is 11 % (wall rule) against 33 % (rod rule) and 49 % (default) on the hook, 1 % (wall rule) against 8 % (rod rule) and 9 % (default) on the bracket and 5 % (wall rule) against 25 % (rod rule) and 36 % (default) on the ladle. With the rod rule the request is met by less of the material; that is a loosening of the requirement, and the consent of the user applies. Rods: the wall-rule results hold round rod-like members from the request to 1.76 x the request, as the rod-rule results and the default do (share of the design material, wall rule / rod rule / default: hook 14 / 11 / 10 %, bracket 6 / 3 / 3 %, ladle 12 / 11 / 10 %), so 1.76 x the request is a continuum reference size and not a measured limit of these designs. The sizes are for the interior of the design space: the face layer, where the filter has one side, holds 89 % of the hook material and 37 % of the bracket material under the wall rule, and the interior sets of the hook are small (68 to 162 voxels)."),
    Card("settings.robust", "yaml", "method", 1,
         "`eroded`: the volume is on the nominal design and the stiffness on an eroded design, so a thin member survives only if it stays stiff after a small loss of "
         "material. `wall`: the same, aimed at the wall of `print.min_wall_mm`. `off` (default): one design.", "measured",
         values="off | eroded | wall (YAML `true` and `false` are read too). Each pass has one solve (the eroded design); one more solve at the end gives the "
                "nominal compliance. A stress limit or a buckling row with `eroded` or `wall` is refused when the run starts (the text gives the options). "
                "`wall` needs `print.min_wall_mm` (w, in mm; without it the YAML is an error with options). It is `eroded` plus three things: eta_e defaults to 0.7 "
                "(`settings.eta_eroded` wins); the filter radius is R = max(w / L(eta_e), 1.5 voxels), with L(eta_e) the width of the nominal wall per radius of the 3D hat kernel "
                "(L(0.7) = 0.7885; R = 1.522 mm at w 1.2 mm; `settings.filter_radius` wins, and `settings.min_member_mm` is only reported); and the density filter is padded with void "
                "outside the design space and at keep-out regions, so a wall at a face of the design space is as thick as an inner wall (fixed and load voxels stay solid). "
                "A known effect of the padding: less of the part lies on the faces of the design space. On the hook the share of the part surface within 0.75 mm of a design-space face "
                "went from 31.0 % (default) to 24.9 % (wall rule); nothing came loose (1 body, the load path joined). A mounting face on the envelope can then sit lower: check the result at that face, or make that face a solid region. "
                "The radius follows the grid of the run: a fine pass computes max(w / L, 1.5 voxels) again for its voxel; an explicit `settings.filter_radius` stays in mm. "
                "`eroded` alone lets a wall at a face survive at half the thickness, because the filter renormalises over the design space. When `print.min_wall_mm` is set and the mode is not `wall`, "
                "`check_problem` gives a fact with these numbers. The mode is not on by itself.",
         symptoms=(Symptom("icicles, spikes, thin spurs", "`cleanup_list` (spur candidates); `robust` gave no gain against small loops at thin ends (F2b)", facts=("cleanup_list",)),),
         evidence=(*_E1_ROBUST, Evidence("robust: eroded (default eta)", "look nearly the same; small loops at thin ends stay (no gain against icicles)",
                            "weight -7 %, tap -6 %", STAND, "F2b vs F1a"),
                   Evidence("robust: wall, print.min_wall_mm 1.2 (R 1.522 mm, eta_e 0.7, padded filter), 80 passes, voxel 0.5 mm; against the default radius",
                            "share of the surface area with a wall under 1.2 mm 4.145 % -> 0.000 %",
                            "compliance 1.253 -> 1.352 (+7.9 %), 636 s -> 679 s (x1.07), peak 5.51 -> 4.86 GiB, 1 body, mass 27.13 g both (the cause of the lower memory was not measured)", WALL_HOOK, "wall rule, step 1 (prototype)"),
                   Evidence("robust: wall, print.min_wall_mm 1.2, the same hook, gpu-mg on the Intel Arc iGPU (this code, not the prototype)",
                            "share of the surface area with a wall under 1.2 mm 0.000 %, 1 body",
                            "compliance 1.3524, mass 27.13 g, peak 5.18 GiB, 1,187 s on a busier host", WALL_HOOK, "wall rule, step 2 (reproduced)"),)),
    Card("settings.eta_eroded", "yaml", "method", 1,
         "The threshold of the eroded design in robust mode. A higher value erodes more: a thin member needs a thicker section to survive.", "measured",
         values="0.5 < eta < 1; default 0.55 with `robust: eroded` and 0.7 with `robust: wall`. Only with `robust: eroded` or `wall` (otherwise a fact, not an error). The effect grows with beta. A robust run with a stress row or a buckling row is "
                "refused when it starts (CONTRACTS.md section 22b), so this key was not tried on the stress case or the buckling case. At voxel 2 mm and filter radius 3 mm the log warns for 0.60 (a wall of 1.98 mm, 0.99 voxels) and for 0.75 (2.65 mm, 1.3 voxels): "
                "a limit below 2 voxels is at the grid scale, so robust mode has little effect on the member size. The options in the warning are `settings.min_member_mm`, a larger `settings.filter_radius` or a smaller voxel size.",
         evidence=(*_E1_ROBUST, *_E2_ETA),
         price_note="The pull_off optimizer stiffness falls steadily with eta on 2 mm: x0.979, x0.967, x0.958, x0.941 for 0.55, 0.60, 0.65, 0.75 (hang x0.999 to x0.982, mass x0.995 to x0.997). P2 pull_off is "
                    "x0.947, x0.964, x0.940, x0.944 in the same order: 3.6 to 6 % under B0, outside the P2 band of +-3.5 %, and it does not follow eta in order. The number that the key moves most is the gap between "
                    "the eroded and the nominal compliance (`metrics.robust`): 2.0, 5.1, 9.8 and 27.0 % above the nominal one. The robust part costs 0.25 s of the run; a try is a new run (layer 1, 20 to 21 s here)."),
    Card("settings.mesh_method", "yaml", "method", 3,
         "How the final mesh is smoothed: `sdf` (volume-preserving smoothing of a signed distance field, exact region shapes) or `blur` (Gaussian blur, "
         "one iso level, Taubin). It acts in the post step only.", "measured",
         values="sdf | blur; default sdf. `topopt run --mesh-method` wins over the key. The sdf step falls back to blur when the fine grid has more "
                "than 30 million voxels (`metrics.mesh.fallback`). The mass changes between the two methods: sdf is x0.912 of blur at voxel 2 mm and x0.950 at 1.2 mm on the hook; on the bracket and the ladle the blur mass is the lower one (blur against sdf x0.992 on the bracket, x0.986 on the fitted bracket and x0.980 on the ladle; E3).",
         cost=RESUME_COST, call=RESUME_CALL,
         symptoms=(Symptom("rough surface, lumpy, faceted", "`metrics.mesh.method`, `metrics.mesh.volume_error_pct`, `metrics.mesh.local_loss`, "
                           "`metrics.mesh.stl_roundtrip.watertight`", facts=("metrics.mesh.method", "metrics.mesh.volume_error_pct", "metrics.mesh.local_loss", "metrics.mesh.stl_roundtrip.watertight")),
                   Symptom("loses thin parts, mass differs", "`metrics.mesh.local_loss.deep_lost_mm3` and `metrics.mesh.islands_removed`", facts=("metrics.mesh.local_loss.deep_lost_mm3", "metrics.mesh.islands_removed")),
                   Symptom("blur mesh though sdf was set, mesh step skipped or failed", "`metrics.mesh.fallback` (null in a normal run; else the fact of the skipped SDF step: `message`, `voxels`, `limit`, `options`, or `error` "
                           "for an exception) and `metrics.mesh.requested` against `metrics.mesh.method`; the run log has a warning", facts=("metrics.mesh.fallback", "metrics.mesh.requested"),
                           when="`metrics.mesh.fallback` is not null (the SDF step did not run and the blur path made the mesh)"),
                   Symptom("a piece is missing from the STL, a loose body vanished", "`metrics.mesh.extra_bodies_removed` (count) and `metrics.mesh.extra_bodies_volume_mm3`: bodies that do not touch a frozen region "
                           "are left out of the sdf mesh and out of the volume target; the key is in the record of the sdf method only; the run log has a warning",
                           facts=("metrics.mesh.extra_bodies_removed", "metrics.mesh.extra_bodies_volume_mm3"), when="`metrics.mesh.extra_bodies_removed` is above 0"),
                   STL_JOIN_SYMPTOM),
         evidence=(*_E1_MESH, *_E3_MESH, _resume_row("sdf -> blur, resume of the finished run", "result/final.stl and preview.bin rewritten; mass_final_g 25.72 -> 28.21",
                               "6.2 s", "mesh_method_blur"),),
         price_note="Read P2: the optimizer value is the same for both methods (it comes from the field, not from the mesh: the last snapshot is byte-equal), and P2 of the blur mesh is x0.571 / x0.784 of the sdf mesh, with the mass x1.097. The printed part is the mesh. "
                    "On three parts (E3; P2 objective ratio of the blur run against the sdf run of the same part): the optimizer ratios are x1.000 on every part (the loop is the same) and P2 is lower on all three, the same sign: "
                    "hook x0.669 (hang x0.571, pull_off x0.784), bracket x0.968 (fitted bracket x0.938) and ladle x0.009 (scoop x0.005, scrape x0.007, stir x0.071), a P2 objective loss of 33.1 % (hook), 3.2 % (bracket), 6.2 % "
                    "(fitted bracket) and 99.1 % (ladle). The mass ratio differs in sign: hook x1.097, bracket x0.992 (fitted x0.986), ladle x0.980. The ladle value comes from one run, cause not isolated (the facts "
                    "of its verify log are in the row E3/LAD-MBb); x0.005 is not a measured stiffness of a printable part."),
    Card("settings.envelope_fill", "yaml", "method", 3,
         "Fills the staircase of the design space surface. A face of the design space that is slanted to the grid axes gives steps in the density field, and the mesh shows them "
         "as panels. With the key on, the `sdf` mesh step grows the sharp field to that surface before the smoothing, and the cut to the envelope then makes the face flat. "
         "The mass target does not change. It acts in the post step only.", "code",
         values="off | auto | on; default off, the mesh of the versions before the fill. `topopt run --envelope-fill` wins over the key. `auto` runs the fill, and skips it when it would add more than 5 % of the volume target inside "
                "the design space. `on` runs it also above that share. `off` (default) does not run it. It works only where the envelope is known (a closed design space STL and STL files for the fixed, "
                "load and solid regions) and with `mesh_method: sdf`; else `metrics.mesh.envelope_fill.reason` says why it did not run. `metrics.mesh.envelope_fill.grown_share_pct` gives the share. "
                "When `on` makes sense: `auto` skipped the fill, and the part has slanted faces with visible steps. The cost of `on` above the cap: the mass-neutral calibration pays for the growth "
                "by thinning the members, and on a sparse part where the fill is a large share of the mass the thin members break away. Measured on the 2 mm cantilever of the optimizer tests (12 passes; the fill adds 36 % of the target inside the design space): without a bore, `on` gives a loose body of 2.8 mm3 that the mesh step removes (with `off`: none). "
                "With a bore on the same part, `on` removes no body (`off` removes one of 4.7 mm3), and the stock ring of the bore is 0.0 mm3 against 0.3 mm3 with `off` (`metrics.interfaces.items[].stock_volume_mm3`). "
                "When `auto` or `on` make sense: a part with slanted faces whose look counts more than a few percent of stiffness. When `off` (the default) stays right: a part gets worse with the fill (thin walls under the print limit, a lost feature), or the stiffness at the mass counts more than flat faces, above all on a coarse grid. The cost in stiffness: the grown skin carries little load, and the members pay for it. Measured with `verify` on one run of `examples/ladle/ladle.yaml` (with the fill against `off`, the same mass): at the example settings (3.25 mm, 80 passes, the fill adds 2.3 %) the stiffness of the three load cases falls by 1.7 to 3.6 %; at 6.5 mm and 30 passes (3.5 %) it falls by 7 to 21 %, and the stress by volume rises by 28 to 42 %. "
                "The fill costs post-step time: side by side on one host, the drone arm 334 s with the fill against 191 s without it, `examples/hook/hook.yaml` at 1.2 mm 93 s against 65 s. "
                "When the fill ran, `metrics.mesh.envelope_fill.note` and one warning line of the post step say the cost: material moves from the members to the skin, and at the same mass the stiffness can drop. "
                "Cost near the cap: on `examples/interfaces/bracket.yaml` at 1.5 mm (12 passes) the fill adds 4.46 % of the target, below the 5 % cap, and the level shifts by -0.157 mm, so every member is thinner by that much. The growth depth and the overshoot are internal.",
         cost=RESUME_COST, call=RESUME_CALL),
    Card("settings.part_only_min_film_mm", "yaml", "both", 3,
         "With `regions.not_printed`: cuts off the thin wall (the film) that stays at a contact face of `result/part_only.stl`, where the wall behind the contact "
         "face of a removed body is thinner than this value. It acts in the post step only. Default off.", "measured",
         kind_note="a wall thickness that the user names is a requirement; a value you pick to clean the film is a method setting",
         values="number > 0, mm, no upper bound; absent or null is off (YAML `1e-2` is read as a number). It needs `regions.not_printed`: with the key alone no file is made, "
                "and `topopt check` and the run log give the fact. The step recommends no value. `metrics.part_only.contact.<label>.thinnest_wall_mm` is the lower end of the film, "
                "not a value to use (the hook tip has 0.000033 mm, and 0.001 mm cuts 2.0 of its 485.5 mm2); `area_below_voxel_mm2` is the scale of the whole film (292.6 mm2 of the tip). "
                "A way to find a value: write one, resume the finished run, read `thin_area_mm2`, `cut_area_mm2` and `thin_area_after_mm2` of each body, and change the value if the film "
                "that you want gone is still there. Hazards (CONTRACTS.md 9.1): the cut area can be larger than the thin area (the repair of pinch vertices); `film_removed_mm3` also holds "
                "material that is not film (12 to 22 % of the removed volume on the hook); a value equal to the voxel size left one non-manifold edge on the hook (`watertight` false, "
                "`pinches_left` 2); a value that removes every piece is an error of the step, and the option is to remove the key. The resume rewrites `result/` of the run folder: copy "
                "the folder first to keep the old files. Read the facts of `metrics.part_only` (thin area, cut area, `thin_area_after_mm2`, `pinches_left`) and the check of the file "
                "(`verify_run` with `part_only=true`): the value is the wall thickness that the user wants gone, and the numbers say what that costs at the contact faces. "
                "The stiffness values of the rows are of the tet model of the file, not of the optimizer. The hook and bar numbers of this card were " + BEFORE_F4_27B + ".",
         cost=RESUME_COST, call=RESUME_CALL,
         symptoms=(Symptom("film at the plate, thin skin on the contact face, paper-thin wall, sliver, extra film where the bolted body was",
                           "`metrics.part_only.contact.<label>.area_below_voxel_mm2` and `.thinnest_wall_mm` for each removed body (there with the setting off too)",
                           facts=("metrics.part_only.contact.<label>.area_below_voxel_mm2", "metrics.part_only.contact.<label>.thinnest_wall_mm"),
                           when="`metrics.part_only.contact.<label>.area_below_voxel_mm2` is above 0 for a removed body"),
                   Symptom("the mesher refuses part_only.stl, pinched or non-manifold file, almost no stiffness in the check of the part-only file",
                           "`metrics.part_only.pinches_left` (with the setting), `metrics.part_only.watertight` and `metrics.part_only.shortest_edge_mm`; the stiffness of the "
                           "tet model of the file is no fact of the result yet (read `verify_run` with `part_only=true`)",
                           facts=("metrics.part_only.pinches_left", "metrics.part_only.watertight", "metrics.part_only.shortest_edge_mm"),
                           when="`metrics.part_only.pinches_left` is above 0, or `metrics.part_only.watertight` is false")),
         evidence=(Evidence("0.25 on the hook with `regions.not_printed` (3 removed bodies), resume of the finished run",
                            "thin area 187.7 mm2 (173 triangles), cut area 286.9 mm2 (207 triangles), summed over the 3 bodies; `topopt verify --part-only` of the file (on a scratch merge): "
                            "stiffness 559 N/mm (hang) and 1131 N/mm (pull-off), against 0 and 0 N/mm for the file without the setting (the film carries the load)",
                            "the post step only, no pass: 7.8 s on a quiet host, 12.0 s on a loaded one; a new run took 16 s", NOT_PRINTED, "CONTRACTS 9.1: hook 0.25; " + BEFORE_F4_27B),
                   Evidence("0.5 on the same hook", "thin area of the tip 242.2 mm2 (of 485.5 mm2); `topopt verify --part-only`: stiffness 488 N/mm (hang) and 411 N/mm (pull-off); one new wedge "
                            "triangle of the keep-out body stays (0.135 mm2, wall 0.44 mm)", "the post step only, as above", NOT_PRINTED, "CONTRACTS 9.1: hook 0.5; " + BEFORE_F4_27B),
                   Evidence("0.05 on the same hook", "thin area 30.95 mm2, cut area 90.66 mm2; 3.49 mm3 of material removed against 0.58 mm3 for the thin triangles alone; the mesher on the file: "
                            "the default size fails (the prepared surface crosses itself at 6 places), 3.0 mm gives a mesh", "the post step only, as above", NOT_PRINTED, "CONTRACTS 9.1: hook 0.05; " + BEFORE_F4_27B),
                   Evidence("2.0 on the same hook (= the voxel size)", "one non-manifold edge after the float32 reload: `watertight` false, `pinches_left` 2; the log line and a note say so",
                            "the post step only, as above", NOT_PRINTED, "CONTRACTS 9.1: hook 2.0; " + BEFORE_F4_27B),
                   Evidence("0.05, 0.1 and 0.2 on the bar of `tests/test_part_only.py` (voxel 1 mm)",
                            "the mesher on the file at 4 sizes (1.0, 1.2, 1.5, 2.0 mm): 4 of 4 give a mesh at 0.05, 0 of 4 at 0.1, 4 of 4 at 0.2, and 0 of 4 without the setting; the area with a wall "
                            "under the value (8 sample points on each triangle) is 12.0, 6.1 and 4.2 mm2 against 33.3, 54.5 and 72.8 mm2 without it",
                            "the post step only: 5.5 s on a quiet host, 8.4 s on a loaded one; a new run 6 s", NOT_PRINTED, "CONTRACTS 9.1: bar 0.05, 0.1, 0.2; " + BEFORE_F4_27B))),
    Card("settings.gpu_device", "yaml", "method", 1,
         "The OpenCL device of the solver gpu-mg: auto, a device number, or a part of a device name. Read by gpu-mg only.", "code",
         values="auto | number | text; default auto (the first GPU, else the first accelerator, else the first CPU device)."),
    Card("settings.gpu_case", "yaml", "method", 1,
         "The precision case of gpu-mg: B (the outer PCG in 64-bit; the device needs 64-bit float) or C (32-bit on the device, refinement on the CPU). "
         "Read by gpu-mg only.", "code",
         values="auto | B | C; default auto (B if the device reports 64-bit float, else C). A forced B on a device without it is an error."),
    Card("settings.gpu_tol", "yaml", "method", 1,
         "The relative residual that each solve of the optimizer loop reaches (gpu-mg only).", "code",
         values="0 < v < 1; default 1e-6 (YAML 1.1 reads 1e-6 as text; the loader accepts it). A changed value is a changed input on a resume.",
         symptoms=(Symptom("gpu run stops: did not reach the relative residual", "the error text names the options; `gpu_mg.jsonl` in the run folder "
                           "has the residual history", facts=()),)),
    Card("settings.gpu_backward_error", "yaml", "method", 1,
         "A solve is also accepted when its normwise backward error is at most this number (the stop test of gpu-mg).", "code",
         values="0 <= v < 1e-6; default 1e-15; 0 turns the rule off. It applies to the loop and to the stress step. A changed value is a changed input "
                "on a resume."),
    Card("settings.gpu_max_iter", "yaml", "method", 1,
         "The number of PCG iterations of the first pass of a gpu-mg solve (case B). Case C has no first pass.", "code",
         values="integer >= 1; default 1000; no upper bound. The text of a failed solve offers a larger value; a resume after that edit needs "
                "`allow_changed_inputs=true`."),
    Card("settings.perimeter_max_mm2", "yaml", "both", 1,
         "A bound on the surface area P of the part, counted in voxel faces: one more constraint row. A lower bound gives fewer bumps and a smoother "
         "shape.", "measured",
         kind_note="requirement when the user or a print rule states it; method when it is a look lever",
         values="number > 0, mm2; default none. A bound below the smallest possible P is refused with the options (`check_problem.perimeter.p_min_mm2`). "
                "P is about 1.24 times the area of final.stl in one measured problem (CONTRACTS.md section 26.3).",
         symptoms=(Symptom("rough, bumpy, too much surface", "`run.json` key `perimeter` (`p_final_mm2` against `max_mm2`) and `check_problem.perimeter`", facts=("run.perimeter.p_final_mm2", "run.perimeter.max_mm2", "check_problem.perimeter")),),
         evidence=_E2_PERIMETER,
         price_note="On the hook at 2 mm the smallest possible perimeter is 6776.0 mm2 (`topopt check`) and the plain run has 16030.6 mm2. A bound at 4.4 times the smallest does nothing (the run equals B0); at 2.07 times it "
                    "binds and P2 falls by 15 to 16 %, more than the optimizer (x0.918 / x0.939) shows; at 1.62 times it binds, the volume row goes slack, the stiffness falls to less than half and the part splits in 2 bodies. "
                    "The perimeter is counted in voxel faces, so it is not the area of final.stl. The row adds nothing to the seconds per pass (median 0.33 to 0.41 s against 0.35 s for B0)."),
    Card("settings.objective_scale", "yaml", "method", 1,
         "Where the optimizer takes the scale of the objective: `stage` (default) at pass 0 and again at the first pass of each beta stage; "
         "`first` at pass 0 only (the behaviour before P3a).", "measured",
         values="stage | first; default stage. A beta jump removes volume. With `first` the scaled gradient can be very small and the volume comes "
                "back slowly. CONTRACTS.md section 13 records, on a small CPU copy of the stand: after the last jump the volume is 0.73 of the limit with "
                "`first` and 0.94 with `stage`. On the hook (E1/OSF) `used_thresholded` was 0.989 with `first` and 0.995 with `stage`: the volume did not collapse in either run. "
                "A change from or to `first` is a change of the inputs on a resume.",
         symptoms=(Symptom("lighter than allowed, volume not used", "`metrics.verdict.facts.volume.used_projected` and `history.jsonl` "
                           "`stage_end.volume` at the end of each beta stage", facts=("metrics.verdict.facts.volume.used_projected", "history.stage_end.volume")),),
         evidence=_E1_OSCALE),
)


# --- mirror, limits, buckling, print ------------------------------------------------------------------------------------------
_MIRROR = (
    Card("mirror", "yaml", "requirement", 1,
         "Symmetry planes: the optimizer averages the design with its mirror image, so the part is symmetric about each plane. A part that "
         "the user wants symmetric needs it; a plane adds a constraint and so costs some stiffness for a part that is not symmetric.", "measured",
         values="list of at most 3 planes with distinct axes; empty or null is no key. The planes are exact on the voxel grid (the plane "
                "moves the grid origin by less than one voxel). A voxel with no partner (an orphan) stays a free variable.",
         symptoms=(Symptom("not symmetric, lopsided, one side different", "`check_problem.mirror` (`planes[].label_mismatches`, `planes[].orphans`) "
                           "before the run; `run.json` key `mirror.end` (`rho_deviation`) after it", facts=("check_problem.mirror.planes[].label_mismatches", "check_problem.mirror.planes[].orphans", "run.mirror.end.rho_deviation")),),
         evidence=_E2_MIRROR,
         price_note="The price is the asymmetry of the load, not the key. With the real `pull_off` load (a side part; the loads are not mirrored) the pull_off optimizer stiffness is x0.886 (face parity) and x0.946 (centre parity) at "
                    "2 mm and x0.921 at 1.2 mm, and P2 x0.869, x0.890 and x0.866: 8 to 13 % of the pull_off stiffness. The `hang` stiffness (a symmetric load), the mass and the seconds per pass did not move (median pass 0.327 s and 0.332 s "
                    "against 0.346 s for B0). With a symmetric load the mirror cost about 1 % of the optimizer stiffness (x0.990 / x0.995 face, x0.992 / x0.995 centre) and changed the design (flip 18.2 %). A mirror moves the grid "
                    "origin along y by less than one voxel, so a mirror run is not comparable voxel by voxel with a plain run: `compare_runs` says the grids differ and gives no flip. The mirror error of final.stl "
                    "(`topopt.ratchet.mirror_error_mm`, the measure of the ratchet rule `mirror`) is not zero for a symmetric part; it follows the orphan voxels."),
    Card("mirror[].normal", "yaml", "requirement", 1,
         "The plane: the axis of its normal and its coordinate on that axis.", "code",
         values="`normal`: x | y | z (any case), or a vector with one non-zero component; required. A plane has no direction, so a sign has no meaning. "
                "A normal that is not along an axis is refused; the error says to turn the part and the loads. `at_mm`: a finite number, mm; required.",
         also=("mirror[].at_mm",)),
    Card("mirror[].align", "yaml", "method", 1,
         "Where the plane lies in the voxel grid: on a voxel face (`face`) or through the voxel centres (`centre`). `auto` takes the parity with "
         "fewer label mismatches.", "measured",
         values="auto | face | centre; default auto. Each plane costs one more voxelization (the other parity). A given `face` or `centre` keeps its parity, "
                "and `check_problem.mirror.planes[].other_parity` has the counts of the other one. The hook example has 134 mismatches with "
                "the face parity and 0 with the centre parity at 1.2 mm (CONTRACTS.md section 26.1).",
         symptoms=(Symptom("not symmetric, lopsided", "`check_problem.mirror.planes[].label_mismatches` (`face` and `centre`), `check_problem.mirror.planes[].orphans` and, after the run, `run.mirror.end.rho_deviation`; "
                           "for a given `face` or `centre`, `check_problem.mirror.planes[].other_parity` (`parity`, `label_mismatches`, `orphans` of the other parity), and a `mirror:` line when it has fewer label mismatches",
                           facts=("check_problem.mirror.planes[].label_mismatches", "check_problem.mirror.planes[].orphans", "run.mirror.end.rho_deviation")),),
         evidence=_E2_ALIGN,
         price_note="`auto` is the default for a reason at 2 mm: forcing `centre` left 570 orphan voxels, added a layer to the grid and gave a mirror error of final.stl of 3.36 mm (symmetric load: 1.94 mm) against 0.462 mm "
                    "(0.471 mm) for `auto`, which chose `face`; neither parity has 0 label mismatches at 2 mm (50 for face, 1140 for centre). At 1.2 mm `auto` chose `centre` with 0 mismatches and 0 orphans (mirror error 0.201 mm, the "
                    "floor of this measure at that grid). A forced `align` can make a mirror run that is not a mirror: read the label mismatches and the orphans in `check_problem` before the run."),
)

_LIMITS = (
    Card("limits", "yaml", "requirement", 1,
         "Motion limits of the finished part: the displacement of a region, and the relative shift and tilt of two regions. Each limit is a "
         "constraint row of the optimizer, and it asks for more material when the stiffest part at the allowed volume does not meet it.", "code",
         values="mapping with the lists `displacement` and `alignment`; both optional. A region is a `regions.measure` probe or a `regions.load` name. "
                "A limit that the design cannot meet at the volume is a fact, not an error. Each row and each case costs one adjoint solve in each pass.",
         symptoms=(Symptom("moves too much, droops, sags, too soft at the tip", "`metrics.rows[]` (`value_end`, `limit`, `met`) for each limit row; "
                           "`metrics.stiffness_cases[].N_per_mm`", facts=("metrics.rows[].value_end", "metrics.rows[].limit", "metrics.rows[].met", "metrics.stiffness_cases[].N_per_mm")),
                   Symptom("tilts, shifts, misaligned", "`metrics.rows[]` of kind `shift` and `tilt` (`value_end`, `limit`, `met`)", facts=("metrics.rows[].value_end", "metrics.rows[].limit", "metrics.rows[].met")))),
    Card("limits.displacement", "yaml", "requirement", 1,
         "The list of displacement rows: the mean displacement of a region stays within a limit.", "code",
         values="list of mappings; the pair (sense, name) is unique. Under `settings.objective: mass`, a row on a probe region does not hold "
                "material: add a stress or compliance limit. The measured rows (limits of 0.1, 0.022 and 0.019 mm on the 2 mm hook) and their price are on the card `limits.displacement[].limit_mm`."),
    Card("limits.displacement[].limit_mm", "yaml", "requirement", 1,
         "The largest allowed mean displacement of the region, in mm. With `direction` it limits the motion along that vector; `sense` says "
         "whether the motion either way (abs), or only one way (max, min) counts.", "measured",
         values="number > 0, mm (a limit of 0 is not allowed). `direction`: any non-zero vector (it is normalised); without it the row limits the norm "
                "of the mean displacement vector. `sense`: abs (default) | max | min; max and min need a direction. The report says met for "
                "g <= 0.005, so a row 0.5 % over its limit is met. Loosen the limit only with the consent of the user." + " `verify_run(gate=true)` reads this as the limit of the tet value (card `verify_run:gate`). The gate test is strict (no 0.5 % slack).",
         also=("limits.displacement[].direction", "limits.displacement[].sense"),
         symptoms=(Symptom("displacement limit not met, the tip still moves too much", "`metrics.rows[]` of kind `displacement` (`name`, `value_end`, `limit`, `met`, `g_end`) and the line `constraint not met` "
                           "in `run.log`", facts=("metrics.rows[].met", "metrics.rows[].value_end", "metrics.rows[].limit", "metrics.rows[].g_end"),
                           when="`metrics.rows[].met` is false for a row of kind `displacement`; on the hook (E2/LDc) a limit 21 % under the value of the plain run was not met (0.02070 mm against 0.019 mm)"),),
         evidence=_E2_DISPLACEMENT,
         price_note="A limit moves stiffness between the cases: 0.022 mm (8 % under the 0.02397 mm of B0) was met with hang x0.665 and pull_off x1.064 (P2 x0.701 / x1.084); 0.019 mm was NOT met (value 0.02070 mm, `feasible` false) with "
                    "hang x0.570 and pull_off x1.112. At 2 mm and 40 passes the smallest pull_off displacement that was reached is near 0.0207 mm. `topopt check` prints no line for this row (no estimate of the smallest "
                    "possible displacement); the run log says `constraint not met`. The row adds 14 % to 39 % to the seconds per pass (median 0.48 s and 0.39 s against 0.35 s; a rough figure on a loaded host). "
                    "A limit above the value of B0 (0.1 mm) is not active and gives B0 within the E1 bands."),
    Card("limits.displacement[].region", "yaml", "requirement", 1,
         "The region whose mean displacement is limited, and the cases in which the row applies.", "code",
         values="`region`: a `regions.measure` or `regions.load` name; required. `cases`: one case name or a list; default every load case (one "
                "row for each case). `name`: text; default the region name.",
         also=("limits.displacement[].cases", "limits.displacement[].name")),
    Card("limits.displacement[].price", "yaml", "method", 1,
         "The MMA price of a violation of this row. A higher price makes the optimizer meet the limit before it keeps the volume.", "code",
         values="number > 0; default 1000. At the price 1000 the volume row and the limit row share the miss; with price 100 the volume stays at its "
                "target and only the limit row is unmet (CONTRACTS.md section 31).",
         symptoms=(Symptom("limit not met, still moves too much", "`metrics.rows[].met` and `g_end`; the line `constraint not met` in `run.log`", facts=("metrics.rows[].met", "metrics.rows[].g_end")),),
         also=("limits.alignment[].price",)),
    Card("limits.alignment", "yaml", "requirement", 1,
         "The list of alignment rows: the shift and the tilt of two regions against each other (across an axis) stay within limits.", "code",
         values="list of mappings; the name is unique. A rigid motion of the whole part gives shift 0 and tilt 0. The fit of a region needs nodes "
                "that are not on one line (an error names the options)."),
    Card("limits.alignment[].shift_limit_mm", "yaml", "requirement", 1,
         "The largest relative shift of the two regions across `axis`, in mm, and the largest relative tilt in rad. One row for each case and each limit.",
         "code",
         values="`shift_limit_mm`: number > 0, mm. `tilt_limit_rad`: number > 0, rad. At least one of the two is needed. Loosen a limit only with "
                "the consent of the user." + " `verify_run(gate=true)` reads this as the limit of the tet value (card `verify_run:gate`). The gate test is strict.",
         symptoms=(Symptom("misaligned, bores do not line up, binds", "`metrics.rows[]` of kind `shift` and `tilt` (`value_end`, `limit`, `met`)", facts=("metrics.rows[].value_end", "metrics.rows[].met")),),
         also=("limits.alignment[].tilt_limit_rad",)),
    Card("limits.alignment[].region_a", "yaml", "requirement", 1,
         "The two regions that are compared, and the cases in which the row applies.", "code",
         values="`region_a`, `region_b`: two different `regions.measure` or `regions.load` names; required. `name`: default <region_a>-<region_b>. "
                "`cases`: as for the displacement rows.",
         also=("limits.alignment[].region_b", "limits.alignment[].name", "limits.alignment[].cases")),
    Card("limits.alignment[].axis", "yaml", "requirement", 1,
         "The axis across which the shift and the tilt are measured. The shift along the axis and the twist about it are not limited.", "code",
         values="any non-zero vector of 3 numbers (it is normalised); required."),
)

_BUCKLING = (
    Card("buckling", "yaml", "method", 3,
         "The linear buckling check of the final design, and the optional buckling row of the optimizer. Off by default; the check is a voxel "
         "estimate, not a verdict.", "measured",
         values="mapping; keys on the cards `buckling.*`. The check needs a direct solver (cholesky or pardiso); with another solver it is skipped "
                "with a fact. `topopt run --buckling` also turns it on.",
         cost=RESUME_COST, call=RESUME_CALL,
         symptoms=(Symptom("buckles, bends sideways, slender, wobbly", "`metrics.buckling.lowest_load_factor` and `metrics.buckling.cases[].load_factors` "
                           "(the multiplier on the load case; null when the check did not run: `metrics.buckling.status`)", facts=("metrics.buckling.lowest_load_factor", "metrics.buckling.cases[].load_factors", "metrics.buckling.status")),),
         evidence=(_resume_row("buckling.check true (+ modes 3), resume of the finished run", "metrics.buckling is new (lowest load factor 123.085 for the "
                               "case pull_off)", "12.5 s", "buckling_check"), *_E2_BK_CHECK),),
    Card("buckling.check", "yaml", "method", 3,
         "Turns the buckling check of the final design on. It runs after the post step and writes `metrics.buckling`.", "measured",
         values="true | false; default false. `topopt run --buckling` and `--no-buckling` win over the key. `topopt buckling <run>` repeats the check "
                "on a finished run (CLI only; there is no MCP tool).",
         cost=RESUME_COST, call=RESUME_CALL,
         symptoms=(Symptom("buckles, bends sideways, slender", "`metrics.buckling.lowest_load_factor`; `metrics.buckling.status`", facts=("metrics.buckling.lowest_load_factor", "metrics.buckling.status")),),
         evidence=(_resume_row("buckling.check true (+ modes 3)", "metrics.buckling is new", "12.5 s", "buckling_check"),
                   Evidence("`topopt buckling <run> --modes 2` on a finished run", "metrics.buckling is rewritten; history is equal by sha256",
                            "6.4 s (the check itself), peak 530 MB", HOOK, "P7-1/l3/cli_buck"), *_E2_BK_CHECK)),
    Card("buckling.density", "yaml", "method", 3,
         "Which density the check uses: `printed` (density 0.5 or more is solid, the rest is void) or `gray` (the density of the last snapshot).", "measured",
         values="printed | gray; default printed. The row of the optimizer uses the gray density of the optimizer. Both densities can have a low "
                "mode that is not a buckling mode of a solid member (see `lowest_mode_energy_share_rho_lt_half`).",
         cost=RESUME_COST, call=RESUME_CALL,
         evidence=(_resume_row("buckling density gray (+ check true, modes 2)", "metrics.buckling.density is gray; the check ran", "16.9 s",
                               "buckling_density_gray"), *_E2_BK_DENSITY)),
    Card("buckling.modes", "yaml", "method", 3,
         "How many of the lowest load factors the check gives for each case. The row of the optimizer also takes the modes into its KS sum.", "measured",
         values="integer >= 1; default 6. With `buckling.min_factor` set, the row reads this key in the passes: a change is then a new run (layer 1). "
                "The resume fact of a finished run reads the layer of the card only, so it does not name this key.",
         cost=RESUME_COST, call=RESUME_CALL,
         evidence=(_resume_row("buckling modes 3 with the check", "metrics.buckling.cases[].load_factors has 3 values", "12.5 s", "buckling_check"), *_E2_BK_MODES),
         price_note="The number of modes is a cost key, not a design key: 2, 6 and 12 modes in the row gave a byte-equal final.stl (E2/BKm2, E2/BK0, E2/BKm12), and the lowest factors of the check are the same numbers "
                    "for 2, 6 and 12 modes. 2 modes did not save time in the row (loop 184.5 s against 171.5 s and 181.4 s), 12 modes cost 2.7 to 2.9 times (490.4 s). In the check alone (`topopt buckling`) "
                    "2 modes took 0.7 times and 12 modes 1.7 times the wall of 6 modes (6.8 s, 16.6 s, 9.8 s to 10.5 s). The value of `modes` of the row is also the default of the check."),
    Card("buckling.max_gb", "yaml", "method", 3,
         "The budgets of the check: the memory in GB (1e9 bytes) with `max_gb`, and the time of one eigen solve of one case with `max_seconds`. A check "
         "whose memory estimate is above the budget is not started.", "measured",
         values="`max_gb`: number > 0; default the RAM of this machine. `max_seconds`: number > 0; default 120. With `buckling.min_factor` set, the row "
                "reads both in the passes: a change is then a new run (layer 1). The resume fact of a finished run reads the layer of the card only, so it does "
                "not name these keys.",
         cost=RESUME_COST, call=RESUME_CALL,
         symptoms=(Symptom("buckling check skipped, no load factor", "`metrics.buckling.status` (`skipped`) with `facts` and `options`", facts=("metrics.buckling.status", "metrics.buckling.facts", "metrics.buckling.options")),),
         evidence=(_resume_row("max_gb 0.01 (+ check true)", "metrics.buckling.status skipped; the fact says the estimate is 0.6 GB; the options name "
                               "a larger voxel size or `topopt buckling --max-gb`", "10.6 s", "buckling_max_gb_small"), *_E2_BK_BUDGET),
         also=("buckling.max_seconds",)),
    Card("buckling.min_factor", "yaml", "requirement", 1,
         "Turns the buckling row of the optimizer on: the lowest buckling load factor of each load case stays at or above this multiplier on the load "
         "case. A met row has a lowest factor of at least the value (a row just met can be up to 4.5 % above it for 6 modes and q 40).", "measured",
         cost="a run with the row on the hook at 2 mm and 40 passes took 171 to 229 s of loop (177 to 235 s wall), 10.4 to 13.8 times the 16.55 s of the same run with no row, also when the row is far from active "
              "(100: 11.4 times); 12 modes took 498 s. The row is evaluated at every pass while the last lowest factor is at most 1.5 times `min_factor` (CONTRACTS.md section 39), so a request far under the factor of the part "
              "is cheap only when the factor is above 1.5 times the request (for B0 a request under about 82; derived from the rule, not run). `topopt check` says that the time estimate does not include the row. "
              "The seconds follow the host load (load average 22 to 36 on 22 cores); the ratio is the more portable figure. Add P2 (5.1 to 7.4 s) and the check (6.2 to 13.0 s) to a try.",
         values="number > 0; no default. The row needs a direct solver (cholesky or pardiso); with amg or gpu-mg, or with `settings.robust: eroded` or `wall`, "
                "the run is refused at the start with the options. A row costs one eigen solve for each case at an evaluated pass. The user's safety "
                "demand is a requirement; tighten it with a report and its price. The key has a second reader: `verify_run(gate=true)` reads it as the lowest allowed "
                "load factor of the tet model and turns the buckling check of `verify` on (card `verify_run:gate`); a YAML with the key gets the row of the optimizer and the limit of the gate together.",
         symptoms=(Symptom("buckles, bends sideways, slender, wobbly", "`metrics.buckling.lowest_load_factor`; the history rows `buckling:<case>` "
                           "(`g`, `ks_offset`)", facts=("metrics.buckling.lowest_load_factor", "history.constraints[].ks_offset")),),
         evidence=_E2_BK_MIN,
         price_note="The optimizer sees no price: hang x0.987 to x0.999, pull_off x0.997 to x1.006 and mass x0.998 to x1.000 for 100, 150, 150.15 and 250. The time is the price (see Cost). P2 shows a fall of the `pull_off` "
                    "stiffness of about 8 % at 150 (x0.917, twice), 150.15 (x0.922) and 250 (x0.915), and in two runs with another `ks_q` (x0.937 at 20, x0.921 at 80), against x1.033 at 100. The 250 run is one where the "
                    "row was slack at the last pass (g -0.0713), so the fall is not tied to a binding row at the end. The P2 band is +-3.5 %, so the 8 % is outside it, it repeated in 4 different designs "
                    "(x0.915 to x0.922), and the cause was not isolated: read 'no stiffness price' as a finding of the optimizer only. The factor of the printed design is higher than the "
                    "request when the row is slack (100 gave 154.1, 250 gave 293.6) and about 1 % under it when the row binds (150 gave 148.3). The end state is a jittery function of the request: 150.15 in place of 150 "
                    "(+0.1 %) moved the gray `pull_off` factor at the last pass from 149.5 to 161.7 (+8.2 %) and 0.72 % of the voxels. A request under the factor of B0 (100 against 123.1) still moved the design "
                    "(flip 6.2 %, printed factor 154.1): the row has a gradient while its value is under the limit. Two runs of the same YAML (BK0, BK1) gave a byte-equal final.stl."),
    Card("buckling.ks_q", "yaml", "method", 1,
         "The KS sharpness of the buckling row. A higher value follows the lowest mode closer.", "measured",
         values="number > 0; default 40. Without `buckling.min_factor` it has no effect (`check_problem` says so as a fact). On the hook at 2 mm, 20 and 80 gave the same design as 40 "
                "within 0.4 % of the voxels: no effect found.",
         evidence=_E2_BK_Q),
)

_PRINT = (
    Card("print", "yaml", "requirement", 3,
         "The printer and the print rules: build direction, bed, printer type, overhang angle, and limits for the print check. The optimizer ignores "
         "this block; the post step and `print_check` read it.", "code",
         values="mapping; keys on the cards `print.*`. The build direction, the bed, the printer type and the angle act in the post step (layer 3, "
                "measured on their cards). The limit keys are read by `print_check` (layer 4); the post step does not read them.",
         cost=RESUME_COST, call=RESUME_CALL),
    Card("print.build_direction", "yaml", "requirement", 3,
         "The up direction of the print (away from the plate), the printer type and the bed. They set the overhang area, the support estimate, "
         "the plate contact and the bed fit.", "measured",
         values="`build_direction`: x | y | z with an optional sign (+x, -z), or a vector of 3 numbers; default +z. `printer_type`: fdm | sla | lpbf | sls; "
                "default fdm; sls needs no support. `bed_mm`: one number or [x, y], mm; default 220 x 220. The print check does not turn the part: "
                "`check_problem.print.options` gives the numbers for the six axis directions.",
         cost=RESUME_COST, call=RESUME_CALL,
         symptoms=(Symptom("needs supports, overhangs, will not print", "`metrics.print.configured` (`overhang_area_mm2`, `support_volume_mm3`) and "
                           "`metrics.print.options[]` (the numbers of each axis direction)", facts=("metrics.print.configured.overhang_area_mm2", "metrics.print.configured.support_volume_mm3", "metrics.print.options[].overhang_area_mm2")),
                   Symptom("does not fit the bed, too big", "`metrics.print.configured.bed_fit`", facts=("metrics.print.configured.bed_fit",))),
         evidence=(_resume_row("build_direction +x, printer_type sla, bed 100", "metrics.print.* changed (overhang, support, bed fit); no other file", "6.6 s",
                               "print_build_direction"),),
         also=("print.printer_type", "print.bed_mm")),
    Card("print.printer_model", "yaml", "requirement", 3,
         "The printer, by name. The tool finds it in the printer catalog and fills `print.bed_mm` and `print.printer_type` when the YAML does not give them "
         "(their source reads `profile:<id>`), and the print report compares the part height with the build height of the printer. A key that the YAML gives wins.", "code",
         values="text: an id of the catalog (`bambu-lab-p1s`), or a name or alias (`Bambu P1S`, `x1c`); case, punctuation and vendor words do not matter. A text that is not one "
                "printer is an error that names the candidates or the suggestions; `find_printer(name)` shows the match, the candidates and the numbers that the profile gives. "
                "A printer that the catalog does not have comes from the user file `topopt-printers.yaml` in the work folder (`save_printer` writes it, with the source of each value). "
                "A number that the catalog does not state is null, with a note, and the key stays at its default. No default: without it the bed is the assumed 220 x 220 mm "
                "and the height is not limited.",
         cost=RESUME_COST, call=RESUME_CALL),
    Card("print.self_supporting_angle_deg", "yaml", "requirement", 3,
         "The angle from the plate under which a face counts as an overhang. It wins over the default of the printer type.", "measured",
         values="number, 0 to 90 degrees; default the value of the printer type (45 for fdm, sla, lpbf and sls).",
         cost=RESUME_COST, call=RESUME_CALL,
         symptoms=(Symptom("overhangs, needs supports", "`metrics.print.configured.overhang_area_mm2` and `support_volume_mm3`", facts=("metrics.print.configured.overhang_area_mm2", "metrics.print.configured.support_volume_mm3")),),
         evidence=(_resume_row("self_supporting_angle_deg 45 -> 60", "metrics.print.configured.overhang_area_mm2 2413.7 -> 2826.7, support_volume_mm3 "
                               "52599.8 -> 59574.7; no other file changed", "14.3 s", "print_angle"),)),
    Card("print.min_wall_mm", "yaml", "requirement", 4,
         "The thinnest wall or member that the printer makes. The print check measures the share of the surface area whose chord is under it, and "
         "fails when the share is above `min_wall_area_pct`.", "measured",
         values="number > 0, mm; no default. With no limit the state is `unknown`. A limit under 3 x the resolution gives `unknown` (the reason "
                "gives the resolution that would pass). `settings.min_member_mm` is a request to the optimizer and measures something else (CONTRACTS.md "
                "section 32). `settings.robust: wall` takes its wall w from this key and aims the optimizer at it (card `settings.robust`). Call: `print_check(print_settings={\"min_wall_mm\": ...})` or `topopt printcheck <run> --set min_wall_mm=...`.",
         cost="seconds: one `printcheck` of the hook example took 5.5 s (thickness 3.8 s)",
         call="`print_check` with `print_settings`, or `topopt printcheck <run or STL> --set KEY=VALUE`",
         symptoms=(Symptom("too thin, will not print, breaks when handled", "`print_check` `checks[wall_thickness]` (`state`, `value.area_below_limit_pct`) "
                           "and `metrics.print.thinnest_member` (`thinnest_member_mm`, `area_below_mm2`)", facts=("print_check.checks[].value.area_below_limit_pct", "metrics.print.thinnest_member.thinnest_member_mm", "metrics.print.thinnest_member.area_below_mm2")),),
         evidence=(Evidence("print.min_wall_mm in the YAML, resume of the finished run", "nothing in metrics.json changes: the post step does not read the limit keys",
                            "6.4 s", HOOK, "P7-1/l3/print_min_wall"),
                   Evidence("`topopt printcheck` with max_overhang_area_mm2 100 and min_wall_mm 2.5", "overhang fail, wall_thickness fail, support_volume and "
                            "plate_contact unknown (no limit)", "5.5 s", HOOK, "P7-1/l3/base"))),
    Card("print.min_wall_area_pct", "yaml", "requirement", 4,
         "The share of the surface area (0 to 100) that may have a chord under `min_wall_mm` before the wall check fails.", "code",
         values="number 0 to 100; default 0.5 (assumed: the report labels it so). Read by `print_check`.",
         call="`print_check` with `print_settings`, or `topopt printcheck <run or STL> --set KEY=VALUE`"),
    Card("print.max_bodies", "yaml", "requirement", 4,
         "The number of separate bodies that is allowed in the print.", "code",
         values="integer >= 1; default 1 (assumed). Read by `print_check`. A body inside a void counts as a body.",
         call="`print_check` with `print_settings`, or `topopt printcheck <run or STL> --set KEY=VALUE`",
         symptoms=(Symptom("falls apart, loose pieces, floating bits", "`print_check` `checks[bodies]` (`value.count`) and `metrics.mesh.islands_removed`", facts=("print_check.checks[].value.count", "metrics.mesh.islands_removed")),)),
    Card("print.max_closed_void_mm3", "yaml", "requirement", 4,
         "The largest closed void (a cavity with no opening) that is allowed; 0 allows none.", "code",
         values="number >= 0, mm3; no default. Without it a void gives `unknown` and the reason says what the process does with it. Read by `print_check`.",
         call="`print_check` with `print_settings`, or `topopt printcheck <run or STL> --set KEY=VALUE`"),
    Card("print.max_overhang_area_mm2", "yaml", "requirement", 4,
         "The limits of the print check for the overhang area, the support estimate (an upper bound) and the area that rests on the plate.", "measured",
         values="`max_overhang_area_mm2`: number >= 0, mm2, no default; fail above it. `max_support_volume_mm3`: number >= 0, mm3, no default; fail above it. "
                "`min_plate_contact_mm2`: number >= 0, mm2, no default; fail below it. With no limit the state is `unknown`, and `pass` when the number is zero. "
                "Read by `print_check`, not by the post step.",
         call="`print_check` with `print_settings`, or `topopt printcheck <run or STL> --set KEY=VALUE`",
         symptoms=(Symptom("needs supports, overhangs", "`print_check` `checks[overhang]` and `metrics.print.configured.overhang_area_mm2`", facts=("print_check.checks[].state", "metrics.print.configured.overhang_area_mm2")),
                   Symptom("does not stand on the plate, small footprint", "`print_check` `checks[plate_contact]` (`value.area_mm2`)", facts=("print_check.checks[].value.area_mm2",))),
         evidence=(Evidence("print.max_overhang_area_mm2 100 in the YAML, resume of the finished run", "nothing in metrics.json changes (only timing)",
                            "12.8 s", HOOK, "P7-1/l3/print_limit_overhang"),),
         also=("print.max_support_volume_mm3", "print.min_plate_contact_mm2")),
)


# --- interfaces, springs, joints ----------------------------------------------------------------------------------------------
_INTERFACES = (
    Card("interfaces", "yaml", "requirement", 1,
         "Exact shapes that the post step unites with the final mesh or cuts out of it, after the smoothing: a round bore with its exact radius, a "
         "boss, a tube, a flat face, or a closed STL solid. With machining stock the step makes two meshes: `final.stl` (as printed) and `nominal.stl`.",
         "measured",
         values="list of mappings; the order is the order of the boolean steps. The optimizer sees the shapes in the voxel labels: a cut shape is a "
                "keep-out region, a union shape is a solid body with no force. A union shape that does not touch the part makes the matrix singular. A "
                "keep-out region wins over a union shape.",
         symptoms=(Symptom("bore not round, hole too small, face not flat, rough hole", "`metrics.interfaces.items[]` (`as_printed.measured.deviation_mm`, "
                           "`within_tolerance`, `coverage`) and `metrics.interfaces.applied`", facts=("metrics.interfaces.items[].as_printed.measured.<measure>.deviation_mm", "metrics.interfaces.items[].as_printed.measured.<measure>.within_tolerance", "metrics.interfaces.items[].as_printed.measured.<measure>.coverage", "metrics.interfaces.applied")),
                   Symptom("shapes are not exact, the interface step failed, holes as rough as the grid", "`metrics.interfaces.error` (null when the exact step ran; else the error text) with `metrics.interfaces.applied`: "
                           "the step failed and the mesh keeps the shape of the mesh method, with no exact interface; the run log has the warning 'interface step failed'",
                           facts=("metrics.interfaces.error", "metrics.interfaces.applied"), when="`metrics.interfaces.error` is not null (`applied` is then false)"),),
         evidence=(_resume_row("interfaces[].stock_mm changed (bracket example), resume of the finished run", "labels.bin, final.stl, nominal.stl, preview.bin "
                               "and stress.npz are rewritten; the optimizer does not run again", "6.4 s", "if_stock_changed", BRACKET),
                   _resume_row("interfaces[].radius_mm 3.0 -> 3.5 (bore), resume of the finished run", "the same files are rewritten; the design is NOT "
                               "adapted to the new bore", "6.7 s", "if_bore_radius_changed", BRACKET))),
    Card("interfaces[].kind", "yaml", "requirement", 1,
         "The kind of the shape and its name: bore (cut a cylinder), boss (unite a cylinder), tube (unite a ring and cut its hole), face (cut away "
         "everything on one side of a plane), stl (unite or cut a closed STL solid).", "code",
         values="`kind`: bore | boss | tube | face | stl; required. `name`: unique text; required. A bore, boss or tube needs `point`, `axis`, "
                "`radius_mm`, `length_mm` (a tube also `inner_radius_mm`); a face needs `point` and `normal`; an stl needs `file` and `op`.",
         also=("interfaces[].name",)),
    Card("interfaces[].point", "yaml", "requirement", 1,
         "Where the shape is: `point` is the centre of the first end face of a cylinder, or a point on the finished plane of a face. `axis` is the "
         "direction along the cylinder, `normal` the direction of the face cut, `length_mm` the length of the cylinder.", "code",
         values="`point`, `axis`, `normal`: lists of 3 numbers (mm); `axis` and `normal` are made unit vectors, and the zero vector is refused. "
                "`length_mm`: number > 0, mm. A change moves the shape: the design needs to adapt, so it is a new run (layer 1).",
         also=("interfaces[].axis", "interfaces[].normal", "interfaces[].length_mm")),
    Card("interfaces[].radius_mm", "yaml", "requirement", 1,
         "The finished radius of a bore or a boss, or the outer radius of a tube. `inner_radius_mm` is the radius of the hole of a tube.", "measured",
         values="number > 0, mm. For a tube `inner_radius_mm` is less than `radius_mm`. A change of the radius in a resume of a finished run re-applies the "
                "exact shape, but the design is not adapted to it: make a new run.",
         symptoms=(Symptom("bore too small, hole does not fit the pin", "`metrics.interfaces.items[].as_printed.measured.measured_mm` against `expected_mm`", facts=("metrics.interfaces.items[].as_printed.measured.<measure>.measured_mm", "metrics.interfaces.items[].as_printed.measured.<measure>.expected_mm")),),
         evidence=(_resume_row("radius_mm 3.0 -> 3.5 (bore), resume of the finished run", "the mesh and labels change; the design is NOT adapted to the "
                               "new bore", "6.7 s", "if_bore_radius_changed", BRACKET),),
         also=("interfaces[].inner_radius_mm",)),
    Card("interfaces[].stock_mm", "yaml", "requirement", 3,
         "The machining stock: material that the as-printed shape keeps on the surface that a machine cuts later. A bore is smaller by the stock, a boss "
         "larger, a face moves by the stock.", "measured",
         values="number >= 0, mm; default 0; less than `radius_mm` of a bore and less than `inner_radius_mm` of a tube. An stl entry has no stock. "
                "Caution: in a resume of a finished run the labels of the stock voxels change but the optimizer does not run, so the design is not adapted to "
                "the new stock; for a small change this is the cheap check, for a large change make a new run. The reply key `resume` of the MCP tool "
                "`run_topopt` compares the YAML keys and the region files only (the server builds no problem before the run), so a change that moves only the "
                "voxel labels is not in the reply; the run log line names it.",
         cost=RESUME_COST, call=RESUME_CALL,
         symptoms=(Symptom("no stock to machine, too little material at the bore", "`metrics.interfaces.stock_volume_mm3` and `items[].as_printed.measured`", facts=("metrics.interfaces.stock_volume_mm3", "metrics.interfaces.items[].as_printed.measured")),),
         evidence=(_resume_row("stock_mm changed (bracket example)", "labels.bin, final.stl, nominal.stl, preview.bin and stress.npz are rewritten; the optimizer "
                               "does not run again", "6.4 s", "if_stock_changed", BRACKET),)),
    Card("interfaces[].full_surface", "yaml", "requirement", 1,
         "The surface of the interface has to be complete: a bond socket, a bearing seat, a press fit. After the run the tool measures the share "
         "of the full area that the finished part has (check D of CONTRACTS.md section 23) and writes a test and a warning. The design does not "
         "change, and the run does not stop for it.", "code",
         values="true or false; default false. On a bore, a boss, a tube and a face with `radius_mm`; true on an stl entry or on a face without "
                "`radius_mm` is a loader error that gives the options. The test is met when the full coverage is 0.98 or more "
                "(`interfaces.FULL_COVERAGE_MIN`). A YAML without the key gives the same run folder as before.",
         symptoms=(Symptom("socket cut open, bearing seat not complete, press fit has a gap",
                           "`metrics.verdict.interface_tests[].met` and `full_coverage` of the interface; the warning text is in "
                           "`metrics.interfaces.warnings` (`run_result` key `interface_warnings`)",
                           facts=("metrics.verdict.interface_tests[].met", "metrics.verdict.interface_tests[].full_coverage",
                                  "metrics.interfaces.warnings"),
                           when="`met` is false: the finished part has less than 0.98 of the full area"),)),
    Card("interfaces[].file", "yaml", "requirement", 1,
         "The closed STL solid of an `stl` interface, and the boolean step: `union` adds the solid, `cut` takes it out. No stock.", "code",
         values="`file`: STL path, relative to the YAML file. `op`: union | cut.", also=("interfaces[].op",)),
)

_SPRINGS = (
    Card("springs", "yaml", "requirement", 1,
         "Spring supports: a load region that is held by springs in place of a fixed body. A spring adds stiffness to the part at the region (a spring in "
         "series with the pad); it adds no degree of freedom.", "code",
         values="list of entries, one for each region. A YAML with neither a fixed region nor a spring (or joint) is refused: six rigid modes are free. "
                "The voxels of a spring region are solid load voxels. A spring acts on all exposed faces of the region, side faces too: shape the region so "
                "that only the wanted face is free (`run.json` `springs.regions[].faces_by_direction` shows it). `topopt verify` models the spring too (`k int N^T N dA` on the tet facets "
                "that look away from the design, `verify/supports_p2.py`); its result has `supports` with the force of each spring on the part for each case.",
         symptoms=(Symptom("mount too soft, wobbles, floppy support", "`metrics.stiffness_cases[].N_per_mm` (it includes the spring, so it is the mounted part) and "
                           "`check_problem.extended.springs`", facts=("metrics.stiffness_cases[].N_per_mm", "check_problem.extended.springs")),
                   Symptom("spring too weak, nearly free", "`check_problem.warnings` (a spring below 1e-6 * E * h) and `metrics.reactions.cases[].residual_rel`", facts=("check_problem.warnings", "metrics.reactions.cases[].residual_rel")),
                   Symptom("mount force, spring force, load on the mount", "`topopt verify`: the force and the moment of each spring on the part for each case, and the force of all supports in the balance",
                           facts=("verify.supports.cases.<case>.springs.<region>.force_N", "verify.cases.<case>.equilibrium.support_force_N", "verify.cases.<case>.equilibrium.support_moment_Nmm")))),
    Card("springs[].region", "yaml", "requirement", 1,
         "The load region that the spring holds. The region may have a force, or none.", "code",
         values="a name of `regions.load`; required; one entry for each region."),
    Card("springs[].stiffness", "yaml", "requirement", 1,
         "The stiffness of the spring: the total stiffness of the region (`stiffness`, N/mm for each axis), or the stiffness of each unit of face area "
         "(`stiffness_per_area`, N/mm3).", "code",
         values="exactly one of the two. `stiffness`: [kx, ky, kz], numbers >= 0, N/mm, not all zero (a zero on one axis is valid). `stiffness_per_area`: one "
                "number or [kx, ky, kz], N/mm3. No rotational stiffness. A spring on one axis only leaves the modes along the other axes free: the run stops "
                "with their names. The value is a fact about the user's mount: a change alters the physical problem, so it needs the user's consent. "
                "`topopt verify` reads the same key: `stiffness` is divided by the area of the tet facets of the region, `stiffness_per_area` is used as it is.",
         symptoms=(Symptom("verify stiffness differs from the run, mount stiffer or softer in verify", "`topopt verify`: the bonded area of the tet model over the bonded area of the voxel run "
                           "(`stiffness_per_area` gives a total stiffness that moves with it)", facts=("verify.supports.springs[].area_over_voxel_area",)),),
         also=("springs[].stiffness_per_area",)),
)

_JOINTS = (
    Card("joints", "yaml", "requirement", 1,
         "Joints: an Euler-Bernoulli beam from a load region to a fixed far point, for a part that hangs on a rod, a pin or a tube that is not in the voxel grid. "
         "The beam sees the region as one rigid body.", "code",
         values="list of entries, one for each region. Each joint adds 6 degrees of freedom (a bordered matrix; about +24 to 38 % of the factor time on the hook "
                "example, CONTRACTS.md section 38). A part held by joints only is valid. `topopt verify` models the joint (bond to a rigid body and the same beam, `verify/supports_p2.py`; its result has `supports` with the 6 numbers of each bond for each case). The beam has no shear: `length_over_depth` "
                "in `run.json` `joints.joints[]` says how short it is.",
         symptoms=(Symptom("rod too soft, mount flexes, hangs loose", "`metrics.stiffness_cases[].N_per_mm` (it includes the beam) and `check_problem.extended.joints`", facts=("metrics.stiffness_cases[].N_per_mm", "check_problem.extended.joints")),
                   Symptom("beam nearly free", "`check_problem.warnings` (a smallest beam stiffness below 1e-6) and `check_problem.extended.joints.joints[].beam_min_translation_over_Eh`", facts=("check_problem.warnings", "check_problem.extended.joints.joints[].beam_min_translation_over_Eh")),
                   Symptom("bond force, mount force, adhesive load, how hard the part pulls on the rod", "`topopt verify`: the force and the moment of each bond on the part for each case, and the force of all supports in the balance",
                           facts=("verify.supports.cases.<case>.joints.<region>.bond_force_N", "verify.supports.cases.<case>.joints.<region>.bond_moment_Nmm",
                                  "verify.cases.<case>.equilibrium.support_force_N", "verify.cases.<case>.equilibrium.support_moment_Nmm")),
                   Symptom("verify says the part is loose, a body is not held", "`topopt verify`: the facets of a joint or spring region on each body of the mesh (a body with none and no fixed node stops the check)",
                           facts=("verify.mesh.bodies.bodies[].support_facets",)),
                   Symptom("verify bond area differs from the run", "`topopt verify`: the bonded area of the tet model over the bonded area of the voxel run",
                           facts=("verify.supports.joints[].area_over_voxel_area",)))),
    Card("joints[].region", "yaml", "requirement", 1,
         "The load region whose exposed faces are bonded to the beam, and the far point where the beam is fixed.", "code",
         values="`region`: a name of `regions.load`; required; one joint for each region. `to`: [x, y, z], mm; required; the beam runs from the reference point "
                "to it, and it has a length above 0.", also=("joints[].to",)),
    Card("joints[].section", "yaml", "requirement", 1,
         "The section of the beam: round (`d`), tube (`od`, `id`), rect (`b`, `h`) or props (`A`, `Iy`, `Iz`, `J`).", "code",
         values="mapping with `kind` and its keys, mm and mm2/mm4, all above 0; `id` is less than `od`. A thicker section makes a stiffer mount. For rect, `b` runs "
                "along the section y axis and `h` along z.",
         also=("joints[].section.kind", "joints[].section.d", "joints[].section.od", "joints[].section.id", "joints[].section.b", "joints[].section.h",
               "joints[].section.A", "joints[].section.Iy", "joints[].section.Iz", "joints[].section.J")),
    Card("joints[].origin", "yaml", "requirement", 1,
         "The reference point of the rigid body (the start of the beam), and the direction of the section y axis.", "code",
         values="`origin`: [x, y, z], mm; default the area centroid of the bonded faces. `orient`: [x, y, z]; default the global axis least parallel to the beam; "
                "it matters for a section that is not round, and a vector parallel to the beam is an error.", also=("joints[].orient",)),
    Card("joints[].bond_stiffness_per_area", "yaml", "method", 1,
         "The stiffness of the bond between the region and the rigid body, per unit of face area. A finite bond puts some strain energy in the bond.", "code",
         values="number > 0, N/mm3; default c * E / h with c = 30 (`joints.BOND_C`), E of the run material, h the voxel size. `bond_energy_share` in "
                "`metrics.reactions` is the share of the strain energy in the bond; the stiffer the bond, the closer to a rigid body (and the harder for amg). `topopt verify` uses the same key "
                "with h the voxel size of the run, and the beam has no geometric stiffness in its buckling check.",
         symptoms=(Symptom("joint too soft because of the bond", "`metrics.reactions.cases[].joints.<region>.bond_energy_share` and `run.json` `joints.joints[].bond_share_max`; in `topopt verify` the same shares "
                           "for the tet model", facts=("metrics.reactions.cases[].joints.<region>.bond_energy_share", "run.joints.joints[].bond_share_max",
                                                       "verify.supports.cases.<case>.joints.<region>.bond_energy_share", "verify.supports.joints[].bond_share_max")),)),
    Card("joints[].E", "yaml", "requirement", 1,
         "The material of the beam. The default is the material of the run.", "code",
         values="`E`: number > 0, MPa. `nu`: -1 < nu < 0.5. Not the material of the part.", also=("joints[].nu",)),
)


# --- acceptance (the ratchet) and bore_checks (verify) ------------------------------------------------------------------------
RATCHET_CALL = "`ratchet_run` (then `ratchet_status`; `ratchet_approve` when `owner_gate` is true)"
VERIFY_CALL = "`verify_run` (CLI `topopt verify <run>`)"

_ACCEPTANCE = (
    Card("acceptance", "yaml", "requirement", 4,
         "The rules by which the ratchet keeps or rejects a candidate edit of a finished part (a mark of the Refine page, a part that another tool made). "
         "The ratchet judges each candidate with the print check and with `verify`, and keeps it only if every rule of the case passes.", "code",
         values="mapping with `rules` (required) and optional `mesh_size_mm`, `percentile`, `owner_gate`. A YAML without the block loads as before and the ratchet "
                "refuses to judge: no rules. A rule that is not listed is off. A rule that is on and has no input is `no_input`, and it counts as failed.",
         call=RATCHET_CALL,
         symptoms=(Symptom("edit made it worse, not allowed, keep or reject", "`ratchet_status` records: `decision` (ACCEPT, REJECT, BASELINE) and `checks[]` "
                           "(`rule`, `state`, `value`, `limit`, `limit_source`)", facts=("ratchet.decision", "ratchet.checks[].state")),)),
    Card("acceptance.rules", "yaml", "requirement", 4,
         "The list of rules, one entry for each rule: `{rule: <name>, ...options}`. The name picks the test; the other keys are the options of that rule.",
         "code",
         values="rule names: watertight | bodies | loose_island | stress | stiffness | mass | print_gate | symmetry | displacement | alignment | buckling. A rule "
                "listed twice, an unknown rule and an empty list are errors. `watertight`, `bodies` and `print_gate` have no options; `displacement`, `alignment` and "
                "`buckling` read the tet rows of `limits` and the tet buckling factor of `verify` (a listed rule `buckling` makes the judge run the buckling check); a listed rule with no input "
                "(no rows in the YAML, a skipped check) is `no_input`. A number in a rule has a source: `source` for the numbers of the rule, or "
                "`<key>_source` for one number; without a source the record prints 'source not recorded'.",
         call=RATCHET_CALL, also=("acceptance.rules[].rule", "acceptance.rules[].source")),
    Card("acceptance.mesh_size_mm", "yaml", "method", 4,
         "The mesh size of `verify` for the baseline and for every candidate, and the volume percentile of the stress.", "code",
         values="`mesh_size_mm`: number > 0, mm; default the size that `verify` picks for the baseline. `percentile`: default 99.9. The peak of the stress moves "
                "with the mesh, so one size is used for the whole ratchet; a root with another size is refused.",
         call=RATCHET_CALL, also=("acceptance.percentile",)),
    Card("acceptance.owner_gate", "yaml", "requirement", 4,
         "true: the runner stops before it promotes a kept candidate and waits for `ratchet_approve`.", "code",
         values="true | false; default false.", call=RATCHET_CALL),
    Card("acceptance.rules[].max_mm3", "yaml", "requirement", 4,
         "The rule `loose_island`: the volume of the loose bodies (all bodies but the largest) is at most `max_mm3`; with `relative_to_parent: true` the "
         "increase over the parent is.", "code",
         values="`max_mm3`: number >= 0, mm3. `relative_to_parent`: true | false. Needs the print check.",
         call=RATCHET_CALL, symptoms=(Symptom("loose bits, floating islands, falls apart", "`ratchet_status` `facts.loose_mm3` and `checks[loose_island]`", facts=("ratchet.facts.loose_mm3", "ratchet.checks[].state")),),
         also=("acceptance.rules[].max_mm3_source", "acceptance.rules[].relative_to_parent")),
    Card("acceptance.rules[].margin", "yaml", "requirement", 4,
         "The rule `stress`: the p99.9 (or `percentile`) stress by volume of each case against the limit of the case times `margin`, or against the parent.", "code",
         values="`margin`: 0 < m <= 1. `against_parent`: absolute (default) | not_worse | cap_or_parent. `tolerance_pct`: number >= 0, the move to the worse side of "
                "the parent value that is allowed: a rise of the stress (and of the rows of the rules `displacement` and `alignment`), a fall of the factor for the rule `buckling` (no default: with none, "
                "every move fails). `against_parent`, `tolerance_pct` and `cases` are keys of those three rules too. `cases`: the load cases to judge. The loader refuses `tolerance_pct` with `absolute`, and "
                "`margin` with `not_worse`. Needs `verify`. Tighten `margin` freely; loosen it only with the consent of the user.",
         call=RATCHET_CALL,
         symptoms=(Symptom("edit raises the stress, too much stress", "`ratchet_status` `facts.cases.<case>.p_by_volume_MPa` and `checks[stress]`", facts=("ratchet.facts.cases.<case>.p_by_volume_MPa", "ratchet.checks[].state")),),
         also=("acceptance.rules[].margin_source", "acceptance.rules[].against_parent", "acceptance.rules[].tolerance_pct",
               "acceptance.rules[].tolerance_pct_source", "acceptance.rules[].cases")),
    Card("acceptance.rules[].max_drop_pct", "yaml", "requirement", 4,
         "The rule `stiffness`: the stiffness of each case drops by at most this percent against the parent.", "code",
         values="number >= 0, percent. Needs `verify`.", call=RATCHET_CALL,
         symptoms=(Symptom("edit makes it softer, loses stiffness", "`ratchet_status` `facts.cases.<case>.stiffness_N_per_mm` and `checks[stiffness]` (`delta_pct`)", facts=("ratchet.facts.cases.<case>.stiffness_N_per_mm", "ratchet.checks[].delta_pct")),),
         also=("acceptance.rules[].max_drop_pct_source",)),
    Card("acceptance.rules[].must_fall", "yaml", "requirement", 4,
         "The rule `mass`: the mass falls (`must_fall: true`), or rises by at most `max_rise_pct` percent against the parent. Exactly one of the two.", "code",
         values="`must_fall`: true. `max_rise_pct`: number >= 0, percent. The mass is the print-check volume times `material.density`.", call=RATCHET_CALL,
         symptoms=(Symptom("edit adds weight, too heavy", "`ratchet_status` `facts.mass_g` and `checks[mass]` (`delta_pct`)", facts=("ratchet.facts.mass_g", "ratchet.checks[].delta_pct")),),
         also=("acceptance.rules[].max_rise_pct", "acceptance.rules[].max_rise_pct_source")),
    Card("acceptance.rules[].min_factor", "yaml", "requirement", 4,
         "The rule `buckling`: the lowest tet load factor of each case (`load_factors[0]`, the measured value; the judge runs `verify` with the buckling check) is at least `min_factor`, "
         "or does not fall below the parent value. The factor without the rigid-body motions of the part (`rigid_body.load_factors`) is a fact in the verify result, and the rule does not read it. "
         "The rules `displacement` and `alignment` read the rows of `limits` on the tets in the same way.",
         "code", values="`min_factor`: number > 0 (not with `against_parent: not_worse`). `against_parent`: absolute (default) | not_worse | cap_or_parent. "
         "`tolerance_pct`: number >= 0 (not with `absolute`), the fall of the factor below the parent value that is allowed. `cases`: the load cases to judge. "
         "The floor is the user's safety demand: tighten `min_factor` freely, with a report; loosen it only with the consent of the user. The rules `displacement` and `alignment` "
         "take their limit from the rows of `limits` (no key of their own).", call=RATCHET_CALL,
         symptoms=(Symptom("edit lowers the buckling load factor, the edit made it buckle, more slender after the edit, the edit shifted the seats",
                           "`ratchet_status` `facts.cases.<case>.buckling_lowest_factor` and `checks[buckling]` (`delta_pct`; for the rows of the rules `displacement` and `alignment` the check `value` and `limit`)",
                           facts=("ratchet.facts.cases.<case>.buckling_lowest_factor", "ratchet.checks[].state")),),
         also=("acceptance.rules[].min_factor_source",)),
    Card("acceptance.rules[].max_mm", "yaml", "requirement", 4,
         "The rule `symmetry`: the mirror error of the part is at most `max_mm` (the largest distance from a mirror image of a vertex to the surface). The YAML "
         "needs a top-level `mirror`.", "code",
         values="`max_mm`: number >= 0, mm. A symmetric part gives about the size of the triangles, not zero.", call=RATCHET_CALL,
         also=("acceptance.rules[].max_mm_source",)),
)

_BORE_CHECKS = (
    Card("bore_checks", "yaml", "requirement", 4,
         "Turns on the bore checks of `verify`: the stress of each load case at the wall of each bore, the Goodman fatigue factor with the mean stress of a press fit. "
         "They give facts, not verdicts. The optimizer ignores the block.", "code",
         values="mapping; needs `bores` and `goodman` or `press_fit`. Absent: the checks are off. `verify_run` has no argument for them: the YAML decides "
                "(CLI `--bore-checks`, `--no-bore-checks`). A physical number needs its source text (`<key>_source`, or one `source` for the entry). `n` below 1 is "
                "a fact, not a limit of topopt.",
         call=VERIFY_CALL,
         symptoms=(Symptom("bore fails, fatigue crack at the hole, hole tears out", "`verify.json`, each mesh level: `goodman.lowest_n`, `goodman.rows[]` (`n_goodman`, "
                           "`peak_over_yield`)", facts=("verify.goodman.lowest_n", "verify.goodman.rows[].n_goodman", "verify.goodman.rows[].peak_over_yield")),)),
    Card("bore_checks.bores", "yaml", "requirement", 4,
         "The bores to check: one entry for each, with a name and the geometry of its contact range.", "code",
         values="list of one entry or more; the names are unique. An entry has either `interface: <name>` (takes the geometry from the `bore` entry of `interfaces`; the "
                "report says the C2 `length_mm` is then the contact length) or the four keys `point`, `axis`, `radius_mm`, `length_mm`.",
         call=VERIFY_CALL, also=("bore_checks.bores[].name", "bore_checks.bores[].interface")),
    Card("bore_checks.bores[].point", "yaml", "requirement", 4,
         "The geometry of the contact range of a bore: `point` is the centre of its first end face, `axis` the direction along the bore, `radius_mm` the finished "
         "radius, `length_mm` the length of the range.", "code",
         values="`point`, `axis`: lists of 3 numbers, mm (`axis` is made a unit vector). `radius_mm`, `length_mm`: numbers > 0, mm; `length_mm` is above "
                "2 x `end_margin_mm`. `radius_mm_source`, `length_mm_source`: text.",
         call=VERIFY_CALL, symptoms=(Symptom("no bore facets found, bore not checked", "`verify.json`, each mesh level: `bore_facts[]` (`status`, `funnel`, `facts`, `options`)", facts=("verify.bore_facts[].status", "verify.bore_facts[].funnel", "verify.bore_facts[].facts"),
                          when="`status` is `empty` (no wall facet, or no sample point) or `error` (an exception: `stage` and `message` say where); `ok` is checked"),),
         also=("bore_checks.bores[].axis", "bore_checks.bores[].radius_mm", "bore_checks.bores[].radius_mm_source", "bore_checks.bores[].length_mm",
               "bore_checks.bores[].length_mm_source")),
    Card("bore_checks.bores[].diametral_interference_mm", "yaml", "requirement", 4,
         "The diametral interference of the press fit of this bore (the radial interference is half of it). Only with `press_fit`.", "code",
         values="number, mm, with a source. A bore needs it when `press_fit` is present, and it is an error without `press_fit`.", call=VERIFY_CALL,
         also=("bore_checks.bores[].diametral_interference_mm_source",)),
    Card("bore_checks.bores[].mean_stress_mpa", "yaml", "requirement", 4,
         "The mean stress `sm` of the Goodman rows of this bore when there is no `press_fit` block. Without it `sm` is 0.", "code",
         values="number, MPa, with a source. Not allowed together with `press_fit`. `source` covers the numbers of the entry that have no own source.",
         call=VERIFY_CALL, also=("bore_checks.bores[].mean_stress_mpa_source", "bore_checks.bores[].source")),
    Card("bore_checks.radius_tolerance_mm", "yaml", "method", 4,
         "The knobs that pick the wall facets and the sample points of a bore: the tolerance on the vertex radius, the margin from the ends, the normal "
         "limit and the width of the sample band. They are not physical data.", "code",
         values="`radius_tolerance_mm` > 0 (default 0.35); `end_margin_mm` >= 0 (default 0.35); `normal_limit` in (0, 1] (default 0.35); `sample_band_mm` > 0 "
                "(default 1.2). The report gives the value used and its origin. A source is optional. A radius tolerance of 0.2 mm removed the first facet row at the "
                "mouth (21 % of the area) on the two-joint part (2.5 mm tets).",
         call=VERIFY_CALL, symptoms=(Symptom("bore value moves with the mesh, scatter", "`verify.json`, each mesh level: `bore_facts[]` (`near_miss`, `sample_set`)", facts=("verify.bore_facts[].near_miss", "verify.bore_facts[].sample_set")),),
         also=("bore_checks.radius_tolerance_mm_source", "bore_checks.end_margin_mm", "bore_checks.end_margin_mm_source", "bore_checks.normal_limit",
               "bore_checks.normal_limit_source", "bore_checks.sample_band_mm", "bore_checks.sample_band_mm_source")),
    Card("bore_checks.goodman", "yaml", "requirement", 4,
         "The Goodman fatigue check: n = 1 / (sa / Se + sm / Su) with the von Mises statistic `sa` of each load case at the bore wall (taken fully reversed) and the "
         "mean stress `sm`.", "code",
         values="mapping with `materials` (one entry or more; the label is free text). `Se` is the endurance limit as the user gives it, fully corrected: the tool "
                "applies no factor and has no preset.",
         call=VERIFY_CALL, also=("bore_checks.goodman.materials",)),
    Card("bore_checks.goodman.materials.<label>.su_mpa", "yaml", "requirement", 4,
         "The strengths of one material for the Goodman rows: ultimate (`su_mpa`), yield (`sy_mpa`), endurance (`se_mpa`).", "code",
         values="numbers, MPa; all three are required, each with a source (`<key>_source`, or one `source` for the entry). The tool does not check the content of a source.",
         call=VERIFY_CALL,
         also=("bore_checks.goodman.materials.<label>.su_mpa_source", "bore_checks.goodman.materials.<label>.sy_mpa",
               "bore_checks.goodman.materials.<label>.sy_mpa_source", "bore_checks.goodman.materials.<label>.se_mpa",
               "bore_checks.goodman.materials.<label>.se_mpa_source", "bore_checks.goodman.materials.<label>.source")),
    Card("bore_checks.press_fit", "yaml", "requirement", 4,
         "The press fit: a cup that is a linear two-sided Winkler spring on the wall facets of each bore. It gives the mean stress `sm` of the Goodman rows.", "code",
         values="mapping with `cups` (one entry or more). The contact pressure is p = k (d_r - u_r); there is no contact iteration. Each bore needs "
                "`diametral_interference_mm`.",
         call=VERIFY_CALL, also=("bore_checks.press_fit.cups",)),
    Card("bore_checks.press_fit.cups[].k_mpa_per_mm", "yaml", "requirement", 4,
         "One cup variant by the spring stiffness for each unit of area (`k_mpa_per_mm`, above 0).", "code",
         values="number > 0, MPa/mm, with a source. A cup has this form, or the form `e_mpa`, `nu`, `wall_mm`; not both, not neither. `name`: unique text. "
                "YAML 1.1 reads `1e6` as text; the loader accepts it.",
         call=VERIFY_CALL, also=("bore_checks.press_fit.cups[].k_mpa_per_mm_source", "bore_checks.press_fit.cups[].name")),
    Card("bore_checks.press_fit.cups[].e_mpa", "yaml", "requirement", 4,
         "One cup variant by the cup wall: Young's modulus, Poisson's ratio and wall thickness.", "code",
         values="`e_mpa` > 0, MPa; `nu` in [0, 0.5); `wall_mm` > 0 and at most the bore radius; all three are needed, each with a source (`<key>_source`, or one "
                "`source` for the entry).",
         call=VERIFY_CALL,
         also=("bore_checks.press_fit.cups[].e_mpa_source", "bore_checks.press_fit.cups[].nu", "bore_checks.press_fit.cups[].nu_source",
               "bore_checks.press_fit.cups[].wall_mm", "bore_checks.press_fit.cups[].wall_mm_source", "bore_checks.press_fit.cups[].source")),
)


# --- engine tools -------------------------------------------------------------------------------------------------------------
# The key is the MCP tool, or the CLI subcommand where a tool has no MCP form (`buckling`, `refine`), or `<tool>:<argument>` for an argument that
# is a lever of its own. The run ids `P7-1/l3/<name>` are runs of PR-1 on the hook example; the times are wall times on a loaded host.
_TOOLS = (
    Card("run_topopt", "tool", "method", 1,
         "Starts a topology optimization run in the background and returns at once. A new `out_dir` makes a new run from the YAML as it is on disk; "
         "every layer 1 key acts through it.", "measured",
         values="`yaml_path`, `out_dir` (default `runs/live`), `override_lock`, `resume`, `allow_changed_inputs`, `continue_from` with `more_iterations` (layer 2), "
                "`fine_from` or `fine_pass` with `fine_voxel`, `fine_passes`, `fine_tol` (layer 2), `stop_rules` (true | false | none: the YAML decides), `proceed_without_brief`. A YAML with a case file and no current brief gives `state: awaiting_user` with the facts and the options. "
                "Only one run, verify or ratchet job goes at a time. Watch it with `run_status`; read it with `run_result`.",
         cost="the run time of the problem: `check_problem.time_estimate.total_s`; the hook example (40 passes, 2.0 mm, pardiso) took 27.0 s",
         call="`run_topopt(yaml_path, out_dir)` after `check_problem`; CLI `topopt run <yaml> --out <dir>`",
         symptoms=(Symptom("takes too long", "the estimate before the run", facts=("check_problem.time_estimate.total_s",)),),
         evidence=(Evidence("base run of the hook example (hook_short.yaml)", "a finished run with result/ and metrics.json", "27.0 s", HOOK, "P7-1/l3/base"),)),
    Card("run_topopt:resume", "tool", "method", 2,
         "Continues a stopped or killed run from its last checkpoint. The history and the snapshots after the checkpoint are dropped. A run that was "
         "killed gives the same result as a run that was not killed (CONTRACTS.md: bit-equal with one thread and a direct solver).", "code",
         values="`resume=true` with the same `out_dir` and the same YAML. If the YAML or a region file changed since the checkpoint, the reply is an error "
                "with the changes in `resume.facts` and nothing is written: see `run_topopt:allow_changed_inputs`. A run that `stop_run` stopped continues "
                "with no override. A resume of a FINISHED run repeats its post step only; `run_topopt:continue_from` adds passes to a finished run in a new folder.",
         cost="the passes that are left",
         call="`run_topopt(yaml_path, out_dir, resume=true)`; CLI `topopt run <yaml> --out <dir> --resume`",
         symptoms=(Symptom("run stopped, killed, interrupted", "the state of the run", facts=("run_status",)),)),
    Card("run_topopt:continue_from", "tool", "method", 2,
         "Adds passes to a run in a NEW run folder. The folder named here is the parent: it is only read and does not change. The new run starts from the last checkpoint of the "
         "parent (pass K) and has K + `more_iterations` passes; `settings.iterations` of the YAML is not used.", "measured",
         values="goes with `more_iterations` and not with `resume`; `out_dir` is another folder (not the parent, not inside it): else an error with options and nothing is written. "
                "A parent that finished (its checkpoint is complete) is replayed: pass K runs again with its update step, then the passes go on; with one thread and a direct solver the "
                "result equals that of one run of K + N passes (CONTRACTS.md 24.3; more threads: close, not equal). A parent that did not finish (stopped, killed, error) goes on as a fork "
                "from its checkpoint; a parent that still runs is not refused (the run reads its checkpoint once). The YAML and the region files are read again and compared with the "
                "checkpoint: a changed key is a refusal with the change and options, or goes on with `allow_changed_inputs=true`, and the new passes then use the new values. A key of "
                "the passes takes effect here, which a resume of a finished run (layer 3) does not do (measured: volume_fraction 0.25). Only `settings.iterations` is no change. "
                "The beta schedule goes by the absolute pass number (the hook stays at its last stage, beta 8 from pass 34). A stop rule that ended the parent is judged again at "
                "pass K (`stop_rules=false` turns the rules off). The run time counts from the start of the parent. The new folder gets a copy of the history and of the snapshots up "
                "to pass K, about K times the size of the grid in bytes; the fact before the copy gives the bytes. The reply has `continue` (parent, iteration, iterations, "
                "pass_runs_again, step_settings, facts).",
         cost="measured on the finished hook run (voxel 2 mm, 40 passes): 14.3 to 25.7 s wall for 10 or 20 more passes in 4 runs (host load 29 to 42 on 22 cores), 34.0 s for a new run "
              "of the 40 passes in the same hour. Of the time of the new run (11.9 to 23.1 s) the passes took 6.6 to 15.5 s (pass K again included; 0.5 to 0.9 s a pass here, 0.40 s "
              "on a quiet host) and the post step 5.3 to 7.5 s. The passes grow with the grid (3.09 s a pass at voxel 1.2 mm, E1); the rest does not",
         call="`run_topopt(yaml_path, out_dir=NEW, continue_from=PARENT, more_iterations=N)`; CLI `topopt run <yaml> --out NEW --continue-from PARENT --more-iterations N` "
              "(add `allow_changed_inputs=true` / `--allow-changed-inputs` for a changed key)",
         symptoms=(Symptom("not converged, needs more passes, still grey, ended too early, did it finish",
                           "`metrics.verdict.state` (`iteration_limit`), `metrics.verdict.converged` and `metrics.verdict.measures.objective_change.met`",
                           facts=("metrics.verdict.state", "metrics.verdict.converged", "metrics.verdict.measures.objective_change.met"),
                           when="`metrics.verdict.converged` is false"),
                   Symptom("keep the old run, try the next step in a new folder, branch from this run",
                           "the wish of the user; `list_runs` gives the run folders and `run_status` the `checkpoint` of the parent", facts=("list_runs", "run_status")),
                   Symptom("the part did not change after the resume, changed a setting of the passes on a finished run",
                           "the reply of `run_topopt` with `resume=true` and `allow_changed_inputs=true` has `resume.pass_keys` (the changed keys that act in the passes)",
                           facts=("run_topopt",))),
         evidence=(Evidence("the finished base run B0 continued with `more_iterations` 10 and no changed input: pass 40 runs again, then 41 to 50",
                            "verdict iteration_limit -> converged (objective change 1.7e-3 -> 2.2e-5, threshold 1e-4); gray 0.0250 -> 0.0247; flip 0.07 %; the last snapshot and "
                            "final.stl of the two runs have equal sha256",
                            "wall 14.3 s and 21.4 s (host load 29 and 41); optimizer hang x1.001, pull_off x1.001, mass x1.000; inside the noise (all three optimizer values)",
                            HOOK, "P7-1/r3/C10, P7-1/r3/C10b"),
                   Evidence("the same continuation with `settings.volume_fraction` 0.2 -> 0.25 in the YAML and `allow_changed_inputs`, 10 passes",
                            "the part changes: gray 0.0269; flip 25.3 % (15.8 % against E1/VF25, a new run of 40 passes); verdict iteration_limit (objective change 7.1e-3); without the "
                            "override the call is refused with the change and the options, and nothing is written",
                            "wall 15.1 s; optimizer hang x1.224, pull_off x1.191, mass x1.145 (E1/VF25: x1.242, x1.207, x1.147 in a new run)", HOOK, "P7-1/r3/CV25, E1/VF25"),
                   Evidence("control: the same YAML as a new run (layer 1), 40 passes", "equal to B0 byte for byte (snapshot it_0040.bin and final.stl)",
                            "wall 34.0 s (host load 37 to 41)", HOOK, "P7-1/r3/S40")),
         price_note="A wall time here is the fixed part of a run plus the passes, and the load of the host moved the same input by a factor of about 1.5 (14.3 and 21.4 s). Compare with "
                    "a new run of the same problem: `check_problem` gives its estimate (`time_estimate.total_s`). The price of the design is the ratio of the results against the parent "
                    "(`compare_runs`)."),
    Card("run_topopt:more_iterations", "tool", "method", 2,
         "With `continue_from`: the number of passes to add. The new run has K + N passes, K being the pass of the checkpoint of the parent.", "measured",
         values="a whole number of 1 or more, with no upper limit (a wrong value is an error with options, and nothing is written). `settings.iterations` of the YAML is not used. It goes "
                "with `continue_from` and not with `resume`. 10 and 20 passes on the finished hook run gave the same part within the noise and the verdict `converged` (below).",
         cost="the passes you add (0.5 to 0.9 s each on the hook example here) and the fixed part of a run (see `run_topopt:continue_from`); N = 10 and N = 20 took 14.3 to 21.4 s and 19.8 to 25.7 s",
         call="`run_topopt(yaml_path, out_dir=NEW, continue_from=PARENT, more_iterations=N)`; CLI `--more-iterations N`",
         symptoms=(Symptom("how many more passes, run it longer", "`metrics.verdict.measures.objective_change.value` against its threshold, and `metrics.verdict.state`",
                           facts=("metrics.verdict.measures.objective_change.value", "metrics.verdict.measures.objective_change.threshold", "metrics.verdict.state")),),
         evidence=(Evidence("N = 20 against N = 10 on the same parent (the finished base run B0)",
                            "verdict converged in both (objective change 1.6e-5 against 2.2e-5); gray 0.0245 against 0.0247; flip against the parent 0.13 % against 0.07 %; the stiffness differs "
                            "by 0.05 % or less",
                            "wall 25.7 s and 19.8 s against 14.3 s and 21.4 s: the 10 passes more do not show above the host load; optimizer hang x1.000, pull_off x1.001, mass x1.000; "
                            "inside the noise (all three optimizer values)", HOOK, "P7-1/r3/C20, P7-1/r3/C20b, P7-1/r3/C10"),)),
    Card("run_topopt:fine_from", "tool", "method", 2,
         "Starts a NEW run on a finer voxel grid from the design of a finished parent run (a fine pass). The parent folder is only read. The new run maps the design of the parent to the new "
         "grid by place, fits the volume to the target with one uniform shift, and starts with new optimizer memory. The beta schedule goes on from the pass of the parent "
         "(`shift_schedule`). The run ends when its loss settles (`fine_tol`) or at the maximum (`fine_passes`).", "measured",
         values="`fine_from` is the parent `out_dir` and `out_dir` is another folder (not the parent, not inside it). The same YAML works: the voxel of the fine pass is the voxel that the "
                "part needs (`check_problem.fine_pass.voxel_mm`) and overrides `settings.voxel_size`; `fine_voxel` sets it by hand. The fine pass takes its design from the checkpoint of "
                "the parent (the field `x`) and falls back to the snapshot density with a fact. A parent that did not finish is refused with the options. The YAML and the region files "
                "are read again and compared with the parent: a changed key other than the voxel, the passes and the beta schedule is a refusal with the change, or goes on with "
                "`allow_changed_inputs=true`. The default filter radius (1.5 voxels) follows the voxel of the fine pass, with a fact; a radius that the YAML gives (`settings.filter_radius`, `settings.min_member_mm`) stays the same in mm. Goes with `fine_voxel`, `fine_passes` and "
                "`fine_tol`, and not with `resume`, `continue_from`, `more_iterations` or `fine_pass`. A pass at the fine voxel costs about as much as a late pass of a run from zero "
                "(measured on the arm). The reply has `fine` (parent, iteration, voxel_mm, max_passes, tolerance, facts) and `run.json` has `fine_from`. Equal voxel: use `continue_from`.",
         cost="measured on the arm: 31 snapshots at 0.7 mm, 1918 s wall on a loaded host, after the parent run of 953 s (together 2871 s against 3462 s for a run from zero); on the hook: 91 s "
              "after 52 s for the parent, against 184 s. The price is the run time of the fine pass plus that of the parent, which comes first; a wall time moved with the host load by a factor of 2 or more, so compare the pass counts at the fine voxel. `topopt check --fine-from PARENT` and `check_problem.fine_pass.estimate` give an estimate of the time and the memory",
         call="`run_topopt(yaml_path, out_dir=NEW, fine_from=PARENT)`; CLI `topopt run <yaml> --out NEW --fine-from PARENT`; plan only: `topopt check <yaml> --fine-from PARENT`",
         symptoms=(Symptom("thin feature, thin wall, too coarse, voxel too big for the part, member thinner than the voxel, finer result from the finished run",
                           "`check_problem.fine_pass.recommend` is true: the voxel of the part (`voxel_mm`) is smaller than the voxel of the run, and `limiting` names the feature that sets it; "
                           "`run_result.fine_pass` and `run.part_voxel` give the same after the run",
                           facts=("check_problem.fine_pass.recommend", "check_problem.fine_pass.voxel_mm", "check_problem.fine_pass.limiting", "run.part_voxel.recommend", "run_result"),
                           when="`recommend` is true"),
                   Symptom("what did the fine pass start from, which parent, which voxel",
                           "`run.fine_from` (run_dir, iteration, voxel_mm, parent_voxel_mm, schedule_offset, source) in the fine run", facts=("run.fine_from.run_dir", "run.fine_from.voxel_mm",
                           "run.fine_from.source"), when="the run is a fine pass: the key `fine_from` exists")),
         evidence=(Evidence("fine pass A from the 1.0 mm parent to 0.7 mm: start design `x`, beta 8 held, 30 passes",
                            "weighted 596.71 at pass 30 against 596.95 for the cold run of 80 passes (-0.04 %); 617, 606, 602 and 599 at pass 5, 10, 15 and 20; the three cases at "
                            "pass 30 are within 2.6 % of the cold run; grayness 0.2223 against 0.2235; volume 0.30742 against 0.30739",
                            "31 snapshots at 0.7 mm against 81; wall 1918 s against 3462 s (host load 25 to 47); with the parent run (953 s) 2871 s; peak memory 2.17 GiB against 2.11 GiB",
                            "drone arm", "FP/arm-A, FP/arm-cold, FP/arm-parent"),
                   Evidence("fine pass B, the same start with beta 4 until pass 14, then 8",
                            "worse than A at the same pass (795 at pass 5 against 617 for A), jumps at the change of beta, and ends at 600.78, 0.7 % above A at pass 30: the continuation "
                            "rule holds beta where the parent ended",
                            "wall 1864 s, 31 snapshots", "drone arm", "FP/arm-B"),
                   Evidence("fine pass from the hook parent (2.0 mm, 40 passes) to 1.2 mm, beta 8 held, 16 passes",
                            "weighted 1.3900 against 1.3612 for the cold run (+2.1 %); the case `pull_off` differs most (+4.2 %); with beta 4 first 1.3802 (+1.4 %); grayness 0.0155 "
                            "against 0.0164",
                            "17 snapshots at 1.2 mm against 41; wall 91 s and CPU 626 s against 184 s and 915 s (host load differs); the parent run (52 s) comes before",
                            "hook fine pass", "FP/hook-fine-b8, FP/hook-fine-b4, FP/hook-cold-12, FP/hook-parent"))),
    Card("run_topopt:fine_pass", "tool", "method", 2,
         "The one-command form of the fine pass: the normal run goes into `out_dir`, then the fine pass goes into the sibling folder `<out_dir>-fine`. The reply names both folders. If the "
         "voxel of the part is not smaller than the voxel of the run, the call writes a fact and ends after the first run, unless `fine_voxel` is given and finer than the run: it wins.", "code",
         values="`fine_pass=true`; goes with `fine_voxel`, `fine_passes`, `fine_tol`, and not with `fine_from`, `resume`, `continue_from` or `more_iterations`. `settings.iterations` of the "
                "YAML is the pass count of the first run. One background process does both runs, so the server counts as busy until the fine pass ends; `run_status` names the run that goes "
                "on (`fine_run`). A fine pass starts only with `fine_pass` or `fine_from`: a plain run does not start one.",
         cost="the run time of the first run plus a fine pass; `check_problem.fine_pass.estimate` gives the time and the memory of the fine pass, as estimates",
         call="`run_topopt(yaml_path, out_dir, fine_pass=true)`; CLI `topopt run <yaml> --out <dir> --fine-pass`",
         symptoms=(Symptom("run it fine in one step, refine automatically, both runs", "`check_problem.fine_pass.recommend` is true", facts=("check_problem.fine_pass.recommend",
                           "check_problem.fine_pass.voxel_mm"), when="`recommend` is true"),)),
    Card("run_topopt:fine_voxel", "tool", "method", 2,
         "With `fine_from` or `fine_pass`: the voxel of the fine pass in mm. Default: the voxel that the part needs.", "code",
         values="number > 0, mm. The default is the voxel that the part needs: the largest voxel with 2.8 voxels across the thinnest fixed, load, solid or keep-out region and 3 voxels "
                "across `min_member_mm`, rounded down to 0.05 mm. The thickness is twice the radius of the largest sphere inside the region. The fine pass tests that voxel on its own grid and goes down in steps of 0.05 mm if a region has too few voxels across there (the log says it). `check_problem.fine_pass.voxel_mm` is the measured value, one number, without that test. A value equal to the voxel of the parent is refused (the options name `continue_from`). A coarser value than the "
                "parent is allowed and gives a fact. The voxel guards (`max_grid_voxels`, `max_active_voxels`) and the memory estimate apply to the fine grid.",
         cost="the voxel count grows with the cube of the ratio of the voxels: `check_problem.fine_pass.estimate` gives the time and the memory for the default voxel; `topopt check --fine-from "
              "PARENT --fine-voxel H` gives them for H",
         call="`run_topopt(yaml_path, out_dir, fine_from=PARENT, fine_voxel=H)`; CLI `--fine-voxel H`",
         symptoms=(Symptom("which voxel for the fine pass, how fine, 2.8 voxels across", "`check_problem.fine_pass.voxel_mm` and the feature that sets it (`limiting`)",
                           facts=("check_problem.fine_pass.voxel_mm", "check_problem.fine_pass.limiting")),)),
    Card("run_topopt:fine_passes", "tool", "method", 2,
         "With `fine_from` or `fine_pass`: the MAXIMUM number of passes of the fine pass. The fine pass ends earlier when its loss settles (`fine_tol`).", "measured",
         values="a whole number of 1 or more. Default: the pass count of the parent run. `settings.iterations` of the YAML stays the pass count of the first run. The log says when the "
                "maximum ended the fine pass before the loss settled (state `iteration_limit`); a fine pass that settled has state `stopped_by_rule` and the rule `settled`. The beta schedule "
                "goes on from the pass of the parent (`shift_schedule`).",
         cost="the passes at the fine voxel (31 snapshots took 1918 s on the arm at 0.7 mm; one pass is about as expensive as a late pass of a run from zero)",
         call="`run_topopt(yaml_path, out_dir, fine_from=PARENT, fine_passes=N)`; CLI `--fine-passes N`",
         symptoms=(Symptom("how many passes, did the fine pass settle, ended too early, ended at the maximum",
                           "`metrics.verdict.state` (`stopped_by_rule` with the rule `settled`, or `iteration_limit` at the maximum) and `metrics.verdict.measures.objective_change.value` "
                           "against the tolerance `run.fine_from.tolerance`",
                           facts=("run.fine_from.max_passes", "run.fine_from.tolerance"), when="the run is a fine pass: the key `fine_from` exists"),),
         evidence=(Evidence("fine pass A on the arm (parent 1.0 mm to 0.7 mm), weighted compliance by pass",
                            "726, 617, 606, 602, 599 and 596.7 at pass 0, 5, 10, 15, 20 and 30; the run from zero has 8531, 1758, 1203, 1109, 1078 and 1019 at the same pass numbers, and 596.95 "
                            "at pass 80",
                            "30 passes at 0.7 mm; 1918 s wall on a loaded host", "drone arm", "FP/arm-A, FP/arm-cold"),
                   Evidence("hook fine pass at 1.2 mm", "within 1.4 % to 2.1 % of the cold run in 16 passes", "16 passes: 91 s to 134 s wall", "hook fine pass",
                            "FP/hook-fine-b8, FP/hook-fine-b4, FP/hook-cold-12"))),
    Card("run_topopt:fine_tol", "tool", "method", 2,
         "With `fine_from` or `fine_pass`: the tolerance of the stop rule `settled`. The fine pass ends when the largest relative change of the objective between two passes in the last "
         "`window` passes, at one beta, is at or below it and the constraints are met.", "measured",
         values="number > 0. Without it the YAML key `settings.fine_pass.objective_change` gives the value, else the default `1e-3` (`config.FINE_OBJECTIVE_CHANGE`); "
                "`run.fine_from.tolerance_from` says which (`option`, `yaml` or `default`). The window is `settings.stop_rules.window` (default 5). The rule is on in a fine pass unless `stop_rules=false` "
                "turns every stop rule off (then the fine pass runs to its maximum). It does not use the grayness test or the last beta stage test, so it can fire where `converged` cannot (the arm has grayness 0.22). The log and "
                "the verdict text say which test fired. A smaller value runs more passes. The tolerance is in `run.json` (`fine_from.tolerance`).",
         cost="the passes after the loss is flat: a replay of the history of the arm run shows 9 passes less at 1e-3 than at 5e-4",
         call="`run_topopt(yaml_path, out_dir, fine_from=PARENT, fine_tol=T)`; CLI `--fine-tol T`",
         symptoms=(Symptom("stopped too early, stopped too late, runs on after it is flat, the loss is still moving", "`metrics.verdict.measures.objective_change.value` against `run.fine_from.tolerance`, "
                           "and `metrics.verdict.state`", facts=("run.fine_from.tolerance",), when="the run is a fine pass: the key `fine_from` exists"),),
         evidence=_FINE_TOL_EVIDENCE),
    Card("run_topopt:allow_changed_inputs", "tool", "both", 3,
         "With `resume`: goes on although the YAML or a region file changed since the checkpoint. On a FINISHED run no pass is left, so only the post step runs "
         "again: the post step reads the changed key and rewrites the result. This is the cheap route for the keys of layer 3. With `continue_from` the new passes "
         "use the changed value: that is the route for a changed key of the passes (`run_topopt:continue_from`).", "measured",
         kind_note="follows the key that changed: its card says if it is a requirement or a method setting",
         values="The post step reads: `settings.mesh_method`, `settings.envelope_fill`, `print.build_direction`, `print.printer_type`, `print.bed_mm`, `print.self_supporting_angle_deg`, "
                "`regions.not_printed`, `settings.part_only_min_film_mm`, `material.density` (the mass numbers only), the thresholds of `settings.stop_rules` (the verdict), "
                "`buckling.*` of the check, `name`, `assumptions`. Hazards, all measured: a key that acts in the passes (`settings.penal`, `settings.volume_fraction`, a region) is "
                "ACCEPTED and the part does NOT change, but `run.json` records the new value; a changed `interfaces` entry or a removed keep-out file changes the "
                "labels and the mesh, and the optimizer does not adapt the design; `settings.iterations` is refused (the log says: this version does not make a "
                "finished run longer). The log has one `resume:` line that names each changed key.",
         cost=RESUME_COST, call=RESUME_CALL,
         symptoms=(Symptom("did my change run again, is the result new", "the result lists each resume of the run; for a post-step-only resume `history.jsonl` and "
                           "the checkpoint stay equal", facts=("metrics.verdict.resume_points",)),),
         evidence=(_resume_row("no change (control)", "the numbers of metrics.json stay equal; metrics.json, run.json, verdict.json and refine.html are rewritten", "6.1 s", "control_nochange"),
                   _resume_row("settings.penal 3.5 (negative control)", "the part does NOT change; only the stress display (result/stress.npz) changes",
                               "8.5 s", "control_penal_negative"),
                   _resume_row("settings.volume_fraction 0.25 (negative control)", "the part does NOT change; run.json records 0.25 for a design made at 0.2",
                               "7.1 s", "control_volume_negative"),
                   _resume_row("material.density 1.24 -> 1.5", "metrics.mass_final_g 25.72 -> 31.11 (mass numbers only)", "7.6 s", "density"))),
    Card("stop_run", "tool", "method", 2,
         "Asks a running run to stop at the end of its current pass. It writes `job.stop` and sends no signal. The run saves its verdict (`stopped_by_request`), has "
         "no post step and no `result/`; `run_topopt` with `resume=true` continues it with no override.", "code",
         values="`run_dir` (default: the run that this server started). A long pass (a big grid, a buckling row) delays the stop by up to one pass. A solve "
                "that hangs does not read the request: the reply has the pid.",
         cost="up to one pass", call="`stop_run(run_dir)`; CLI `topopt stop <run>`",
         symptoms=(Symptom("stop it, going wrong, wrong settings", "the state of the run", facts=("run_status",)),)),
    Card("check_problem", "tool", "method", 1,
         "Loads and voxelizes a problem YAML and solves nothing. It gives facts, estimates and warnings: grid, voxel counts, the volume basis, masses, time and memory "
         "estimates for each solver, lost force, connectivity, solid pull, print facts, mirror and perimeter facts.", "measured",
         values="`yaml_path`. The reply has `derived` (forces, volume basis, `min_feature_note`, `filter`), `time_estimate`, `memory_estimate`, `backends`, `warnings`, `fine_pass` (the voxel that the part needs; `recommend` is true when the run voxel is too coarse; see `run_topopt:fine_from`). "
                "A warning is a fact with options and blocks nothing, except the one that starts with 'settings.solver is' (the solver cannot run here): `run_topopt` then stops with the same message. A warning that starts with 'fact: settings.iterations' says that the passes end before the last beta stage (card `settings.beta_schedule`).",
         cost="seconds; the hook example took 1.5 s", call="`check_problem(yaml_path)`; CLI `topopt check <yaml> [--json]`",
         symptoms=(Symptom("will it take long, will it fit in memory", "the estimates before the run; the rows give the estimate and the wall time of runs of the bracket and the ladle (E3): the wall time was 1.2 times the estimate "
                           "on the bracket and 2.2 and 2.5 times on the ladle, on a shared host", facts=("check_problem.time_estimate.total_s",
                           "check_problem.memory_estimate.bytes")),
                   Symptom("force lost, load sits on the support", "the lost force before the run", facts=("check_problem.lost_force.max_lost_percent",))),
         evidence=(Evidence("check of the hook example (hook_short.yaml)", "the reply with grid, estimates and warnings", "1.5 s (`topopt check --json`)", HOOK,
                            "P7-1/l3/check_hook"),
                   Evidence("the estimate of `topopt check` for examples/ladle/ladle.yaml (3.25 mm, 80 passes, three load cases), against the run LAD-B0 of the same file",
                            "the check estimated 160.5 s in all (`derived.time_estimate_s` of run.json; 1.67 s for each pass); the run took 400.7 s of wall time, of which 381.2 s in the loop (4.71 s for each record). The report "
                            "reads this as an estimate for another model of the machine, not as an error of the check",
                            "wall time 2.5 times the estimate; host load average (1, 5 and 15 min) 21.51 23.92 27.16 at the start and 27.71 29.62 29.03 at the end", LAD_EX, "E3/LAD-B0"),
                   Evidence("the same file with `voxel_size` 4.0 mm (the coarse copy), against the run LAD-D0",
                            "the check estimated 55.1 s in all (0.37 s for each pass); the run took 118.8 s of wall time (1.34 s for each record)",
                            "wall time 2.2 times the estimate; host load average (1, 5 and 15 min) 24.92 27.73 28.41 at the start and 21.33 25.92 27.69 at the end", LAD4, "E3/LAD-D0"),
                   Evidence("the estimate of `topopt check` for examples/interfaces/bracket.yaml (40 passes), against the run BR-B0",
                            "grey base (BR-B0 ends at beta 2, gray 0.129); the check estimated 29.5 s in all (`derived.time_estimate_s` of run.json); the run took 36.0 s of wall time, of which 28.6 s in the loop; the byte-equal repeat BR-B1 took 25.3 s",
                            "wall time 1.2 times the estimate (BR-B0); host load average (1, 5 and 15 min) 21.90 24.52 27.58 at the start and 22.17 24.38 27.43 at the end", BR_E3, "E3/BR-B0, E3/BR-B1"),)),
    Card("build_regions", "tool", "method", 1,
         "Builds region STL files and a problem YAML from a part spec given as data (design space, fixed regions, loads, keep-outs, solid bodies, springs, joints). "
         "It returns `check` and `warnings` of the new YAML.", "code",
         values="`spec` (a dict or text), `name`. Use it when the user describes a part in words. A spec key that the YAML has is a lever of its own: see the cards of "
                "the keys.",
         cost="not measured", call="`build_regions(spec, name)`, then state the assumptions, then `run_topopt`"),
    Card("marks_to_hints", "tool", "method", 1,
         "Turns the marks of the Refine page into facts, options and a proposed YAML in a NEW folder. It edits no mesh. A remove mark becomes a keep-out STL, an add mark a "
         "forced-solid region, a smooth mark only facts and options. The proposal needs a new run.", "measured",
         values="`marks` (file, JSON text or object), `run_dir`, `out_dir`, `options` (remove: keep_out | skip; add: solid | skip; radius_scale; protect_regions; "
                "clip_to_design; per_mark), `overwrite`, `detail`. A mark is a point of emphasis, not an absolute. A mark inside an earlier hint has status `no_effect`.",
         cost="seconds: one mark on the hook example took 9.6 s; then `run_topopt` with the proposed YAML and a new `out_dir`, then `compare_runs`",
         call="`marks_to_hints(marks, run_dir)`, then `run_topopt`",
         evidence=(Evidence("the one mark of `cleanup_list` on the hook example", "status no_effect; a proposed YAML in a new folder", "9.6 s", HOOK,
                            "P7-1/l3/hints_hook"),)),
    Card("apply_marks", "tool", "method", 4,
         "Edits the finished part at the marks of the Refine page, in a NEW folder (`edited.stl`, `apply_marks.json`). Remove is a cut, add is a closing, smooth is a "
         "fairing. It runs no FE solve, so the stiffness and the stress of the edited part are not known, and it gives no verdict.", "measured",
         values="`marks`, `run_dir`, `out_dir`, `options` (radius_scale, pitch_mm, frozen_keep_mm, parent_check report | enforce, smooth_mode, remove_mode, mirror_plane, "
                "skip, per_mark), `overwrite`, `allow_stale`. Fixed, load and keep-out regions are frozen: an edit near them is clipped, and the clip is a fact. A step "
                "that gives a part that is not closed, or that splits it, is rolled back. Judge the result with `verify_run` or `ratchet_run`. The run folder and final.stl are not changed; "
                "`edited.stl` and `apply_marks.json` go to `out_dir`. A flat triangle is split, so the edited mesh has more triangles (24494 against 20586 on B0). A `remove` mark took 81 % of the volume that "
                "the detector listed. E2 did not run `verify` on an `edited.stl`; the ratchet job with the stiffness rule did judge this same edit (see `ratchet_run`).",
         cost="seconds: one mark on the hook example took 2.5 s; 2.4 to 3.1 s for a smooth, an add and a remove mark in E2 (host load average 12 to 43)", call="`apply_marks(marks, run_dir)`",
         symptoms=(Symptom("extra film here, remove this bit, smooth this, bump", "the candidates of the cleanup list", facts=("cleanup_list",)),),
         evidence=(Evidence("the one mark of `cleanup_list` on the hook example", "status applied; edited.stl in a new folder", "2.5 s", HOOK, "P7-1/l3/apply_marks_hook"), *_E2_APPLY)),
    Card("cleanup_list", "tool", "method", 4,
         "Lists the geometric cleanup candidates of a finished part as a marks file in a NEW folder (`marks.json`, `cleanup.json`): loose bodies, dead-end spurs, thin "
         "bumps, uneven struts, flat webs. It edits nothing and judges nothing.", "measured",
         values="`run_dir`, `out_dir`, `options` (pitch_mm 0.5, max_candidates 10, kinds body | spur | bump | member, order, the limits of the detectors), "
                "`estimate_only`. When a loose body exists, the file holds the bodies only. Each mark has the detector, the operator, the thresholds and the sizes; the "
                "applicability says how well the detectors fit the part. The marks go to `apply_marks` or `ratchet_run`. The detectors work on their own grid (`pitch_mm`, 0.5 mm), not on the grid of the run: "
                "2,165,566 voxels and a model of 733 MB for the 2 mm hook. With `verify/fields.npz` in the run folder each candidate has a stress ratio (0.402 on the hook); without it the candidate has no stress "
                "facts and a note says to run `topopt verify` first.",
         cost="seconds: the hook example took 3.0 s (the docstring: 2 s on the short hook, 8 s on the ladle); E2 on a copy of the short hook: 3.1 s with `verify/`, 3.7 s without, 0.63 s with `estimate_only` "
              "(host load average 12 to 43)", call="`cleanup_list(run_dir)`",
         symptoms=(Symptom("icicles, spikes, spurs, thin bumps, loose bits, film, webs", "the candidates in the reply", facts=("cleanup_list",)),),
         evidence=(Evidence("the hook example", "the part is one body; 1 mark in the file (operator remove cut); 15 candidates near frozen regions excluded", "3.0 s", HOOK, "P7-1/l3/cleanup_hook"), *_E2_CLEANUP)),
    Card("ratchet_run", "tool", "method", 4,
         "Judges candidate edits of a finished part and keeps the ones that pass the rules of the YAML block `acceptance` (the ratchet). The baseline is judged first; a kept "
         "candidate is the new parent.", "measured",
         values="`run_dir`, `source` (`marks:FILE` or `stl:FOLDER`) or `marks`, `root`, `estimate_only` (default true: the rules, what is missing, the time), `max_candidates`, "
                "`mesh_size_mm`, `stop_before_promote`, `overwrite`, `rebaseline`. The rules and every limit come from the YAML; the log names the source of each. A rule "
                "that is on and has no input fails every candidate. The estimate names the number of judge calls and verify calls and the verify time only: the time of printcheck and apply_marks is not in it, "
                "and for one candidate it covered about a third of the job (6.6 s of verify against a job of 18.7 s). The start part is judged first, inside the job: 3.7 s with the rule `mass` and 8.4 s with the rule `stiffness` (the "
                "ratchet log of E2/T-ratchet-M and E2/T-ratchet-S).",
         cost="seconds, in the background after a start call of 4.6 to 4.8 s: one candidate on the 2 mm hook took 15.5 s in all with the rule `mass` and 23.3 s with the rule `stiffness` (which runs `verify`); "
              "`estimate_only` took 3.7 s and 4.3 s (host load average 12 to 43); peak memory 0.33 and 0.53 GB. Call: `ratchet_run(run_dir, source, estimate_only=false)`, then `ratchet_status`",
         call="`ratchet_run(run_dir, source, estimate_only=false)`, then `ratchet_status`",
         evidence=_E2_RATCHET,
         symptoms=(Symptom("is the edit allowed, keep or reject", "the decision and the checks of the log", facts=("ratchet.decision", "ratchet.checks[].state")),)),
    Card("ratchet_approve", "tool", "requirement", 4,
         "The opt-in owner gate: promotes the accepted candidate that waits, or drops it (`decline`). It writes the decision of the owner to the log and starts no job.", "code",
         values="`root`, `id` (default: the one that waits), `decline`. Ask the user first. After an approval call `ratchet_run` again with the same root.",
         cost="seconds", call="`ratchet_approve(root, id)` after the user's word"),
    Card("verify_run", "tool", "method", 4,
         "Checks the final STL of a finished run with quadratic tets, independent of the voxel optimizer: von Mises stress for each load case. It meshes `final.stl` (`nominal.stl` "
         "when there is stock), puts each fixed region as supports (a joint or a spring region of the YAML holds the part too) and each load as a traction.", "measured",
         values="`run_dir`, `estimate_only` (default true: memory and time for each mesh size), `mesh_size_mm`, `levels_mm`, `allowable_mpa` (no default: it adds a `limit` block "
                "and exit code 3 when the p99.9 is above it; card `verify_run:allowable_mpa`), `gate` (a pass or fail from the limits of the YAML, exit code 3 when it fails; card `verify_run:gate`), `part_only` (see `verify_run:part_only`), "
                "`percentile` (99.9), `solver`, `max_gb`, `force`, `region_band_mm` (the band of the region rule; default 0.5 voxel), `yaml_path` (the problem YAML, when run.json names a file that is gone), "
                "`near_region_mm` (default 2 mm: the distance for the stress numbers without the region patches), `load_sensitivity` (default true: also the load at a band of 0 and of 2 times the band). "
                "It needs gmsh. The load cases are solved alone (the case weights are not "
                "used). The peak moves with the mesh at sharp features; p99.9 by volume is the main number. A real run starts in the background: `verify_status`. It refuses a YAML with a moment, a remote "
                "force (`at`), a pressure or a per-axis support (fixed axes) (CONTRACTS.md section 26.2). It models springs and joints (CONTRACTS.md sections 35, 38): the result has the key `supports` (the bond force and moment of each joint, the force of each spring), "
                "and the press-fit stage is skipped for a joint. "
                "The result of `verify_status` (the content of verify.json) has these keys. "
                "Result keys (verify_status, when done): ok, exit_code (0 done, 3 limit failed), run, yaml_vs_run (YAML against run.json), material, stl, band_mm, backend, extras, memory_limit, estimates_before_meshing, mesh (nodes, tets, dofs, quality, P2 volume over STL volume), regions (per region: band, counts, area, centroid, counts_by_band; load regions also normal_check), supports (joint and spring forces), memory, timing_s, cases (per load case: von_mises with p50, p95, p99, p99.9 by volume, peak_qp_MPa, peak_vertex_MPa, peak_qp_over_p99_9, the same without bad tets, near_regions (p99.9 without the points near the regions; the share of the top 0.1 % volume near each), top_elements; linear_solve, equilibrium, forces with the resultant error and load_centre (optimizer centre, P2 traction centroid, distance, moment ratio)), load_band_sensitivity, levels, limit, gate, files. Every real result also has partial and coverage (CONTRACTS.md section 44). "
                "YAML blocks add keys: buckling.check: buckling (mode_facts, rigid_body), buckling_check; limits rows: motion; bore_checks: bore_checks, bore_facts, goodman, pressfit if press_fit. No pressfit with a joint.",
         cost="seconds to minutes: the hook example at the default size took 12 s (the mesh 7.3 s; " + BEFORE_F4_27B + ")", call="`verify_run(run_dir, estimate_only=false)`, then `verify_status`",
         symptoms=(Symptom("will it break, strong enough, stress too high, weak spot", "the stress of each case in the result",
                           facts=("verify.cases.<case>.von_mises.p99_9_by_volume_MPa", "verify.cases.<case>.von_mises.peak_qp_MPa")),
                   Symptom("the lowest buckling mode, local or global mode, swings on its joints, soft joint, which region moves most",
                           "the buckling record of the result: the lowest load factor (`verify.buckling.lowest_load_factor`, the measured one, which the gate and the ratchet read); the facts of each mode "
                           "(`verify.buckling.cases[].mode_facts`: local or global, the region with the largest motion; `kind` is null when the supports hold half of the energy or more); and, for a part "
                           "that joints or springs hold with no fixed region, the factor without the rigid-body motions of the part (`verify.buckling.cases[].rigid_body`: a fact next to the lowest factor, "
                           "not read by the gate)",
                           facts=("verify.buckling.lowest_load_factor", "verify.buckling.cases[].mode_facts", "verify.buckling.cases[].rigid_body"),
                           when="`verify.buckling.cases[].rigid_body` exists: the lowest mode is mostly a motion of the part on its supports"),),
         evidence=(Evidence("verify of the hook example, default size, no allowable (" + BEFORE_F4_27B + ")", "case pull_off: p99.9 1.66 MPa, peak at quadrature points 2.98 MPa; case hang: p99.9 "
                            "0.076 MPa; no limit block", "12 s", HOOK, "P7-1/l3/verify_hook"), *_E2_VERIFY_BK)),
    Card("verify_run:part_only", "tool", "method", 4,
         "Checks `result/part_only.stl` (the part without the bodies of `regions.not_printed`) in place of `final.stl` or `nominal.stl`. A removed body is not in the mesh: it "
         "becomes a boundary condition at its contact face.", "measured",
         values="`part_only=true` (CLI `topopt verify <run> --part-only`); the default does not change. It needs `result/part_only.stl`: a run without `regions.not_printed`, a cut that "
                "failed or a file that is gone gives an error with the fact (from the block `part_only` of `run.json`) and the options. The region rule is the same (the same band, no new "
                "number): a removed `fixed` region clamps the nodes that the rule selects (no node selected: the body clamps nothing, `boundary_condition: none`); a removed `load` region "
                "gives a uniform traction on the facets that the rule selects (none selected: the error of a load region that no body touches); a removed `solid` or `keep_out` region "
                "gives no boundary condition. A body with no fixed node and no joint or spring facet is a singular matrix: exit 1 with the facts. The result has the block `part_only` (file, sha256, `run_record`, "
                "`yaml_vs_run`, notes) and, for each removed body, the nodes, facets and areas that the rule selects and the contact area of the cut (`selected_area_over_contact`). "
                "Without the option, a run that has the file gets one note that names the option and the file that is meshed. With machining stock the file keeps the stock. The film at "
                "a contact face is in the mesh: `settings.part_only_min_film_mm` cuts it, and the hook shows why (below). The stiffness of the check is of the tet model of the file with the "
                "film as it is: read it together with `metrics.part_only.contact` (the thin area of each body) before you take it for the stiffness of the printed part. "
                "The film numbers of the rows below were " + BEFORE_F4_27B + ".",
         cost="the cost of `verify_run` (the hook example at the default size took 12 s, " + BEFORE_F4_27B + "); the contact-area booleans add `timing_s.part_only_contact_s`, a measured fact in the result",
         call="`verify_run(run_dir, part_only=true, estimate_only=false)`, then `verify_status`; CLI `topopt verify <run> --part-only`",
         symptoms=(Symptom("check the part that is printed, stress without the plate, the bolted body is not in the print",
                           "the run has the file: `metrics.part_only.applied` and `metrics.part_only.file`; the result of the check has `verify.part_only` (file, sha256, notes)",
                           facts=("metrics.part_only.applied", "metrics.part_only.file", "verify.part_only.file", "verify.part_only.yaml_vs_run"),
                           when="`metrics.part_only.applied` is true"),),
         evidence=(Evidence("`topopt verify --part-only` on the hook run with the tip removed and no film setting (voxel 2 mm, mesh size 3 mm)",
                            "the load goes into the film around the removed tip (the top-stress elements lie in it): the stiffness of the tet model is 0.47 to 0.58 N/mm against 1295 to 1317 N/mm "
                            "of the optimizer; the boundary-condition facts are right for that mesh, but the stiffness is not that of the part without the film",
                            "not timed in the record", NOT_PRINTED, "CONTRACTS 9.1: hook verify; " + BEFORE_F4_27B),
                   Evidence("`selected_area_over_contact` of the removed bodies",
                            "1.45 (bar, plate), 1.47 (bar, boss), 0.96 (hook, plate), 1.20 (hook, tip), 22.6 (hook, the keep-out hole, which touches the part along an edge only); "
                            "above 1 where a film stays; on exact boxes the contact areas equal the face areas", "not timed in the record", NOT_PRINTED, "CONTRACTS 9.1: ratio; " + BEFORE_F4_27B))),
    Card("verify_run:gate", "tool", "method", 4,
         "Turns on the hard gate of `verify`: a pass or a fail (`met`) for each check and each load case, against the limits that the YAML already has. It adds no limit and changes none.", "code",
         values="`gate=true` (CLI `topopt verify <run> --gate`). The default does not change `verify.json` or the exit codes. The gate reads these keys as the limit of the TET value: "
                "`load_cases[].stress_limit` (or `allowable_mpa` / `safety_factor`) against the stress by volume at `percentile` (99.9); `buckling.min_factor` against the lowest tet load factor "
                "of each case (`load_factors[0]`, the measured value; the gate turns the buckling check on; a case with no compression is listed as not gated; when the lowest mode of a part on soft joints or springs is a swing on the supports, "
                "the record of the case also shows `projected_value`, the factor without the rigid-body motions of the part, as a fact: the gate does not read it, because a higher number in place of the measured one would loosen `buckling.min_factor`); `limits.displacement[].limit_mm` and `limits.alignment[].shift_limit_mm` / `.tilt_limit_rad` "
                "against the tet value of each row. A check with no limit is listed in `not_gated`. The test is strict: a value on the limit passes. A check that the gate asked for and that could not run (a skipped "
                "check, a missing row) has `met: null` and fails the gate. Exit code 3 when the gate fails. `allowable_mpa` of the call replaces the stress limit of every case (card `verify_run:allowable_mpa`). "
                "With `levels_mm` the gate reads the finest level and gives the change from the level before. The limits of the YAML are for the voxel model, and the tet value differs from the voxel value by a "
                "measured amount (hook, one part, one host: stress 2.45 to 2.67 times, buckling factor 0.61 to 0.79 times, a displacement row 1.23 to 1.63 times; CONTRACTS.md section 16, \"The gate\"), "
                "so a design that met its limit on the voxel model can fail the gate. The gate does not change a limit, and it does not cover fatigue, the peaks at sharp features or printing defects.",
         cost="the cost of `verify_run`, plus the buckling check when the YAML has `buckling.min_factor` (about 9 s more on the hook at the default size; CONTRACTS.md section 33)",
         call="`verify_run(run_dir, gate=true, estimate_only=false)`, then `verify_status`; CLI `topopt verify <run> --gate`",
         symptoms=(Symptom("does it meet the limits, pass or fail, is it within the limits, is the part allowed",
                           "the verdict of the gate in the result: `verify.gate.passed`, the checks that did not pass in `verify.gate.failed`, and the value, limit and margin of each check in `verify.gate.checks`; "
                           "the checks that were left out are in `verify.gate.not_gated`",
                           facts=("verify.gate.passed", "verify.gate.failed", "verify.gate.checks.<kind>.<name>.met", "verify.gate.checks.<kind>.<name>.margin", "verify.gate.not_gated[].reason"),
                           when="`verify.gate.passed` is false: a check is not met, or a check that the gate asked for could not run (`met` is null)"),)),
    Card("verify_run:allowable_mpa", "tool", "requirement", 4,
         "Your allowable stress for the check: the stress by volume at `percentile` of each load case is compared with it, and the exit code is 3 when a case is above it. With `gate=true` it replaces the stress limit "
         "of every load case that the YAML gives.", "code",
         values="`allowable_mpa` (CLI `--allowable-mpa X`): number > 0, MPa; no default. Tighten it freely, with a report. A value above a stress limit of the YAML loosens that requirement, and that needs the consent of "
                "the user. With `gate=true` the record of each case that the call changed keeps `replaced_limit`, `replaced_source` and `loosened` (true when the new limit is above the replaced one): the gate reports the change and "
                "refuses nothing. Without `gate` the result gets the block `limit`, and the limits of the YAML are not read. Data sheet ranges: `topopt://materials/reference`.",
         cost="the cost of `verify_run`", call="`verify_run(run_dir, allowable_mpa=X, estimate_only=false)`; with the gate: `verify_run(run_dir, gate=true, allowable_mpa=X, estimate_only=false)`",
         symptoms=(Symptom("the gate passes with my allowable, is the limit looser than the YAML, a higher allowable than the YAML",
                           "`verify.gate.checks.stress.<case>.loosened` and the replaced limit in `verify.gate.checks.stress.<case>.replaced_limit`",
                           facts=("verify.gate.checks.stress.<case>.loosened", "verify.gate.checks.stress.<case>.replaced_limit"),
                           when="`verify.gate.checks.stress.<case>.loosened` is true: the call replaced a limit of the YAML with a larger one"),)),
    Card("buckling", "tool", "method", 4,
         "CLI only (no MCP tool): repeats the linear buckling check on a finished run. It rebuilds the problem from the YAML, reads the last snapshot and replaces the key "
         "`buckling` of `result/metrics.json`.", "measured",
         values="`topopt buckling <run_dir> [--modes N] [--density printed|gray] [--max-seconds S] [--max-gb G] [--yaml FILE]`. The YAML block `buckling` gives the defaults; the "
                "options win for this call. Exit code 3: it wrote a fact and no load factors. It needs a direct solver. A second call while the check of the same run is running writes the same key `buckling` of `result/metrics.json`.",
         cost="seconds: the hook example took 6.4 s (peak 530 MB); `history.jsonl` stays equal. E2, whole command on a loaded host: 9.8 s and 10.5 s at voxel 2 mm (6 modes), 6.8 s with 2 modes, 16.6 s with 12, "
              "10.2 s with gray density, 47.3 s and 1.8 GB at the public size of 1.2 mm (32.2 s with gray density)", call="CLI `topopt buckling <run_dir>`",
         symptoms=(Symptom("buckles, bends sideways, slender, wobbly", "the lowest load factor in the result", facts=("metrics.buckling.lowest_load_factor",
                           "metrics.buckling.status")),),
         evidence=(Evidence("`topopt buckling <run> --modes 2` on the finished hook run", "metrics.buckling rewritten: lowest load factor 123.085 (pull_off); history equal by sha256",
                            "6.4 s, peak 530 MB", HOOK, "P7-1/l3/cli_buck"), *_E2_BK_CHECK, *_E2_BK_MODES[2:], *_E2_BK_DENSITY, *_E2_BK_BUDGET)),
    Card("print_check", "tool", "method", 4,
         "Checks a finished part for printing. Geometry only: no solve, no run. For each check it gives the value, the limit with its source, a state (pass, fail, unknown) and "
         "a reason. It changes no file; the gate is opt-in.", "measured",
         values="`path` (an STL or a run folder), `yaml_path`, `print_settings` (a dict with keys of the `print` block; it wins), `resolution_mm`, `strict`. The checks are mesh_closed, "
                "bodies, closed_voids, bed_fit, overhang, support_volume, plate_contact, wall_thickness. A `pass` is not a promise that a print works. A `min_wall_mm` under 3 times the voxel size of the run leaves "
                "the wall check unknown, and the reply names the resolution that a pass needs.",
         cost="seconds: the hook example took 5.5 s (thickness 3.8 s); E2 on a copy of the short hook: 3.2 to 3.9 s (the thickness estimate 2.13 to 2.77 s over three calls; host load average 12 to 43)",
         call="`print_check(path, print_settings={...})`; CLI `topopt printcheck <run or STL> [--set KEY=VALUE]`",
         symptoms=(Symptom("will it print, needs supports, too thin", "the state of each check", facts=("print_check.checks[].state", "print_check.summary.failed")),),
         evidence=(Evidence("`topopt printcheck` with max_overhang_area_mm2 100 and min_wall_mm 2.5 on the hook run", "pass 4, fail 2 (overhang, wall_thickness), unknown 2 "
                            "(support_volume, plate_contact: no limit)", "5.5 s", HOOK, "P7-1/l3/base"), *_E2_PRINT)),
    Card("refine", "tool", "method", 4,
         "CLI only (no MCP tool): writes the Refine page `result/refine.html` and `result/stress.npz` of a finished run, a page to mark the final part. `topopt run` calls it after the "
         "post step.", "measured",
         values="`topopt refine <run_dir> [--yaml FILE]`. It exits 1 with one line when `final.stl` is missing or the YAML changed after the run. The stress is a voxel estimate.",
         cost="seconds: the hook example took 2.3 s", call="CLI `topopt refine <run_dir>`",
         evidence=(Evidence("`topopt refine` on the finished hook run", "refine.html and stress.npz written", "2.3 s", HOOK, "P7-1/l3/refine_hook"),)),
    Card("compare_runs", "tool", "method", 4,
         "Two runs side by side: the setup diff and the metrics diff, from files. It writes nothing. A is the base and B the change; `delta` is B - A. "
         "With `variants` it gives a table of a base and any number of variants.", "measured",
         values="`run_a`, `run_b`, `check_labels` (null: when the grid has at most 10 million voxels). The table form takes `run_a` (the base) and `variants` (a list of run folders) "
                "instead of `run_b`: one row per run (mass, stiffness per load case from the optimizer, verdict, verify, print facts, the path of the newest montage PNG) and `vs_base`, the setup diff "
                "of each variant against the base. It makes no image: `render_part` does. The two-run reply has `setup`, `metrics` (rows with a, b, delta, delta_pct), `inputs`, "
                "`notes`, `summary`. A load-case difference makes the compliance not comparable (a note). What it does not show: `run.json` does not record material.E, material.nu, settings.penal, "
                "settings.beta_schedule and settings.move_limit, so a change of them is in `setup.yaml_diff` only (the YAML is read now, not proven to be the file that ran). `settings.robust` and `settings.eta_eroded` "
                "are in `setup.yaml_diff` only: for E2/RE60 `setup.run_json_diff` was empty and the summary lines had no setup line. A mirror run moves the grid origin: `setup.labels_diff.comparable` is false and "
                "no label difference is given, and the summary line names only the first six of 32 keys of `run.json`. With `check_labels` false the files on disk are not voxelized again (0.12 s against 0.65 to 0.67 s): "
                "the label difference between the two runs is still given, and `inputs.<run>.labels_vs_disk.checked` is false. A `regions.solid` block shows as 'design->load' in the label transitions (a solid region has "
                "the label `load`) and as `regions.solid` in `setup.run_json_diff`.",
         cost="seconds: the hook example took 0.6 s; E2: 0.65 to 0.67 s with the label check and 0.12 s with `check_labels` false (host load average 12 to 43). "
              "Table form, a base and two variants, finished runs of the hook and ladle examples, read only, 3 calls each (host load average 9.4): with the label check 0.44 s, 0.11 s and 0.13 s (hook) and 0.21 to 0.22 s (ladle); "
              "with `check_labels` false 0.03 s; the reply is 7.9 KB (hook) and 8.2 KB (ladle) of JSON",
         call="`compare_runs(run_a, run_b)`, or `compare_runs(run_a, variants=[...])` for a table",
         symptoms=(Symptom("did it get better, what changed, price of the change", "the metrics rows of the two runs", facts=("metrics.mass_final_g",
                           "metrics.stiffness_cases[].N_per_mm")),),
         evidence=(Evidence("two runs of the hook example: sdf and blur", "summary: mass_final_g 25.72 -> 28.21 (+9.7 %), triangles +34.5 %, the labels are equal", "0.6 s", HOOK,
                            "P7-1/l3/compare_blur"), *_E2_COMPARE)),
    Card("render_part", "tool", "method", 4,
         "Renders a part (a run, or STL files) to PNG images, with no display. The facts name the views, each mesh, its watertight state and the share of back faces.", "measured",
         values="`run_dir` or `stl_paths`, `roles`, `views` (iso, front, back, right, left, top, bottom, or NAME:dx,dy,dz), `section` (for example x=30), `size`, `regions` (ghost | "
                "solid | off), `max_triangles`, `inline`. Back faces are magenta: a flipped or open surface. The same input gives the same PNG bytes.",
         cost="seconds: the hook example, 4 views, took 6.1 s through the CLI `topopt render` (the docstring: 2 to 4 s for 85,000 triangles)", call="`render_part(run_dir)`",
         evidence=(Evidence("`topopt render` on the finished hook run", "4 views as a montage", "6.1 s", HOOK, "P7-1/l3/render_hook"),)),
)

CARDS: tuple[Card, ...] = (*_TOP_LEVEL, *_REGIONS, *_LOAD_CASES, *_MATERIAL, *_SETTINGS, *_MIRROR, *_LIMITS, *_BUCKLING, *_PRINT,
                           *_INTERFACES, *_SPRINGS, *_JOINTS, *_ACCEPTANCE, *_BORE_CHECKS, *_TOOLS)
