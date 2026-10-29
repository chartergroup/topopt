"""A resume of a finished run with a changed key that acts in the passes (P7 round 2, CONTRACTS.md section 41.8).

A finished run has no pass left. A resume with `--allow-changed-inputs` (MCP: `allow_changed_inputs=true`) runs the post step again.
A key that the post step reads (for example `settings.mesh_method`) takes effect. A key that the passes read (for example
`settings.penal`) is accepted, no pass runs, and the design stays as the old values made it. This module names the second kind as a fact,
with the options: a new run, or `continue_from` (round 3), which runs passes with the new values. It refuses nothing and holds no number.

The table of lever cards decides which key acts in the passes: a YAML card of layer 1 (a global setting: a new run). A card of layer 3
(the post step reads it) or layer 4 (a local edit or judgement) is not named. A region file, and the voxels that the regions fill, act in the
passes too. The function is separate from `topopt/checkpoint.py` (which calls it in `plan_resume`) so that the code there stays small.
"""
from __future__ import annotations

import re

from .levers_cards import CARDS

# The key path at the start of a change line of `checkpoint.diff_inputs`: `settings.penal: 3.0 -> 3.5`, `load_cases[0].forces[0].force: ...`.
_CHANGE_KEY = re.compile(r"^([A-Za-z_][\w\[\]\.<>-]*): ")
_REGIONS = "regions"
_REGION_FILE = "region file "
_LABEL_LINES = ("the voxel labels are not the same", "the design voxels are not the same")

OPTIONS = [
    "run_topopt without resume=true and with a new out_dir: a new run with the new values (the cost is a whole run; check_problem gives "
    "derived.time_estimate_s)",
    "run_topopt with continue_from=<this run folder>, more_iterations=N, allow_changed_inputs=true and a new out_dir: a new folder that links to this run "
    "(which is only read); the passes go on from its last checkpoint with the new values (a changed grid is refused; the cost is the N passes and a fixed "
    "part, and the card run_topopt:continue_from has a measured row)",
    "give the old values back in the YAML and resume: the record then agrees with the part",
    "keep this resume for a key that only the post step reads (the card of the key, layer 3 in the resource topopt://levers, says so)",
]


def _layers() -> dict[str, int]:
    """Key path (a card key or a key in `also`) -> layer, for the YAML cards."""
    return {k: c.layer for c in CARDS if c.group == "yaml" for k in (c.key, *c.also)}


def _card_key(key: str, layers: dict[str, int]) -> str | None:
    """The card key of a changed YAML key path: `[0]` is `[]`, and a path with no card takes the card of its parent (`a.b[][]` -> `a.b`)."""
    k = re.sub(r"\[\d+\]", "[]", key)
    while k:
        if k in layers:
            return k
        k = k[:k.rindex("[")] if k.endswith("]") else k.rpartition(".")[0]
    return None


def pass_key_note(changes: list[str], complete: bool, labels_compared: bool = False) -> dict | None:
    """For a finished run (`complete`): the changed inputs that act in the passes, else None.

    `changes` are the lines of `ResumePlan.changes`. The result has `keys` (the card keys, in the order of the lines), `fact` (one sentence
    of text) and `options`.

    The passes read the voxel labels, not the region files. `labels_compared` says that the plan compared the labels with the checkpoint (it had
    the problem, and the checkpoint holds the labels hash): then a region file with other content that moves no label is no key of the passes.
    A file that is only under another path (the same bytes) is never one. Without the comparison (the MCP reply builds no problem) the file is
    named, and the text says that the voxels were not compared.
    """
    if not complete or not changes:
        return None
    layers = _layers()
    moved = any(line.startswith(_LABEL_LINES) for line in changes)
    keys, shown = [], []
    for line in changes:
        is_file = line.startswith(_REGION_FILE)
        if is_file or line.startswith(_LABEL_LINES):
            if is_file and ("the content is the same" in line or (labels_compared and not moved)):
                continue   # the same bytes under another path, or a file that fills the same voxels: no pass reads a difference
            key, text = _REGIONS, re.split(r" has | is | are |: ", line, maxsplit=1)[0]
            if is_file and not (labels_compared or moved):
                text += " (this reply does not compare the voxels that it fills)"
        else:
            m = _CHANGE_KEY.match(line)
            key = _card_key(m.group(1), layers) if m else None
            if key is None or layers[key] != 1:
                continue
            text = f"{m.group(1)} ({line[m.end():].split(' (the YAML', 1)[0]})"
        keys.append(key)
        shown.append(text)
    if not keys:
        return None
    fact = ("the run is finished, so no pass runs again, and the design stays as the old values made it. "
            f"The changed inputs that act in the passes: {'; '.join(shown)}. "
            "The part, the history and the checkpoint are from the passes that ran with the old values; the record can name the new values. "
            "A key that only the post step reads (layer 3 on its card) takes effect in this resume.")
    return {"keys": list(dict.fromkeys(keys)), "fact": fact, "options": list(OPTIONS)}
