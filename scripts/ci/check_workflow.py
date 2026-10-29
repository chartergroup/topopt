#!/usr/bin/env python3
"""Check a GitHub Actions workflow file without `actionlint`: a YAML parse and a check of the step keys.

    .venv/bin/python scripts/ci/check_workflow.py [.github/workflows/tests.yml]
    .venv/bin/python scripts/ci/check_workflow.py .github/workflows/intel-mac.yml
    .venv/bin/python scripts/ci/check_workflow.py .github/actions/plugin-eval/action.yml

The path is optional. Without it the script reads `.github/workflows/tests.yml`, else the old inactive
path `ci/github-workflows/tests.yml` (CONTRACTS.md section 27.1).
Each file has its own rule for `on` (`EVENT_RULES`, chosen by the file name). All files run by hand only: `tests.yml`, `intel-mac.yml`, `release.yml`, `evals.yml` and any other workflow file need workflow_dispatch and no other event (`tests.yml` also has workflow_call, so that `release.yml` can run it). A push or a pull
request must never start a run.

`actionlint` is not installed on the machines of this project, so this script checks what a typo
most often breaks. It is not a full linter. It checks:
- the file parses, and has `name`, `on` (the events of the rule of the file) and `jobs`;
- each job has `runs-on` and a list of `steps`; a job with `needs` names jobs that exist;
- each step has one of `uses` and `run`, only known keys, and (here) a `name`;
- a `uses` value is `owner/repo@ref` (a ref is given), or `./.github/actions/<name>` with an `action.yml` in the repo;
- `${{ matrix.X }}` names a key of the matrix of the job, `${{ steps.X }}` names an `id` of an earlier step,
  `${{ needs.X }}` names a job of `needs`, `${{ inputs.X }}` names an input of the file, and every `${{` is closed;
- each `scripts/ci/<file>` in a `run` exists in the repo, and `legs.toml` has every leg label of the matrix;
- each upload-artifact step has a name with the leg label, `retention-days`, and `if: always()` or `!cancelled()`.
A composite action file (`action.yml`) gets the same step checks, and also: `runs.using` is `composite`, each input has a
`description`, and each `run` step has a `shell`.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, CiError, load_spec, run_main  # noqa: E402
from plan_legs import MATRIX_KEYS  # noqa: E402

DEFAULT_WORKFLOWS = (".github/workflows/tests.yml", "ci/github-workflows/tests.yml")    # the live path, then the path of the branch
# The rule for `on` of each workflow file, by file name: (events that `on` must have, True if `on` may have no other event).
# A file with another name runs by hand only: OTHER_RULE.
DEFAULT_RULE_NAME = "tests.yml"
EVENT_RULES = {
    "tests.yml": (("workflow_dispatch", "workflow_call"), True),   # workflow_call: release.yml runs the tests first
    "intel-mac.yml": (("workflow_dispatch",), True),
    "release.yml": (("workflow_dispatch",), True),
    "evals.yml": (("workflow_dispatch",), True),        # each case is a paid model call
}
OTHER_RULE = (("workflow_dispatch",), True)
ACTION_NAMES = ("action.yml", "action.yaml")           # a composite action file, checked by check_action
STEP_KEYS = {"id", "if", "name", "uses", "run", "shell", "with", "env", "continue-on-error", "timeout-minutes", "working-directory"}
JOB_KEYS = {"name", "runs-on", "needs", "if", "strategy", "steps", "env", "defaults", "timeout-minutes", "outputs",
            "permissions", "concurrency", "continue-on-error", "container", "services", "uses", "with", "secrets"}
TOP_KEYS = {"name", "on", "env", "jobs", "permissions", "concurrency", "defaults", "run-name"}
ACTION_TOP_KEYS = {"name", "description", "author", "inputs", "outputs", "runs", "branding"}
EXPR = re.compile(r"\$\{\{(.*?)\}\}", re.S)
REMOTE_USES = re.compile(r"[\w.-]+/[\w./-]+@[\w.-]+")
LOCAL_USES = re.compile(r"\./\.github/actions/[\w.-]+")


def default_workflow(root: Path = ROOT) -> Path:
    """The workflow file to check: the live path if the file is there, else the path that the branch holds."""
    for rel in DEFAULT_WORKFLOWS:
        if (root / rel).is_file():
            return root / rel
    raise CiError("no workflow file at " + " or ".join(DEFAULT_WORKFLOWS))


def _on(doc: dict):
    """PyYAML reads the key `on` as the boolean True. Return its value."""
    return doc.get("on", doc.get(True))


def matrix_keys(job: dict) -> set[str]:
    """The keys that `matrix.<key>` can name: the include entries, and the plain matrix axes."""
    m = ((job.get("strategy") or {}).get("matrix"))
    if isinstance(m, str):          # `matrix: ${{ fromJSON(...) }}`: the keys are those that plan_legs.py writes
        return set(MATRIX_KEYS)
    keys = set()
    for k, v in (m or {}).items():
        if k in ("include", "exclude"):
            if isinstance(v, str):
                return set(MATRIX_KEYS)
            for e in v:
                keys |= set(e)
        else:
            keys.add(k)
    return keys


def event_rule(name: str) -> tuple[tuple[str, ...], bool]:
    """The rule for `on` of the workflow file with this name: (required events, True if no other event is allowed)."""
    return EVENT_RULES.get(name, OTHER_RULE)


def check_workflow(doc: dict, root: Path = ROOT, leg_labels: set[str] | None = None, name: str = DEFAULT_RULE_NAME) -> list[str]:
    """Return a list of problem texts. An empty list means that the checks pass. `name` is the file name: it chooses the rule for `on`."""
    p: list[str] = []
    if not isinstance(doc, dict):
        return ["the file is not a YAML mapping"]
    for k in doc:
        if k not in TOP_KEYS and k is not True:
            p.append(f"unknown top-level key {k!r}")
    if not doc.get("name"):
        p.append("no `name`")
    on = _on(doc)
    if not isinstance(on, dict):
        p.append("`on` must be a mapping of events")
    else:
        required, exact = event_rule(name)
        for ev in required:
            if ev not in on:
                p.append(f"`on` has no {ev}")
        if exact:
            p += [f"`on` has the event {ev}, and the rule of {name} allows only: {', '.join(required)}" for ev in on if ev not in required]
    jobs = doc.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        return p + ["no `jobs`"]
    for jid, job in jobs.items():
        where = f"job {jid}"
        for k in job:
            if k not in JOB_KEYS:
                p.append(f"{where}: unknown key {k!r}")
        if "uses" in job:       # a job that calls a reusable workflow of this repo: no runs-on, no steps
            if not str(job["uses"]).startswith("./.github/workflows/") or "steps" in job or "runs-on" in job:
                p.append(f"{where}: `uses` must be a ./.github/workflows/ file, and the job has no `steps` or `runs-on`")
            continue
        if "runs-on" not in job:
            p.append(f"{where}: no `runs-on`")
        needs = job.get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        for n in needs:
            if n not in jobs:
                p.append(f"{where}: needs {n!r}, which is not a job")
        steps = job.get("steps")
        if not isinstance(steps, list) or not steps:
            p.append(f"{where}: no `steps` list")
            continue
        p += step_problems(steps, where, root, matrix_keys(job), set(needs), workflow_inputs(doc))
    if leg_labels is not None:
        for jid, job in jobs.items():
            m = (job.get("strategy") or {}).get("matrix")
            if isinstance(m, dict):
                for e in m.get("include", []) if not isinstance(m.get("include"), str) else []:
                    if e.get("label") and e["label"] not in leg_labels:
                        p.append(f"job {jid}: matrix label {e['label']!r} is not a leg of legs.toml")
    return p


def workflow_inputs(doc: dict) -> set[str]:
    """The input names that `${{ inputs.X }}` can name: the inputs of workflow_dispatch and of workflow_call."""
    on = _on(doc)
    if not isinstance(on, dict):
        return set()
    names: set[str] = set()
    for ev in ("workflow_dispatch", "workflow_call"):
        names |= set(((on.get(ev) or {}).get("inputs") or {}))
    return names


def step_problems(steps: list, where: str, root: Path, mkeys: set[str], needs: set[str], inputs: set[str],
                  composite: bool = False) -> list[str]:
    """The problems of the steps of a job, or of a composite action (`composite`: each `run` step needs a `shell`)."""
    p: list[str] = []
    seen_ids: set[str] = set()
    for i, st in enumerate(steps, 1):
        sw = f"{where}, step {i} ({st.get('name', 'no name') if isinstance(st, dict) else 'no name'})"
        if not isinstance(st, dict):
            p.append(f"{sw}: a step must be a mapping")
            continue
        for k in st:
            if k not in STEP_KEYS:
                p.append(f"{sw}: unknown key {k!r}")
        if ("uses" in st) == ("run" in st):
            p.append(f"{sw}: a step needs exactly one of `uses` and `run`")
        if not st.get("name"):
            p.append(f"{sw}: no `name`")
        if "uses" in st:
            uses = str(st["uses"])
            if LOCAL_USES.fullmatch(uses):
                if not any((root / uses / n).is_file() for n in ACTION_NAMES):
                    p.append(f"{sw}: `uses: {uses}` has no action.yml in the repo")
            elif not REMOTE_USES.fullmatch(uses):
                p.append(f"{sw}: `uses: {uses}` is not owner/repo@ref")
        if "with" in st and "uses" not in st:
            p.append(f"{sw}: `with` is only for `uses`")
        if composite and "run" in st and not st.get("shell"):
            p.append(f"{sw}: a `run` step of a composite action needs a `shell`")
        if "timeout-minutes" in st and not isinstance(st["timeout-minutes"], int):
            p.append(f"{sw}: timeout-minutes must be an integer")
        text = "\n".join(str(v) for v in [st.get("run"), st.get("if"), st.get("name"), st.get("working-directory")] + list((st.get("with") or {}).values()) + list((st.get("env") or {}).values()) if v is not None)
        p += [f"{sw}: {m}" for m in expression_problems(text, mkeys, seen_ids, needs, inputs)]
        for f in re.findall(r"scripts/ci/([\w.-]+)", str(st.get("run", ""))):
            if not (root / "scripts" / "ci" / f).is_file():
                p.append(f"{sw}: scripts/ci/{f} does not exist")
        if str(st.get("uses", "")).startswith("actions/upload-artifact@"):
            w = st.get("with") or {}
            if "retention-days" not in w:
                p.append(f"{sw}: upload-artifact has no retention-days")
            if "matrix.label" not in str(w.get("name", "")) and "label" in mkeys:
                p.append(f"{sw}: the artifact name does not include matrix.label")
            if not re.search(r"always\(\)|!cancelled\(\)", str(st.get("if", ""))):
                p.append(f"{sw}: upload-artifact does not run on failure (add `if: always()`)")
        if st.get("id"):
            seen_ids.add(st["id"])
    return p


def check_action(doc: dict, root: Path = ROOT) -> list[str]:
    """Return the problems of a composite action file (action.yml). An empty list means that the checks pass."""
    if not isinstance(doc, dict):
        return ["the file is not a YAML mapping"]
    p = [f"unknown top-level key {k!r}" for k in doc if k not in ACTION_TOP_KEYS]
    for k in ("name", "description"):
        if not doc.get(k):
            p.append(f"no `{k}`")
    inputs = doc.get("inputs") or {}
    for name, spec in inputs.items():
        if not isinstance(spec, dict) or not spec.get("description"):
            p.append(f"input {name}: no `description`")
    runs = doc.get("runs")
    if not isinstance(runs, dict) or runs.get("using") != "composite":
        return p + ["`runs.using` must be composite"]
    steps = runs.get("steps")
    if not isinstance(steps, list) or not steps:
        return p + ["no `runs.steps` list"]
    return p + step_problems(steps, "action", root, set(), set(), set(inputs), composite=True)


def expression_problems(text: str, matrix: set[str], step_ids: set[str], needs: set[str], inputs: set[str] | None = None) -> list[str]:
    """`inputs`: the declared inputs; None does not check `inputs.X`."""
    p = []
    if text.count("${{") != text.count("}}"):
        p.append("`${{` and `}}` do not match")
    for expr in EXPR.findall(text):
        for key in re.findall(r"\bmatrix\.([\w-]+)", expr):
            if key not in matrix:
                p.append(f"`matrix.{key}` is not a key of the matrix of this job")
        for sid in re.findall(r"\bsteps\.([\w-]+)", expr):
            if sid not in step_ids:
                p.append(f"`steps.{sid}` is not the id of an earlier step")
        for n in re.findall(r"\bneeds\.([\w-]+)", expr):
            if n not in needs:
                p.append(f"`needs.{n}` is not in `needs` of this job")
        for name in re.findall(r"\binputs\.([\w-]+)", expr) if inputs is not None else []:
            if name not in inputs:
                p.append(f"`inputs.{name}` is not an input of this file")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workflow", nargs="?", help="path of the workflow file (default: " + " or ".join(DEFAULT_WORKFLOWS) + ")")
    args = ap.parse_args(argv)
    path = Path(args.workflow) if args.workflow else default_workflow()
    try:
        doc = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise CiError(f"{path}: cannot parse: {e}")
    if path.name in ACTION_NAMES:
        problems = check_action(doc, ROOT)
        for t in problems:
            print(f"{path}: {t}")
        if problems:
            raise CiError(f"{len(problems)} problem(s) in {path}")
        print(f"OK: {path} parses; composite action, {len(doc['runs']['steps'])} steps, step keys and references checked")
        return 0
    problems = check_workflow(doc, ROOT, set(load_spec()["legs"]), path.name)
    for t in problems:
        print(f"{path}: {t}")
    if problems:
        raise CiError(f"{len(problems)} problem(s) in {path}")
    jobs = doc["jobs"]
    print(f"OK: {path} parses; {len(jobs)} job(s) ({', '.join(jobs)}), "
          f"{sum(len(j.get('steps', [])) for j in jobs.values())} steps, step keys and references checked")
    return 0


if __name__ == "__main__":
    run_main(main)
