"""scripts/ci/release_assets.py and its wiring: the built files of a release (the wheel, the sdist, the .mcpb) go up to the
GitHub release, and the run fails when the release does not list them (#206). The .mcpb goes up a second time as topopt.mcpb,
a byte-equal copy with a name that does not change (#409). `gh` and the build are fakes here: the release workflow cannot run in
a test."""
from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "ci"))
try:
    import release_assets
    from _common import CiError
finally:
    sys.path.pop(0)

WORKFLOW = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
CONFIG = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["semantic_release"]
VERSION = "1.2.3"
NAMES = [f"topopt-{VERSION}-py3-none-any.whl", f"topopt-{VERSION}.tar.gz", f"topopt-{VERSION}.mcpb"]
STABLE = "topopt.mcpb"
ALL = [*NAMES, STABLE]
LISTED_MCPB = b"the .mcpb bytes on the release"      # what `gh release download` gives for a listed .mcpb


def make_source(tmp_path: Path, version: str = VERSION) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "pyproject.toml").write_text(f'[project]\nname = "topopt"\nversion = "{version}"\n')
    return src


def make_dist(dist: Path, version: str = VERSION, mcpb: bool = True) -> None:
    (dist / "mcpb").mkdir(parents=True, exist_ok=True)
    (dist / f"topopt-{version}-py3-none-any.whl").write_bytes(b"w")
    (dist / f"topopt-{version}.tar.gz").write_bytes(b"s")
    if mcpb:
        (dist / "mcpb" / f"topopt-{version}.mcpb").write_bytes(b"m")


class FakeRun:
    """Records the commands. `gh release upload` adds the file names (and sizes) to the release; `gh release view` lists them.
    `drop` names files that the release never lists (an upload that GitHub lost). `build` makes the files in `dist`.
    `listed` are on the release from the start, each with the size of LISTED_MCPB unless `sizes` gives another;
    `gh release download` writes LISTED_MCPB."""

    def __init__(self, dist: Path, drop: tuple[str, ...] = (), build: bool = True, listed: tuple[str, ...] = (),
                 sizes: dict[str, int] | None = None):
        self.dist, self.drop, self.do_build = dist, drop, build
        self.calls: list[tuple[list[str], Path | None]] = []
        self.uploaded: list[str] = ["README.md", "CHANGELOG.md", *listed]
        self.size = {n: len(LISTED_MCPB) for n in self.uploaded} | (sizes or {})
        self.sent: dict[str, bytes] = {}

    def __call__(self, cmd, cwd=None, capture=False):
        self.calls.append((list(cmd), cwd))
        if cmd[:3] == ["gh", "release", "upload"]:
            assert "--clobber" not in cmd          # a listed file is never replaced: the script uploads only what is missing
            for c in cmd[4:]:
                self.uploaded.append(Path(c).name)
                self.sent[Path(c).name] = Path(c).read_bytes()
                self.size[Path(c).name] = len(self.sent[Path(c).name])
        elif cmd[:3] == ["gh", "release", "download"]:
            name, folder = cmd[cmd.index("--pattern") + 1], Path(cmd[cmd.index("--dir") + 1])
            assert name in self.uploaded
            (folder / name).write_bytes(LISTED_MCPB)          # the script makes the folder: dist/ is empty in a run that builds nothing
        elif cmd[:3] == ["gh", "release", "view"]:
            return json.dumps({"assets": [{"name": n, "size": self.size[n]} for n in self.uploaded if n not in self.drop]})
        elif cmd[:2] == ["uv", "build"] and self.do_build:
            make_dist(self.dist, mcpb=False)
        elif "packaging/mcpb/build.py" in cmd and self.do_build:
            make_dist(self.dist)
        return ""

    def commands(self):
        return [" ".join(c) for c, _ in self.calls]


def uploads(run):
    return [[Path(c).name for c in cmd[4:]] for cmd, _ in run.calls if cmd[:3] == ["gh", "release", "upload"]]


def test_the_built_files_go_up_and_the_release_lists_them(tmp_path):
    src, dist = make_source(tmp_path), tmp_path / "dist"
    make_dist(dist)
    run = FakeRun(dist)
    files = release_assets.attach(f"v{VERSION}", src, dist, run)
    assert [f.name for f in files] == ALL
    assert uploads(run) == [NAMES, [STABLE]]                                 # topopt.mcpb goes up after the versioned file is listed
    assert run.sent[STABLE] == run.sent[NAMES[2]] == b"m"                    # the copy has the bytes of the .mcpb of the build
    assert next(c for c, _ in run.calls if c[2] == "upload")[3] == f"v{VERSION}"
    assert [c[2] for c, _ in run.calls] == ["view", "upload", "view", "upload", "view"]   # a list before and after each upload
    assert not any(c[:2] == ["uv", "build"] for c, _ in run.calls)          # the files were there: no build


def test_a_repeat_replaces_nothing_and_builds_nothing(tmp_path):
    """A new build gives other bytes for the wheel and the sdist, so a file that the release lists is never uploaded again."""
    src, dist = make_source(tmp_path), tmp_path / "dist"
    run = FakeRun(dist, listed=tuple(ALL))                                   # the release has all four; dist/ is empty
    release_assets.attach(f"v{VERSION}", src, dist, run)
    assert uploads(run) == [] and not any(c[:2] == ["uv", "build"] for c, _ in run.calls)
    assert not any(c[:3] == ["gh", "release", "download"] for c, _ in run.calls)


def test_a_release_with_the_three_files_gets_the_stable_name_from_its_own_mcpb(tmp_path):
    """The backfill of a release made before #409: only topopt.mcpb goes up, with the bytes of the listed .mcpb (no new build)."""
    src, dist = make_source(tmp_path), tmp_path / "dist"
    run = FakeRun(dist, listed=tuple(NAMES))                                 # dist/ is empty, as in the attach job
    files = release_assets.attach(f"v{VERSION}", src, dist, run)
    assert [f.name for f in files] == ALL
    assert uploads(run) == [[STABLE]] and run.sent[STABLE] == LISTED_MCPB
    assert not any(c[:2] == ["uv", "build"] for c, _ in run.calls)
    dl = next(c for c, _ in run.calls if c[:3] == ["gh", "release", "download"])
    assert dl[3] == f"v{VERSION}" and dl[dl.index("--pattern") + 1] == NAMES[2]


def test_a_rebuilt_mcpb_does_not_feed_the_copy_when_the_release_lists_one(tmp_path):
    """The wheel is missing, so the run builds; the .mcpb of that build has other bytes than the listed one, and the copy must
    not take them."""
    src, dist = make_source(tmp_path), tmp_path / "dist"
    run = FakeRun(dist, listed=(NAMES[1], NAMES[2]))
    release_assets.attach(f"v{VERSION}", src, dist, run)
    assert uploads(run) == [[NAMES[0]], [STABLE]] and run.sent[STABLE] == LISTED_MCPB
    names = run.commands()
    assert names.index(next(c for c in names if "packaging/mcpb/build.py" in c)) < names.index(next(c for c in names if "release download" in c))


def test_only_the_files_that_the_release_lacks_go_up(tmp_path):
    src, dist = make_source(tmp_path), tmp_path / "dist"
    make_dist(dist)
    run = FakeRun(dist, listed=(NAMES[0],))
    release_assets.attach(f"v{VERSION}", src, dist, run)
    assert uploads(run) == [NAMES[1:], [STABLE]]


def test_files_of_other_versions_and_other_names_stay_out(tmp_path):
    src, dist = make_source(tmp_path), tmp_path / "dist"
    make_dist(dist)
    make_dist(dist, version="0.2.0")
    (dist / f"xtopopt-{VERSION}.tar.gz").write_bytes(b"x")
    (dist / f"topopt-1{VERSION}-py3-none-any.whl").write_bytes(b"x")
    run = FakeRun(dist)
    release_assets.attach(f"v{VERSION}", src, dist, run)
    assert uploads(run) == [NAMES, [STABLE]]


@pytest.mark.parametrize("lost", [NAMES[2], STABLE])
def test_the_run_fails_when_the_release_does_not_list_either_mcpb(tmp_path, lost):
    src, dist = make_source(tmp_path), tmp_path / "dist"
    make_dist(dist)
    run = FakeRun(dist, drop=(lost,))
    with pytest.raises(CiError, match=rf"does not list {re.escape(lost)} after the upload"):
        release_assets.attach(f"v{VERSION}", src, dist, run)
    assert (STABLE in [n for u in uploads(run) for n in u]) == (lost == STABLE)   # no copy goes up while the versioned file is missing


class StopRun(FakeRun):
    """The first upload stops with an error (a network break)."""

    def __call__(self, cmd, cwd=None, capture=False):
        if cmd[:3] == ["gh", "release", "upload"]:
            self.calls.append((cmd, cwd))
            raise CiError("gh release upload failed with exit code 1")
        return super().__call__(cmd, cwd, capture)


def test_topopt_mcpb_never_goes_up_in_the_call_of_the_versioned_files(tmp_path):
    """An upload that stops leaves no topopt.mcpb without its topopt-X.Y.Z.mcpb: the next run would upload a new build under the
    versioned name, and only the size would show the difference."""
    src, dist = make_source(tmp_path), tmp_path / "dist"
    make_dist(dist)
    run = StopRun(dist)
    with pytest.raises(CiError, match="upload failed"):
        release_assets.attach(f"v{VERSION}", src, dist, run)
    assert uploads(run) == [NAMES]


def test_a_release_with_topopt_mcpb_but_no_versioned_file_fails_before_any_upload(tmp_path):
    src, dist = make_source(tmp_path), tmp_path / "dist"
    make_dist(dist)
    run = FakeRun(dist, listed=(NAMES[0], NAMES[1], STABLE))
    with pytest.raises(CiError, match=rf"lists {STABLE} but not {re.escape(NAMES[2])}.*delete {STABLE}"):
        release_assets.attach(f"v{VERSION}", src, dist, run)
    assert uploads(run) == [] and not any(c[:2] == ["uv", "build"] for c, _ in run.calls)


class LossyRun(FakeRun):
    """The upload of topopt.mcpb arrives with another size."""

    def __call__(self, cmd, cwd=None, capture=False):
        out = super().__call__(cmd, cwd, capture)
        if cmd[:3] == ["gh", "release", "upload"] and STABLE in self.size:
            self.size[STABLE] = 7
        return out


@pytest.mark.parametrize("run_of", [lambda d: FakeRun(d, listed=tuple(ALL), sizes={STABLE: 7}), lambda d: LossyRun(d, listed=tuple(NAMES))],
                         ids=["listed-before-the-run", "changed-on-the-upload"])
def test_the_run_fails_when_the_two_mcpb_assets_differ_in_size(tmp_path, run_of):
    src, dist = make_source(tmp_path), tmp_path / "dist"
    with pytest.raises(CiError, match=rf"{re.escape(NAMES[2])} with {len(LISTED_MCPB)} bytes and {STABLE} with 7 bytes"):
        release_assets.attach(f"v{VERSION}", src, dist, run_of(dist))


def test_two_mcpb_assets_of_other_sizes_fail_the_run_before_it_uploads_the_rest(tmp_path):
    src, dist = make_source(tmp_path), tmp_path / "dist"
    make_dist(dist)
    run = FakeRun(dist, listed=(NAMES[1], NAMES[2], STABLE), sizes={STABLE: 7})   # the wheel is missing
    with pytest.raises(CiError, match=rf"{STABLE} with 7 bytes"):
        release_assets.attach(f"v{VERSION}", src, dist, run)
    assert uploads(run) == []


def test_missing_files_are_built_from_the_source_first(tmp_path):
    src, dist = make_source(tmp_path), tmp_path / "dist"
    make_dist(dist, mcpb=False)                                              # a new run after a failed one: the .mcpb is not there
    run = FakeRun(dist)
    release_assets.attach(f"v{VERSION}", src, dist, run)
    builds = [(c, cwd) for c, cwd in run.calls if c[:2] == ["uv", "build"] or "packaging/mcpb/build.py" in c]
    assert [c[:2] for c, _ in builds] == [["uv", "build"], [sys.executable, "-P"]]
    assert all(cwd == src for _, cwd in builds)                              # the build of the tag checkout
    assert builds[0][0][-2:] == ["--out-dir", str(dist)] and builds[1][0][-2:] == ["--out", str(dist / "mcpb")]
    assert run.commands().index(" ".join(builds[-1][0])) < next(i for i, c in enumerate(run.commands()) if "release upload" in c)


def test_a_build_that_makes_no_mcpb_fails_before_the_upload(tmp_path):
    src, dist = make_source(tmp_path), tmp_path / "dist"
    run = FakeRun(dist, build=False)
    with pytest.raises(CiError, match=rf"did not make .*topopt-{re.escape(VERSION)}\.mcpb"):
        release_assets.attach(f"v{VERSION}", src, dist, run)
    assert not any("release upload" in c for c in run.commands())


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2", "v1.2.3-rc1", "latest", "v1.2.3; rm -rf /"])
def test_the_tag_must_be_vxyz(tmp_path, tag):
    with pytest.raises(CiError, match="vX.Y.Z"):
        release_assets.attach(tag, make_source(tmp_path), tmp_path / "dist", FakeRun(tmp_path / "dist"))


def test_the_source_must_hold_the_version_of_the_tag(tmp_path):
    src, dist = make_source(tmp_path, version="1.2.4"), tmp_path / "dist"
    make_dist(dist)
    with pytest.raises(CiError, match="has the version 1.2.4, not 1.2.3"):
        release_assets.attach(f"v{VERSION}", src, dist, FakeRun(dist))


def test_the_error_paths_give_a_clear_failure(tmp_path):
    with pytest.raises(CiError, match="no-such-tool-206 is not installed"):
        release_assets._run(["no-such-tool-206", "release", "view"])
    with pytest.raises(CiError, match="false release upload failed with exit code 1"):
        release_assets._run(["false", "release", "upload"])

    def not_json(cmd, cwd=None, capture=False):
        return "release not found"
    with pytest.raises(CiError, match="gave no asset list"):
        release_assets.attach(f"v{VERSION}", make_source(tmp_path), tmp_path / "dist", not_json)


def test_the_files_are_the_ones_of_the_publish_patterns(tmp_path):
    make_dist(tmp_path / "dist")
    files = release_assets.built_files(tmp_path / "dist", VERSION)
    assert [f.name for f in files] == NAMES
    pats = CONFIG["publish"]["dist_glob_patterns"]
    for f in files:
        assert any(f.relative_to(tmp_path).match(p) for p in pats), f


# ---- the wiring in the release workflow ----

def job(name: str) -> str:
    """The text of one job of release.yml."""
    body = WORKFLOW.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  (\w[\w-]*):\n", body, flags=re.M)[1:]
    jobs = dict(zip(parts[::2], parts[1::2]))
    return jobs[name]


def step_of(job_text: str, name: str) -> str:
    for part in re.split(r"^      - name: ", job_text, flags=re.M)[1:]:
        if part.split("\n", 1)[0].strip() == name:
            return part
    raise AssertionError(f"no step {name!r}")


def test_the_release_job_attaches_the_files_after_the_release_and_before_latest():
    rel = job("release")
    names = [p.split("\n", 1)[0].strip() for p in re.split(r"^      - name: ", rel, flags=re.M)[1:]]
    assert names.index("Release") < names.index("Attach the files to the release") < names.index("Move the branch latest")
    s = step_of(rel, "Attach the files to the release")
    assert "inputs.dry_run == 'false'" in s and "steps.release.outputs.released == 'true'" in s
    assert 'scripts/ci/release_assets.py --tag "$TAG"' in s and "steps.release.outputs.tag" in s
    assert "inputs.attach_to_tag == ''" in rel.split("steps:", 1)[0]       # an attach run makes no release
    mcpb = step_of(rel, "Install the mcpb CLI (the build checks the manifest of the .mcpb with it)")
    assert re.search(r"npm install -g @anthropic-ai/mcpb@\d+\.\d+\.\d+", mcpb) and "inputs.dry_run == 'false'" in mcpb
    assert names.index("Install the mcpb CLI (the build checks the manifest of the .mcpb with it)") < names.index("Release")


def test_the_attach_job_builds_the_tag_and_attaches_its_files():
    a = job("attach")
    assert "inputs.attach_to_tag != ''" in a.split("steps:", 1)[0] and "github.ref_name == 'main'" in a.split("steps:", 1)[0]
    s = step_of(a, "Build and attach the files of the tag")
    assert 'git worktree add --detach "$RUNNER_TEMP/release-tag" "refs/tags/$TAG"' in s
    assert '--tag "$TAG" --source "$RUNNER_TEMP/release-tag"' in s
    assert "${{ inputs.attach_to_tag }}" in s and "semantic-release" not in a and "latest" not in a
    assert re.search(r"uses: actions/checkout@\S+\n\s+with:\n\s+fetch-depth: 0", a)    # the tags must be there
    assert re.search(r"npm install -g @anthropic-ai/mcpb@\d+\.\d+\.\d+", step_of(a, "Install the mcpb CLI for the build"))
    assert re.search(r"attach_to_tag:\n\s+description: .+\n\s+type: string\n\s+default: \"\"", WORKFLOW)


def test_no_step_of_the_release_workflow_may_fail_quietly():
    assert "continue-on-error" not in WORKFLOW


def test_releasing_md_says_how_the_files_get_attached():
    releasing = (REPO / "RELEASING.md").read_text(encoding="utf-8")
    assert "dist/mcpb/*.mcpb\"` from `dist_glob_patterns`" not in releasing      # the patterns do not drive the upload
    for needle in ("## Attach the files to an existing release", "release_assets.py", "attach_to_tag=v0.3.0", "Attach the files to the release",
                   "does not upload the built files", "never replaces", "attach_to_tag=vX.Y.Z", "mcpb validate"):
        assert needle in releasing, needle


STABLE_LINK = "https://github.com/chartergroup/topopt/releases/latest/download/topopt.mcpb"


def test_the_docs_name_the_link_to_the_newest_extension():
    """#409: RELEASING.md and the Claude Desktop part of the install doc give the link without a version."""
    releasing = (REPO / "RELEASING.md").read_text(encoding="utf-8")
    assert STABLE_LINK in releasing and "same size" in releasing
    install = (REPO / "docs" / "reference" / "install.md").read_text(encoding="utf-8")
    desktop = install.split("\n## Claude Desktop\n", 1)[1].split("\n## ", 1)[0]
    assert STABLE_LINK in desktop and "`topopt.mcpb`" in desktop
    assert release_assets.STABLE_MCPB == STABLE_LINK.rsplit("/", 1)[1]
