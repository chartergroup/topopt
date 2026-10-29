"""Tests for the Claude Desktop extension (`.mcpb`): the manifest, the build script and the zip.

The zip is made by packaging/mcpb/build.py with the standard library only. If the `mcpb` CLI is on PATH, one test also
checks the manifest against its schema. A test that starts the unpacked bundle through its launcher needs the network
(about 200 MB of packages): it runs only when the environment variable TOPOPT_TEST_MCPB_LAUNCH is 1.
"""
import ast
import asyncio
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_PY = re.search(r"^LAUNCHER_PYTHON=(\d+\.\d+)$", (ROOT / "scripts" / "topopt-env.sh").read_text(), re.M).group(1)   # the one source (#422)
BUILD = ROOT / "packaging" / "mcpb" / "build.py"
TEMPLATE = ROOT / "packaging" / "mcpb" / "manifest.json"
IDS = ["lead-design", "cad", "fea", "print", "finish"]


@pytest.fixture(scope="module")
def pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("mcpb_build_script", BUILD)  # not `build`: that is a PyPI package
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_bundle(out: Path) -> Path:
    r = subprocess.run([sys.executable, str(BUILD), "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    bundles = list(out.glob("*.mcpb"))
    assert len(bundles) == 1, bundles
    return bundles[0]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return build_bundle(tmp_path_factory.mktemp("mcpb"))


def read_manifest(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("manifest.json"))


# --- the committed manifest ---------------------------------------------------------------------------------------

def test_template_has_the_required_fields_and_the_package_version(pyproject):
    m = json.loads(TEMPLATE.read_text())
    for key in ("manifest_version", "name", "version", "description", "author", "server"):
        assert m[key], key
    assert m["author"]["name"]
    assert m["version"] == pyproject["project"]["version"]  # the build script writes the same value
    server = m["server"]
    assert server["type"] in {"uv", "python", "node", "binary"} and server["mcp_config"]["command"]
    assert m["license"] == "MIT"


def test_template_starts_the_launcher_and_resolves_the_entry_point():
    """#206: the host starts the launcher, the same first start as the plugin (#370): it connects at once with all the tools and
    sets up its Python in the background. `/bin/sh` runs it, so the mode of the unpacked file does not matter."""
    m = json.loads(TEMPLATE.read_text())
    server = m["server"]
    assert server["type"] == "binary" and m["manifest_version"] == "0.4"
    assert server["entry_point"] == "scripts/topopt-env.sh" and (ROOT / server["entry_point"]).is_file()
    assert server["mcp_config"]["command"] == "/bin/sh"
    assert server["mcp_config"]["args"] == ["${__dirname}/" + server["entry_point"], "mcp"]


def test_the_extension_runs_on_the_python_of_the_launcher_and_names_no_host_runtime():
    """PY-1: the launcher makes its environment with a uv-managed Python (LAUNCHER_PYTHON, 3.13 since #422), whatever `requires-python` allows. The host needs no
    Python, so the manifest names no runtime (a host could take `runtimes.python` as a need of the machine)."""
    m = json.loads(TEMPLATE.read_text())
    assert "runtimes" not in m["compatibility"]
    launcher = (ROOT / "scripts" / "topopt-env.sh").read_text()
    assert re.search(r'"\$UV" venv --python "\$LAUNCHER_PYTHON" --managed-python', launcher)
    assert f"Python {LAUNCHER_PY} and about 200 MB" in m["long_description"]


def test_template_declares_macos_arm64_and_no_windows():
    m = json.loads(TEMPLATE.read_text())
    platforms = m["compatibility"]["platforms"]
    assert "darwin" in platforms and "win32" not in platforms  # the server uses POSIX calls
    meta = m["_meta"]["io.github.chartergroup.topopt"]["architectures"]
    assert meta["darwin"] == ["arm64"] and meta["linux"] == ["x86_64"]
    assert set(platforms) == set(meta)


def test_template_gives_the_server_a_work_folder():
    """Claude Desktop starts the server with a working folder that the user does not know. The user can pick one.

    The option has no default: Claude Desktop does not replace a variable inside the default of an option (the server got the text
    `${DOCUMENTS}/topopt` and made the folder in the extension folder). The default is in `mcp_config.env`, where the host replaces
    `${HOME}`; the server uses it when the option is empty (`topopt/workfolder.py`).
    """
    m = json.loads(TEMPLATE.read_text())
    option = m["user_config"]["workspace"]
    assert option["type"] == "directory" and option["title"] == "Work folder" and option["required"] is False
    assert "default" not in option and "Documents/topopt" in option["description"]       # the text says what an empty option does
    env = m["server"]["mcp_config"]["env"]
    assert env["TOPOPT_WORKSPACE"] == "${user_config.workspace}"
    assert env["TOPOPT_DEFAULT_WORKSPACE"] == "${HOME}/Documents/topopt"
    assert "${" not in json.dumps({k: v for k, v in option.items() if k != "description"})   # no variable inside the option


def test_template_text_has_no_habitual_caps_and_no_talk_material():
    import re
    text = TEMPLATE.read_text()
    assert not re.search(r"\b(must|never)\b", text, re.I)
    assert not re.search(r"t-shirt|\bshirt\b|run mcp|tuesday|on stage|audience|demo day", text, re.I)


def test_output_folder_is_ignored_by_git(script):
    rel = script.DEFAULT_OUT.relative_to(ROOT).as_posix()
    assert "dist/" in (ROOT / ".gitignore").read_text().split() and rel.startswith("dist/")
    if shutil.which("git") and (ROOT / ".git").exists():
        r = subprocess.run(["git", "check-ignore", "-q", f"{rel}/topopt-0.1.0.mcpb"], cwd=ROOT)
        assert r.returncode == 0, f"{rel} is not in .gitignore"
        tracked = subprocess.run(["git", "ls-files", "*.mcpb"], cwd=ROOT, capture_output=True, text=True).stdout
        assert tracked.strip() == ""  # some tools cannot publish a binary file


# --- the build ---------------------------------------------------------------------------------------------------

def test_the_bundle_is_a_zip_with_the_manifest_and_the_server_entry(bundle, pyproject):
    version = pyproject["project"]["version"]
    assert bundle.name == f"topopt-{version}.mcpb"
    with zipfile.ZipFile(bundle) as zf:
        assert zf.testzip() is None
        names = set(zf.namelist())
        m = json.loads(zf.read("manifest.json"))
    assert {"manifest.json", "pyproject.toml", m["server"]["entry_point"], "topopt_mcp/team.py",
            "topopt/__init__.py", "topopt/cli.py", "topopt/dashboard/index.html", "topopt/refine/template.html",
            "tools/check_part.py", "examples/hook/hook.yaml", "CONTRACTS.md", "topopt/data/printers.yaml",
            "topopt/data/materials.yaml", "scripts/topopt-env.sh", "requirements.lock", "requirements-macos-x86_64.lock",
            "topopt_mcp/handoff.py", "topopt_mcp/launcher/modern.tools.json", "topopt_mcp/launcher/legacy.tools.json"} <= names
    assert {f"topopt_mcp/roles/{i}.md" for i in IDS} <= names
    assert m["version"] == version  # from pyproject.toml
    assert "runtimes" not in m["compatibility"]


def test_the_bundle_holds_no_cache_binary_or_test_file(bundle):
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    assert names == sorted(names)
    for name in names:
        assert "__pycache__" not in name and not name.endswith((".pyc", ".dylib", ".so", ".mcpb")), name
        assert not name.startswith(("tests/", "recordings/", "runs/", ".git", ".venv", "dist/", "agents/", "skills/")), name
        assert not name.endswith("local.json")
    assert max(zipfile.ZipFile(bundle).infolist(), key=lambda i: i.file_size).file_size < 2_000_000


def test_the_manifest_in_the_bundle_lists_every_tool_and_the_role_prompts(bundle):
    m = read_manifest(bundle)
    server_tree = ast.parse((ROOT / "topopt_mcp" / "server.py").read_text())
    decorated = [n.name for n in server_tree.body if isinstance(n, ast.FunctionDef)
                 and any("tool" in ast.unparse(d) and "mcp" in ast.unparse(d) for d in n.decorator_list)]
    assert [t["name"] for t in m["tools"]] == decorated
    assert "team_guide" in decorated and "run_topopt" in decorated
    assert all(t["description"] for t in m["tools"])
    assert m["tools_generated"] is True
    from topopt_mcp import team
    roles = team.read_roles(ROOT / "topopt_mcp" / "roles")
    assert m["prompts"] == [{"name": r.id, "description": r.summary, "text": r.text} for r in roles.values()]


def test_the_bundle_pyproject_has_the_pins_of_the_repo_and_the_mcp_pin(bundle, pyproject):
    with zipfile.ZipFile(bundle) as zf:
        meta = tomllib.loads(zf.read("pyproject.toml").decode())
    project, repo = meta["project"], pyproject["project"]
    assert (project["name"], project["version"], project["requires-python"]) == (
        repo["name"], repo["version"], repo["requires-python"])
    assert project["dependencies"] == repo["dependencies"] + repo["optional-dependencies"]["mcp"]
    assert any(d.startswith("mcp==") for d in project["dependencies"])  # an install of the folder without the extra gets the server
    assert project["scripts"] == repo["scripts"]
    assert meta["build-system"] == pyproject["build-system"]
    assert meta["tool"]["setuptools"] == pyproject["tool"]["setuptools"]
    assert "roles/*.md" in meta["tool"]["setuptools"]["package-data"]["topopt_mcp"]
    # the extras of the repo stay: `uv run` installs none of them unless asked, and the offers of `environment`
    # (`uv pip install -e '<folder>[pardiso]'`) need them in the pyproject of the unpacked folder
    assert project["optional-dependencies"] == repo["optional-dependencies"]
    assert "pardiso" in project["optional-dependencies"]
    lock = [ln.strip() for ln in (ROOT / "requirements.lock").read_text().splitlines() if "==" in ln]
    constraints = meta["tool"]["uv"]["constraint-dependencies"]
    intel_lock = {ln.split("==")[0]: ln.strip() for ln in
                  (ROOT / "requirements-macos-x86_64.lock").read_text().splitlines() if "==" in ln and not ln.startswith("#")}
    other, mac = "platform_system != 'Darwin' or platform_machine != 'x86_64'", "platform_system == 'Darwin' and platform_machine == 'x86_64'"
    expected = []
    for pin in lock:  # the tested versions, for the sub-packages; for an x86_64 Mac (V5) the older pins, with markers
        alt = intel_lock.get(pin.split("==")[0])
        expected += [f"{pin}; {other}", f"{alt}; {mac}"] if alt and alt != pin else [pin]
    assert constraints == expected
    assert sorted(c.split("==")[0] for c in constraints if c.endswith(mac)) == ["cryptography", "llvmlite", "numba", "numpy"]
    assert "numba==0.62.1; platform_system == 'Darwin' and platform_machine == 'x86_64'" in constraints
    assert "numba==0.67.0; platform_system != 'Darwin' or platform_machine != 'x86_64'" in constraints


def test_the_bundle_pyproject_copies_the_license_fields_and_checks_the_files(script, pyproject):
    meta = json.loads(json.dumps(pyproject))  # a copy
    meta["project"]["license"] = "MIT"
    meta["project"]["license-files"] = ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    names = ["LICENSE", "THIRD_PARTY_NOTICES.md", "topopt/__init__.py"]
    project = tomllib.loads(script.bundle_pyproject(meta, names))["project"]
    assert project["license"] == "MIT" and project["license-files"] == ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    with pytest.raises(SystemExit, match="THIRD_PARTY_NOTICES.md"):  # setuptools stops on a pattern with no file
        script.bundle_pyproject(meta, ["LICENSE"])
    plain = tomllib.loads(script.bundle_pyproject(pyproject))["project"]  # the repo as it is: no field in, none out
    assert ("license" in plain) == ("license" in pyproject["project"])


def test_python_below_3_11_gets_one_line_with_the_fact_and_the_uv_command(script):
    line = script.python_problem((3, 9, 6))
    assert line and "\n" not in line
    assert "3.11" in line and "Python 3.9.6" in line
    assert f"uv run --no-project --python {LAUNCHER_PY} packaging/mcpb/build.py" in line
    assert script.python_problem((3, 11, 0)) is None and script.python_problem((3, 12, 13)) is None
    assert script.python_problem() is None  # this Python runs the tests


def test_python_3_9_stops_before_tomllib_with_that_line_and_no_traceback():
    """Fake the version: the guard sits before `import tomllib`, so the script stops with the line and exit code 1."""
    code = ("import runpy, sys\n"
            "sys.version_info = (3, 9, 6, 'final', 0)\n"
            "runpy.run_path(sys.argv[1], run_name='__main__')\n")
    r = subprocess.run([sys.executable, "-c", code, str(BUILD)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "Traceback" not in r.stderr and r.stdout == ""
    assert len(r.stderr.strip().splitlines()) == 1
    assert "Python 3.9.6" in r.stderr and f"uv run --no-project --python {LAUNCHER_PY} packaging/mcpb/build.py" in r.stderr


@pytest.fixture
def checkout(tmp_path):
    """A small copy of the repo (the files that the build reads) in a new git repo, with local files around it."""
    top = tmp_path / "repo"
    skip = shutil.ignore_patterns("__pycache__", "*.pyc", "runs", ".venv", "dist")
    for rel in ("topopt", "topopt_mcp", "examples", "tools", "packaging/mcpb"):
        shutil.copytree(ROOT / rel, top / rel, ignore=skip)
    (top / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "topopt-env.sh", top / "scripts" / "topopt-env.sh")
    for name in ("pyproject.toml", "requirements.lock", "requirements-macos-x86_64.lock", "CONTRACTS.md"):
        shutil.copy(ROOT / name, top / name)
    for pattern in ("LICENSE*", "THIRD_PARTY_NOTICES*"):   # the files that `license-files` of pyproject.toml names
        for path in ROOT.glob(pattern):
            shutil.copy(path, top / path.name)
    (top / ".gitignore").write_text("dist/\nscratch/\n__pycache__/\n")
    return top


def run_build(top: Path, out: Path):
    r = subprocess.run([sys.executable, str(top / "packaging" / "mcpb" / "build.py"), "--out", str(out)],
                       capture_output=True, text=True, cwd=top)
    assert r.returncode == 0, r.stdout + r.stderr
    return zipfile.ZipFile(next(out.glob("*.mcpb"))).namelist(), r.stderr


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_a_git_checkout_ships_tracked_files_only(checkout, tmp_path):
    git = ["git", "-C", str(checkout)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    (checkout / "examples" / "hook" / "my_notes.txt").write_text("local note")          # untracked
    (checkout / "topopt" / ".mcp.json").write_text("{}")                                  # untracked
    (checkout / "examples" / "hook" / "scratch").mkdir()
    (checkout / "examples" / "hook" / "scratch" / "big.npz").write_bytes(b"0" * 1000)    # ignored by .gitignore
    names, stderr = run_build(checkout, tmp_path / "out")
    assert "topopt_mcp/server.py" in names and "examples/hook/hook.yaml" in names
    assert not [n for n in names if "my_notes" in n or n.endswith(".mcp.json") or "scratch" in n]
    assert "left out 3 file(s) that git does not track" in stderr
    assert "my_notes.txt" in stderr and ".mcp.json" in stderr and "big.npz" in stderr


def test_a_folder_without_git_ships_every_file_and_says_so(checkout, tmp_path):
    (checkout / "examples" / "hook" / "my_notes.txt").write_text("local note")
    names, stderr = run_build(checkout, tmp_path / "out")  # no .git here: the tarball case
    assert "examples/hook/my_notes.txt" in names
    assert "no git checkout here" in stderr


@pytest.mark.parametrize("name", ["scripts/topopt-env.sh", "requirements.lock", "requirements-macos-x86_64.lock"])
def test_the_build_stops_when_a_file_of_the_launcher_is_missing(checkout, tmp_path, name):
    (checkout / name).unlink()
    r = subprocess.run([sys.executable, str(checkout / "packaging" / "mcpb" / "build.py"), "--out", str(tmp_path / "out")],
                       capture_output=True, text=True, cwd=checkout)
    assert r.returncode != 0 and not list((tmp_path / "out").glob("*.mcpb"))
    assert name in r.stderr


def test_the_zip_has_every_snapshot_file_of_the_launcher(bundle):
    """The status server answers from these files (#370). Without one, for example `legacy.initialize.json`, it answers `initialize`
    with an error, and the first start in Claude Desktop fails."""
    names = set(zipfile.ZipFile(bundle).namelist())
    snapshot = {p.relative_to(ROOT).as_posix() for p in (ROOT / "topopt_mcp" / "launcher").glob("*.json")}
    assert len(snapshot) == 10 and snapshot <= names


def test_the_zip_has_only_the_launcher_of_the_scripts(bundle):
    """scripts/ holds the launcher and the developer and CI scripts. Only the launcher goes in."""
    names = zipfile.ZipFile(bundle).namelist()
    assert {n for n in names if n.startswith("scripts/")} == {"scripts/topopt-env.sh"}


def test_the_zip_has_only_the_tools_that_the_texts_name(bundle):
    """tools/ ships an allow-list: the developer scripts (fake_run.py, benchmarks) stay out."""
    names = zipfile.ZipFile(bundle).namelist()
    in_zip = {n for n in names if n.startswith("tools/")}
    assert in_zip == {"tools/check_part.py"}
    named = set()
    for path in [*(ROOT / "topopt_mcp" / "roles").glob("*.md"), ROOT / "topopt_mcp" / "server.py"]:
        named |= set(re.findall(r"tools/[A-Za-z_]+\.(?:py|sh)", path.read_text()))
    assert named and named <= in_zip, f"a role text or the server names a tool that the zip lacks: {named - in_zip}"


def test_scripts_in_the_zip_are_executable_and_the_other_files_are_not(bundle):
    modes = {i.filename: (i.external_attr >> 16) & 0o777 for i in zipfile.ZipFile(bundle).infolist()}
    assert modes["topopt/native/build.sh"] == 0o755 and modes["scripts/topopt-env.sh"] == 0o755
    assert {m for n, m in modes.items() if not n.endswith(".sh")} == {0o644}


def test_the_build_is_reproducible(bundle, tmp_path):
    again = build_bundle(tmp_path)
    assert again.read_bytes() == bundle.read_bytes()


def test_the_bundle_serves_the_same_role_texts_as_the_repo(bundle, tmp_path):
    """Unpack the zip and read the roles from it with the code of the zip (no dependency but the standard library)."""
    target = tmp_path / "unpacked"
    zipfile.ZipFile(bundle).extractall(target)
    code = ("import json, sys\nsys.path.insert(0, sys.argv[1])\nfrom topopt_mcp import team\n"
            "print(json.dumps({k: v.text for k, v in team.read_roles().items()}))")
    out = subprocess.run([sys.executable, "-I", "-c", code, str(target)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    from topopt_mcp import team
    assert json.loads(out.stdout) == {k: v.text for k, v in team.read_roles(ROOT / "topopt_mcp" / "roles").items()}


@pytest.mark.skipif(shutil.which("mcpb") is None, reason="the mcpb CLI is not installed (npm install -g @anthropic-ai/mcpb)")
def test_the_manifest_passes_the_mcpb_schema(bundle, tmp_path):
    (tmp_path / "manifest.json").write_bytes(zipfile.ZipFile(bundle).read("manifest.json"))
    r = subprocess.run(["mcpb", "validate", str(tmp_path / "manifest.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# --- the launch (network) ----------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("TOPOPT_TEST_MCPB_LAUNCH") != "1",
                    reason="set TOPOPT_TEST_MCPB_LAUNCH=1 (needs the network: about 200 MB of packages)")
def test_unpacked_bundle_starts_with_the_command_of_the_manifest(bundle, tmp_path):
    """Run the `mcp_config` command of the manifest on the unpacked zip, as the host does, and talk MCP to it: all the tools at
    once, the setup status during the setup, then the answers of the server in the same session (#206, #370). The machine is a
    clean one: an empty home folder (no uv cache, no Python of uv) and the short PATH of a Mac app, so the launcher gets its own uv."""
    import time

    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    target = tmp_path / "unpacked"
    zipfile.ZipFile(bundle).extractall(target)       # like an unzip that drops the modes: `/bin/sh` runs the launcher anyway
    cfg = read_manifest(bundle)["server"]["mcp_config"]
    work = tmp_path / "work"
    subst = {"${__dirname}": str(target), "${user_config.workspace}": str(work), "${HOME}": str(tmp_path / "home")}

    def sub(text):
        for k, v in subst.items():
            text = text.replace(k, v)
        return text

    # The host gives none of these: the default folder of the environment, no wait, and no PYTHONPATH (a test runner can put the
    # checkout there, and the server would then import the code of the checkout instead of the code of the bundle). No setting of
    # uv and no cache folder of the runner either: the home folder is empty, as on a new machine.
    home = tmp_path / "home"
    home.mkdir()
    short_path = "/usr/bin:/bin"
    env = {k: v for k, v in os.environ.items()
           if k not in {"TOPOPT_ENV_DIR", "EVAL_TOPOPT_ENV_DIR", "CLAUDE_PLUGIN_DATA", "TOPOPT_MCP_WAIT", "PYTHONPATH"}
           and not k.startswith(("UV_", "XDG_"))}
    env.update(HOME=str(home), PATH=short_path)
    params = StdioServerParameters(command=cfg["command"], args=[sub(a) for a in cfg["args"]],
                                   env={**env, **{k: sub(v) for k, v in cfg.get("env", {}).items()}}, cwd=str(tmp_path))
    expected = [t["name"] for t in read_manifest(bundle)["tools"]]

    async def go():
        async with Client(params) as client:
            at_once = [t.name for t in (await client.list_tools()).tools]
            prompts = {p.name for p in (await client.list_prompts()).prompts}
            first = (await client.call_tool("environment", {})).content[0].text
            end = time.monotonic() + 840
            while True:
                res = await client.call_tool("environment", {})
                try:
                    answer = res.structured_content or json.loads(res.content[0].text)
                except ValueError:
                    answer = None                                  # the text of the setup status
                if isinstance(answer, dict) and "code_folder" in answer:
                    break
                assert time.monotonic() < end, res.content[0].text
                await asyncio.sleep(5)
            guide = (await client.call_tool("team_guide", {"role": "cad"})).content[0].text
            later = [t.name for t in (await client.list_tools()).tools]
            return at_once, prompts, first, answer, guide, later

    at_once, prompts, first, env_answer, guide, later = asyncio.run(asyncio.wait_for(go(), 900))
    assert sorted(at_once) == sorted(expected) == sorted(later) and prompts == set(IDS)
    assert first.startswith("topopt is still setting up its Python environment")   # a new folder: the launcher answers first
    assert guide.startswith("You are the CAD engineer")
    assert Path(env_answer["code_folder"]).resolve() == target.resolve()
    assert env_answer["work_folder"]["path"] == str(work.resolve()) and env_answer["work_folder"]["source"] == "TOPOPT_WORKSPACE"
    # no uv on the short PATH (and the launcher adds only these folders): the launcher downloaded its pinned uv and checked it
    if shutil.which("uv", path=f"{short_path}:/opt/homebrew/bin:/usr/local/bin") is None:
        log = (target / ".plugin-env" / "setup.log").read_text()
        assert "uv is not installed: downloading uv" in log and re.search(r"uv \S+ is ready", log), log[:2000]
    # the environment is in the extension folder, and every installed package that requirements.lock names has the version of the lock
    python = target / ".plugin-env" / "venv" / "bin" / "python"
    listing = subprocess.run([str(python), "-c", "import importlib.metadata as m, json; "
                              "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))"],
                             capture_output=True, text=True, check=True).stdout
    installed = {k.lower().replace("_", "-"): v for k, v in json.loads(listing).items()}
    lock = {k.lower().replace("_", "-"): v for k, v in
            (ln.strip().split("==") for ln in (ROOT / "requirements.lock").read_text().splitlines() if "==" in ln)}
    assert {k: v for k, v in installed.items() if k != "topopt"} == lock      # the environment of the plugin: the lock and topopt
    # the extras stay in the pyproject, and the launcher installs none of them
    assert not {"pypardiso", "mkl", "gmsh"} & set(installed)
