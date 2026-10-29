"""The work folder rule (topopt/workfolder.py, CONTRACTS.md section 10.1): the resolver, the notes in the replies of the MCP server,
the folder of `runinfo.key_root`, the runs in an old place, and the real server over stdio with the text that Claude Desktop gave.

Nothing here starts a solver. The in-process tests replace the environment probe, so they do not touch OpenCL. The stdio test runs
the real `environment` tool: run pytest with a CPU OpenCL device (pocl) if the machine has a GPU that you do not want to use.
"""
import asyncio
import datetime
import inspect
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(Path(__file__).parent))

from helpers import recorded_hook_run  # noqa: E402
from topopt import runinfo, workfolder  # noqa: E402
from topopt_mcp import server  # noqa: E402

DESKTOP_TEXT = "Open Settings > Extensions > topopt, choose the Work folder with the folder button"
NOT_ROOT = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can write in any folder")


@pytest.fixture
def places(tmp_path):
    """A fake home folder and a fake current folder (the extension folder of the host)."""
    home, ext = tmp_path / "home", tmp_path / "ext"
    home.mkdir()
    ext.mkdir()
    return home, ext


def resolve(places, **env):
    home, ext = places
    return workfolder.resolve(env, cwd=ext, home=home)


# --- the resolver ----------------------------------------------------------------------------------------------

def test_the_owner_value_is_replaced_and_the_server_says_so(places):
    home, _ = places
    f = resolve(places, TOPOPT_WORKSPACE="${DOCUMENTS}/topopt")
    assert f.path == (home / "Documents" / "topopt").resolve()
    assert f.source == "TOPOPT_WORKSPACE" and f.raw == "${DOCUMENTS}/topopt" and f.expanded == ["${DOCUMENTS}"]
    assert f.fix_needed == [] and len(f.notes) == 1
    assert '"${DOCUMENTS}/topopt"' in f.notes[0] and str(f.path) in f.notes[0]


@pytest.mark.parametrize("text, tail, used", [
    ("${HOME}/x", "x", ["${HOME}"]),
    ("~/x", "x", ["~"]),
    ("~", "", ["~"]),
    ("${DESKTOP}/x", "Desktop/x", ["${DESKTOP}"]),
    ("${DOWNLOADS}/x", "Downloads/x", ["${DOWNLOADS}"]),
    ("${HOME}${/}x", "x", ["${HOME}", "${/}"]),
    ("${HOME}${pathSeparator}x", "x", ["${HOME}", "${pathSeparator}"]),
    ("${DOCUMENTS}/x ", "Documents/x ", ["${DOCUMENTS}"]),          # a path keeps its blanks (a folder name can end in a blank)
])
def test_the_known_variables_and_the_tilde(places, text, tail, used):
    home, _ = places
    f = resolve(places, TOPOPT_WORKSPACE=text)
    assert f.path == (home / tail).resolve() and f.expanded == used and f.source == "TOPOPT_WORKSPACE"
    assert f.fix_needed == []
    assert bool(f.notes) == any(u.startswith("${") for u in used)   # a tilde was always replaced: no note for it


@pytest.mark.parametrize("value", [None, "", "   ", "${user_config.workspace}", " ${user_config.other_key} "])
def test_an_option_that_is_not_set_gives_the_default_with_no_note(places, value):
    home, ext = places
    env = {"TOPOPT_DEFAULT_WORKSPACE": str(home / "Documents" / "topopt")}
    if value is not None:
        env["TOPOPT_WORKSPACE"] = value
    f = workfolder.resolve(env, cwd=ext, home=home)
    assert f.path == (home / "Documents" / "topopt").resolve() and f.source == "TOPOPT_DEFAULT_WORKSPACE"
    assert f.fix_needed == [] and f.notes == []


def test_the_default_gets_the_same_replacement(places):
    home, _ = places
    f = resolve(places, TOPOPT_WORKSPACE="${user_config.workspace}", TOPOPT_DEFAULT_WORKSPACE="${HOME}/Documents/topopt")
    assert f.path == (home / "Documents" / "topopt").resolve() and f.source == "TOPOPT_DEFAULT_WORKSPACE"
    assert f.fix_needed == [] and len(f.notes) == 1 and "TOPOPT_DEFAULT_WORKSPACE" in f.notes[0]


def test_an_unknown_variable_gives_the_default_and_a_fix_needed_text(places):
    home, _ = places
    f = resolve(places, TOPOPT_WORKSPACE="${FOO}/x", TOPOPT_DEFAULT_WORKSPACE="${HOME}/Documents/topopt")
    assert f.path == (home / "Documents" / "topopt").resolve() and f.source == "TOPOPT_DEFAULT_WORKSPACE"
    assert f.raw == "${FOO}/x"
    assert len(f.fix_needed) == 1
    text = f.fix_needed[0]
    assert '"${FOO}/x"' in text and str(f.path) in text            # the cause (the raw text), the folder in use
    assert DESKTOP_TEXT in text and "Cmd+Q" in text and "A new chat or a new Code tab session is not enough" in text   # the fix of the Desktop extension


def test_blanks_are_cut_only_to_test_for_an_empty_option(places):
    home, _ = places
    default = str(home / "Documents" / "topopt")
    for value in ("   ", "\t", " ${user_config.workspace}\n"):                    # not set: the default
        f = resolve(places, TOPOPT_WORKSPACE=value, TOPOPT_DEFAULT_WORKSPACE=default)
        assert f.source == "TOPOPT_DEFAULT_WORKSPACE" and f.fix_needed == []
    f = resolve(places, TOPOPT_WORKSPACE=str(home / "work "))                      # the blank is part of the name
    assert f.path.name == "work " and f.source == "TOPOPT_WORKSPACE"
    g = resolve(places, TOPOPT_WORKSPACE="${HOME}/work ")
    assert g.path == (home / "work ").resolve() and f.fix_needed == [] and g.fix_needed == []
    h = resolve(places, TOPOPT_WORKSPACE="${FOO}/x ")                              # the text in the reply is the text that was given
    assert '"${FOO}/x "' in h.fix_needed[0]


def test_a_tilde_name_that_cannot_be_replaced_is_a_value_that_cannot_be_used(places):
    home, ext = places
    default = str(home / "Documents" / "topopt")
    f = resolve(places, TOPOPT_WORKSPACE="~no-such-user-p9a/x", TOPOPT_DEFAULT_WORKSPACE=default)
    assert f.path == (home / "Documents" / "topopt").resolve() and f.source == "TOPOPT_DEFAULT_WORKSPACE"
    assert len(f.fix_needed) == 1 and '"~no-such-user-p9a/x"' in f.fix_needed[0] and "starts with ~" in f.fix_needed[0]
    assert "Claude Desktop did not replace it" not in f.fix_needed[0]             # it is not a variable of the host
    assert DESKTOP_TEXT in f.fix_needed[0] and f.notes == []                      # not a relative path in the current folder
    g = resolve(places, TOPOPT_WORKSPACE="~no-such-user-p9a/x")
    assert g.path == ext.resolve() and g.source == "current folder" and len(g.fix_needed) == 1
    assert workfolder.FIX_OTHER in g.fix_needed[0]


def test_two_values_that_cannot_be_used_are_both_named_and_the_fix_is_said_once(places, monkeypatch, probe):
    home, ext = places
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(ext)
    monkeypatch.setenv("TOPOPT_WORKSPACE", "${FOO}/x")
    monkeypatch.setenv("TOPOPT_DEFAULT_WORKSPACE", "${BAR}/y")
    f = workfolder.resolve()
    assert f.path == ext.resolve() and f.source == "current folder" and len(f.fix_needed) == 1
    text = f.fix_needed[0]
    assert '"${FOO}/x"' in text and '"${BAR}/y"' in text and "TOPOPT_DEFAULT_WORKSPACE" in text
    assert text.count(DESKTOP_TEXT) == 1 and text.count("The server used") == 1 and text.endswith(workfolder.FIX_DESKTOP)
    for reply in (server.run_status("nothing"), server.environment()):
        assert reply["work_folder_note"].count(DESKTOP_TEXT) == 1                 # in the reply too
    assert server.start_line(f).count(DESKTOP_TEXT) == 1


def test_the_variables_are_case_sensitive_as_in_the_spec(places):
    f = resolve(places, TOPOPT_WORKSPACE="${Documents}/topopt")
    assert f.fix_needed and f.raw == "${Documents}/topopt" and f.source == "current folder"


def test_an_unknown_variable_without_a_default_gives_the_current_folder(places):
    _, ext = places
    f = resolve(places, TOPOPT_WORKSPACE="${FOO}/x")
    assert f.path == ext.resolve() and f.source == "current folder"
    assert len(f.fix_needed) == 1 and '"${FOO}/x"' in f.fix_needed[0]
    assert workfolder.FIX_OTHER in f.fix_needed[0]
    assert "Settings > Extensions" not in f.fix_needed[0]          # no Desktop extension started this server


def test_a_part_of_a_user_config_variable_is_not_an_empty_option(places):
    f = resolve(places, TOPOPT_WORKSPACE="${user_config.workspace}/x")
    assert f.fix_needed and f.source == "current folder"


def test_a_default_that_cannot_be_used_is_named(places):
    _, ext = places
    f = resolve(places, TOPOPT_DEFAULT_WORKSPACE="${FOO}")
    assert f.path == ext.resolve() and f.source == "current folder"
    assert len(f.fix_needed) == 1 and "TOPOPT_DEFAULT_WORKSPACE" in f.fix_needed[0]


def test_a_relative_path_starts_at_the_current_folder_and_the_server_says_so(places):
    _, ext = places
    f = resolve(places, TOPOPT_WORKSPACE="Documents/topopt")
    assert f.path == (ext / "Documents" / "topopt").resolve() and f.source == "TOPOPT_WORKSPACE"   # as before
    assert f.fix_needed == [] and len(f.notes) == 1 and "not a full path" in f.notes[0] and str(f.path) in f.notes[0]


def test_a_relative_path_under_the_desktop_extension_says_what_the_current_folder_is(places):
    home, ext = places
    f = resolve(places, TOPOPT_WORKSPACE="Documents/topopt", TOPOPT_DEFAULT_WORKSPACE=str(home / "Documents" / "topopt"))
    assert f.path == (ext / "Documents" / "topopt").resolve() and f.fix_needed == [] and len(f.notes) == 1   # still an info note, as before
    note = f.notes[0]
    assert "The current folder of the server is the folder of the extension, and an update of the extension can remove it." in note
    assert f"To keep the runs, type a full path in the Work folder setting, for example {home / 'Documents' / 'topopt'}." in note
    assert str(f.path) in note and "not a full path" in note
    other = resolve(places, TOPOPT_WORKSPACE="Documents/topopt").notes[0]       # a client that is not the extension
    assert "folder of the extension" not in other and "for example" not in other and "A full path does not depend" in other


def test_no_env_var_gives_the_current_folder_and_no_text(places):
    _, ext = places
    f = resolve(places)
    assert f.path == ext.resolve() and f.source == "current folder" and f.raw is None and f.expanded == []
    assert f.fix_needed == [] and f.notes == [] and f.old_runs is None and f.exists and f.writable


def test_a_plain_absolute_path_is_the_same_as_before(places, tmp_path):
    target = tmp_path / "some" / "where"
    f = resolve(places, TOPOPT_WORKSPACE=str(target), TOPOPT_DEFAULT_WORKSPACE="/ignored")
    assert f.path == Path(str(target)).expanduser().resolve()      # the rule of the old workspace()
    assert f.source == "TOPOPT_WORKSPACE" and f.expanded == [] and f.fix_needed == [] and f.notes == []
    assert f.exists is False and f.writable is True                # not there yet, and it can be made


def test_resolve_creates_nothing(places, tmp_path):
    f = resolve(places, TOPOPT_WORKSPACE="${DOCUMENTS}/topopt/deep")
    assert not f.path.exists() and not (places[0] / "Documents").exists()


def test_check_false_reads_the_text_only(places):
    f = workfolder.resolve({"TOPOPT_WORKSPACE": "${DOCUMENTS}/x"}, cwd=places[1], home=places[0], check=False)
    assert f.exists is None and f.writable is None and f.old_runs is None and f.blocked is None


@NOT_ROOT
def test_a_folder_that_cannot_be_written_is_a_fix_needed_text(places, tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        f = resolve(places, TOPOPT_WORKSPACE=str(locked))
        assert f.exists is True and f.writable is False and f.blocked and f.blocked in f.fix_needed
        assert str(locked) in f.blocked and "cannot write" in f.blocked
        g = resolve(places, TOPOPT_WORKSPACE=str(locked / "new" / "deeper"))   # it cannot be made
        assert g.exists is False and g.writable is False and "cannot create" in g.blocked and str(locked) in g.blocked
    finally:
        locked.chmod(0o700)


def deny(monkeypatch, number=None):
    """Every `is_dir` below a folder named Documents fails with the error that macOS gives when the user said no to a folder."""
    import errno
    real = Path.is_dir

    def is_dir(self, *args, **kwargs):
        if "Documents" in self.parts:
            raise PermissionError(number or errno.EPERM, "Operation not permitted")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_dir", is_dir)


def test_a_macos_privacy_block_under_the_home_folder_names_the_setting(places, monkeypatch):
    home, _ = places
    deny(monkeypatch)
    desktop = {"TOPOPT_WORKSPACE": "${DOCUMENTS}/topopt", "TOPOPT_DEFAULT_WORKSPACE": "${HOME}/Documents/topopt"}
    with monkeypatch.context() as mac:
        mac.setattr(sys, "platform", "darwin")
        f = workfolder.resolve(desktop, cwd=places[1], home=home)
    assert f.exists is False and f.writable is False and f.blocked in f.fix_needed
    assert "the server cannot read it (Operation not permitted)" in f.blocked
    assert "System Settings > Privacy & Security > Files and Folders" in f.blocked
    assert DESKTOP_TEXT in f.blocked and str(f.path) in f.blocked                  # the cause, the setting, then the fix
    assert f.blocked.count("System Settings") == 1                                  # one option text
    # Linux, or another error number: the cause stays as it was, with no macOS text
    with monkeypatch.context() as other:
        other.setattr(sys, "platform", "linux")                  # not the platform of the machine: a Mac would add the macOS text
        linux = workfolder.resolve(desktop, cwd=places[1], home=home)
    assert "Operation not permitted" in linux.blocked and "System Settings" not in linux.blocked
    deny(monkeypatch, number=13)
    with monkeypatch.context() as mac:
        mac.setattr(sys, "platform", "darwin")
        assert "System Settings" not in workfolder.resolve(desktop, cwd=places[1], home=home).blocked
    # a folder outside the home folder has no macOS text, also on a Mac
    deny(monkeypatch)
    outside = places[1].parent / "Documents" / "kept"
    with monkeypatch.context() as mac:
        mac.setattr(sys, "platform", "darwin")
        g = workfolder.resolve({"TOPOPT_WORKSPACE": str(outside)}, cwd=places[1], home=home)
    assert g.blocked and "Operation not permitted" in g.blocked and "System Settings" not in g.blocked


def test_a_file_in_place_of_the_folder_is_a_fix_needed_text(places, tmp_path):
    (tmp_path / "afile").write_text("x")
    f = resolve(places, TOPOPT_WORKSPACE=str(tmp_path / "afile"))
    assert f.writable is False and "is a file" in f.blocked
    g = resolve(places, TOPOPT_WORKSPACE=str(tmp_path / "afile" / "sub"))
    assert g.writable is False and "is a file" in g.blocked


def test_a_value_that_expands_to_a_file_is_a_fix_needed_text_and_a_refusal(places, monkeypatch, probe):
    home, ext = places
    (home / "Documents").mkdir()
    (home / "Documents" / "afile").write_text("x")
    (home / "p.yaml").write_text("name: x\n")
    env = {"TOPOPT_WORKSPACE": "${DOCUMENTS}/afile", "TOPOPT_DEFAULT_WORKSPACE": "${HOME}/Documents/topopt"}
    f = workfolder.resolve(env, cwd=ext, home=home)
    assert f.source == "TOPOPT_WORKSPACE" and f.path == (home / "Documents" / "afile").resolve()   # the value is usable text: no other folder
    assert f.exists is True and f.writable is False and "is a file, not a folder" in f.blocked and f.blocked in f.fix_needed
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(ext)
    for name, text in env.items():
        monkeypatch.setenv(name, text)
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    server._current.clear()
    assert "is a file, not a folder" in server.run_topopt(str(home / "p.yaml"))["error"]
    b = server.build_regions({"name": "a", "regions": {}})
    assert b["ok"] is False and "is a file, not a folder" in b["error"] and "proc" not in server._current
    assert "is a file, not a folder" in server.run_status("nothing")["work_folder_note"]


def test_the_facts_have_the_keys_of_the_contract(places):
    facts = resolve(places, TOPOPT_WORKSPACE="${DOCUMENTS}/topopt").facts()
    assert set(facts) == {"path", "source", "raw", "expanded", "exists", "writable", "fix_needed", "notes", "old_runs"}
    json.dumps(facts)


# --- the scratch workspace of a Code tab session (observed 2026-10-23, Claude Desktop 3.6556.2) ------------------------

def scratch_folder(base: Path) -> Path:
    folder = base / "Library" / "Application Support" / "Claude" / "scratch-workspaces" / "id1" / "id1" / "scratch-2026-10-23-ab12cd"
    folder.mkdir(parents=True)
    return folder


def test_a_scratch_workspace_of_claude_desktop_is_an_info_note_not_a_refusal(places, tmp_path):
    home, _ = places
    scratch = scratch_folder(tmp_path / "mac")
    f = workfolder.resolve({}, cwd=scratch, home=home)          # the Code tab starts the plugin server in the scratch folder
    assert f.path == scratch.resolve() and f.source == "current folder"
    assert f.fix_needed == [] and f.blocked is None and f.writable is True      # a fact from the path: nothing is refused
    assert len(f.notes) == 1 and str(scratch.resolve()) in f.notes[0]
    assert "removes this folder when the session ends" in f.notes[0]
    assert "Open the session in a project folder that you keep. " + workfolder.change_text({}) in f.notes[0]
    # a folder deeper inside it is the same; TOPOPT_WORKSPACE inside it too
    assert workfolder.resolve({"TOPOPT_WORKSPACE": str(scratch / "work")}, cwd=home, home=home).notes
    # a set folder outside of it has no note, also when the current folder is the scratch folder
    elsewhere = workfolder.resolve({"TOPOPT_WORKSPACE": str(tmp_path / "kept")}, cwd=scratch, home=home)
    assert elsewhere.notes == []
    # the names alone are not enough: the two parts follow each other, and something follows them
    other = tmp_path / "scratch-workspaces" / "Claude" / "x"
    other.mkdir(parents=True)
    assert workfolder.resolve({}, cwd=other, home=home).notes == []
    last = tmp_path / "mac2" / "Claude" / "scratch-workspaces"
    last.mkdir(parents=True)
    assert workfolder.resolve({}, cwd=last, home=home).notes == []


def test_the_scratch_workspace_note_is_in_three_replies_only(tmp_path, monkeypatch, probe):
    scratch = scratch_folder(tmp_path / "mac")
    monkeypatch.chdir(scratch)
    for name in ("TOPOPT_WORKSPACE", "TOPOPT_DEFAULT_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)
    text = workfolder.resolve().notes[0]
    assert server.list_runs()["work_folder_note"] == text and server.environment()["work_folder_note"] == text
    assert server.environment()["work_folder"]["notes"] == [text]
    assert "work_folder_note" not in server.run_status("nothing")


# --- runs in the old place -------------------------------------------------------------------------------------

def fake_old_run(ext: Path, text: str, name="r1"):
    old = ext / text
    (old / "runs" / name).mkdir(parents=True)
    (old / "runs" / name / "status.json").write_text(json.dumps({"state": "done"}))
    return old


def test_runs_in_the_old_place_are_reported_and_not_moved(places):
    home, ext = places
    old = fake_old_run(ext, "${DOCUMENTS}/topopt")
    f = resolve(places, TOPOPT_WORKSPACE="${DOCUMENTS}/topopt", TOPOPT_DEFAULT_WORKSPACE=str(home / "Documents" / "topopt"))
    assert f.old_runs == {"path": str(old.resolve()), "run_folders": 1}
    assert any(n.startswith(f"Runs from before are in {old.resolve()}.") and "An update of the extension can remove this folder" in n
               and "Copy them to the work folder if you want to keep them." in n for n in f.notes)
    assert (old / "runs" / "r1" / "status.json").is_file() and not f.path.exists()   # nothing moved or copied, nothing made


def test_the_sentence_about_an_update_of_the_extension_is_only_for_the_desktop_extension(places):
    _, ext = places
    old = fake_old_run(ext, "${DOCUMENTS}/topopt")
    f = resolve(places, TOPOPT_WORKSPACE="${DOCUMENTS}/topopt")             # no TOPOPT_DEFAULT_WORKSPACE: another client
    note = [n for n in f.notes if n.startswith("Runs from before")]
    assert note == [f"Runs from before are in {old.resolve()}. Copy them to the work folder if you want to keep them."]


def test_runs_in_the_old_place_of_an_empty_option(places):
    _, ext = places
    old = fake_old_run(ext, "${user_config.workspace}")
    f = resolve(places, TOPOPT_WORKSPACE="${user_config.workspace}", TOPOPT_DEFAULT_WORKSPACE=str(places[0] / "Documents" / "topopt"))
    assert f.old_runs and f.old_runs["path"] == str(old.resolve())


def test_the_user_case_a_typed_folder_and_runs_in_the_text_of_the_first_default(places):
    """The user typed ~/Documents/topopt after the update. The runs of the first hours are in `<extension>/${DOCUMENTS}/topopt`."""
    home, ext = places
    old = fake_old_run(ext, "${DOCUMENTS}/topopt")
    desktop = str(home / "Documents" / "topopt")
    f = resolve(places, TOPOPT_WORKSPACE="~/Documents/topopt", TOPOPT_DEFAULT_WORKSPACE=desktop)
    assert f.source == "TOPOPT_WORKSPACE" and f.path == (home / "Documents" / "topopt").resolve() and f.fix_needed == []
    assert f.old_runs == {"path": str(old.resolve()), "run_folders": 1}
    assert [n for n in f.notes if n.startswith("Runs from before")] == [
        f"Runs from before are in {old.resolve()}. An update of the extension can remove this folder. "
        "Copy them to the work folder if you want to keep them."]
    assert resolve(places, TOPOPT_WORKSPACE="~/Documents/topopt").old_runs is None       # another client: its text had no ${DOCUMENTS}
    assert resolve(places, TOPOPT_WORKSPACE="", TOPOPT_DEFAULT_WORKSPACE=desktop).old_runs == f.old_runs   # an empty option too


def test_old_runs_in_two_places_with_more_than_one_run_are_all_reported(places):
    home, ext = places
    first = fake_old_run(ext, "${DESKTOP}/topopt", "r1")
    fake_old_run(ext, "${DESKTOP}/topopt", "r2")
    second = fake_old_run(ext, "${DOCUMENTS}/topopt", "r3")
    f = resolve(places, TOPOPT_WORKSPACE="${DESKTOP}/topopt", TOPOPT_DEFAULT_WORKSPACE=str(home / "Documents" / "topopt"))
    assert f.old_runs == {"path": str(first.resolve()), "run_folders": 2,
                          "also": [{"path": str(second.resolve()), "run_folders": 1}]}
    said = [n for n in f.notes if n.startswith("Runs from before")]
    assert len(said) == 2 and str(first.resolve()) in said[0] and str(second.resolve()) in said[1]
    json.dumps(f.facts())


def test_the_folder_in_use_is_not_an_old_place_also_through_a_link(places):
    home, ext = places
    docs = home / "Documents"
    (docs / "topopt" / "runs" / "r1").mkdir(parents=True)
    (docs / "topopt" / "runs" / "r1" / "status.json").write_text("{}")
    (ext / "${DOCUMENTS}").symlink_to(docs)             # <extension>/${DOCUMENTS}/topopt is the folder in use
    f = resolve(places, TOPOPT_WORKSPACE="${DOCUMENTS}/topopt", TOPOPT_DEFAULT_WORKSPACE=str(docs / "topopt"))
    assert f.path == (docs / "topopt").resolve() and f.old_runs is None
    assert not [n for n in f.notes if n.startswith("Runs from before")]


def test_two_texts_that_lead_to_one_place_report_it_once(places):
    home, ext = places
    docs = home / "Documents"
    fake_old_run(docs, "topopt")
    (ext / "${DOCUMENTS}").symlink_to(docs)
    (ext / "${DESKTOP}").symlink_to(docs)
    f = resolve(places, TOPOPT_WORKSPACE="${DESKTOP}/topopt", TOPOPT_DEFAULT_WORKSPACE=str(home / "somewhere"))
    assert f.old_runs == {"path": str((docs / "topopt").resolve()), "run_folders": 1}      # no "also": it is one place
    assert len([n for n in f.notes if n.startswith("Runs from before")]) == 1


def test_no_old_runs_when_there_are_none_or_the_folder_is_the_same(places):
    _, ext = places
    assert resolve(places, TOPOPT_WORKSPACE="${DOCUMENTS}/topopt").old_runs is None            # the old folder is not there
    (ext / "${DOCUMENTS}" / "topopt").mkdir(parents=True)
    assert resolve(places, TOPOPT_WORKSPACE="${DOCUMENTS}/topopt").old_runs is None            # there, and empty
    fake_old_run(ext, "plain/topopt")
    assert resolve(places, TOPOPT_WORKSPACE="plain/topopt").old_runs is None                   # a text with no ${ : it is the folder in use


# --- the same folder in the server and in the checkpoint ----------------------------------------------------------

CASES = [
    {},
    {"TOPOPT_WORKSPACE": "${DOCUMENTS}/topopt"},
    {"TOPOPT_WORKSPACE": "${HOME}/x", "TOPOPT_DEFAULT_WORKSPACE": "${HOME}/d"},
    {"TOPOPT_WORKSPACE": "~/x"},
    {"TOPOPT_WORKSPACE": "${user_config.workspace}", "TOPOPT_DEFAULT_WORKSPACE": "${HOME}/Documents/topopt"},
    {"TOPOPT_WORKSPACE": ""},
    {"TOPOPT_WORKSPACE": "${FOO}/x", "TOPOPT_DEFAULT_WORKSPACE": "${HOME}/Documents/topopt"},
    {"TOPOPT_WORKSPACE": "${FOO}/x"},
    {"TOPOPT_WORKSPACE": "rel/x"},
    {"TOPOPT_WORKSPACE": "ABS/x"},
]


@pytest.mark.parametrize("env", CASES)
def test_key_root_gives_the_folder_of_the_server(env, places, monkeypatch, tmp_path):
    home, ext = places
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(ext)
    for name in ("TOPOPT_WORKSPACE", "TOPOPT_DEFAULT_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)
    for name, text in env.items():
        monkeypatch.setenv(name, str(tmp_path / "abs" / "x") if text == "ABS/x" else text)
    yaml = server.workspace() / "runs" / "p" / "c.yaml"
    assert runinfo.key_root(yaml) == server.workspace()            # a YAML in the work folder: the work folder
    assert runinfo.key_root(tmp_path / "elsewhere" / "c.yaml") == (tmp_path / "elsewhere").resolve()
    assert server.workspace() == workfolder.resolve().path


# --- the replies of the server -------------------------------------------------------------------------------------

@pytest.fixture
def desktop(places, monkeypatch):
    """The server as the Desktop extension starts it on a Mac that has the old setting `${DOCUMENTS}/topopt`."""
    home, ext = places
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(ext)
    monkeypatch.setenv("TOPOPT_WORKSPACE", "${DOCUMENTS}/topopt")
    monkeypatch.setenv("TOPOPT_DEFAULT_WORKSPACE", "${HOME}/Documents/topopt")
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    server._current.clear()
    yield home / "Documents" / "topopt"
    server._current.clear()


@pytest.fixture
def probe(monkeypatch):
    """Replace the environment probe (OpenCL and more) with a small dict; the new keys are made by the server."""
    import topopt.environment as env_module
    monkeypatch.setattr(env_module, "environment_data", lambda yaml=None: {"python": {"executable": sys.executable}, "ok": None})
    monkeypatch.delenv("TOPOPT_PYTHON", raising=False)


def test_environment_has_the_work_folder_and_the_server(desktop, probe):
    r = server.environment()
    wf = r["work_folder"]
    assert wf["path"] == str(desktop.resolve()) and wf["source"] == "TOPOPT_WORKSPACE" and wf["raw"] == "${DOCUMENTS}/topopt"
    assert wf["expanded"] == ["${DOCUMENTS}"] and wf["writable"] is True and wf["fix_needed"] == [] and len(wf["notes"]) == 1
    assert r["server"]["pid"] == os.getpid()
    started = datetime.datetime.strptime(r["server"]["started_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    assert started <= datetime.datetime.now(datetime.timezone.utc)
    assert r["server"]["started_utc"] == server.STARTED_UTC       # taken once, at import
    assert r["work_folder_note"] == wf["notes"][0]


def test_started_utc_is_the_time_taken_once_at_import(desktop, probe, monkeypatch):
    monkeypatch.setattr(server, "STARTED_UTC", "2001-02-03T04:05:06Z")
    assert server.environment()["server"] == {"pid": os.getpid(), "started_utc": "2001-02-03T04:05:06Z"}   # no new time at each call


def test_list_runs_and_run_topopt_carry_the_info_note(desktop, probe, tmp_path):
    note = workfolder.resolve().notes[0]
    assert server.list_runs()["work_folder_note"] == note
    (desktop).mkdir(parents=True)
    (desktop / "p.yaml").write_text("name: x\n")
    r = server.run_topopt("p.yaml")
    assert r["work_folder_note"] == note and r["workspace"] == str(desktop.resolve())
    server._current["proc"].wait()


def test_a_tool_without_a_fix_needed_text_has_no_note(desktop, probe):
    assert "work_folder_note" not in server.run_status("nothing")         # the info note is in three tools only
    assert "work_folder_note" not in server.case_status("nothing.yaml") and "work_folder_note" not in server.run_history()


def test_the_normal_case_has_no_note_in_any_reply(places, monkeypatch, probe):
    home, ext = places
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(ext)
    monkeypatch.delenv("TOPOPT_WORKSPACE", raising=False)
    monkeypatch.setenv("TOPOPT_DEFAULT_WORKSPACE", str(home / "Documents" / "topopt"))   # the extension with an empty option (Desktop replaced ${HOME})
    for reply in (server.list_runs(), server.environment(), server.run_status("nothing")):
        assert "work_folder_note" not in reply
    assert server.environment()["work_folder"]["notes"] == []


def test_every_other_tool_carries_the_fix_needed_text_while_it_persists(places, monkeypatch, probe):
    home, ext = places
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(ext)
    monkeypatch.setenv("TOPOPT_WORKSPACE", "${FOO}/topopt")
    monkeypatch.setenv("TOPOPT_DEFAULT_WORKSPACE", "${HOME}/Documents/topopt")
    text = workfolder.resolve().fix_needed[0]
    for reply in (server.run_status("nothing"), server.list_runs(), server.run_history(), server.case_status("nothing.yaml"),
                  server.environment()):
        assert reply["work_folder_note"].startswith(text)
    assert '"${FOO}/topopt"' in text and str(home / "Documents" / "topopt") in text
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(home / "fixed"))                  # the person fixed the setting
    assert "work_folder_note" not in server.run_status("nothing")


def test_every_registered_tool_is_wrapped_and_the_texts_are_unchanged():
    tools = asyncio.run(server.mcp.list_tools())
    assert len(tools) >= 20
    for tool in tools:
        fn = getattr(server, tool.name)
        assert hasattr(fn, "__wrapped__"), tool.name                                  # the note is added in one place
        assert tool.description == inspect.cleandoc(fn.__wrapped__.__doc__), tool.name   # the docstring of the tool, cleaned the same way on every Python (#388)
        assert "work_folder_note" not in json.dumps(tool.inputSchema if hasattr(tool, "inputSchema") else tool.input_schema), tool.name


def test_the_environment_description_stays_within_the_platform_cut():
    # Clients cut a tool description at 4,096 characters; the work_folder and server sentence moved to the key guide (#174).
    from topopt_mcp import replies
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    assert len(tools["environment"].description) <= 4096
    assert "`server`: the pid and start time" in replies.guide("environment")


def test_a_reply_that_is_not_a_dict_is_not_changed(places, monkeypatch):
    monkeypatch.setenv("TOPOPT_WORKSPACE", "${FOO}/x")
    monkeypatch.setenv("HOME", str(places[0]))
    assert isinstance(server.team_guide(), str)


@NOT_ROOT
def test_a_folder_that_cannot_be_written_is_refused_where_the_server_writes(tmp_path, monkeypatch):
    locked = tmp_path / "locked"
    locked.mkdir()
    (tmp_path / "p.yaml").write_text("name: x\n")
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    server._current.clear()
    locked.chmod(0o500)
    try:
        monkeypatch.setenv("TOPOPT_WORKSPACE", str(locked))
        r = server.run_topopt(str(tmp_path / "p.yaml"))
        assert "cannot write in it" in r["error"] and str(locked) in r["error"]
        assert workfolder.FIX_OTHER in r["error"]
        assert "work_folder_note" not in r                              # the error holds the text: it is not said twice
        assert "proc" not in server._current and not (locked / "runs").exists()
        # a run folder outside the work folder does not need it
        r = server.run_topopt(str(tmp_path / "p.yaml"), str(tmp_path / "elsewhere"))
        assert "The server cannot use the work folder" not in r.get("error", "") and "pid" in r
        server._current["proc"].wait()
        # a folder that cannot be made: build_regions refuses too, and the other tools carry the text
        monkeypatch.setenv("TOPOPT_WORKSPACE", str(locked / "work"))
        b = server.build_regions({"name": "a", "regions": {}})
        assert b["ok"] is False and "cannot create" in b["error"] and str(locked) in b["error"] and "work_folder_note" not in b
        assert "cannot create" in server.run_status("nothing")["work_folder_note"]
        assert not (locked / "work").exists()
    finally:
        locked.chmod(0o700)
        server._current.clear()


def lose_current_folder(monkeypatch, tmp_path):
    """Go into a folder and remove it (an extension folder after an update, a scratch workspace at the end of a session)."""
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    with pytest.raises(OSError):
        Path.cwd()


def test_a_removed_current_folder_does_not_matter_with_a_full_path(tmp_path, monkeypatch, probe):
    target = tmp_path / "kept"
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(target))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    lose_current_folder(monkeypatch, tmp_path)
    assert server.workspace() == target.resolve()
    assert runinfo.key_root(target / "p" / "c.yaml") == target.resolve()
    assert runinfo.key_root(tmp_path / "elsewhere" / "c.yaml") == (tmp_path / "elsewhere").resolve()
    for reply in (server.list_runs(), server.run_status("nothing"), server.environment()):
        assert isinstance(reply, dict) and "work_folder_note" not in reply          # a dict tool reply works, and says nothing
    assert server.list_runs()["workspace"] == str(target.resolve())
    f = workfolder.resolve()
    assert f.fix_needed == [] and f.notes == [] and f.path == target.resolve() and f.writable is True


def test_a_removed_current_folder_with_a_variable_in_the_full_path_looks_for_no_old_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TOPOPT_WORKSPACE", "${DOCUMENTS}/topopt")
    lose_current_folder(monkeypatch, tmp_path)
    f = workfolder.resolve()
    assert f.path == (tmp_path / "home" / "Documents" / "topopt").resolve() and f.old_runs is None and f.fix_needed == []


def test_a_removed_current_folder_with_no_work_folder_gives_a_fix_needed_text(tmp_path, monkeypatch, probe):
    for name in ("TOPOPT_WORKSPACE", "TOPOPT_DEFAULT_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)
    lose_current_folder(monkeypatch, tmp_path)
    f = workfolder.resolve()                                    # no exception
    assert len(f.fix_needed) == 1 and "the current folder of the server is gone" in f.fix_needed[0]
    assert workfolder.FIX_OTHER in f.fix_needed[0]
    assert f.blocked == f.fix_needed[0] and f.exists is False and f.writable is False
    assert workfolder.resolve(check=False).fix_needed == f.fix_needed        # the text does not need the disk
    assert server.start_line(f).count("fix needed: ") == 1
    reply = server.environment()
    assert reply["work_folder_note"] == f.fix_needed[0] and reply["work_folder"]["fix_needed"] == f.fix_needed
    assert runinfo.key_root(tmp_path / "elsewhere" / "c.yaml") == (tmp_path / "elsewhere").resolve()
    # a relative setting needs the current folder too
    monkeypatch.setenv("TOPOPT_WORKSPACE", "rel/x")
    g = workfolder.resolve()
    assert len(g.fix_needed) == 1 and '"rel/x"' in g.fix_needed[0] and "not a full path" in g.fix_needed[0] and g.blocked
    # the Desktop extension: the default is a full path, so it works with no current folder
    monkeypatch.setenv("TOPOPT_DEFAULT_WORKSPACE", str(tmp_path / "default"))
    monkeypatch.delenv("TOPOPT_WORKSPACE")
    h = workfolder.resolve()
    assert h.path == (tmp_path / "default").resolve() and h.source == "TOPOPT_DEFAULT_WORKSPACE" and h.fix_needed == []


@NOT_ROOT
def test_a_work_folder_that_cannot_be_written_works_when_runs_can_be_written(tmp_path, monkeypatch, probe):
    """The case of the reviewer (probe_runs_writable): ws is 0o500, ws/runs is 0o700. The run goes in runs/, so it starts."""
    ws = tmp_path / "ws"
    (ws / "runs").mkdir(parents=True)
    (ws / "p.yaml").write_text("name: x\n")
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(ws))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    server._current.clear()
    ws.chmod(0o500)
    try:
        f = workfolder.resolve()
        assert f.exists is True and f.writable is False                      # the fact stays
        assert f.blocked is None and f.fix_needed == []                      # no refusal and no fix_needed
        assert len(f.notes) == 1 and "lock files of verify_run and ratchet_run" in f.notes[0] and str(ws / "runs") in f.notes[0]
        r = server.run_topopt("p.yaml")
        assert "error" not in r and r["pid"] and r["run_dir"] == str(ws.resolve() / "runs" / "live")
        assert r["work_folder_note"] == f.notes[0]                          # an info note, in the three tools only
        server._current["proc"].wait()
        assert (ws / "runs" / "live.stdout").exists()
        assert "work_folder_note" not in server.run_status("nothing")
        facts = server.environment()["work_folder"]
        assert facts["writable"] is False and facts["fix_needed"] == [] and facts["notes"] == f.notes
        b = server.build_regions({"name": "a", "regions": {}})              # the spec is wrong, but the folder is not the reason
        assert "cannot use the work folder" not in b.get("error", "") and "cannot write the folder" not in b.get("error", "")
    finally:
        ws.chmod(0o700)
        server._current.clear()


@NOT_ROOT
def test_a_runs_folder_that_cannot_be_written_is_refused_with_its_own_name(tmp_path, monkeypatch, probe):
    ws = tmp_path / "ws"
    (ws / "runs").mkdir(parents=True)
    (ws / "p.yaml").write_text("name: x\n")
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(ws))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    server._current.clear()
    (ws / "runs").chmod(0o500)
    try:
        assert workfolder.resolve().fix_needed == []                         # the work folder can be written
        r = server.run_topopt("p.yaml")
        assert f"cannot write the folder {ws.resolve() / 'runs'}" in r["error"] and workfolder.FIX_OTHER in r["error"]
        assert "proc" not in server._current and not (ws / "runs" / "live.stdout").exists()
        assert "cannot write the folder" in server.build_regions({"name": "a", "regions": {}})["error"]
        assert "work_folder_note" not in r and "work_folder_note" not in server.run_status("nothing")
    finally:
        (ws / "runs").chmod(0o700)
        server._current.clear()


# --- a work folder that is not made yet is the normal first state of a Desktop install -------------------------------

@pytest.fixture
def fresh_desktop(places, tmp_path, monkeypatch, probe):
    """The Desktop extension with an empty option on a new install: `<home>/Documents/topopt` is not there, and the YAML and the
    run folder are outside it (the case 1 of the review)."""
    home, ext = places
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(ext)
    monkeypatch.setenv("TOPOPT_WORKSPACE", "${user_config.workspace}")
    monkeypatch.setenv("TOPOPT_DEFAULT_WORKSPACE", str(home / "Documents" / "topopt"))
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "p.yaml").write_text("name: x\n")
    server._current.clear()
    yield home / "Documents" / "topopt", outside
    server._current.clear()


def test_a_run_starts_when_the_work_folder_is_not_made_yet(fresh_desktop):
    work, outside = fresh_desktop
    assert not work.exists() and workfolder.resolve().writable is True
    r = server.run_topopt(str(outside / "p.yaml"), str(outside / "out"))
    assert "error" not in r and r["pid"] and "work_folder_note" not in r
    server._current["proc"].wait()
    assert work.is_dir()                       # made just before the child started in it


def test_export_step_starts_when_the_work_folder_is_not_made_yet(fresh_desktop):
    work, outside = fresh_desktop
    r = server.export_step(str(outside / "nothing"), estimate_only=True)     # the child reports the missing run folder
    assert "did not start" not in r.get("error", "") and "[Errno 2]" not in r.get("error", "")
    assert r["ok"] is False and r.get("exit_code") == 1
    assert work.is_dir()


RECORDING = recorded_hook_run()


@pytest.mark.skipif(not (RECORDING / "result" / "final.stl").is_file(), reason="needs the recorded hook run recordings/hook_*")
def test_verify_run_starts_when_the_work_folder_is_not_made_yet(tmp_path, monkeypatch):
    import shutil
    import signal
    from topopt.verify import run as vrun
    d = tmp_path / "elsewhere" / "hook"                                   # a finished run outside the work folder
    (d / "result").mkdir(parents=True)
    shutil.copy(RECORDING / "run.json", d / "run.json")
    for name in ("final.stl", "metrics.json"):
        shutil.copy(RECORDING / "result" / name, d / "result" / name)
    work = tmp_path / "fresh"
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(work))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.setenv("TOPOPT_MCP_VERIFY_CMD", f"{sys.executable} {REPO}/tests/serve_verify_fake.py {{run_dir}} ok --step 0.25 --job topopt-verify")
    monkeypatch.setattr(vrun, "extras_facts", lambda: {"gmsh": {"state": "ok", "version": "4.15.2", "detail": "", "fix": ""},
                                                       "scikit-fem": {"state": "ok", "version": "x"}, "cholespy": {"state": "ok", "version": None}})
    server._verify.clear()
    try:
        res = server.verify_run(run_dir=str(d), estimate_only=False, yaml_path=str(REPO / "examples" / "hook" / "hook.yaml"))
        assert res["ok"] and res["started"] and "[Errno" not in str(res.get("error", ""))
        assert work.is_dir()
    finally:
        proc = server._verify.get("proc")
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        server._verify.clear()


def test_ratchet_run_starts_when_the_work_folder_is_not_made_yet(tmp_path, monkeypatch):
    import shlex
    import signal
    import yaml
    sys.path.insert(0, str(Path(__file__).parent))
    from ratchet_helpers import stl_folder
    from test_feedback import fake_run, make_case
    ypath = make_case(tmp_path / "in")
    doc = yaml.safe_load(ypath.read_text())
    doc["acceptance"] = {"rules": [{"rule": "watertight"}, {"rule": "bodies"}, {"rule": "mass", "must_fall": True}]}
    ypath.write_text(yaml.safe_dump(doc, sort_keys=False))
    run = fake_run(tmp_path / "elsewhere" / "a", ypath)                  # a finished run outside the work folder
    stl_folder(tmp_path / "cands", 3)
    work = tmp_path / "fresh"
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(work))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.setenv("TOPOPT_MCP_RATCHET_CMD", f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)' topopt-ratchet-fake")
    server._ratchet.clear()
    try:
        res = server.ratchet_run(run_dir=str(run), source=f"stl:{tmp_path / 'cands'}", estimate_only=False)
        assert res["ok"] and res["started"] and "[Errno" not in str(res.get("error", ""))
        assert work.is_dir()
    finally:
        proc = server._ratchet.get("proc")
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        server._ratchet.clear()


@NOT_ROOT
def test_a_run_folder_outside_the_work_folder_that_cannot_be_written_is_left_to_the_tool(tmp_path, monkeypatch, probe):
    ws, outside = tmp_path / "ws", tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    (ws / "p.yaml").write_text("name: x\n")
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(ws))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    server._current.clear()
    outside.chmod(0o500)
    try:
        r = server.run_topopt("p.yaml", str(outside / "out"))             # as before: the tool's own error, not the work folder text
        assert "error" in r and "work folder" not in r["error"] and workfolder.FIX_OTHER not in r["error"]
        assert "proc" not in server._current
    finally:
        outside.chmod(0o700)
        server._current.clear()


@NOT_ROOT
def test_a_work_folder_that_cannot_be_made_gives_the_work_folder_text_not_the_system_error(tmp_path, monkeypatch, probe):
    locked = tmp_path / "locked"
    locked.mkdir()
    (tmp_path / "p.yaml").write_text("name: x\n")
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(locked / "work"))
    monkeypatch.delenv("TOPOPT_DEFAULT_WORKSPACE", raising=False)
    monkeypatch.setenv("TOPOPT_MCP_CMD", f"{sys.executable} -c pass")
    server._current.clear()
    locked.chmod(0o500)
    try:
        # a run folder outside the work folder is not refused, but the child cannot start in a folder that is not there
        r = server.run_topopt(str(tmp_path / "p.yaml"), str(tmp_path / "out"))
        assert "cannot create it" in r["error"] and str(locked) in r["error"] and "[Errno" not in r["error"]
        assert "proc" not in server._current
        e = server.export_step(str(tmp_path / "nothing"), estimate_only=True)
        assert e["ok"] is False and "cannot create it" in e["error"] and "[Errno" not in e["error"]
        assert not (locked / "work").exists()
        # when the system refuses a folder that the check called writable, the text names the folder and the fix
        monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError(1, "Operation not permitted")))
        monkeypatch.setenv("TOPOPT_WORKSPACE", str(tmp_path / "other"))
        text = server._make_work_folder()
        assert str(tmp_path / "other") in text and "Operation not permitted" in text and workfolder.FIX_OTHER in text
    finally:
        locked.chmod(0o700)
        server._current.clear()


def test_the_probe_of_another_python_is_a_read_and_starts_in_an_existing_folder(places, monkeypatch):
    import types
    home, ext = places
    work = home / "Documents" / "topopt"
    monkeypatch.setenv("TOPOPT_WORKSPACE", str(work))
    started = []

    def fake_run(cmd, **kwargs):
        started.append(kwargs["cwd"])
        return types.SimpleNamespace(returncode=1, stdout="", stderr="stop")

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    server._probe_environment("/other/python", None)
    assert started == [str(home)] and not (home / "Documents").exists()      # the nearest existing parent; nothing made
    work.mkdir(parents=True)
    server._probe_environment("/other/python", None)
    assert started[-1] == str(work.resolve())                                # the work folder, when it is there


def test_the_start_line_names_the_folder_its_source_and_the_texts(places):
    line = server.start_line(resolve(places, TOPOPT_WORKSPACE="${FOO}/x"))
    assert "\n" not in line and line.startswith(f"python: {sys.executable}; work folder: {places[1].resolve()} (from current folder)")
    assert "fix needed: " in line and '"${FOO}/x"' in line
    quiet = server.start_line(resolve(places))
    assert "fix needed" not in quiet and "note:" not in quiet


# --- the real server over stdio ----------------------------------------------------------------------------------------

def payload(res):
    data = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if data:
        return data.get("result", data) if set(data) == {"result"} else data
    return json.loads(res.content[0].text)


def test_stdio_server_with_the_text_that_claude_desktop_gave(tmp_path):
    """The server as Desktop starts it: the extension folder is the current folder, HOME is a fake home, and the option holds
    the text `${DOCUMENTS}/topopt`. Runs from the old server are in `<extension folder>/${DOCUMENTS}/topopt`."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    home, ext = tmp_path / "home", tmp_path / "extension"
    home.mkdir()
    old = fake_old_run(ext, "${DOCUMENTS}/topopt")
    expected = (home / "Documents" / "topopt").resolve()
    env = dict(os.environ, HOME=str(home), TOPOPT_WORKSPACE="${DOCUMENTS}/topopt", PYTHONPATH=str(REPO))
    env["TOPOPT_DEFAULT_WORKSPACE"] = str(home / "Documents" / "topopt")
    params = StdioServerParameters(command=sys.executable, args=["-m", "topopt_mcp.server"], env=env, cwd=str(ext))
    errfile = tmp_path / "server.stderr"

    async def go():
        with open(errfile, "w") as errlog:
            async with stdio_client(params, errlog=errlog) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    env_reply = payload(await session.call_tool("environment", {}))
                    runs = payload(await session.call_tool("list_runs", {}))
                    status = payload(await session.call_tool("run_status", {"run_dir": "nothing"}))
                    return env_reply, runs, status

    env_reply, runs, status = asyncio.run(asyncio.wait_for(go(), 300))
    wf = env_reply["work_folder"]
    assert wf["path"] == str(expected) and wf["source"] == "TOPOPT_WORKSPACE" and wf["expanded"] == ["${DOCUMENTS}"]
    assert wf["old_runs"] == {"path": str(old.resolve()), "run_folders": 1}
    assert any("Runs from before are in" in n for n in wf["notes"])
    assert env_reply["server"]["pid"] > 0 and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", env_reply["server"]["started_utc"])
    assert runs["workspace"] == str(expected) and runs["runs"] == []        # the runs of the old place are not listed, and not moved
    assert 'The work folder setting is "${DOCUMENTS}/topopt".' in runs["work_folder_note"] and str(expected) in runs["work_folder_note"]
    assert "work_folder_note" not in status                                 # an info note is in three replies only
    assert not expected.exists() and (old / "runs" / "r1" / "status.json").is_file()
    start = [ln for ln in errfile.read_text().splitlines() if "work folder:" in ln]
    assert start and f"work folder: {expected} (from TOPOPT_WORKSPACE)" in start[0] and "note: " in start[0]
