"""Which factor `topopt verify` uses, with a faked platform (CONTRACTS.md section 16, "Factor order")."""
import sys

import pytest

from topopt.config import UserError
from topopt.verify import factor
from topopt.verify.errors import SolverUnavailable


def fake(monkeypatch, *, mac=False, linux=False, shim=True, pypardiso=True, cholespy=True, superlu=True):
    """Fake the platform: which of the four factors can run.

    `cholespy` is True (loads), False (not installed) or a text (installed, but it fails to load with this error).
    """
    monkeypatch.setattr(factor, "is_macos_arm64", lambda: mac)
    monkeypatch.setattr(factor, "is_linux_x86_64", lambda: linux)
    monkeypatch.setattr(factor, "_shim_problem", lambda: None if shim else "RuntimeError: libaccel_chol.dylib not found")
    monkeypatch.setattr(factor, "pypardiso_problem", lambda: None if pypardiso else ("missing", "No module named 'pypardiso'"))
    if cholespy is True:
        monkeypatch.setattr(factor, "cholespy_problem", lambda: None)
    elif cholespy is False:
        monkeypatch.setattr(factor, "cholespy_problem", lambda: ("missing", "No module named 'cholespy'"))
    else:
        monkeypatch.setattr(factor, "cholespy_problem", lambda: ("load_failed", cholespy))
    monkeypatch.setattr(factor, "_superlu_problem", lambda: None if superlu else "ImportError: no scipy")
    return factor.factor_availability()


def test_auto_takes_cholesky_on_a_mac_when_the_shim_loads(monkeypatch):
    avail = fake(monkeypatch, mac=True, shim=True)
    assert factor.resolve_factor(None, avail) == "cholesky"
    assert factor.resolve_factor("auto", avail) == "cholesky"


def test_auto_skips_cholesky_on_a_mac_when_the_shim_does_not_load(monkeypatch):
    avail = fake(monkeypatch, mac=True, shim=False)
    assert not avail["cholesky"]["available"]
    assert "does not load" in avail["cholesky"]["note"]
    assert factor.resolve_factor(None, avail) == "cholespy"        # pardiso is Linux only


def test_auto_takes_superlu_last_when_no_other_factor_runs(monkeypatch):
    """The order is cholesky, pardiso, cholespy, superlu. SuperLU (scipy) is the fallback of last resort."""
    assert factor.AUTO_ORDER == ("cholesky", "pardiso", "cholespy", "superlu")
    for kw in (dict(linux=True, pypardiso=False, cholespy=False), dict(mac=True, shim=False, cholespy=False),
               dict(cholespy=False), dict(linux=True, pypardiso=False, cholespy="ImportError: libstdc++.so.6 too old")):
        avail = fake(monkeypatch, **kw)
        assert factor.resolve_factor(None, avail) == "superlu", kw
        assert avail["superlu"]["available"] and "last fallback" in avail["superlu"]["note"]


def test_superlu_is_not_taken_when_a_faster_factor_runs(monkeypatch):
    for kw in (dict(mac=True), dict(linux=True), dict(linux=True, pypardiso=False)):
        avail = fake(monkeypatch, **kw)
        assert factor.resolve_factor(None, avail) != "superlu", kw
        assert factor.resolve_factor("superlu", avail) == "superlu"      # the user can still name it


def test_auto_takes_pardiso_on_linux_x86_64_when_it_loads(monkeypatch):
    avail = fake(monkeypatch, linux=True, pypardiso=True)
    assert factor.resolve_factor(None, avail) == "pardiso"


def test_auto_takes_cholespy_on_linux_without_pypardiso(monkeypatch):
    avail = fake(monkeypatch, linux=True, pypardiso=False)
    assert "pypardiso is not installed" in avail["pardiso"]["note"]
    assert factor.resolve_factor(None, avail) == "cholespy"


def test_auto_takes_cholespy_on_another_platform(monkeypatch):
    avail = fake(monkeypatch)                                       # for example Linux arm64
    assert factor.resolve_factor(None, avail) == "cholespy"
    assert "Apple silicon only" in avail["cholesky"]["note"]
    assert "Linux x86_64 only" in avail["pardiso"]["note"]


def test_pypardiso_is_not_asked_off_linux_x86_64(monkeypatch):
    def boom():
        raise AssertionError("pypardiso must not be imported here")
    fake(monkeypatch, mac=True)
    monkeypatch.setattr(factor, "pypardiso_problem", boom)
    avail = factor.factor_availability()
    assert avail["pardiso"]["available"] is False


def test_auto_never_takes_amg(monkeypatch):
    for kw in (dict(mac=True), dict(linux=True), dict(linux=True, pypardiso=False), dict()):
        avail = fake(monkeypatch, **kw)
        assert factor.resolve_factor(None, avail) in factor.FACTOR_NAMES
    assert "amg" not in factor.FACTOR_NAMES
    assert "amg" not in factor.AUTO_ORDER


def test_auto_with_nothing_available_is_a_user_error_with_every_fact(monkeypatch):
    """No direct factor can run: `SolverUnavailable`, a `UserError` (exit code 1), not a `VerifyError` (exit code 2)."""
    avail = fake(monkeypatch, cholespy=False, superlu=False)
    with pytest.raises(SolverUnavailable) as e:
        factor.resolve_factor(None, avail)
    text = str(e.value)
    assert isinstance(e.value, UserError) and not isinstance(e.value, factor.VerifyError)
    for name in factor.FACTOR_NAMES:
        assert f"{name}:" in text
    assert "verify" in text and "Options" in text and "scipy" in text
    assert e.value.facts["requested"] == "auto" and set(e.value.facts["factors"]) == set(factor.FACTOR_NAMES)


def test_cholespy_that_fails_to_load_gets_the_error_text_not_the_install_hint(monkeypatch):
    """The form of `solver_choice.resolve_solver` for pypardiso: only a missing package gets the install hint."""
    err = "ImportError: libstdc++.so.6: version `GLIBCXX_3.4.32' not found"
    avail = fake(monkeypatch, linux=True, pypardiso=False, cholespy=err)
    note = avail["cholespy"]["note"]
    assert not avail["cholespy"]["available"]
    assert "installed but does not load" in note and err in note and "reinstall" in note.lower()
    assert "Install it" not in note and "[verify]" not in note
    assert factor.resolve_factor(None, avail) == "superlu"                 # and the choice is a fact, not a stop


def test_cholespy_that_is_missing_gets_the_install_hint(monkeypatch):
    avail = fake(monkeypatch, linux=True, pypardiso=False, cholespy=False)
    note = avail["cholespy"]["note"]
    assert "not installed" in note and "does not load" not in note
    cmd = factor.verify_install_command()                       # the command names the Python that runs topopt (as for pypardiso)
    assert f"install the extra with: {cmd}" in note and "--python" in cmd and sys.executable.replace("'", "") in cmd.replace("'", "")
    assert "[verify]" in cmd                                    # a source tree gets the extra


def test_cholespy_problem_tells_missing_from_load_failed(monkeypatch):
    import builtins
    real = builtins.__import__

    def make(exc):
        def fake_import(name, *a, **k):
            if name == "cholespy":
                raise exc
            return real(name, *a, **k)
        return fake_import
    monkeypatch.setattr(builtins, "__import__", make(ModuleNotFoundError("No module named 'cholespy'", name="cholespy")))
    assert factor.cholespy_problem()[0] == "missing"
    monkeypatch.setattr(builtins, "__import__", make(ModuleNotFoundError("No module named 'libx'", name="libx")))
    assert factor.cholespy_problem()[0] == "load_failed"                  # a module under cholespy is missing
    monkeypatch.setattr(builtins, "__import__", make(ImportError("libstdc++.so.6: version GLIBCXX_3.4.32 not found")))
    kind, text = factor.cholespy_problem()
    assert kind == "load_failed" and "GLIBCXX_3.4.32" in text and "ImportError" in text


def test_unknown_name_raises_valueerror(monkeypatch):
    avail = fake(monkeypatch, linux=True)
    with pytest.raises(ValueError, match="unknown solver 'foo'"):
        factor.resolve_factor("foo", avail)
    with pytest.raises(ValueError, match="amg is not offered for verify") as e:
        factor.resolve_factor("amg", avail)
    text = str(e.value)
    assert "iterative method" in text and "low-memory option" in text
    assert "nullspace" not in text and "did not converge" not in text       # the old reason was wrong (review m2)


def test_explicit_name_is_taken_as_given_and_overrides_the_order(monkeypatch):
    avail = fake(monkeypatch, linux=True, pypardiso=True)
    assert factor.resolve_factor("cholespy", avail) == "cholespy"     # not the auto choice (pardiso)
    assert factor.resolve_factor("pardiso", avail) == "pardiso"


def test_explicit_name_that_cannot_run_here_gives_the_facts_and_the_options(monkeypatch):
    avail = fake(monkeypatch, linux=True, pypardiso=True)
    with pytest.raises(SolverUnavailable) as e:
        factor.resolve_factor("cholesky", avail)
    assert isinstance(e.value, UserError)                                  # exit code 1: the user asked for the impossible
    assert "Apple silicon only" in str(e.value) and "--solver pardiso" in str(e.value) and "--solver superlu" in str(e.value)
    assert e.value.facts["requested"] == "cholesky"


def test_unknown_solver_is_a_user_error_for_the_command(tmp_path):
    from topopt.verify.run import verify_run
    with pytest.raises(UserError):
        verify_run(tmp_path, solver="foo")                          # bad usage comes first: exit code 1


def test_availability_and_resolve_use_the_real_platform_without_error():
    avail = factor.factor_availability()
    assert set(avail) == set(factor.FACTOR_NAMES)
    for v in avail.values():
        assert isinstance(v["available"], bool) and v["note"]
    assert avail["superlu"]["available"]                                   # scipy is a main dependency
    assert factor.resolve_factor(None, avail) in factor.FACTOR_NAMES       # so a solver is always there


def test_the_core_solver_has_no_new_backend():
    """The verify factors sit beside `LinearSolver`; the solver of the optimizer keeps its three methods."""
    from topopt.solver import LinearSolver
    with pytest.raises(ValueError):
        LinearSolver("cholespy")
