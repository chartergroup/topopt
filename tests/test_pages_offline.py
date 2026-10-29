"""#396: the pages that topopt gives to a browser name no host but this computer, so the browser asks only the topopt server.

- The page files, the dashboard (`topopt/dashboard/`) and the template of the refine page (`topopt/refine/template.html`), name an
  `http://` or `https://` address only on this computer (`127.0.0.1`, `localhost`, `[::1]`).
- The copied libraries (the `vendor/` folders) hold a few addresses that load nothing. `VENDOR_URLS` names each one, so a new copy
  with a new address, or a new copied file, is a changed test.
- No HTML, CSS or JS of the package, also HTML inside Python code, loads a file from another host: a tag with `src`, `href`, `data`
  or `poster`, CSS `url(...)` and `@import`, JS `import ... from`, `import(...)` and `fetch(...)`.
"""
import re
from pathlib import Path

from topopt.refine.page_scripts import OLD_CDN_TAGS

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "topopt" / "dashboard"
REFINE_TEMPLATE = ROOT / "topopt" / "refine" / "template.html"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
URL = re.compile(r"""https?://[^\s"'`<>()\\]+""")
# A namespace name (an identifier, not a load), and a link in the text of a console warning of three.js r160.
VENDOR_URLS = {
    "topopt/dashboard/vendor/three.module.min.js": {"http://www.w3.org/1999/xhtml",
                                                    "https://discourse.threejs.org/t/updates-to-lighting-in-three-js-r155/53733"},
    "topopt/refine/vendor/three.min.js": {"http://www.w3.org/1999/xhtml"},
    "topopt/refine/vendor/OrbitControls.js": set(),
}

# A tag that makes the browser load a file (an <a> link is a click, not a load).
LOAD_TAG = re.compile(r"<(?:script|link|img|iframe|source|video|audio|embed|object)\b[^>]*>", re.IGNORECASE)
TAG_URL = re.compile(r"""\b(?:src|href|data|poster)\s*=\s*\\?["']?https?://([^/"'\\\s>]+)""", re.IGNORECASE)   # \" in Python code too
# CSS and JS forms that load a file.
CODE_URL = re.compile(r"""(?:\burl\(\s*["']?|@import\s+(?:url\(\s*)?["']?|\bfrom\s+["']|\bimport\s*\(\s*["']|\bfetch\s*\(\s*["'])"""
                      r"""https?://([^/"'\s)]+)""")


def is_vendored(path: Path) -> bool:
    return "vendor" in path.relative_to(ROOT).parts


def urls(text: str) -> set[str]:
    return {u.rstrip(".,;:") for u in URL.findall(text)}


def host(url: str) -> str:
    """The host of a URL without its port: `127.0.0.1` of `http://127.0.0.1:8765/`, `[::1]` of `http://[::1]:8765/`."""
    netloc = re.split(r"[/?#]", url.split("://", 1)[1], maxsplit=1)[0]
    return re.match(r"\[[^\]]*\]|[^:]*", netloc).group(0)


def page_files() -> list[Path]:
    """The files that the dashboard server serves (every file of `topopt/dashboard/` but the copied libraries) and the refine
    template."""
    served = [p for p in DASHBOARD.rglob("*") if p.is_file() and not is_vendored(p) and "__pycache__" not in p.parts]
    return [REFINE_TEMPLATE, *sorted(served)]


def test_the_pages_name_no_host_but_this_computer():
    outside = {}
    for path in page_files():
        for url in urls(path.read_text(errors="replace")):
            if host(url) not in LOCAL_HOSTS:
                outside.setdefault(str(path.relative_to(ROOT)), []).append(url)
    assert not outside, f"a page names an outside host; copy the file into the package (vendor/) instead: {outside}"


def test_the_copied_libraries_hold_only_the_known_addresses():
    found = {str(p.relative_to(ROOT)): urls(p.read_text(encoding="utf-8", errors="replace"))
             for base in (ROOT / "topopt", ROOT / "topopt_mcp") for p in base.rglob("*")
             if p.is_file() and is_vendored(p) and "__pycache__" not in p.parts}
    assert set(found) == set(VENDOR_URLS), "a copied file is new or gone: list it in VENDOR_URLS with the addresses that it holds"
    for path, known in VENDOR_URLS.items():
        assert found[path] == known, f"{path}: the addresses changed; check that none of them loads a file"


# It names the CDN tags of the pages that an older topopt made, to put the copied files in their place (tests/test_refine.py). The scan
# leaves out these two tags and reads the rest of the file.
OLD_TAGS_MODULE = ROOT / "topopt" / "refine" / "page_scripts.py"


def load_hosts() -> set[str]:
    """The hosts that a file of the package loads a file from: HTML tags in .html and .py files, and the CSS and JS forms in .html,
    .css and .js files. A copied library (a local path) has no host."""
    hosts = set()
    for base in (ROOT / "topopt", ROOT / "topopt_mcp"):
        for path in base.rglob("*"):
            if path.suffix not in {".html", ".css", ".js", ".py"} or is_vendored(path):
                continue
            text = path.read_text(errors="replace")
            if path == OLD_TAGS_MODULE:
                for tag in OLD_CDN_TAGS.values():
                    assert text.count(tag.decode()) == 1, tag
                    text = text.replace(tag.decode(), "")
            for tag in LOAD_TAG.findall(text):
                hosts.update(TAG_URL.findall(tag))
            if path.suffix != ".py":
                hosts.update(CODE_URL.findall(text))
    return hosts


def test_no_file_of_the_package_loads_a_page_file_from_another_host():
    outside = {h for h in load_hosts() if host("https://" + h) not in LOCAL_HOSTS}
    assert not outside, f"HTML, CSS or JS of the package loads a file from {sorted(outside)}; copy the file into the package"
