"""The scripts that the refine page holds inline: three.js r128 and its OrbitControls (#396).

The files in `vendor/` are copies from the npm package three 0.128.0 (THIRD_PARTY_NOTICES.md). `build.py` puts them into each new page, so
the page works with no network, from file:// and from the dashboard, and the browser asks no other host for a file. A page that an older
topopt made loads the same two files from cdn.jsdelivr.net. `inline_old_page` puts the copies in place of its two tags, so the dashboard
serves such a page with no outside load too. This module imports nothing heavy: the dashboard server uses it.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR_SCRIPTS = {"%%THREE_JS%%": "vendor/three.min.js", "%%ORBIT_CONTROLS_JS%%": "vendor/OrbitControls.js"}
# The two tags of each page that an older topopt made: topopt/refine/template.html had them from its first version to #396.
OLD_CDN_TAGS = {
    "%%THREE_JS%%": b'<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>',
    "%%ORBIT_CONTROLS_JS%%": b'<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>',
}


def script_text(mark: str) -> str:
    """The text of the copied file for a template mark."""
    return (HERE / VENDOR_SCRIPTS[mark]).read_text(encoding="utf-8")


def inline_old_page(page: bytes) -> bytes:
    """Put the copied scripts in place of the CDN tags of a page that an older topopt made. Another page comes back as it is."""
    for mark, tag in OLD_CDN_TAGS.items():
        if tag in page:   # the first one is the tag in the head: the data of the run comes after it
            page = page.replace(tag, b"<script>\n" + script_text(mark).encode("utf-8") + b"\n</script>", 1)
    return page
