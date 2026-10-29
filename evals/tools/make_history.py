"""Make a small history file (context.history_file) from a Claude Code session transcript.

The session file of an eval run (kept with --keep-temp, under
<kept dir>/config/projects/<cwd>/<session>.jsonl) also holds the system prompt, API request
blobs and account facts. This script keeps only the user and assistant messages, links each
message to the last kept message before it, and puts placeholders in place of local paths.

Usage:
    python3 evals/tools/make_history.py <session.jsonl> <plugin root> > evals/<case>/history.jsonl
"""
import json
import re
import sys

KEEP = {"user", "assistant"}
DROP_KEYS = {"promptId", "promptSource", "turnOrigin", "turnPosition", "requestId"}


def main(src, plugin_root):
    by_uuid = {}
    entries = []
    for line in open(src, encoding="utf-8"):
        d = json.loads(line)
        if "uuid" in d:
            by_uuid[d["uuid"]] = d
        entries.append(d)

    kept = set()
    out = []
    for d in entries:
        if d.get("type") not in KEEP:
            continue
        parent = d.get("parentUuid")
        while parent and parent not in kept:
            parent = (by_uuid.get(parent) or {}).get("parentUuid")
        d = {k: v for k, v in d.items() if k not in DROP_KEYS}
        d["parentUuid"] = parent
        kept.add(d["uuid"])
        out.append(d)

    text = "\n".join(json.dumps(d, ensure_ascii=False) for d in out) + "\n"
    text = text.replace(plugin_root, "/Users/me/topopt-demo")
    for bad in re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
        if not bad.endswith(("example.com", ".stl", ".yaml")):
            text = text.replace(bad, "user@example.com")
    sys.stdout.write(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
