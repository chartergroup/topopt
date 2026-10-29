// Small rules of the dashboard page. No DOM, no three.js: a node test can import it.
//
// The demo screen is the code panel (the optimizer source, with the active step marked) and the row "Pull-off stiffness per gram".
// It shows only when the run has a load case named `pull_off` (the recorded hook run has one), or when the page address has `?demo=1`.
// Otherwise the side panel shows the log and the result rows.
export function talkScreen(search, run) {
  if (new URLSearchParams(search || "").get("demo") === "1") return true;
  return !!run && Array.isArray(run.load_cases) && run.load_cases.some((c) => c && c.name === "pull_off");
}

// The line under "Part file" in the result panel: the run folder name and the path of the file in it.
// `info` is the answer of /api/part_file. It gives null while the file is not there (nothing is shown then).
export function partFileText(info) {
  if (!info || !info.exists || !info.run_name || !info.file) return null;
  return `${info.run_name}/${info.file}`;
}
