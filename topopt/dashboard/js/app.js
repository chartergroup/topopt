// Dashboard client. No build step. Only local files (no CDN).
import * as THREE from "../vendor/three.module.min.js";
import { componentBoxes } from "./boxes.js";
import { labelPlaces, sameArrowGroups } from "./labels.js";
import { partFileText, talkScreen } from "./talk.js";

const $ = (id) => document.getElementById(id);
const STEP_ORDER = ["setup", "assemble", "solve", "sensitivity", "filter", "update", "snapshot"];

const S = {
  run: null, labels: null, source: null,
  status: null, statusAt: 0,
  history: [], shownIter: -1, meshBusy: false,
  finalShown: false, doneStarted: 0,
  activeStep: null, gen: 0, refineFound: false,
  talk: false,   // the demo screen: the code panel and the pull-off row (see talk.js)
};

// ---------------------------------------------------------------- helpers
async function getJSON(url) {
  try {
    const r = await fetch(url, { cache: "no-store" });
    return r.ok ? await r.json() : null;
  } catch (e) { return null; }
}
// /api/run, with a missing file (HTTP 404: the run folder has no run.json) told from a network error or another HTTP error
// (the server restarts: `topopt run --dashboard` ends as a new program that takes over the socket). Only a missing file resets the view.
async function getRun() {
  try {
    const r = await fetch("/api/run", { cache: "no-store" });
    if (r.ok) return { run: await r.json() };
    return { missing: r.status === 404 };
  } catch (e) { return { missing: false }; }
}
async function getBuf(url) {
  try {
    const r = await fetch(url, { cache: "no-store" });
    return r.ok ? { buf: await r.arrayBuffer(), it: r.headers.get("X-Iteration") } : null;
  } catch (e) { return null; }
}
const fmtTime = (s) => {
  s = Math.max(0, Math.floor(s));
  return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
};
const fmtNum = (v) => (v >= 1000 ? v.toFixed(0) : v >= 100 ? v.toFixed(1) : v.toFixed(2));

// ---------------------------------------------------------------- 3D scene
const canvas = $("gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 1, 5000);
camera.up.set(0, 0, 1); // z is up
window.__dbg = { scene };
const spin = new THREE.Group(); // turns around the vertical axis
scene.add(spin);
const decor = new THREE.Group(); // boxes, arrows, outline (hidden on the final screen)
spin.add(decor);
const meshHolder = new THREE.Group();
spin.add(meshHolder);

scene.add(new THREE.HemisphereLight(0xdbe8ff, 0x1a2230, 1.0));
const key = new THREE.DirectionalLight(0xffffff, 2.4); key.position.set(0.6, -1, 1.2); scene.add(key);
const rim = new THREE.DirectionalLight(0x6fb6ff, 1.2); rim.position.set(-1, 0.8, 0.4); scene.add(rim);

const previewMat = new THREE.MeshStandardMaterial({ color: 0x9cc4ff, roughness: 0.45, metalness: 0.15 });
// No clearcoat: a sharp highlight on voxel steps looks like white specks.
const finalMat = new THREE.MeshStandardMaterial({ color: 0xdfe6f1, roughness: 0.55, metalness: 0.0 });

let center = new THREE.Vector3(), radius = 60, floor = null;
const CASE_COLORS = [0xffb020, 0xff5d8f, 0x5be3a4, 0xb28cff];

// The spin axis goes through `c`. The camera fits a sphere of radius `r` around it.
function setPivot(c, r) {
  center = c.clone();
  radius = r;
  spin.position.copy(center);
  meshHolder.position.copy(center).negate();
  decor.position.copy(center).negate();
  resize();
}

// A label of one line: the names of `parts` ([{text, color}, ...]), each in its colour, with a dim dot between two names. The canvas is
// 256 x 64 px, wider if the text needs it; `userData.aspect` is its width over its height.
function makeLabel(parts) {
  const font = "bold 40px -apple-system, Helvetica, Arial, sans-serif", sep = " \u00b7 ";
  const cv = document.createElement("canvas");
  let g = cv.getContext("2d");
  g.font = font;
  const widths = parts.map((p) => g.measureText(p.text).width), sepW = g.measureText(sep).width;
  const textW = widths.reduce((a, b) => a + b, 0) + sepW * (parts.length - 1);
  cv.width = Math.max(256, Math.ceil(textW + 24)); cv.height = 64;
  g = cv.getContext("2d");                 // a new canvas size resets the context
  g.font = font; g.textAlign = "left"; g.textBaseline = "middle";
  const runs = [];
  let x = (cv.width - textW) / 2;
  parts.forEach((p, i) => {
    if (i) { runs.push([sep, x, "#6f8197"]); x += sepW; }
    runs.push([p.text, x, "#" + p.color.toString(16).padStart(6, "0")]); x += widths[i];
  });
  g.lineWidth = 8; g.strokeStyle = "#0b0f16";
  runs.forEach(([t, at]) => g.strokeText(t, at, 34));                 // all outlines first, so no outline covers a letter
  runs.forEach(([t, at, c]) => { g.fillStyle = c; g.fillText(t, at, 34); });
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.renderOrder = 5;
  sp.userData.aspect = cv.width / cv.height;
  return sp;
}

// One arrow for each load case, at the load centre (at `at` for a remote force), in the true force direction.
function addLoadArrows(size) {
  const len = Math.max(size.length() * 0.18, 8);
  const drawn = [];    // the cases with an arrow
  (S.run.load_cases || []).forEach((lc, ci) => {
    const color = CASE_COLORS[ci % CASE_COLORS.length];
    // The first entry with a resultant force. A case of moments only (a zero force) has no arrow.
    const f = (lc.forces || []).find((e) => new THREE.Vector3(...e.force).length() >= 1e-12);
    if (!f) return;
    const dir = new THREE.Vector3(...f.force).normalize();
    // A remote force starts where the YAML says (`at`), else at the centre of its region.
    const c = f.at ? new THREE.Vector3(...f.at) : f.center ? new THREE.Vector3(...f.center) : center.clone();
    const tail = c.clone().addScaledVector(dir, -len); // the tip touches the load centre
    // A thick arrow: a cylinder shaft and a cone head. The +y axis of both
    // points along the force.
    const arrow = new THREE.Group();
    const mat = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 });
    const headL = len * 0.32, shaftL = len - headL, r = len * 0.045;
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(r, r, shaftL, 12), mat);
    shaft.position.y = shaftL / 2;
    const head = new THREE.Mesh(new THREE.ConeGeometry(r * 3, headL, 16), mat);
    head.position.y = shaftL + headL / 2;
    arrow.add(shaft, head);
    arrow.traverse((n) => { n.renderOrder = 4; });
    arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    arrow.position.copy(tail);
    arrow.userData.keep = true;
    decor.add(arrow);
    drawn.push({ name: lc.name, color, tail, dir, ci });
  });
  // Cases with their arrows at the same place (the later arrow hides the earlier one; tails closer than the radius of the arrow head)
  // share ONE label with all their names, at the place of the label of the first.
  const arrows = drawn.map((d) => ({ tail: d.tail.toArray(), dir: d.dir.toArray(), ci: d.ci }));
  const groups = sameArrowGroups(arrows, len * 0.135);
  const labels = groups.map((group) => {
    const lab = makeLabel(group.map((i) => ({ text: drawn[i].name, color: drawn[i].color })));
    lab.scale.set(len * 0.4 * lab.userData.aspect, len * 0.4, 1);
    lab.userData.keep = true;
    decor.add(lab);
    return lab;
  });
  // Past the tail of the arrow; a label that would cover an earlier one moves on, away from the tip (labels.js).
  labelPlaces(arrows, groups, len, labels.map((l) => l.scale.x)).forEach((at, i) => labels[i].position.set(...at));
}

// Remove everything of the old run from the scene.
function clearScene() {
  swapMesh(null, previewMat);
  while (decor.children.length) {
    const o = decor.children[0];
    decor.remove(o);
    o.traverse((n) => {
      if (n.geometry) n.geometry.dispose();
      if (n.material) { if (n.material.map) n.material.map.dispose(); n.material.dispose(); }
    });
  }
  if (floor) { scene.remove(floor); floor.geometry.dispose(); floor = null; }
  decor.visible = true;
}

function buildSceneFromRun() {
  const [nx, ny, nz] = S.run.shape, h = S.run.h, o = S.run.origin;
  const lab = S.labels;
  // The bounding box (in voxels) of labels 1..3. Labels 2 (fixed regions) and 3 (load and solid regions) get one box for each
  // connected part, so two feet apart are two boxes and not one slab between them.
  const bb = { all: null };
  const upd = (key, i, j, k) => {
    const b = bb[key] || (bb[key] = [i, j, k, i, j, k]);
    b[0] = Math.min(b[0], i); b[1] = Math.min(b[1], j); b[2] = Math.min(b[2], k);
    b[3] = Math.max(b[3], i); b[4] = Math.max(b[4], j); b[5] = Math.max(b[5], k);
  };
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
    const base = (i * ny + j) * nz;
    for (let k = 0; k < nz; k++) {
      const c = lab[base + k];
      if (c === 1 || c === 2 || c === 3) upd("all", i, j, k);
    }
  }
  const all = bb.all || [0, 0, 0, nx - 1, ny - 1, nz - 1];
  const toWorld = (b) => ({
    min: new THREE.Vector3(o[0] + h * b[0], o[1] + h * b[1], o[2] + h * b[2]),
    max: new THREE.Vector3(o[0] + h * (b[3] + 1), o[1] + h * (b[4] + 1), o[2] + h * (b[5] + 1)),
  });
  const A = toWorld(all);
  const size = A.max.clone().sub(A.min);
  setPivot(A.min.clone().add(A.max).multiplyScalar(0.5), size.length() / 2);

  // Faint outline of the design space.
  const outline = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x, size.y, size.z)),
    new THREE.LineBasicMaterial({ color: 0x35506f, transparent: true, opacity: 0.8 }));
  outline.position.copy(A.min.clone().add(A.max).multiplyScalar(0.5));
  decor.add(outline);

  const addBox = (b, color) => {
    if (!b) return null;
    const w = toWorld(b), pad = 0.35 * h;
    const sz = w.max.clone().sub(w.min).addScalar(2 * pad);
    const g = new THREE.BoxGeometry(sz.x, sz.y, sz.z);
    const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
      color, transparent: true, opacity: 0.6, roughness: 0.6, emissive: color, emissiveIntensity: 0.25,
      depthWrite: false }));
    m.position.copy(w.min.clone().add(w.max).multiplyScalar(0.5));
    m.renderOrder = 2;
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color }));
    m.add(e);
    decor.add(m);
    return w;
  };
  componentBoxes(lab, nx, ny, nz, 2).forEach((b) => addBox(b, 0x3b82f6));
  componentBoxes(lab, nx, ny, nz, 3).forEach((b) => addBox(b, 0xffb020));
  addLoadArrows(size);

  // Floor grid below the model.
  floor = new THREE.GridHelper(Math.max(size.x, size.y) * 2.4, 24, 0x2a3b52, 0x1b2738);
  floor.rotation.x = Math.PI / 2;
  floor.position.set(center.x, center.y, A.min.z - size.z * 0.45);
  scene.add(floor);

  resize();
}

function resize() {
  const view = $("view");
  const w = view.clientWidth, h = view.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  const vfov = THREE.MathUtils.degToRad(camera.fov);
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
  const dist = (radius * 1.08) / Math.sin(Math.min(vfov, hfov) / 2);
  camera.position.set(center.x + dist * 0.2, center.y - dist * 0.9, center.z + dist * 0.45);
  camera.lookAt(center.x, center.y, center.z - radius * 0.05);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe($("view"));

// Make a geometry from the mesh binary format.
function geometryFromBuffer(buf) {
  const dv = new DataView(buf);
  const nv = dv.getUint32(0, true), nt = dv.getUint32(4, true);
  if (nv === 0 || nt === 0) return null;
  const pos = new Float32Array(buf.slice(8, 8 + nv * 12));
  const idx = new Uint32Array(buf.slice(8 + nv * 12, 8 + nv * 12 + nt * 12));
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  return g;
}

// Average each normal with the normals of its neighbours. This hides the
// small voxel steps in the shading. It does not move any vertex.
function smoothNormals(g, passes = 3) {
  const idx = g.index.array, nrm = g.attributes.normal;
  const n = nrm.count;
  let a = nrm.array, b = new Float32Array(a.length);
  for (let p = 0; p < passes; p++) {
    b.set(a); // each vertex counts once for itself
    for (let t = 0; t < idx.length; t += 3) {
      for (let e = 0; e < 3; e++) {
        const i = idx[t + e] * 3, j = idx[t + (e + 1) % 3] * 3;
        b[i] += a[j]; b[i + 1] += a[j + 1]; b[i + 2] += a[j + 2];
        b[j] += a[i]; b[j + 1] += a[i + 1]; b[j + 2] += a[i + 2];
      }
    }
    for (let i = 0; i < n * 3; i += 3) {
      const l = Math.hypot(b[i], b[i + 1], b[i + 2]) || 1;
      b[i] /= l; b[i + 1] /= l; b[i + 2] /= l;
    }
    [a, b] = [b, a];
  }
  nrm.array.set(a);
  nrm.needsUpdate = true;
}

// Swap the mesh in one step, with a short fade, so there is no flash.
let currentMesh = null;
function swapMesh(geom, material) {
  const old = currentMesh;
  currentMesh = null;
  if (geom) {
    const m = new THREE.Mesh(geom, material);
    currentMesh = m;
    meshHolder.add(m);
  }
  if (old) {
    meshHolder.remove(old);
    old.geometry.dispose();
  }
}

// ---------------------------------------------------------------- loop
let lastT = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;
  if (S.finalShown) {
    // One turn in 14 seconds, always in the same direction.
    spin.rotation.z += dt * (Math.PI * 2 / 14);
  } else {
    spin.rotation.z = 0.5 * Math.sin(now / 1000 * 0.35) - 0.25; // slow sway
  }
  renderer.render(scene, camera);
  updateTimer();
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- code panel
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const KW = /\b(def|for|in|return|if|else|elif|import|from|as|with|while|not|and|or|None|True|False)\b/g;
function highlight(line) {
  let cm = "";
  const hash = line.indexOf("#");
  if (hash >= 0 && !/["'].*#.*["']/.test(line)) { cm = line.slice(hash); line = line.slice(0, hash); }
  let out = esc(line)
    .replace(/(&quot;|"[^"]*"|'[^']*')/g, '<span class="st">$1</span>')
    .replace(KW, '<span class="kw">$1</span>');
  return out + (cm ? `<span class="cm">${esc(cm)}</span>` : "");
}

function buildCode() {
  const src = S.source;
  $("srcname").textContent = src.path;
  const code = $("code");
  code.innerHTML = "";
  src.lines.forEach((t, i) => {
    const d = document.createElement("div");
    d.className = "ln";
    d.dataset.n = i + 1;
    d.innerHTML = `<span class="no">${i + 1}</span><span class="tx">${highlight(t) || " "}</span>`;
    code.appendChild(d);
  });
  const steps = $("steps");
  steps.innerHTML = "";
  const names = Object.keys(src.steps).sort(
    (a, b) => src.steps[a][0] - src.steps[b][0]);
  names.forEach((n) => {
    const s = document.createElement("span");
    s.textContent = n; s.dataset.step = n;
    steps.appendChild(s);
  });
}

function setActiveStep(step) {
  if (!S.source || step === S.activeStep) return;
  S.activeStep = step;
  const range = S.source.steps[step];
  const lines = $("code").children;
  for (let i = 0; i < lines.length; i++) {
    const n = i + 1;
    lines[i].classList.toggle("on", !!range && n >= range[0] && n <= range[1]);
  }
  const names = [...$("steps").children];
  const at = names.findIndex((s) => s.dataset.step === step);
  names.forEach((s, i) => {
    s.classList.toggle("on", i === at);
    s.classList.toggle("past", at >= 0 && i < at);
  });
  if (range) { // smooth scroll so the block is in the upper third
    const code = $("code");
    const first = lines[range[0] - 1], last = lines[range[1] - 1];
    const blockH = last.offsetTop + last.offsetHeight - first.offsetTop;
    let top = first.offsetTop - (code.clientHeight - blockH) / 3;
    top = Math.max(0, Math.min(top, code.scrollHeight - code.clientHeight));
    code.scrollTo({ top, behavior: "smooth" });
  }
}

// The demo screen is the code panel and the row "Pull-off stiffness per gram" (see talk.js for the rule). Without it the side
// panel shows the log (run.log) and, at the end, the result rows.
function applyTalk(talk) {
  S.talk = talk;
  $("codepanel").classList.toggle("plain", !talk);
  $("code").hidden = !talk;
  $("steps").hidden = !talk;
  $("row-perg").hidden = !talk;
  $("srcname").textContent = talk ? "optimize.py" : "run.log";
}

// ---------------------------------------------------------------- bottom bar
function drawSpark() {
  const c = $("spark"), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (!w || !h) return;
  if (c.width !== w * dpr || c.height !== h * dpr) { c.width = w * dpr; c.height = h * dpr; }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const ys = S.history.map((r) => r.compliance).filter((v) => isFinite(v) && v > 0);
  if (ys.length < 2) return;
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const N = Math.max(S.run ? S.run.iterations : ys.length, ys.length);
  const px = (i) => 6 + (w - 12) * (i / (N - 1));
  const py = (v) => h - 8 - (h - 16) * ((v - lo) / Math.max(hi - lo, 1e-12));
  g.beginPath();
  ys.forEach((v, i) => (i ? g.lineTo(px(i), py(v)) : g.moveTo(px(i), py(v))));
  g.lineWidth = 3; g.lineJoin = "round"; g.strokeStyle = "#4cc2ff"; g.stroke();
  const last = ys.length - 1;
  g.beginPath(); g.arc(px(last), py(ys[last]), 5, 0, 7); g.fillStyle = "#fff"; g.fill();
}

function updateBar() {
  const st = S.status, run = S.run;
  if (run && st) {
    $("b-iter").innerHTML = `${st.iteration}<small> / ${run.iterations}</small>`;
  }
  const last = S.history[S.history.length - 1];
  if (last) {
    $("b-vol").innerHTML = `${(last.volume_fraction * 100).toFixed(1)}<small> %</small>`;
    $("b-comp").textContent = fmtNum(last.compliance);
  }
  drawSpark();
}

function updateTimer() {
  const st = S.status;
  if (!st) return;
  let t = st.elapsed_s;
  if (st.state === "starting" || st.state === "running" || st.state === "postprocess") {
    t += (performance.now() - S.statusAt) / 1000;
  }
  $("b-time").textContent = fmtTime(t);
}

// "about 93 s left": the live estimate of the engine (E1-P2). The server puts it in the status (`time_left`) once the run has made
// 2 passes. The page only shows it; the title has the bound. No estimate (a run before pass 2, a run that is not in its passes): no text.
function leftText(st) {
  const t = st && st.state === "running" ? st.time_left : null;
  if (!t || typeof t.left_s !== "number" || !isFinite(t.left_s)) return { text: "", title: "" };
  const b = t.basis || {};
  return { text: `about ${Math.round(t.left_s)} s left`,
    title: `up to ${Math.round(t.up_to_s)} s` + (b.assumes ? `; ${b.assumes}` : "") };
}

function updateLeft() {
  const el = $("b-left"), l = leftText(S.status);
  el.textContent = l.text; el.title = l.title;      // the span stays in the layout when it is empty: the bar does not move
}

// ---------------------------------------------------------------- state
function setPill(st) {
  const p = $("pill");
  const map = { starting: "STARTING", running: "RUNNING", postprocess: "POST-PROCESS", done: "DONE", error: "ERROR",
    stopped: "STOPPED" };
  p.textContent = (map[st.state] || st.state.toUpperCase()) +
    (st.state === "running" ? "  ·  " + st.step : "");
  p.className = "pill " + (st.state === "done" ? "done" : st.state === "error" ? "error" : st.state === "stopped" ? "stopped" : "");
  const b = $("banner");
  b.hidden = st.state !== "error" && st.state !== "stopped"; // a stop by request shows the status message (the pass, and how to resume)
  b.className = st.state === "stopped" ? "stopped" : "";
  if (!b.hidden) b.textContent = st.message || (st.state === "stopped" ? "The run was stopped by a request." : "The run failed.");
}

async function pollStatus() {
  const gen = S.gen;
  const st = await getJSON("/api/status");
  if (st && S.run && gen === S.gen) {
    const changed = !S.status || st.iteration !== S.status.iteration || st.state !== S.status.state;
    S.status = st; S.statusAt = performance.now();
    setPill(st);
    updateLeft();
    $("replay-badge").hidden = st.mode !== "replay"; // quiet label in replay mode
    if (st.state === "running" || st.state === "starting") setActiveStep(st.step);
    if (st.state === "done" || st.state === "postprocess" || st.state === "stopped") setActiveStep(null);
    if (changed) updateBar();
    if (st.state === "done" && !S.finalShown && !S.doneStarted) showFinal();
  }
  setTimeout(pollStatus, 300);
}

async function pollHistory() {
  const gen = S.gen;
  const h = await getJSON("/api/history");
  if (h && S.run && gen === S.gen) {
    if (h.length !== S.history.length) { S.history = h; updateBar(); }
  }
  setTimeout(pollHistory, S.finalShown ? 2000 : 600);
}

async function pollMesh() {
  const gen = S.gen;
  if (!S.finalShown && S.run) {
    const r = await getBuf("/api/mesh/latest");
    if (r && gen === S.gen && !S.finalShown) {
      const it = Number(r.it);
      if (it !== S.shownIter) {
        S.shownIter = it;
        swapMesh(geometryFromBuffer(r.buf), previewMat);
      }
    }
  }
  setTimeout(pollMesh, 450);
}

async function pollLog() {
  try {
    const r = await fetch("/api/log", { cache: "no-store" });
    if (r.ok) $("log").textContent = (await r.text()).split("\n").slice(S.talk ? -6 : -30).join("\n");
  } catch (e) { /* ignore */ }
  setTimeout(pollLog, 1000);
}

// The refine page is made after the run is done. Show the link when the file exists.
async function pollRefine() {
  const gen = S.gen;
  if (S.finalShown && !S.refineFound) {
    const r = await getJSON("/api/refine");
    if (r && r.exists && gen === S.gen) {
      S.refineFound = true;
      $("refine-link").hidden = false;
    }
  }
  setTimeout(pollRefine, 2000);
}

async function showFinal() {
  S.doneStarted = 1;
  const gen = S.gen;
  const [fin, met, part] = await Promise.all([getBuf("/api/final"), getJSON("/api/metrics"), getJSON("/api/part_file")]);
  if (gen !== S.gen) return;
  if (!fin) { S.doneStarted = 0; return; } // try again at the next status poll
  S.finalShown = true;
  const fg = geometryFromBuffer(fin.buf);
  if (fg) smoothNormals(fg);
  swapMesh(fg, finalMat);
  // On the done screen only the arrows stay. Boxes and outline go.
  decor.children.forEach((o) => { o.visible = !!o.userData.keep; });
  if (fg) { // fit the camera to the final mesh, for the full turn
    fg.computeBoundingBox();
    const bc = fg.boundingBox.getCenter(new THREE.Vector3());
    let r2 = 0;
    const pos = fg.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const dx = pos.getX(i) - bc.x, dy = pos.getY(i) - bc.y, dz = pos.getZ(i) - bc.z;
      r2 = Math.max(r2, dx * dx + dy * dy + dz * dz);
    }
    setPivot(bc, Math.sqrt(r2));
  }
  spin.rotation.z = 0;
  $("legend").style.display = "none";
  if (met) fillMetrics(met);
  fillPartFile(part);
  if (S.talk) $("codepanel").hidden = true;        // the demo screen: the result card has the whole side panel
  else $("codepanel").classList.add("short");      // else the log stays under the result rows
  $("result").hidden = false;
  if (met) {
    $("b-time").textContent = fmtTime(met.run_time_s);
    S.status = { ...S.status, elapsed_s: met.run_time_s, state: "done" };
  }
}

function fillMetrics(m) {
  $("m-before").textContent = m.mass_design_space_g.toFixed(0);
  $("m-after").textContent = m.mass_final_g.toFixed(1);
  $("m-saved").textContent = m.mass_saved_pct.toFixed(0) + " %";
  $("m-stiff").innerHTML = (m.stiffness_cases || []).map((c) =>
    c.N_per_mm == null  // null: the case has a moment, a remote force or a pressure, or applies no load (see `reason`)
      ? `<div class="row" title="${esc(c.reason || "").replace(/"/g, "&quot;")}"><span>Stiffness: ${esc(c.name)}</span><b>n/a</b></div>`
      : `<div class="row"><span>Stiffness: ${esc(c.name)}</span><b>${c.N_per_mm.toFixed(0)} <small>N/mm</small></b></div>`).join("");
  const t = m.run_time_s;
  $("m-time").textContent = `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")} min`;
  // Stiffness per gram for the pull_off case (no compare with the gray start).
  const po = (m.stiffness_cases || []).find((c) => c.name === "pull_off");
  $("m-perg").innerHTML = po && po.N_per_mm != null && m.mass_final_g > 0
    ? `${(po.N_per_mm / m.mass_final_g).toFixed(0)} <small>N/mm per g</small>` : "-";
}

// The part file: the run folder and the path of result/final.stl in it. The full path is in the tooltip.
function fillPartFile(info) {
  const text = partFileText(info);
  $("partfile").hidden = !text;
  if (!text) return;
  $("m-file").textContent = text;
  $("m-file").title = `${info.run_dir}/${info.file}`;
}

// ---------------------------------------------------------------- boot
// Load a run: labels, scene, source, history. Return false if a file is not there yet.
async function loadRun(run) {
  const lab = await getBuf("/api/labels");
  if (!lab) return false;
  S.labels = new Uint8Array(lab.buf);
  S.run = run;
  applyTalk(talkScreen(location.search, run));
  $("title").textContent = run.name || "";
  buildSceneFromRun();
  $("view").classList.remove("waiting");
  S.source = S.talk ? await getJSON("/api/source") : null;
  if (S.source) buildCode();
  S.history = (await getJSON("/api/history")) || [];
  updateBar();
  return true;
}

// Go back to the empty state: no reload of the page.
function resetView() {
  S.gen++;
  S.run = null; S.labels = null; S.status = null; S.history = [];
  S.shownIter = -1; S.finalShown = false; S.doneStarted = 0; S.activeStep = null;
  S.refineFound = false; $("refine-link").hidden = true;
  clearScene();
  spin.rotation.z = 0;
  $("code").innerHTML = ""; $("steps").innerHTML = ""; $("log").textContent = "";
  applyTalk(talkScreen(location.search, null));
  $("codepanel").hidden = false; $("codepanel").classList.remove("short"); $("result").hidden = true; $("partfile").hidden = true;
  $("legend").style.display = "";
  $("banner").hidden = true;
  $("pill").textContent = "WAITING"; $("pill").className = "pill";
  $("replay-badge").hidden = true;
  $("title").textContent = "";
  $("b-iter").textContent = "-"; $("b-time").textContent = "00:00"; updateLeft();
  $("b-vol").textContent = "-"; $("b-comp").textContent = "-";
  drawSpark();
  $("view").classList.add("waiting");
}

// Look at run.json. A new `started_at` (or no file) means a new run.
let watching = false;
async function watchRun() {
  if (!watching) {
    watching = true;
    try {
      const { run, missing } = await getRun();
      if (!run) {
        if (missing && S.run) resetView();   // a network error keeps the view: the next tick asks again
      } else if (!S.run || run.started_at !== S.run.started_at) {
        if (S.run) resetView();
        await loadRun(run); // if labels.bin is not there yet, try again at the next tick
      }
    } finally { watching = false; }
  }
  setTimeout(watchRun, 700);
}

applyTalk(talkScreen(location.search, null));   // before a run is known, only ?demo=1 shows the demo screen
$("view").classList.add("waiting");
requestAnimationFrame(frame);
watchRun(); pollStatus(); pollHistory(); pollMesh(); pollLog(); pollRefine();
window.addEventListener("resize", drawSpark);
