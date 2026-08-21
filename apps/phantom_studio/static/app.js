// Phantom Studio front end.
//
// Division of labour with the server: the server owns physics and ships
// numbers (uint8 slices and volumes plus a value range); this file owns
// nothing but presentation — colour mapping, layout, interaction. That is why
// switching field or colour ramp is instant and never re-runs a build.
//
// Colour rules, applied in both the 2-D slices and the 3-D view so a tissue
// looks the same everywhere:
//   * categorical (labels) — the validated tissue palette from the server;
//   * sequential (c, rho, alpha, beta, impedance) — ONE hue, dark->light,
//     generated in OKLCH so lightness is monotone by construction. Never a
//     rainbow: a rainbow ramp invents banding that reads as structure.

import { VolumeView } from "/volume.js";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  server: null,
  build: null,
  raw: { slices: {}, volume: null },
  // Monotonic request tokens. A slider drag fires overlapping fetches that do
  // not return in order; without a token the last RESPONSE wins instead of
  // the last REQUEST, so the view settles on whichever slice was slowest.
  seq: { slices: {}, volume: 0 },
  hidden: new Set(),
  isolated: null,
  view: null,
  busy: false,
  pendingPreview: null,
};

// ───────────────────────────────── colour ──────────────────────────────────

// Minimal OKLCH -> sRGB. Used only for the sequential ramps; the categorical
// palette arrives from the server already validated.
function oklch(L, C, H) {
  const h = (H * Math.PI) / 180;
  const a = C * Math.cos(h);
  const b = C * Math.sin(h);
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
  const lin = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ];
  return lin.map((v) => {
    const c = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(Math.max(v, 0), 1 / 2.4) - 0.055;
    return Math.round(Math.max(0, Math.min(1, c)) * 255);
  });
}

const RAMPS = {
  // dark surface: low values sit near the surface, high values are bright
  blue: { hue: 256, chroma: 0.115, L: [0.22, 0.93] },
  amber: { hue: 68, chroma: 0.125, L: [0.22, 0.93] },
  gray: { hue: 0, chroma: 0.0, L: [0.16, 0.96] },
};

function rampLUT(name) {
  const r = RAMPS[name] || RAMPS.blue;
  const out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const L = r.L[0] + (r.L[1] - r.L[0]) * t;
    // taper chroma at both ends so the extremes stay readable
    const C = r.chroma * Math.sin(Math.PI * (0.15 + 0.7 * t));
    const [R, G, B] = oklch(L, C, r.hue);
    out[i * 4] = R; out[i * 4 + 1] = G; out[i * 4 + 2] = B; out[i * 4 + 3] = 255;
  }
  return out;
}

const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

/** 256-entry RGBA lookup used by BOTH the slice canvases and the ray caster.
 *
 * `field` is passed in, never read from the selector. Every buffer is tagged
 * with the field that produced it, so bytes can only ever be coloured and
 * read back by their own field. Reading the selector here instead lets a
 * slow response land after the user has switched: the old field's bytes get
 * the new field's ramp and range, and the hover readout reports a number
 * that was never in the data.
 */
function buildLUT({ forVolume, field }) {
  const lut = new Uint8Array(256 * 4);
  if (field !== "labels") {
    const ramp = rampLUT($("#ramp").value);
    lut.set(ramp);
    if (forVolume) {
      // Continuous field in 3-D: opacity follows DISTANCE FROM THE BATH, not
      // the raw value. Water's sound speed sits mid-range and fills ~60% of
      // the box, so "opacity rises with value" renders a solid blue block
      // with the breast invisible inside it. Measuring from the bath makes
      // both the faster tissue (skin, muscle) and the slower tissue (fat)
      // stand out, and the water itself vanish.
      const meta = state.raw.volume ? state.raw.volume.meta : null;
      const bath = meta && meta.bath_u8 != null ? meta.bath_u8 : 0;
      const reach = Math.max(bath, 255 - bath) || 1;
      for (let i = 0; i < 256; i++) {
        const d = Math.abs(i - bath) / reach;
        lut[i * 4 + 3] = Math.round(255 * Math.pow(d, 1.4) * 0.7);
      }
    }
    return lut;
  }
  const mats = state.build ? state.build.materials : [];
  for (const m of mats) {
    // A material with no voxels contributes nothing to the image. The
    // `simple` model carries an unused PML slot at id 0; giving it a colour
    // put a stray swatch in the legend for a tissue that is not there.
    if (!m.voxels) continue;
    const [r, g, b] = hex2rgb(m.color);
    const i = m.id * 4;
    lut[i] = r; lut[i + 1] = g; lut[i + 2] = b;
    let a = 255;
    if (state.hidden.has(m.id)) a = 0;
    else if (state.isolated !== null && state.isolated !== m.id) a = 0;
    else if (forVolume) {
      // The coupling bath is the background of the image, not a tissue:
      // opaque here would hide everything behind it.
      a = m.id === couplingId() ? 0 : 190;
    }
    lut[i + 3] = a;
  }
  return lut;
}

function couplingId() {
  const mats = state.build ? state.build.materials : [];
  // Class code 0 IS the immersion medium — match on that, never on the name.
  // Four of the ten detailed tissue names contain "water", and the `simple`
  // model carries a zero-voxel placeholder at id 0 with the same name as the
  // real bath at id 4; matching by name left 52% of that volume opaque in 3-D.
  const m = mats.find((x) => (x.codes || []).includes(0));
  return m ? m.id : -1;   // -1: make nothing transparent rather than guess
}

// ───────────────────────────────── spec ────────────────────────────────────

function readSpec() {
  const props = $$("#noiseProps input:checked").map((el) => el.value);
  return {
    phantom_id: $("#phantomId").value,
    f0_mhz: parseFloat($("#f0").value),
    simplify: {
      tissue_model: $("#tissueModel .active").dataset.value,
      drop_muscle: $("#dropMuscle").checked,
      drop_skin: $("#dropSkin").checked,
      keep_largest_only: $("#keepLargest").checked,
      fill_holes: $("#fillHoles").checked,
      remove_islands_vox: parseInt($("#islands").value, 10),
      close_skin_iterations: parseInt($("#closeSkin").value, 10),
      smooth_iterations: parseInt($("#smooth").value, 10),
    },
    resolution: { dx_mm: parseFloat($("#dx").value), method: $("#resampleMethod").value },
    crop: { mode: $("#cropMode").value, margin_mm: parseFloat($("#margin").value) },
    domain: {
      standoff_mm: parseFloat($("#standoff").value),
      backing_mm: parseFloat($("#backing").value),
      lateral_margin_mm: parseFloat($("#lateral").value),
      fft_friendly: $("#fftFriendly").checked,
    },
    heterogeneity: {
      use_pval: $("#usePval").checked,
      noise_pct: parseFloat($("#noise").value),
      correlation_mm: parseFloat($("#corr").value),
      properties: props.length ? props : ["rho"],
      coupled: $("#coupled").checked,
      tissue_only: $("#tissueOnly").checked,
      seed: parseInt($("#seed").value, 10) || 0,
    },
    name: $("#exportName").value || null,
  };
}

// ───────────────────────────────── net ─────────────────────────────────────

async function api(path, body) {
  const res = await fetch(path, body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : undefined);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

async function apiBinary(path) {
  const res = await fetch(path);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error || msg; } catch { /* binary endpoint, no JSON */ }
    throw new Error(msg);
  }
  const meta = JSON.parse(res.headers.get("X-Meta"));
  return { data: new Uint8Array(await res.arrayBuffer()), meta };
}

function setStatus(text, kind) {
  const pill = $("#statusPill");
  pill.textContent = kind === "busy" ? "working…" : kind === "error" ? "error" : "ready";
  pill.className = "pill" + (kind ? " " + kind : "");
  $("#statusText").textContent = text || "";
}

// ───────────────────────────────── build ───────────────────────────────────

let planTimer = null;
function schedulePlan() {
  clearTimeout(planTimer);
  planTimer = setTimeout(async () => {
    try {
      const p = await api("/api/plan", { spec: readSpec() });
      const over = p.n_voxels > state.server.preview_mvox * 1e6;
      const approx = p.exact ? "" : "≤ ";  // simplification can shrink it
      $("#planLine").innerHTML =
        `${approx}${p.shape.join("×")} @ ${p.dx_mm.toFixed(2)} mm` +
        `<br>${approx}${(p.n_voxels / 1e6).toFixed(1)} Mvox · ~${p.peak_gb.toFixed(2)} GB` +
        (over ? `<br><span class="over">preview will coarsen this</span>` : "");
    } catch (err) {
      $("#planLine").textContent = String(err.message).slice(0, 120);
    }
  }, 180);
}

let previewTimer = null;
function schedulePreview() {
  if (!$("#autoPreview").checked) return;
  clearTimeout(previewTimer);
  previewTimer = setTimeout(() => runBuild(true), 420);
}

async function runBuild(preview) {
  if (state.busy) { state.pendingPreview = preview; return; }
  state.busy = true;
  setStatus(preview ? "building preview" : "building at full resolution", "busy");
  $("#previewBtn").disabled = $("#fullBtn").disabled = true;
  try {
    const build = await api("/api/build", { spec: readSpec(), preview });
    state.build = build;
    state.hidden.clear();
    state.isolated = null;
    renderBuild();
    await Promise.all([loadVolume(), loadAllSlices()]);
    // loadAllSlices re-centres the sliders on a shape change, so the cutaway
    // has to follow them here — not only on user input.
    applyCutaway();
    setStatus(
      `${build.shape.join("×")} @ ${build.dx_mm.toFixed(2)} mm · ` +
      `${(build.n_voxels / 1e6).toFixed(1)} Mvox · ${build.seconds}s` +
      (build.preview ? ` · PREVIEW (requested ${build.requested_dx_mm.toFixed(2)} mm)` : "")
    );
  } catch (err) {
    setStatus(err.message, "error");
  } finally {
    state.busy = false;
    $("#previewBtn").disabled = $("#fullBtn").disabled = false;
    if (state.pendingPreview !== null) {
      const p = state.pendingPreview;
      state.pendingPreview = null;
      runBuild(p);
    }
  }
}

function renderBuild() {
  const b = state.build;
  $("#buildSummary").textContent = b.summary;

  const warn = $("#warnings");
  warn.innerHTML = "";
  if (b.preview) {
    const d = document.createElement("div");
    d.textContent =
      `Preview only: rendered at ${b.dx_mm.toFixed(2)} mm because ${b.requested_dx_mm.toFixed(2)} mm ` +
      `exceeds the interactive voxel budget. Press "Build full" before exporting.`;
    warn.appendChild(d);
  }
  for (const w of b.warnings) {
    const d = document.createElement("div");
    d.textContent = w;
    warn.appendChild(d);
  }

  const log = $("#buildLog");
  log.innerHTML = "";
  for (const line of b.log) {
    const li = document.createElement("li");
    li.textContent = line;
    log.appendChild(li);
  }

  renderLegend();
  renderSnippet();
}

function renderLegend() {
  const b = state.build;
  const table = $("#legend");
  table.innerHTML = "";
  for (const m of b.materials) {
    if (!m.voxels) continue;   // not present in this build
    const tr = document.createElement("tr");
    tr.dataset.id = m.id;
    if (state.hidden.has(m.id)) tr.classList.add("dim");
    if (state.isolated === m.id) tr.classList.add("isolated");
    const alpha = m.alpha_np_m.toFixed(2);
    tr.innerHTML = `
      <td><input type="checkbox" ${state.hidden.has(m.id) ? "" : "checked"}></td>
      <td><span class="sw" style="background:${m.color}"></span></td>
      <td class="nm">${m.name}${m.interpolated ? ' <span class="interp" title="interpolated along the water-content ordering, not a measured tissue">~</span>' : ""}</td>
      <td class="num">${(100 * m.fraction).toFixed(2)}%</td>
      <td class="num">${m.c_mid.toFixed(0)} m/s</td>
      <td class="num">${m.rho_mid.toFixed(0)}</td>
      <td class="num">${alpha} Np/m</td>
      <td class="num">β ${m.beta_mid.toFixed(2)}</td>`;
    tr.querySelector("input").addEventListener("change", (e) => {
      e.stopPropagation();
      if (e.target.checked) state.hidden.delete(m.id); else state.hidden.add(m.id);
      refreshColors();
      renderLegend();
    });
    tr.addEventListener("click", () => {
      state.isolated = state.isolated === m.id ? null : m.id;
      refreshColors();
      renderLegend();
    });
    table.appendChild(tr);
  }
}

function renderSnippet() {
  const b = state.build;
  const name = ($("#exportName").value || b.name) + ".npz";
  // Forward slashes: Python takes them on every platform, and the server may
  // be a Linux box (Colab) even when the browser is not. A hardcoded "\\"
  // produced a snippet that only pasted correctly on Windows.
  const path = `${state.server.export_dir.replace(/\\/g, "/")}/${name}`;
  $("#snippet").textContent =
`import hifusim as hs
from uwcem_phantoms import load_phantom

ph     = load_phantom("${path}")
grid   = ph.grid(pml=hs.PMLSpec(thickness=5e-3))   # ${b.shape.join(" x ")} @ ${b.dx_mm.toFixed(2)} mm
medium = ph.to_medium()                            # ${b.continuous ? "dense per-voxel properties" : "labels + MaterialDB"}

# the same file is also a plain hifusim label volume:
from hifusim.geometry import LabelVolume
vol = LabelVolume.load_npz("${path}")`;
}

// ───────────────────────────────── views ───────────────────────────────────

async function loadVolume() {
  if (!state.view || !state.build) return;
  const field = $("#field").value;
  const token = ++state.seq.volume;
  const { data, meta } = await apiBinary(
    `/api/volume?build=${encodeURIComponent(state.build.key)}&max_dim=150&field=${field}`);
  if (token !== state.seq.volume) return;     // a newer request already won
  state.raw.volume = { data, meta, field };   // set BEFORE buildLUT — it reads bath_u8
  state.view.setVolume(data, meta.shape, meta.extent_mm);
  state.view.setTransfer(buildLUT({ forVolume: true, field }));
}

function sliceFigures() { return $$(".slice"); }

async function loadAllSlices() {
  await Promise.all(sliceFigures().map((fig) => loadSlice(fig)));
}

async function loadSlice(fig) {
  if (!state.build) return;
  const axis = parseInt(fig.dataset.axis, 10);
  const slider = fig.querySelector("input[type=range]");
  const n = state.build.shape[axis];
  if (parseInt(slider.max, 10) !== n - 1) {
    slider.max = String(n - 1);
    slider.value = String(Math.floor(n / 2));
  }
  const field = $("#field").value;
  const token = (state.seq.slices[axis] = (state.seq.slices[axis] || 0) + 1);
  const { data, meta } = await apiBinary(
    `/api/slice?build=${encodeURIComponent(state.build.key)}&axis=${axis}` +
    `&index=${slider.value}&field=${field}`);
  if (token !== state.seq.slices[axis]) return;   // superseded mid-flight
  state.raw.slices[axis] = { data, meta, axis, field };
  paintSlice(fig);
}

function paintSlice(fig) {
  const axis = parseInt(fig.dataset.axis, 10);
  const rec = state.raw.slices[axis];
  if (!rec) return;
  const { data, meta } = rec;
  const canvas = fig.querySelector("canvas");
  canvas.width = meta.width;
  canvas.height = meta.height;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(meta.width, meta.height);
  const lut = buildLUT({ forVolume: false, field: rec.field });
  const bg = hex2rgb("#0b0f14");
  for (let i = 0, p = 0; i < data.length; i++, p += 4) {
    const v = data[i] * 4;
    const a = lut[v + 3] / 255;
    img.data[p] = lut[v] * a + bg[0] * (1 - a);
    img.data[p + 1] = lut[v + 1] * a + bg[1] * (1 - a);
    img.data[p + 2] = lut[v + 2] * a + bg[2] * (1 - a);
    img.data[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);

  const dx = state.build.dx_mm;
  const field = rec.field;
  const pos = (meta.index * dx).toFixed(1);
  fig.querySelector(".slice-meta").textContent =
    `${["x", "y", "z"][axis]} = ${meta.index} / ${meta.n - 1}  (${pos} mm)` +
    (field === "labels" ? "" : `   range ${fmt(meta.vmin)} … ${fmt(meta.vmax)}`);
}

function fmt(v) {
  if (Math.abs(v) >= 1e5) return v.toExponential(2);
  if (Math.abs(v) >= 100) return v.toFixed(0);
  return v.toFixed(3);
}

function refreshColors() {
  for (const fig of sliceFigures()) paintSlice(fig);
  const vol = state.raw.volume;
  if (state.view && vol) state.view.setTransfer(buildLUT({ forVolume: true, field: vol.field }));
}

function applyCutaway() {
  if (!state.view || !state.build) return;
  // The notch corner IS the slice-slider position, so the three faces the
  // cutaway exposes are exactly the three cross-sections shown below it.
  const corner = [0, 1, 2].map((axis) => {
    const fig = sliceFigures().find((f) => parseInt(f.dataset.axis, 10) === axis);
    const s = fig.querySelector("input[type=range]");
    return Math.min(1, Math.max(0, (parseInt(s.value, 10) + 0.5) / state.build.shape[axis]));
  });
  state.view.setNotch(corner, $("#cutaway").checked);
}

// ───────────────────────────── hover readout ───────────────────────────────

function installHover(fig) {
  const wrap = fig.querySelector(".canvas-wrap");
  const canvas = fig.querySelector("canvas");
  const readout = fig.querySelector(".readout");
  wrap.addEventListener("pointermove", (e) => {
    const axis = parseInt(fig.dataset.axis, 10);
    const rec = state.raw.slices[axis];
    if (!rec || !state.build) return;
    // object-fit: contain — map client point back through the letterbox
    const r = canvas.getBoundingClientRect();
    const scale = Math.min(r.width / canvas.width, r.height / canvas.height);
    const ox = (r.width - canvas.width * scale) / 2;
    const oy = (r.height - canvas.height * scale) / 2;
    const px = Math.floor((e.clientX - r.left - ox) / scale);
    const py = Math.floor((e.clientY - r.top - oy) / scale);
    if (px < 0 || py < 0 || px >= canvas.width || py >= canvas.height) {
      readout.textContent = "";
      return;
    }
    const raw = rec.data[py * rec.meta.width + px];
    const field = rec.field;
    if (field === "labels") {
      const m = state.build.materials.find((x) => x.id === raw);
      readout.textContent = m ? `${m.name}  (id ${m.id})` : `id ${raw}`;
    } else {
      const v = rec.meta.vmin + (raw / 255) * (rec.meta.vmax - rec.meta.vmin);
      readout.textContent = `${field} ≈ ${fmt(v)} ${UNITS[field] || ""}`;
    }
  });
}

const UNITS = { c: "m/s", rho: "kg/m³", alpha: "Np/m", beta: "", impedance: "Rayl" };

// ───────────────────────────────── wiring ──────────────────────────────────

function bindSlider(id, fmtFn) {
  const el = $(id);
  const out = $(id + "Val");
  const update = () => { if (out) out.textContent = fmtFn(el.value); };
  el.addEventListener("input", update);
  update();
  return el;
}

async function init() {
  state.server = await api("/api/state");
  $("#citation").textContent = state.server.citation;

  const sel = $("#phantomId");
  for (const p of state.server.catalog) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = `${p.id} — ACR ${p.acr_class} · ${p.shape.join("×")}`;
    sel.appendChild(opt);
  }
  sel.value = state.server.catalog[state.server.catalog.length - 1].id;
  updatePhantomInfo();

  try {
    state.view = new VolumeView($("#gl"));
    const ro = new ResizeObserver(() => state.view.render());
    ro.observe($("#gl"));
  } catch (err) {
    $("#gl").classList.add("hidden");
    const box = $("#glFallback");
    box.classList.remove("hidden");
    box.textContent =
      "3-D view unavailable: " + err.message +
      ". The cross-sections below still work.";
  }

  // ---- sidebar bindings
  bindSlider("#f0", (v) => `${(+v).toFixed(2)} MHz`);
  bindSlider("#dx", (v) => `${(+v).toFixed(2)} mm`);
  bindSlider("#margin", (v) => `${v} mm`);
  bindSlider("#standoff", (v) => `${v} mm`);
  bindSlider("#backing", (v) => `${v} mm`);
  bindSlider("#lateral", (v) => `${v} mm`);
  bindSlider("#islands", (v) => (+v ? `${v} vox` : "off"));
  bindSlider("#smooth", (v) => (+v ? `${v}×` : "off"));
  bindSlider("#closeSkin", (v) => (+v ? `${v}×` : "off"));
  bindSlider("#noise", (v) => (+v ? `${(+v).toFixed(1)} %` : "off"));
  bindSlider("#corr", (v) => `${(+v).toFixed(1)} mm`);

  const MODEL_HINTS = {
    detailed: "All ten UWCEM media numbers kept apart.",
    grouped: "Coupling · skin · muscle · fibroglandular · fat.",
    simple:
      "Ids 1–4 as hifusim.materials.breast_default(). The ID SPACE matches so exports "
      + "drop into existing scripts; the property values are this module's literature "
      + "table, not breast_default's (skin absorption is 2.2x higher).",
  };
  $$("#tissueModel button").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$("#tissueModel button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      $("#tissueModelHint").textContent = MODEL_HINTS[btn.dataset.value];
      onSpecChange();
    });
  });

  sel.addEventListener("change", () => { updatePhantomInfo(); onSpecChange(); });
  $("#fetchBtn").addEventListener("click", async () => {
    setStatus("downloading (this can take a minute)", "busy");
    try {
      const res = await api("/api/fetch", { phantom_id: sel.value });
      state.server.catalog = res.catalog;
      updatePhantomInfo();
      setStatus("download complete");
      onSpecChange();
    } catch (err) { setStatus(err.message, "error"); }
  });

  const inputs = [
    "#f0", "#dx", "#resampleMethod", "#cropMode", "#margin", "#standoff", "#backing",
    "#lateral", "#fftFriendly", "#dropMuscle", "#dropSkin", "#fillHoles", "#keepLargest",
    "#islands", "#smooth", "#closeSkin", "#usePval", "#noise", "#corr", "#coupled",
    "#tissueOnly", "#seed",
  ];
  for (const id of inputs) $(id).addEventListener("change", onSpecChange);
  $$("#noiseProps input").forEach((el) => el.addEventListener("change", onSpecChange));
  // sliders also plan live while dragging (cheap; the build is debounced)
  for (const id of ["#dx", "#margin", "#standoff", "#backing", "#lateral"]) {
    $(id).addEventListener("input", schedulePlan);
  }

  $("#previewBtn").addEventListener("click", () => runBuild(true));
  $("#fullBtn").addEventListener("click", () => runBuild(false));
  $("#exportBtn").addEventListener("click", doExport);
  $("#exportName").addEventListener("input", () => { if (state.build) renderSnippet(); });

  // ---- viewer bindings
  $("#field").addEventListener("change", async () => {
    $("#rampGroup").classList.toggle("hidden", $("#field").value === "labels");
    if (!state.build) return;
    setStatus("switching field", "busy");
    try {
      await Promise.all([loadVolume(), loadAllSlices()]);
      setStatus("");
    } catch (err) { setStatus(err.message, "error"); }
  });
  $("#ramp").addEventListener("change", refreshColors);
  $("#density").addEventListener("input", (e) => {
    $("#densityVal").textContent = (+e.target.value).toFixed(2);
    if (state.view) { state.view.density = +e.target.value; state.view.render(); }
  });
  $("#light").addEventListener("input", (e) => {
    if (state.view) { state.view.light = +e.target.value; state.view.render(); }
  });
  $("#cutaway").addEventListener("change", applyCutaway);
  $("#resetCam").addEventListener("click", () => state.view && state.view.resetCamera());
  $("#rampGroup").classList.add("hidden");

  for (const fig of sliceFigures()) {
    installHover(fig);
    const slider = fig.querySelector("input[type=range]");
    slider.addEventListener("input", () => {
      applyCutaway();
      loadSlice(fig).catch((err) => setStatus(err.message, "error"));
    });
  }

  applyUrlState();
  schedulePlan();
  runBuild(true);
}

/** Let the URL carry the view: ?phantom=012304&field=c&ramp=amber&model=grouped
 *
 * Cheap, and it makes the studio linkable — "look at the sound speed in the
 * dense phantom" becomes a URL instead of six clicks of instructions. Unknown
 * or invalid values are ignored rather than throwing the page away. */
function applyUrlState() {
  const q = new URLSearchParams(window.location.search);
  const setSelect = (sel, value) => {
    if (!value) return;
    const el = $(sel);
    if ([...el.options].some((o) => o.value === value)) el.value = value;
  };
  setSelect("#phantomId", q.get("phantom"));
  setSelect("#field", q.get("field"));
  setSelect("#ramp", q.get("ramp"));
  setSelect("#cropMode", q.get("crop"));
  const model = q.get("model");
  if (model) {
    const btn = $$("#tissueModel button").find((b) => b.dataset.value === model);
    if (btn) {
      $$("#tissueModel button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
    }
  }
  const dx = parseFloat(q.get("dx"));
  if (Number.isFinite(dx) && dx > 0) {
    $("#dx").value = String(dx);
    $("#dxVal").textContent = `${dx.toFixed(2)} mm`;
  }
  $("#rampGroup").classList.toggle("hidden", $("#field").value === "labels");
  updatePhantomInfo();
}

function updatePhantomInfo() {
  const p = state.server.catalog.find((x) => x.id === $("#phantomId").value);
  $("#phantomInfo").textContent =
    `ACR ${p.acr_class}: ${p.acr_description} · ${p.extent_mm.join(" × ")} mm at 0.5 mm ` +
    `· ${(p.n_voxels / 1e6).toFixed(1)} Mvox`;
  $("#fetchRow").classList.toggle("hidden", p.has_mtype && p.has_pval);
}

function onSpecChange() {
  schedulePlan();
  schedulePreview();
}

async function doExport() {
  if (!state.build) return;
  setStatus("writing export", "busy");
  try {
    const res = await api("/api/export", {
      build: state.build.key,
      name: $("#exportName").value || null,
      store_properties: $("#storeProps").value,
    });
    const box = $("#exportResult");
    box.classList.remove("hidden");
    box.textContent = `wrote ${res.path}  (${res.mb} MB)`;
    setStatus("exported");
    renderSnippet();
  } catch (err) {
    setStatus(err.message, "error");
    const box = $("#exportResult");
    box.classList.remove("hidden");
    box.textContent = err.message;
  }
}

init().catch((err) => setStatus(err.message, "error"));
