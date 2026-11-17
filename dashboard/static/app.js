/* Station twin dashboard (live). Reads only from the dashboard server:
 *   /api/config  camera poses + intrinsics (from the sim scene), pivot
 *   /api/status  source health, focus block and anomaly overlays (stubs until a service posts)
 *   /api/cloud   fused D415 point cloud (int16 x5 records), long-polled
 *   /frame/{sim,real}.jpg?after=SEQ  live Top camera frames, pulled one at a time so the rate
 *                     fits the link (the hidden source refreshes once a second)
 */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const css = (el, name, fallback) => getComputedStyle(el).getPropertyValue(name).trim() || fallback;
  const hex = (h) => { const v = parseInt(h.replace("#", ""), 16); return [(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255]; };

  // fetch with a deadline: a half-dead link must not stall a loop forever
  function fetchT(url, ms) {
    const ac = new AbortController(), t = setTimeout(() => ac.abort(), ms);
    return fetch(url, { cache: "no-store", signal: ac.signal }).finally(() => clearTimeout(t));
  }
  const hash = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
  const state = {
    src: hash.src === "real" ? "real" : "sim",
    cam: ["top", "plate", "wrist"].includes(hash.cam) ? hash.cam : "top",
    hud: hash.hud === "1",
    features: false,
    defects: hash.defects === "1",
    focus: hash.focus !== "0",
    scope: hash.scope === "focus" ? "focus" : "all",
    fs: ["cam", "depth"].includes(hash.fs) ? hash.fs : "",
    color: hash.color === "sensor" ? "sensor" : "rgb",
    zoom: hash.zoom === "2",
    // which D415 clouds (points + frustum) are visible; #hide=plate or #hide=top,plate
    show: { top: !(hash.hide || "").split(",").includes("top"), plate: !(hash.hide || "").split(",").includes("plate") },
  };
  let config = null; // from /api/config
  let status = null; // from /api/status
  let cloud = null; // renderer

  // ------------------------------------------------------------------ state → DOM
  function bind(name, text) { $$(`[data-bind="${name}"]`).forEach((e) => (e.textContent = text)); }
  function setOn(el, on) { el.classList.toggle("on", !!on); el.setAttribute("aria-pressed", on ? "true" : "false"); }
  // A status older than STATUS_STALE_MS (local clock) means the server is unreachable: no source is
  // trusted and no overlay is drawn. Overlays must never outlive their data.
  const STATUS_STALE_MS = 4000;
  let statusAt = 0;
  const statusFresh = () => !!status && performance.now() - statusAt < STATUS_STALE_MS;
  const focusOverlay = () => (OverlayEvidence.drawable(status?.focus, state.src, statusFresh()) && status.focus.object_id ? status.focus : null);
  function visibleAnomalies() {
    const a = status?.anomalies;
    if (!state.defects || !OverlayEvidence.drawable(a, state.src, statusFresh())) return [];
    const f = focusOverlay();
    return state.scope === "all" ? a.items : a.items.filter((i) => f && i.object_id === f.object_id);
  }

  function sync() {
    if (typeof wake === "function") wake();
    const b = document.body.classList;
    b.toggle("src-sim", state.src === "sim");
    b.toggle("src-real", state.src === "real");
    b.toggle("opt-defects", state.defects);
    b.toggle("opt-focus", state.focus);
    b.toggle("fs-cam", state.fs === "cam");
    b.toggle("fs-depth", state.fs === "depth");
    $$("[data-src]").forEach((e) => setOn(e, e.dataset.src === state.src));
    $$("[data-cam]").forEach((e) => setOn(e, e.dataset.cam === state.cam));
    $$("[data-opt]").forEach((e) => setOn(e, state[e.dataset.opt]));
    $$("[data-scope]").forEach((e) => setOn(e, e.dataset.scope === state.scope));
    $$("[data-color]").forEach((e) => setOn(e, e.dataset.color === state.color));
    $$("[data-sensor]").forEach((e) => setOn(e, state.show[e.dataset.sensor]));
    $$("[data-zoom]").forEach((e) => setOn(e, state.zoom));
    $$("[data-panel]").forEach((p) => p.classList.toggle("fs", p.dataset.panel === state.fs));
    $$("[data-fs]").forEach((e) => setOn(e, e.closest("[data-panel]")?.dataset.panel === state.fs));
    bind("zoom", state.zoom ? "2×" : "1×");
    bind("cam-title", { top: "Top camera", plate: "Plate camera", wrist: "Wrist camera" }[state.cam] || "Top camera");
    bind("src", `${state.src.toUpperCase()} · ${state.cam.toUpperCase()}`);
    bind("src-long", state.src === "sim" ? "MuJoCo sim" : "Real camera");
    renderStatus();
    drawOverlay();
    cloud?.updateHud();
    cloud?.resize();
  }

  function fmtAge(s) { return s == null ? "–" : s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`; }

  function renderStatus() {
    if (!status) return;
    const fresh = statusFresh(), S = status.sources, b = document.body.classList;
    const realFeed = status.camera_feeds?.real?.[state.cam] || S.real;
    const simFeed = status.camera_feeds?.sim?.[state.cam] || S.sim;
    b.toggle("real-down", !fresh || !realFeed.ok);
    b.toggle("sim-down", !fresh || !simFeed.ok);
    b.toggle("cloud-down", !fresh || !S.cloud.ok);
    const live = $('[data-bind="health"]');
    live.textContent = !fresh ? "Server unreachable" : status.ok ? "Live" : "Degraded";
    live.classList.toggle("down", !fresh || !status.ok);
    if (!fresh) {
      bind("cam-error", "The dashboard server does not answer.");
      bind("cloud-error", "The dashboard server does not answer.");
      bind("focus-label", "–");
      bind("defect-count", "0");
      for (const kind of ["anomalies", "focus"]) {
        $(`[data-stub="${kind}"]`).classList.toggle("show", state[kind === "anomalies" ? "defects" : "focus"]);
        bind(`${kind}-reason`, "Server unreachable");
        bind(`${kind}-state`, "Publisher offline");
      }
      return;
    }
    bind("real-fps", S.real.fps ? `camera ${S.real.fps.toFixed(0)} fps` : "– fps");
    const cam = state.src === "sim" ? simFeed : realFeed;
    const fps = shownFps(state.src);
    const shown = fps == null ? "–" : fps.toFixed(fps < 10 ? 1 : 0);
    const r2s = status.real2sim;
    bind("hud-count", String(r2s?.bricks?.length || 0));
    const feat = status.camera_features?.[state.src]?.[state.cam];
    bind("feat-count", String(feat?.bricks?.length || 0));
    bind("feat-corridor", feat?.corridor?.ok
      ? `pinch u=${feat.corridor.u_pinch}px (gap ${feat.corridor.gap_px}px)`
      : (state.cam === "wrist" ? "jaw corridor + minAreaRect" : "minAreaRect + grasp axis"));
    const realCamName = { top: "D415 Top", plate: "D415 Plate", wrist: "Sonix Wrist" }[state.cam] || "D415 Top";
    bind("cam-detail", state.src === "sim"
      ? (r2s?.ok
        ? `Real2Sim MuJoCo 4.0 (${state.cam}) · ${r2s.bricks?.length || 0} bricks · ${shown} fps shown`
        : `${state.cam} render · ${shown} fps shown`)
      : `${realCamName} · ${shown} fps shown`);
    bind("cam-age", `age ${fmtAge(cam.age_s)}`);
    bind("cam-error", cam.error || (cam.ok ? "" : "The source sends no fresh frames."));
    const cams = Object.entries(S.cloud.cameras || {}).map(([n, c]) => `${n} ${c.ok ? c.points : "×"}${state.show[n] === false ? " (hidden)" : ""}`).join(" · ");
    bind("cloud-detail", `${cams} · age ${fmtAge(S.cloud.age_s)}`);
    bind("cloud-points", S.cloud.ok ? `${S.cloud.points.toLocaleString()} pts` : "down");
    bind("cloud-error", Object.entries(S.cloud.cameras || {}).filter(([, c]) => !c.ok).map(([n, c]) => `${n}: ${c.error}`).join(" · ") || "Waiting for depth.");
    const sim = S.scene.sim;
    bind("sim-state", !S.scene.ok ? "down" : sim?.busy ? "RUNNING" : sim?.stopped ? "STOPPED" : "IDLE");
    const f = focusOverlay();
    bind("focus-label", f ? (f.label || f.object_id) : "–");
    for (const kind of ["anomalies", "focus"]) {
      const o = status[kind];
      $(`[data-stub="${kind}"]`).classList.toggle("show", state[kind === "anomalies" ? "defects" : "focus"]);
      bind(`${kind}-state`, OverlayEvidence.label(o, state.src));
      bind(`${kind}-reason`, o.evidence && o.evidence.mode !== state.src
        ? `Result belongs to ${o.evidence.mode}; hidden in ${state.src} view.` : o.reason || "");
    }
    const e = S && status.anomalies?.evidence;
    bind("anomalies-provenance", e
      ? `${e.method || "Unknown method"} · model ${e.model_version || "unknown"} · ${e.threshold_version ? "threshold " + e.threshold_version : "uncalibrated / unknown threshold"} · capture ${fmtAge(status.anomalies.capture_age_s)}`
      : "Detector method and calibration unknown");
    bind("defect-count", String(visibleAnomalies().length));
  }

  // ------------------------------------------------------------------ controls
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-src],[data-cam],[data-opt],[data-scope],[data-fs],[data-color],[data-zoom],[data-sensor]");
    if (!t) return;
    if (t.dataset.src) state.src = t.dataset.src;
    if (t.dataset.cam) state.cam = t.dataset.cam;
    if (t.dataset.opt) state[t.dataset.opt] = !state[t.dataset.opt];
    if (t.dataset.scope) state.scope = t.dataset.scope;
    if (t.hasAttribute("data-zoom")) state.zoom = !state.zoom;
    if (t.dataset.color) { state.color = t.dataset.color; cloud?.recolor(); }
    if (t.dataset.sensor) { state.show[t.dataset.sensor] = !state.show[t.dataset.sensor]; cloud?.dirty(); }
    if (t.hasAttribute("data-fs")) {
      const p = t.closest("[data-panel]").dataset.panel;
      state.fs = state.fs === p ? "" : p;
    }
    sync();
  });
  document.addEventListener("keydown", (e) => {
    if ((e.target instanceof Element && e.target.matches("input,textarea")) || e.altKey || e.metaKey) return;
    const k = e.key.toLowerCase();
    if (k === "t") state.src = state.src === "sim" ? "real" : "sim";
    else if (k === "escape") state.fs = "";
    else if (k === "h") state.hud = !state.hud;
    else if (k === "w") state.features = !state.features;
    else if (k === "d") state.defects = !state.defects;
    else if (k === "f") state.focus = !state.focus;
    else if (k === "z") state.zoom = !state.zoom;
    else if (state.fs !== "depth" && (k === "1" || k === "2" || k === "3")) {
      state.cam = { "1": "top", "2": "plate", "3": "wrist" }[k];
    } else if (k === "8" || k === "9") { const c = k === "8" ? "top" : "plate"; state.show[c] = !state.show[c]; cloud?.dirty(); }
    else if (cloud && cloud.key(e)) return;
    else return;
    sync();
  });

  // ------------------------------------------------------------------ live camera frames
  const frames = { sim: { seq: 0, cam: "top", shown: [] }, real: { seq: 0, cam: "top", shown: [] } };
  const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
  async function frameLoop(name) {
    const img = $(`img.cam-${name}`), f = frames[name];
    let url = null;
    for (;;) {
      const visible = state.src === name;
      const camSlot = state.cam;
      if (f.cam !== camSlot) {
        f.cam = camSlot;
        f.seq = 0;
        f.shown = [];
      }
      const endpoint = camSlot === "top" ? `/frame/${name}.jpg` : `/frame/${name}/${camSlot}.jpg`;
      try {
        // visible: long-poll the next frame; hidden: take the newest frame once a second
        const r = await fetchT(`${endpoint}?after=${visible ? f.seq : 0}`, 6000);
        if (r.status === 200) {
          const seq = Number(r.headers.get("X-Frame-Seq"));
          if (seq !== f.seq || f.cam !== state.cam) {
            const next = URL.createObjectURL(await r.blob());
            img.src = next;
            f.seq = seq;
            // no await on decode(): it only resolves when the page paints
            const old = url;
            if (old) setTimeout(() => URL.revokeObjectURL(old), 2000);
            url = next;
            f.shown.push(performance.now());
            if (f.shown.length > 20) f.shown.shift();
          } else await r.arrayBuffer();
        } else if (r.status !== 204) await sleep(1000);
      } catch { await sleep(1000); }
      if (state.src !== name && f.cam === state.cam) await waitVisibleOr(1000, name, f);
    }
  }
  const wakers = [];
  function waitVisibleOr(ms, name, f) {
    return new Promise((ok) => {
      const t = setTimeout(ok, ms);
      wakers.push(() => { if (state.src === name || (f && f.cam !== state.cam)) { clearTimeout(t); ok(); } });
    });
  }
  function wake() { wakers.splice(0).forEach((w) => w()); }
  function shownFps(name) {
    const t = frames[name].shown;
    return t.length > 2 && performance.now() - t[t.length - 1] < 3000 ? ((t.length - 1) * 1000) / (t[t.length - 1] - t[0]) : null;
  }

  // ------------------------------------------------------------------ camera overlay
  const camPanel = $('[data-panel="cam"]');
  const overlay = $("canvas.cam-overlay", camPanel);
  function activeCameraModel() {
    const r2s = status?.real2sim;
    if (state.cam === "plate") return r2s?.camera_plate || config?.cameras?.plate || config?.cameras?.top;
    if (state.cam === "wrist") return r2s?.camera_wrist || config?.cameras?.wrist || config?.cameras?.top;
    return r2s?.camera_top || config?.cameras?.top;
  }
  function activeImageSize() {
    const feat = status?.camera_features?.[state.src]?.[state.cam];
    const c = activeCameraModel();
    if (state.cam === "wrist") {
      return { width: feat?.width || c?.width || 640, height: feat?.height || c?.height || 360 };
    }
    return { width: c?.width || config.image.width, height: c?.height || config.image.height };
  }
  function projectTop(p, camOverride) {
    const c = camOverride || activeCameraModel();
    if (!c) return null;
    const R = c.rotation_cv, d = [p[0] - c.pos[0], p[1] - c.pos[1], p[2] - c.pos[2]];
    const q = [0, 1, 2].map((j) => R[0][j] * d[0] + R[1][j] * d[1] + R[2][j] * d[2]);
    return q[2] > 0 ? [c.fx * q[0] / q[2] + c.cx, c.fy * q[1] / q[2] + c.cy] : null;
  }
  function r2sToCloud(p, r2sCam) {
    if (!r2sCam || !config?.cameras?.top) return p;
    const R1 = r2sCam.rotation_cv, P1 = r2sCam.pos, d = [p[0] - P1[0], p[1] - P1[1], p[2] - P1[2]];
    const q = [0, 1, 2].map((j) => R1[0][j] * d[0] + R1[1][j] * d[1] + R1[2][j] * d[2]);
    const R2 = config.cameras.top.rotation_cv, P2 = config.cameras.top.pos;
    return [
      R2[0][0] * q[0] + R2[0][1] * q[1] + R2[0][2] * q[2] + P2[0],
      R2[1][0] * q[0] + R2[1][1] * q[1] + R2[1][2] * q[2] + P2[1],
      p[2],
    ];
  }
  function contentRect(stage, imSize) {
    const W = stage.clientWidth, H = stage.clientHeight, iw = imSize.width, ih = imSize.height;
    if (!state.zoom) {
      const s = Math.min(W / iw, H / ih);
      return { x: (W - iw * s) / 2, y: (H - ih * s) / 2, s };
    }
    // 2x: fill the stage, centred on the focus block, else on the plates (pivot)
    const s = Math.max(W / iw, H / ih) * 2.2;
    const f = state.cam === "top" ? focusOverlay() : null;
    const [cx, cy] = f ? [(f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2] : projectTop(config.pivot) || [iw / 2, ih / 2];
    return { x: Math.min(0, Math.max(W - iw * s, W / 2 - cx * s)), y: Math.min(0, Math.max(H - ih * s, H / 2 - cy * s)), s };
  }
  function drawOverlay() {
    if (!config) return;
    const stage = overlay.parentElement, dpr = devicePixelRatio || 1;
    overlay.width = stage.clientWidth * dpr;
    overlay.height = stage.clientHeight * dpr;
    const g = overlay.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, overlay.width, overlay.height);
    const imSize = activeImageSize();
    const r = contentRect(stage, imSize);
    $$("img", stage).forEach((im) => Object.assign(im.style, {
      left: `${r.x}px`, top: `${r.y}px`, width: `${imSize.width * r.s}px`, height: `${imSize.height * r.s}px`,
    }));
    const P = (x, y) => [r.x + x * r.s, r.y + y * r.s];
    const st = {
      line: parseFloat(css(camPanel, "--box-line", "2")),
      kind: css(camPanel, "--box-style", "rect"),
      font: css(camPanel, "--label-font", "12px monospace"),
      fg: css(camPanel, "--label-fg", "#000"),
      pad: parseFloat(css(camPanel, "--box-pad", "4")),
    };
    if (state.cam === "top") {
      const f = focusOverlay();
      if (state.focus && f) box(g, P, f.box, css(camPanel, "--focus", "#ff6a1a"), st, st.pad, `PROJECTED FOCUS  ${(f.label || f.object_id).toUpperCase()}`, "top");
      for (const a of visibleAnomalies()) {
        const label = ["OBJECT ESTIMATE ·", a.label, a.score != null ? a.score.toFixed(2) : null].filter(Boolean).join(" ");
        box(g, P, a.box, css(camPanel, "--defect", "#ff3b3b"), st, 2, label, "bottom");
      }
    }
  }
  const OBB_EDGES = [
    [0, 1], [1, 3], [3, 2], [2, 0],
    [4, 5], [5, 7], [7, 6], [6, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  function rgbaCss(rgba, a = 1) {
    if (!Array.isArray(rgba) || rgba.length < 3) return `rgba(255,178,127,${a})`;
    return `rgba(${Math.round(rgba[0] * 255)},${Math.round(rgba[1] * 255)},${Math.round(rgba[2] * 255)},${a})`;
  }
  function drawCameraPlates2D(g, P, st, feat) {
    const plates = feat?.plates;
    if (!plates || typeof plates !== "object") return;
    const dpr = devicePixelRatio || 1;
    const lw = (px) => px * dpr;
    const labelMap = { good: "GOOD", defective: "DEFECT", source: "SOURCE" };
    g.save();
    for (const [role, pl] of Object.entries(plates)) {
      const el = pl?.ellipse;
      const center = pl?.center_uv;
      if (!el || !Array.isArray(center) || center.length !== 2) continue;
      const col = rgbaCss(pl.rgba, 0.88);
      const ang = ((el.angle_deg || 0) * Math.PI) / 180;
      const ca = Math.cos(ang);
      const sa = Math.sin(ang);
      const rx = el.rx || 0;
      const ry = el.ry || 0;
      if (rx > 4 && ry > 4) {
        g.strokeStyle = "rgba(0, 0, 0, 0.45)";
        g.lineWidth = lw(2.6);
        g.setLineDash([6 * dpr, 4 * dpr]);
        g.beginPath();
        for (let k = 0; k <= 36; k++) {
          const th = (2 * Math.PI * k) / 36;
          const lx = rx * Math.cos(th);
          const ly = ry * Math.sin(th);
          const [sx, sy] = P(el.cx + lx * ca - ly * sa, el.cy + lx * sa + ly * ca);
          if (k === 0) g.moveTo(sx, sy);
          else g.lineTo(sx, sy);
        }
        g.stroke();
        g.strokeStyle = col;
        g.lineWidth = lw(1.5);
        g.stroke();
        g.setLineDash([]);
      }
      const [cx, cy] = P(center[0], center[1]);
      const arm = role === "source" ? lw(5) : lw(7);
      const rad = role === "source" ? lw(3) : lw(4.5);
      g.strokeStyle = "rgba(0, 0, 0, 0.65)";
      g.lineWidth = lw(3.0);
      g.beginPath();
      g.moveTo(cx - arm, cy); g.lineTo(cx + arm, cy);
      g.moveTo(cx, cy - arm); g.lineTo(cx, cy + arm);
      g.stroke();
      g.beginPath();
      g.arc(cx, cy, rad, 0, Math.PI * 2);
      g.stroke();
      g.strokeStyle = col;
      g.lineWidth = lw(1.6);
      g.beginPath();
      g.moveTo(cx - arm, cy); g.lineTo(cx + arm, cy);
      g.moveTo(cx, cy - arm); g.lineTo(cx, cy + arm);
      g.stroke();
      g.beginPath();
      g.arc(cx, cy, rad, 0, Math.PI * 2);
      g.stroke();

      const tag = labelMap[role] || role.toUpperCase();
      const ly = Math.max(12 * dpr, cy - Math.max(ry * 0.55, lw(16)));
      g.font = `600 ${Math.round(10 * dpr)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      const tw = g.measureText(tag).width + lw(8);
      const th = lw(14);
      g.fillStyle = "rgba(10, 10, 12, 0.72)";
      g.fillRect(cx - tw / 2, ly - th / 2, tw, th);
      g.fillStyle = col;
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.fillText(tag, cx, ly);
    }
    g.restore();
  }
  function drawFeatureAnalysisHud(g, P, st) {
    const feat = status?.camera_features?.[state.src]?.[state.cam];
    if (!feat?.ok) return;
    drawCameraPlates2D(g, P, st, feat);
  }
  function pickReal2SimTargetBrick(r2s) {
    const bricks = r2s?.bricks || [];
    if (!bricks.length) return null;
    const pinch = r2s?.gripper_fk?.pinch_world_m;
    const dist2 = (b) => {
      const p = b?.pos_world_m;
      if (!Array.isArray(p) || !Array.isArray(pinch)) return 1e9;
      return (p[0] - pinch[0]) ** 2 + (p[1] - pinch[1]) ** 2 + (p[2] - pinch[2]) ** 2;
    };
    const inSource = bricks.filter((b) => b.support_region === "source");
    const pool = inSource.length ? inSource : bricks;
    return pool.slice().sort((a, b) => dist2(a) - dist2(b))[0];
  }
  function drawReal2SimHud(g, P, st) {
    const r2s = status.real2sim;
    const r2sCam = activeCameraModel();
    if (!r2sCam) return;
    const dpr = devicePixelRatio || 1;
    const lw = (px) => px * dpr;
    const projR2S = (pt) => projectTop(pt, r2sCam);
    g.save();
    g.lineCap = "round";
    g.lineJoin = "round";
    // 1. 3D projected KALAS plate rings only on top when 2D plate tracking is off (never on wrist)
    if (state.cam === "top" && !state.features) {
      for (const [role, pl] of Object.entries(r2s.plates || {})) {
        const pos = pl.pos_world_m;
        if (!Array.isArray(pos) || pos.length < 3) continue;
        const rad = pl.outer_radius_m || 0.0795;
        const col = rgbaCss(pl.rgba, 0.78);
        g.strokeStyle = col;
        g.lineWidth = lw(1.4);
        g.setLineDash([5 * dpr, 4 * dpr]);
        g.beginPath();
        let started = false;
        for (let k = 0; k <= 36; k++) {
          const th = (2 * Math.PI * k) / 36;
          const uv = projR2S([pos[0] + rad * Math.cos(th), pos[1] + rad * Math.sin(th), pos[2] + 0.008]);
          if (!uv) continue;
          const [sx, sy] = P(uv[0], uv[1]);
          if (!started) { g.moveTo(sx, sy); started = true; } else g.lineTo(sx, sy);
        }
        g.stroke();
        g.setLineDash([]);
      }
    }
    // 2. Singular block of interest: visible cuboid vertices + centroid + parallel-face axis
    const camW = r2sCam.width || 640, camH = r2sCam.height || 480;
    const inFrameUv = (uv) => uv && uv[0] >= 0 && uv[0] <= camW && uv[1] >= 0 && uv[1] <= camH;
    const drawVertexDot = (pt) => {
      g.fillStyle = "rgba(0, 0, 0, 0.75)";
      g.beginPath();
      g.arc(pt[0], pt[1], lw(4.2), 0, Math.PI * 2);
      g.fill();
      g.fillStyle = "#ff8a3d";
      g.beginPath();
      g.arc(pt[0], pt[1], lw(2.8), 0, Math.PI * 2);
      g.fill();
    };
    const drawCentroidTarget = (cx, cy) => {
      g.strokeStyle = "rgba(0, 0, 0, 0.72)";
      g.lineWidth = lw(3.6);
      g.beginPath();
      g.arc(cx, cy, lw(6.0), 0, Math.PI * 2);
      g.stroke();
      g.strokeStyle = "#ffc13b";
      g.lineWidth = lw(2.0);
      g.beginPath();
      g.arc(cx, cy, lw(6.0), 0, Math.PI * 2);
      g.stroke();
      g.fillStyle = "#ffc13b";
      g.beginPath();
      g.arc(cx, cy, lw(2.8), 0, Math.PI * 2);
      g.fill();
    };
    let centroidCanvas = null;
    let centroidWorld = null;
    const wristBrick = state.cam === "wrist" ? status?.camera_features?.[state.src]?.wrist?.bricks?.[0] : null;
    if (wristBrick && Array.isArray(wristBrick.box_uv) && wristBrick.box_uv.length === 4) {
      // On wrist view, use image-locked cuboid vertices & centroid for zero-offset visual servoing
      const proj = wristBrick.box_uv.map((uv) => (inFrameUv(uv) ? P(uv[0], uv[1]) : null));
      g.strokeStyle = "rgba(255, 106, 26, 0.55)";
      g.lineWidth = lw(1.3);
      for (let k = 0; k < 4; k++) {
        const a = proj[k], c = proj[(k + 1) % 4];
        if (!a || !c) continue;
        g.beginPath();
        g.moveTo(a[0], a[1]);
        g.lineTo(c[0], c[1]);
        g.stroke();
      }
      for (const pt of proj) {
        if (pt) drawVertexDot(pt);
      }
      if (Array.isArray(wristBrick.axis_uv) && wristBrick.axis_uv.length === 2) {
        const [a0, a1] = wristBrick.axis_uv.map(([u, v]) => P(u, v));
        g.strokeStyle = "rgba(255, 193, 59, 0.85)";
        g.lineWidth = lw(1.8);
        g.beginPath();
        g.moveTo(a0[0], a0[1]);
        g.lineTo(a1[0], a1[1]);
        g.stroke();
      }
      if (Array.isArray(wristBrick.center_uv) && inFrameUv(wristBrick.center_uv)) {
        const [cx, cy] = P(wristBrick.center_uv[0], wristBrick.center_uv[1]);
        centroidCanvas = [cx, cy];
        drawCentroidTarget(cx, cy);
      }
    } else {
      const b = pickReal2SimTargetBrick(r2s);
      if (b) {
        const corners = b.corners_world_m;
        if (Array.isArray(corners) && corners.length === 8) {
          const proj = corners.map((c) => {
            const uv = projR2S(c);
            return inFrameUv(uv) ? P(uv[0], uv[1]) : null;
          });
          g.strokeStyle = "rgba(255, 106, 26, 0.55)";
          for (const [ia, ib] of OBB_EDGES) {
            if (!proj[ia] || !proj[ib]) continue;
            g.lineWidth = lw(1.3);
            g.beginPath();
            g.moveTo(proj[ia][0], proj[ia][1]);
            g.lineTo(proj[ib][0], proj[ib][1]);
            g.stroke();
          }
          for (const pt of proj) {
            if (pt) drawVertexDot(pt);
          }
          const top4 = corners.slice().sort((u, v) => v[2] - u[2]).slice(0, 4);
          const cxW = 0.25 * (top4[0][0] + top4[1][0] + top4[2][0] + top4[3][0]);
          const cyW = 0.25 * (top4[0][1] + top4[1][1] + top4[2][1] + top4[3][1]);
          const czW = 0.25 * (top4[0][2] + top4[1][2] + top4[2][2] + top4[3][2]);
          centroidWorld = Array.isArray(b.pos_world_m) ? b.pos_world_m : [cxW, cyW, czW];
        } else if (Array.isArray(b.pos_world_m)) {
          centroidWorld = b.pos_world_m;
        }

        if (centroidWorld) {
          const cUv = projR2S(centroidWorld);
          if (inFrameUv(cUv)) {
            const [cx, cy] = P(cUv[0], cUv[1]);
            centroidCanvas = [cx, cy];
            drawCentroidTarget(cx, cy);
          }
        }
      }
    }
    // 3. Static midpoint between the grippers + visual-servoing vector to cuboid centroid
    let gripMidCanvas = null;
    let useImgGripMid = false;
    const fk = r2s.gripper_fk;
    const topGripUv = state.cam === "top" ? status?.camera_features?.[state.src]?.top?.gripper_mid_uv : null;
    if (state.cam === "wrist") {
      // Physically bolted wrist camera: midpoint between the two grippers is static in image space
      gripMidCanvas = P(0.494 * camW, 0.72 * camH);
      useImgGripMid = true;
    } else if (Array.isArray(topGripUv) && topGripUv.length === 2) {
      gripMidCanvas = P(topGripUv[0], topGripUv[1]);
      useImgGripMid = true;
    } else if (fk && Array.isArray(fk.pinch_world_m)) {
      const pinchUv = projR2S(fk.pinch_world_m);
      if (inFrameUv(pinchUv)) {
        gripMidCanvas = P(pinchUv[0], pinchUv[1]);
      }
    }
    if (gripMidCanvas) {
      const [gx, gy] = gripMidCanvas;
      const arm = lw(7.5);
      g.strokeStyle = "rgba(0, 0, 0, 0.75)";
      g.lineWidth = lw(3.8);
      g.beginPath();
      g.moveTo(gx - arm, gy); g.lineTo(gx + arm, gy);
      g.moveTo(gx, gy - arm); g.lineTo(gx, gy + arm);
      g.stroke();
      g.beginPath();
      g.arc(gx, gy, lw(5.2), 0, Math.PI * 2);
      g.stroke();
      g.strokeStyle = "#00e5ff";
      g.lineWidth = lw(2.0);
      g.beginPath();
      g.moveTo(gx - arm, gy); g.lineTo(gx + arm, gy);
      g.moveTo(gx, gy - arm); g.lineTo(gx, gy + arm);
      g.stroke();
      g.beginPath();
      g.arc(gx, gy, lw(5.2), 0, Math.PI * 2);
      g.stroke();

      if (centroidCanvas) {
        const [cx, cy] = centroidCanvas;
        let tag = "";
        let aligned = false;
        if (!useImgGripMid && fk?.pinch_world_m && centroidWorld) {
          const dMm = Math.round(
            1000 * Math.hypot(
              fk.pinch_world_m[0] - centroidWorld[0],
              fk.pinch_world_m[1] - centroidWorld[1],
              fk.pinch_world_m[2] - centroidWorld[2]
            )
          );
          aligned = dMm <= 10;
          tag = aligned ? `ALIGNED (${dMm}mm)` : `SERVO ${dMm}mm`;
        } else {
          const dPx = Math.round(Math.hypot(gx - cx, gy - cy));
          aligned = dPx <= 12;
          tag = aligned ? "ALIGNED" : `SERVO ${dPx}px`;
        }
        g.strokeStyle = aligned ? "#00ff88" : "#ffe600";
        g.lineWidth = lw(2.0);
        g.setLineDash([5 * dpr, 4 * dpr]);
        g.beginPath();
        g.moveTo(gx, gy);
        g.lineTo(cx, cy);
        g.stroke();
        g.setLineDash([]);

        g.font = `600 ${Math.round(10 * dpr)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
        const tw = g.measureText(tag).width + lw(8);
        const th = lw(14);
        const mx = 0.5 * (gx + cx), my = 0.5 * (gy + cy) - lw(12);
        g.fillStyle = "rgba(10,10,12,0.78)";
        g.fillRect(mx - tw / 2, my - th / 2, tw, th);
        g.fillStyle = aligned ? "#00ff88" : "#ffe600";
        g.textAlign = "center";
        g.textBaseline = "middle";
        g.fillText(tag, mx, my);
      }
    }
    g.restore();
  }
  function box(g, P, b, color, st, pad, label, where) {
    let [x0, y0] = P(b[0], b[1]);
    let [x1, y1] = P(b[2], b[3]);
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
    g.strokeStyle = color;
    g.lineWidth = st.line;
    if (st.kind === "corners") {
      const c = Math.max(6, Math.min(x1 - x0, y1 - y0) * 0.28);
      g.beginPath();
      for (const [x, y, dx, dy] of [[x0, y0, 1, 1], [x1, y0, -1, 1], [x0, y1, 1, -1], [x1, y1, -1, -1]]) {
        g.moveTo(x + dx * c, y); g.lineTo(x, y); g.lineTo(x, y + dy * c);
      }
      g.stroke();
    } else g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    if (!label) return;
    g.font = st.font;
    const px = parseFloat((st.font.match(/([\d.]+)px/) || [0, 12])[1]);
    const w = g.measureText(label).width + 10, h = px + 8, ly = where === "top" ? y0 - h - 2 : y1 + 2;
    g.fillStyle = color;
    g.fillRect(x0 - st.line / 2, ly, w, h);
    g.fillStyle = st.fg;
    g.textBaseline = "middle";
    g.fillText(label, x0 + 5 - st.line / 2, ly + h / 2 + 0.5);
  }
  new ResizeObserver(drawOverlay).observe(overlay.parentElement);

  // ------------------------------------------------------------------ point cloud (WebGL)
  function makeCloud() {
    const panel = $('[data-panel="depth"]');
    const canvas = $("canvas.cloud", panel), labels = $("canvas.cloud-labels", panel), gizmo = $("canvas.gizmo", panel);
    const gl = canvas.getContext("webgl", { antialias: true });
    if (!gl) { bind("cloud-error", "WebGL is not available in this browser."); return null; }
    const cams = config.cameras, top = cams.top, pivot = config.pivot;

    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, `attribute vec3 p;attribute vec3 c;uniform mat4 m;uniform float k;uniform float ortho;varying vec3 v;
      void main(){gl_Position=m*vec4(p,1.);gl_PointSize=clamp(ortho>.5?k:k/gl_Position.w,1.,8.);v=c;}`));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, "precision mediump float;varying vec3 v;void main(){gl_FragColor=vec4(v,1.);}"));
    gl.linkProgram(prog);
    const U = (n) => gl.getUniformLocation(prog, n), pa = gl.getAttribLocation(prog, "p"), ca = gl.getAttribLocation(prog, "c");
    const buf = (data, usage = gl.STATIC_DRAW) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, usage); return b; };

    // live cloud buffers
    const pBuf = gl.createBuffer(), cBuf = gl.createBuffer();
    let n = 0, nTop = 0, rgb = new Float32Array(0), sensor = new Uint8Array(0);
    function setPoints(arrayBuffer) {
      // records: int16 x, y, z (mm); uint16 sensor << 15 | RGB555
      // stored grouped by sensor (top first, then plate) so each camera draws as one range
      const xyz = new Int16Array(arrayBuffer), col = new Uint16Array(arrayBuffer), count = xyz.length / 4;
      const pos = new Float32Array(count * 3);
      rgb = new Float32Array(count * 3); sensor = new Uint8Array(count);
      let tops = 0;
      for (let i = 0; i < count; i++) if (!(col[i * 4 + 3] >> 15)) tops++;
      const next = [0, tops];
      for (let i = 0; i < count; i++) {
        const o = i * 4, c = col[o + 3], sid = c >> 15, j = next[sid]++;
        pos[j * 3] = xyz[o] / 1000; pos[j * 3 + 1] = xyz[o + 1] / 1000; pos[j * 3 + 2] = xyz[o + 2] / 1000;
        rgb[j * 3] = ((c >> 10) & 31) / 31; rgb[j * 3 + 1] = ((c >> 5) & 31) / 31; rgb[j * 3 + 2] = (c & 31) / 31;
        sensor[j] = sid;
      }
      n = count;
      nTop = tops;
      gl.bindBuffer(gl.ARRAY_BUFFER, pBuf); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
      recolor();
    }
    function recolor() {
      const a = hex(css(panel, "--sensor-a", "#ff6a1a")), b = hex(css(panel, "--sensor-b", "#f1f1f0"));
      const out = state.color === "sensor" ? new Float32Array(n * 3) : rgb;
      if (state.color === "sensor") for (let i = 0; i < n; i++) out.set(sensor[i] ? b : a, i * 3);
      gl.bindBuffer(gl.ARRAY_BUFFER, cBuf); gl.bufferData(gl.ARRAY_BUFFER, out, gl.DYNAMIC_DRAW);
      dirty();
    }

    // reference lines: table grid, then one frustum per D415 in that sensor's colour
    let lineP, lineC, gridN = 0;
    const frustum = {}; // camera name -> [first vertex, vertex count]
    (function buildLines() {
      const L = [], C = [], seg = (a, b, c) => { L.push(...a, ...b); C.push(...c, ...c); };
      const grid = hex(css(panel, "--cloud-grid", "#2a1a0f"));
      const camColor = { top: hex(css(panel, "--sensor-a", "#ff6a1a")), plate: hex(css(panel, "--sensor-b", "#f1f1f0")) };
      const [[x0, x1], [y0, y1]] = config.workspace;
      for (let x = Math.ceil(x0 * 20) / 20; x <= x1 + 1e-9; x += 0.05) seg([x, y0, 0], [x, y1, 0], grid);
      for (let y = Math.ceil(y0 * 20) / 20; y <= y1 + 1e-9; y += 0.05) seg([x0, y, 0], [x1, y, 0], grid);
      gridN = L.length / 3;
      for (const name of Object.keys(cams)) {
        const first = L.length / 3, camc = camColor[name] || hex(css(panel, "--cloud-cam", "#ff8a45"));
        const c = cams[name], R = c.rotation_cv, s = 0.07, col = (j) => [R[0][j], R[1][j], R[2][j]];
        const [rx, ry, rz] = [col(0), col(1), col(2)];
        const corner = (a, b) => c.pos.map((p, i) => p + s * (rz[i] + a * 0.66 * rx[i] + b * 0.5 * ry[i]));
        const k = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
        k.forEach((q, i) => { seg(c.pos, q, camc); seg(q, k[(i + 1) % 4], camc); });
        frustum[name] = [first, L.length / 3 - first];
      }
      lineP = buf(new Float32Array(L)); lineC = buf(new Float32Array(C));
    })();

    // Dynamic Real2Sim 3D HUD buffers (3 KALAS plate rings + brick 3D OBB wireframes + 6D pose triads)
    const hudP = gl.createBuffer(), hudC = gl.createBuffer();
    let hudN = 0;
    function updateHud() {
      const L = [], C = [], seg = (a, b, c) => { L.push(...a, ...b); C.push(...c, ...c); };
      const r2s = statusFresh() ? status?.real2sim : null;
      const r2sCam = r2s?.camera_top;
      const tc = (pt) => r2sToCloud(pt, r2sCam);
      if (state.hud && r2s?.ok) {
        for (const pl of Object.values(r2s.plates || {})) {
          const pos = pl.pos_world_m;
          if (!Array.isArray(pos) || pos.length < 3) continue;
          const rad = pl.outer_radius_m || 0.0795;
          const col = Array.isArray(pl.rgba) ? pl.rgba.slice(0, 3) : [0.28, 0.82, 0.80];
          for (let k = 0; k < 32; k++) {
            const t0 = (2 * Math.PI * k) / 32, t1 = (2 * Math.PI * (k + 1)) / 32;
            seg(
              tc([pos[0] + rad * Math.cos(t0), pos[1] + rad * Math.sin(t0), pos[2] + 0.006]),
              tc([pos[0] + rad * Math.cos(t1), pos[1] + rad * Math.sin(t1), pos[2] + 0.006]),
              col,
            );
          }
        }
        for (const b of r2s.bricks || []) {
          const corners = b.corners_world_m;
          const col = Array.isArray(b.display_rgba) ? b.display_rgba.slice(0, 3) : [0.95, 0.78, 0.12];
          if (Array.isArray(corners) && corners.length === 8) {
            const cCloud = corners.map(tc);
            for (const [ia, ib] of OBB_EDGES) seg(cCloud[ia], cCloud[ib], col);
          }
          const pos = b.pos_world_m, R = b.R_world_block;
          if (Array.isArray(pos) && Array.isArray(R) && R.length === 3) {
            const p0 = tc(pos);
            const axes = [
              [[R[0][0], R[1][0], R[2][0]], [1.0, 0.28, 0.28]],
              [[R[0][1], R[1][1], R[2][1]], [0.22, 0.90, 0.30]],
              [[R[0][2], R[1][2], R[2][2]], [0.23, 0.62, 1.0]],
            ];
            for (const [ax, acol] of axes) {
              seg(p0, tc([pos[0] + 0.022 * ax[0], pos[1] + 0.022 * ax[1], pos[2] + 0.022 * ax[2]]), acol);
            }
          }
        }
      }
      hudN = L.length / 3;
      gl.bindBuffer(gl.ARRAY_BUFFER, hudP); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(L), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, hudC); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(C), gl.DYNAMIC_DRAW);
      dirty();
    }

    // view: "cam" = exactly the real Top camera; "orbit" = Blender-style around a target
    const R = top.rotation_cv, col = (j) => [R[0][j], R[1][j], R[2][j]];
    const fovBase = 2 * Math.atan(config.image.height / 2 / top.fy), camAspect = config.image.width / config.image.height;
    let fovy = fovBase;
    const view = { mode: "cam", ortho: false, target: [...pivot], yaw: -Math.PI / 2, pitch: 0.6, dist: 0.85 };
    const sub = (a, b) => a.map((v, i) => v - b[i]), add = (a, b) => a.map((v, i) => v + b[i]), mul = (a, s) => a.map((v) => v * s);
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const norm = (a) => mul(a, 1 / Math.hypot(...a));
    function basis() {
      if (view.mode === "cam") return { eye: top.pos, right: col(0), up: mul(col(1), -1), back: mul(col(2), -1), dist: dot(sub(pivot, top.pos), col(2)) };
      const cp = Math.cos(view.pitch), back = [cp * Math.cos(view.yaw), cp * Math.sin(view.yaw), Math.sin(view.pitch)];
      const right = norm(cross([0, 0, 1], back));
      return { eye: add(view.target, mul(back, view.dist)), right, up: cross(back, right), back, dist: view.dist };
    }
    function leaveCam() {
      if (view.mode !== "cam") return;
      const fwd = col(2), t = dot(sub(pivot, top.pos), fwd);
      Object.assign(view, { target: add(top.pos, mul(fwd, t)), dist: t, yaw: Math.atan2(-fwd[1], -fwd[0]), pitch: Math.asin(-fwd[2]), mode: "orbit" });
    }
    function matrix(aspect) {
      const { eye, right, up, back, dist } = basis();
      fovy = aspect < camAspect ? 2 * Math.atan(Math.tan(fovBase / 2) * camAspect / aspect) : fovBase;
      const V = [right[0], up[0], back[0], 0, right[1], up[1], back[1], 0, right[2], up[2], back[2], 0, -dot(right, eye), -dot(up, eye), -dot(back, eye), 1];
      const near = 0.02, far = 5;
      let P;
      if (view.ortho) {
        const h = dist * Math.tan(fovy / 2), w = h * aspect;
        P = [1 / w, 0, 0, 0, 0, 1 / h, 0, 0, 0, 0, -2 / (far - near), 0, 0, 0, -(far + near) / (far - near), 1];
      } else {
        const f = 1 / Math.tan(fovy / 2);
        P = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, (2 * far * near) / (near - far), 0];
      }
      const M = new Float32Array(16);
      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += P[k * 4 + r] * V[c * 4 + k]; M[c * 4 + r] = s; }
      return { M, dist };
    }

    let pending = false;
    function dirty() { if (!pending) { pending = true; requestAnimationFrame(draw); } }
    function resize() {
      const dpr = devicePixelRatio || 1;
      for (const c of [canvas, labels]) { c.width = canvas.clientWidth * dpr; c.height = canvas.clientHeight * dpr; }
      gizmo.width = gizmo.clientWidth * dpr; gizmo.height = gizmo.clientHeight * dpr;
      dirty();
    }
    function draw() {
      pending = false;
      const W = canvas.width, H = canvas.height;
      if (!W || !H) return;
      gl.viewport(0, 0, W, H);
      const bg = hex(css(panel, "--cloud-bg", "#0a0a0b"));
      gl.clearColor(bg[0], bg[1], bg[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(prog);
      const { M, dist } = matrix(W / H);
      gl.uniformMatrix4fv(U("m"), false, M);
      const pxPerM = H / (2 * Math.tan(fovy / 2)), size = parseFloat(css(panel, "--point-size", "0.0026"));
      gl.uniform1f(U("k"), view.ortho ? (size * pxPerM) / dist : size * pxPerM);
      gl.uniform1f(U("ortho"), view.ortho ? 1 : 0);
      gl.enableVertexAttribArray(pa); gl.enableVertexAttribArray(ca);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineP); gl.vertexAttribPointer(pa, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, lineC); gl.vertexAttribPointer(ca, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, gridN);
      for (const [name, [first, count]] of Object.entries(frustum))
        if (state.show[name] !== false && !(name === "top" && view.mode === "cam")) gl.drawArrays(gl.LINES, first, count);
      if (n) {
        gl.bindBuffer(gl.ARRAY_BUFFER, pBuf); gl.vertexAttribPointer(pa, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, cBuf); gl.vertexAttribPointer(ca, 3, gl.FLOAT, false, 0, 0);
        if (state.show.top) gl.drawArrays(gl.POINTS, 0, nTop);
        if (state.show.plate && n > nTop) gl.drawArrays(gl.POINTS, nTop, n - nTop);
      }
      if (state.hud && hudN > 0) {
        gl.bindBuffer(gl.ARRAY_BUFFER, hudP); gl.vertexAttribPointer(pa, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, hudC); gl.vertexAttribPointer(ca, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.LINES, 0, hudN);
      }
      drawLabels(M);
      drawGizmo();
      bind("proj", view.ortho ? "ORTHO" : "PERSP");
      bind("view", view.mode === "cam" ? "TOP CAMERA" : viewName());
      bind("dist", `${Math.round(dist * 1000)} mm`);
    }
    function viewName() {
      const d = (a, b) => Math.abs(a - b) < 0.02;
      if (view.pitch > 1.5) return "TOP";
      if (view.pitch < -1.5) return "BOTTOM";
      if (d(view.pitch, 0) && d(view.yaw, -Math.PI / 2)) return "FRONT";
      if (d(view.pitch, 0) && d(view.yaw, 0)) return "RIGHT";
      return "FREE";
    }
    function drawLabels(M) {
      const g = labels.getContext("2d"), dpr = devicePixelRatio || 1;
      g.clearRect(0, 0, labels.width, labels.height);
      g.font = `600 ${parseFloat(css(panel, "--cloud-label-size", "12")) * dpr}px ${css(panel, "--label-family", "monospace")}`;
      g.fillStyle = css(panel, "--cloud-text", "#ffb27f");
      const put = (p, t) => {
        const x = M[0] * p[0] + M[4] * p[1] + M[8] * p[2] + M[12], y = M[1] * p[0] + M[5] * p[1] + M[9] * p[2] + M[13], w = M[3] * p[0] + M[7] * p[1] + M[11] * p[2] + M[15];
        if (w > 0) g.fillText(t, (x / w * 0.5 + 0.5) * labels.width + 6 * dpr, (0.5 - y / w * 0.5) * labels.height - 6 * dpr);
      };
      if (view.mode !== "cam" && state.show.top) put(top.pos, "D415 TOP");
      if (cams.plate && state.show.plate) put(cams.plate.pos, "D415 PLATE");
      if (state.hud && statusFresh() && status?.real2sim?.ok) {
        const r2sCam = status.real2sim.camera_top;
        const tc = (pt) => r2sToCloud(pt, r2sCam);
        for (const [role, pl] of Object.entries(status.real2sim.plates || {})) {
          const pos = pl.pos_world_m;
          if (!Array.isArray(pos)) continue;
          g.fillStyle = rgbaCss(pl.rgba, 0.92);
          put(tc([pos[0], pos[1], pos[2] + 0.015]), `${role.toUpperCase()} BOWL`);
        }
        const bricks = status.real2sim.bricks || [];
        for (let i = 0; i < bricks.length; i++) {
          const b = bricks[i], pos = b.pos_world_m;
          if (!Array.isArray(pos)) continue;
          g.fillStyle = rgbaCss(b.display_rgba, 0.98);
          const shortPart = (b.part_id || "").replace("duplo_", "");
          put(tc([pos[0], pos[1], pos[2] + 0.022]), `#${i} ${b.color_name || ""} ${shortPart}`);
        }
      }
    }
    function drawGizmo() {
      const g = gizmo.getContext("2d"), W = gizmo.width, H = gizmo.height, dpr = devicePixelRatio || 1;
      g.clearRect(0, 0, W, H);
      const { right, up, back } = basis(), c = [W / 2, H / 2], r = Math.min(W, H) * 0.34, items = [];
      for (const [name, a, v, fb] of [["X", [1, 0, 0], "--gx", "#ff5a5f"], ["Y", [0, 1, 0], "--gy", "#8fd14f"], ["Z", [0, 0, 1], "--gz", "#4d9cff"]])
        for (const s of [1, -1]) {
          const d = mul(a, s);
          items.push({ name, s, color: css(panel, v, fb), x: c[0] + dot(d, right) * r, y: c[1] - dot(d, up) * r, z: dot(d, back) });
        }
      items.sort((a, b) => a.z - b.z);
      gizmo._items = items;
      for (const it of items) {
        if (it.s > 0) { g.strokeStyle = it.color; g.lineWidth = 2 * dpr; g.beginPath(); g.moveTo(...c); g.lineTo(it.x, it.y); g.stroke(); }
        g.beginPath(); g.arc(it.x, it.y, (it.s > 0 ? 9 : 6.5) * dpr, 0, 7);
        g.fillStyle = it.s > 0 ? it.color : "transparent"; g.fill();
        g.strokeStyle = it.color; g.lineWidth = 1.5 * dpr; g.stroke();
        if (it.s > 0) { g.fillStyle = "#000"; g.font = `bold ${10 * dpr}px sans-serif`; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(it.name, it.x, it.y + 0.5); }
      }
    }

    // Blender-like interaction (full screen only)
    function preset(name) {
      if (name === "cam") { view.mode = "cam"; view.ortho = false; return dirty(); }
      leaveCam();
      view.target = [...pivot];
      const P = { top: [-Math.PI / 2, Math.PI / 2 - 1e-4], bottom: [-Math.PI / 2, -Math.PI / 2 + 1e-4], front: [-Math.PI / 2, 0], back: [Math.PI / 2, 0], right: [0, 0], left: [Math.PI, 0] }[name];
      if (!P) return;
      [view.yaw, view.pitch] = P;
      view.dist = 0.85;
      dirty();
    }
    const orbit = (dx, dy) => { leaveCam(); view.yaw -= dx * 0.008; view.pitch = Math.max(-1.5707, Math.min(1.5707, view.pitch + dy * 0.008)); dirty(); };
    const pan = (dx, dy) => {
      leaveCam();
      const { right, up } = basis(), k = (2 * view.dist * Math.tan(fovy / 2)) / canvas.clientHeight;
      view.target = add(view.target, add(mul(right, -dx * k), mul(up, dy * k)));
      dirty();
    };
    const zoom = (f) => { leaveCam(); view.dist = Math.max(0.08, Math.min(3, view.dist * f)); dirty(); };
    let drag = null;
    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) => {
      if (!panel.classList.contains("fs")) return;
      drag = { how: e.shiftKey || e.button === 2 ? "pan" : e.ctrlKey ? "zoom" : "orbit", x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    $$("[data-drag]", panel).forEach((h) => h.addEventListener("pointerdown", (e) => {
      drag = { how: h.dataset.drag, x: e.clientX, y: e.clientY };
      h.setPointerCapture(e.pointerId);
      e.preventDefault();
    }));
    window.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.how === "orbit") orbit(dx, dy); else if (drag.how === "pan") pan(dx, dy); else zoom(Math.exp(dy * 0.01));
    });
    window.addEventListener("pointerup", () => (drag = null));
    canvas.addEventListener("wheel", (e) => { if (!panel.classList.contains("fs")) return; e.preventDefault(); zoom(Math.exp(e.deltaY * 0.0012)); }, { passive: false });
    gizmo.addEventListener("click", (e) => {
      const r = gizmo.getBoundingClientRect(), dpr = devicePixelRatio || 1, x = (e.clientX - r.left) * dpr, y = (e.clientY - r.top) * dpr;
      const hit = (gizmo._items || []).filter((i) => Math.hypot(i.x - x, i.y - y) < 12 * dpr).sort((a, b) => b.z - a.z)[0];
      if (hit) preset({ X1: "right", "X-1": "left", Y1: "back", "Y-1": "front", Z1: "top", "Z-1": "bottom" }[hit.name + hit.s]);
    });
    document.addEventListener("click", (e) => {
      const v = e.target.closest("[data-view]");
      if (v) preset(v.dataset.view);
      if (e.target.closest("[data-proj]")) { leaveCam(); view.ortho = !view.ortho; dirty(); }
    });
    function key(e) {
      if (state.fs !== "depth") return false;
      const map = { 1: e.ctrlKey ? "back" : "front", 3: e.ctrlKey ? "left" : "right", 7: e.ctrlKey ? "bottom" : "top", 0: "cam" };
      if (map[e.key]) { e.preventDefault(); preset(map[e.key]); return true; }
      if (e.key === "5") { leaveCam(); view.ortho = !view.ortho; dirty(); return true; }
      if (e.key === "Home" || e.key === ".") { leaveCam(); view.target = [...pivot]; view.dist = 0.85; dirty(); return true; }
      return false;
    }
    new ResizeObserver(resize).observe(canvas);
    if (hash.view) preset(hash.view);
    if (hash.orbit) { leaveCam(); const [y, p, d] = hash.orbit.split(",").map(Number); Object.assign(view, { yaw: y, pitch: p, dist: d, target: [...pivot] }); }
    return { resize, recolor, key, dirty, setPoints, updateHud, count: () => n, counts: () => ({ top: nTop, plate: n - nTop }), view };
  }

  // ------------------------------------------------------------------ live loops
  async function loadConfig() {
    for (;;) {
      try {
        const r = await fetchT("/api/config", 5000);
        if (r.ok) return await r.json();
        bind("health", "Waiting for sim scene");
      } catch { bind("health", "Server unreachable"); }
      await new Promise((ok) => setTimeout(ok, 1500));
    }
  }
  async function statusLoop() {
    for (;;) {
      try {
        const r = await fetchT("/api/status", 3000);
        if (r.ok) { status = await r.json(); statusAt = performance.now(); }
      } catch { /* statusFresh() turns false after STATUS_STALE_MS */ }
      renderStatus();
      drawOverlay();
      cloud?.updateHud();
      await new Promise((ok) => setTimeout(ok, 1000));
    }
  }
  const cloudState = { seq: 0, updates: 0 };
  async function cloudLoop() {
    for (;;) {
      try {
        const r = await fetchT(`/api/cloud?after=${cloudState.seq}`, 8000);
        if (r.status === 200) {
          cloudState.seq = Number(r.headers.get("X-Cloud-Seq"));
          cloud.setPoints(await r.arrayBuffer());
          cloudState.updates++;
        } else if (r.status !== 204) await new Promise((ok) => setTimeout(ok, 1000));
      } catch { await new Promise((ok) => setTimeout(ok, 1000)); }
    }
  }

  // ------------------------------------------------------------------ icons + clock
  const ICON = {
    fullscreen: '<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/>',
    orbit: '<path d="M20 12a8 8 0 1 1-2.3-5.6"/><path d="M20 4v4.5h-4.5"/>',
    pan: '<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3"/>',
    zoom: '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5M8 10.5h5M10.5 8v5"/>',
  };
  $$("[data-icon]").forEach((e) => e.insertAdjacentHTML("afterbegin",
    `<svg viewBox="0 0 24 24" width="${e.dataset.iconSize || 16}" height="${e.dataset.iconSize || 16}" fill="none" stroke="currentColor" stroke-width="${e.dataset.iconStroke || 1.6}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICON[e.dataset.icon]}</svg>`));
  const tick = () => bind("clock", new Date().toTimeString().slice(0, 8));
  tick();
  setInterval(tick, 1000);

  sync();
  loadConfig().then((c) => {
    config = c;
    frameLoop("sim");
    frameLoop("real");
    cloud = makeCloud();
    window.dashboard = { state, get status() { return status; }, get config() { return config; }, cloudState, cloud, sync, frames, shownFps };
    sync();
    statusLoop();
    if (cloud) cloudLoop();
  });
})();
